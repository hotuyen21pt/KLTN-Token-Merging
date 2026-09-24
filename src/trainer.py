# -*- coding: utf-8 -*-
"""Training loop for generative T5 Aspect Term Extraction."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.inference import predict_aspects_for_records
from src.metrics import evaluate_exact_match, format_metrics
from src.model import T5AspectExtractor


def pick_amp_dtype() -> Optional[torch.dtype]:
    """Chọn dtype cho autocast, hoặc None nghĩa là chạy fp32.

    T5/mT5 được pre-train ở bfloat16 và activations của chúng thường vượt dải
    biểu diễn của fp16 (tối đa 65504) -> tràn thành inf -> loss = NaN ngay từ
    batch đầu tiên. Vì vậy chỉ bật autocast khi GPU hỗ trợ bf16 (Ampere trở
    lên). Trên T4 / P100 (compute 7.5 và 6.0) thì buộc phải chạy fp32.
    """
    if not torch.cuda.is_available():
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None


class ATETrainer:
    """Fine-tune T5ForConditionalGeneration with decoder CrossEntropyLoss."""

    def __init__(
        self,
        model: T5AspectExtractor,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        *,
        learning_rate: float = 3e-4,
        num_epochs: int = 20,
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        patience: int = 4,
        output_dir: Optional[str | Path] = None,
        dev_records: Optional[Sequence[Dict[str, object]]] = None,
        test_records: Optional[Sequence[Dict[str, object]]] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.max_grad_norm = max_grad_norm
        self.patience = patience
        self.output_dir = Path(output_dir) if output_dir else None
        self.dev_records = list(dev_records) if dev_records is not None else None
        self.test_records = list(test_records) if test_records is not None else None

        self.device = model.device
        self.optimizer = torch.optim.AdamW(
            model.model.parameters(),
            lr=learning_rate,
        )
        total_steps = max(1, len(train_loader) * num_epochs)
        warmup_steps = int(total_steps * warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.history: List[Dict[str, object]] = []
        self.best_dev_f1 = -1.0
        self.best_checkpoint_dir: Optional[Path] = None
        self.amp_dtype = pick_amp_dtype()
        self.use_amp = self.amp_dtype is not None
        # GradScaler chỉ cần thiết cho fp16; bf16 có cùng dải số mũ với fp32.
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(self.amp_dtype is torch.float16)
        )
        print(f"[AMP] dtype = {self.amp_dtype or 'fp32 (tắt autocast)'}")

    def _train_epoch(self, epoch: int) -> float:
        self.model.model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.model.parameters(),
                self.max_grad_norm,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            running_loss += float(loss.item())
            n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)
        print(f"[epoch {epoch + 1}/{self.num_epochs}] train_loss={avg_loss:.4f}")
        return avg_loss

    def evaluate_records(
        self,
        records: Sequence[Dict[str, object]],
        split_name: str = "dev",
    ) -> Dict[str, float]:
        preds, golds = predict_aspects_for_records(self.model, records)
        metrics = evaluate_exact_match(preds, golds)
        print(f"[{split_name}] {format_metrics(metrics)}")
        return metrics

    def train(self, *, eval_test: bool = True) -> Dict[str, object]:
        """Huấn luyện ATE.

        ``eval_test=False`` bỏ qua vòng đánh giá test ở cuối — dùng khi phía
        gọi (run_multiseed_ate) tự đánh giá lại trên best checkpoint, tránh
        chạy hai lần trên cùng tập test.
        """
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        no_improve = 0
        for epoch in range(self.num_epochs):
            train_loss = self._train_epoch(epoch)

            dev_metrics: Dict[str, float] = {}
            if self.dev_records is not None:
                dev_metrics = self.evaluate_records(self.dev_records, "dev")
            elif self.dev_loader is not None:
                # Fallback: no raw records supplied; skip generative eval.
                dev_metrics = {}

            epoch_info = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "dev": dev_metrics,
            }
            self.history.append(epoch_info)

            # Chỉ lưu khi thật sự tốt hơn. Trước đây dùng >= nên khi dev F1
            # đứng yên (kể cả 0.0) vẫn ghi lại toàn bộ model mỗi epoch.
            dev_f1 = float(dev_metrics.get("f1", -1.0))
            if dev_f1 > self.best_dev_f1 + 1e-6:
                self.best_dev_f1 = dev_f1
                no_improve = 0
                if self.output_dir is not None:
                    best_dir = self.output_dir / "best"
                    self.model.save(str(best_dir))
                    self.best_checkpoint_dir = best_dir
                    print(f"Saved best checkpoint to {best_dir} (dev F1={dev_f1:.4f})")
            else:
                no_improve += 1
                if self.patience > 0 and no_improve >= self.patience:
                    print(f"Early stopping ở epoch {epoch + 1} "
                          f"(dev F1 không cải thiện {no_improve} epoch liên tiếp, "
                          f"tốt nhất = {self.best_dev_f1:.4f})")
                    break

        if self.output_dir is not None:
            final_dir = self.output_dir / "last"
            self.model.save(str(final_dir))
            print(f"Saved final checkpoint to {final_dir}")

        test_metrics: Dict[str, float] = {}
        if eval_test and self.test_records is not None:
            ckpt = self.best_checkpoint_dir or (self.output_dir / "last" if self.output_dir else None)
            if ckpt is not None and ckpt.is_dir():
                eval_model = T5AspectExtractor.from_pretrained(str(ckpt), device=self.device)
            else:
                eval_model = self.model
            preds, golds = predict_aspects_for_records(eval_model, self.test_records)
            test_metrics = evaluate_exact_match(preds, golds)
            print(f"[test] {format_metrics(test_metrics)}")

        elapsed = time.perf_counter() - t0
        return {
            "history": self.history,
            "best_dev_f1": self.best_dev_f1,
            "best_checkpoint": str(self.best_checkpoint_dir) if self.best_checkpoint_dir else None,
            "test_metrics": test_metrics,
            "wall_time_sec": round(elapsed, 3),
        }
