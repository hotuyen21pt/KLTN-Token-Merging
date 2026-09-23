# -*- coding: utf-8 -*-
"""CLI entry point for generative T5 Aspect Term Extraction training."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.dataset import (
    DEFAULT_DATA_DIR,
    create_dataloaders,
    get_raw_split_records,
)
from src.model import T5AspectExtractor
from src.trainer import ATETrainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train generative T5 Aspect Term Extraction (GAS extraction-style).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing train.apc / dev.apc / test.apc",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "checkpoints" / "gas_t5_ate"),
        help="Checkpoint output directory",
    )
    parser.add_argument("--model-name", type=str, default="t5-base")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-input-length", type=int, default=128)
    parser.add_argument("--max-target-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print(f"Data dir   : {args.data_dir}")
    print(f"Output dir : {args.output_dir}")
    print(f"Model      : {args.model_name}")
    print(f"Device     : {'cuda' if torch.cuda.is_available() else 'cpu'}")

    train_loader, dev_loader, test_loader, tokenizer = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        num_workers=args.num_workers,
    )

    dev_records = get_raw_split_records("dev", args.data_dir)
    test_records = get_raw_split_records("test", args.data_dir)

    model = T5AspectExtractor(model_name=args.model_name)
    if tokenizer is not None:
        model.tokenizer = tokenizer

    trainer = ATETrainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        output_dir=args.output_dir,
        dev_records=dev_records,
        test_records=test_records,
    )
    result = trainer.train()

    print("\n=== Training complete ===")
    print(f"Best dev F1      : {result['best_dev_f1']:.4f}")
    print(f"Best checkpoint  : {result['best_checkpoint']}")
    if result.get("test_metrics"):
        tm = result["test_metrics"]
        print(f"Test P/R/F1      : {tm['precision']:.4f} / {tm['recall']:.4f} / {tm['f1']:.4f}")
    print(f"Wall time (sec)  : {result['wall_time_sec']}")


if __name__ == "__main__":
    main()
