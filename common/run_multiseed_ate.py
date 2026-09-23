# -*- coding: utf-8 -*-
"""Multi-seed ATE training runner for GAS (Generative Aspect Sentiment) T5.

Trains the ATE stage with N seeds.  For each seed:
  - Trains T5-GAS on train.apc / dev.apc (early-stop on dev F1)
  - Saves best checkpoint to checkpoints/gas_t5_ate/seed_<N>/best/
  - Runs inference on test.apc → saves predictions to
      runs_ate/seed_<N>/test_predictions.csv   (sentence, predicted_term, gold_terms)
  - Records P/R/F1 per seed
After all seeds, prints mean±std and saves:
  runs_ate/results_ate_multiseed.csv   ← per-seed metrics
  runs_ate/results_ate_summary.txt     ← aggregated mean±std

The per-seed CSV (sentence, predicted_term, gold_terms) is the input expected by
common/run_multiseed.py when using --ate-csv-dir runs_ate/ for seed-paired e2e evaluation.

Usage (from repo root):
  python common/run_multiseed_ate.py                          # seeds [42, 123, 456]
  python common/run_multiseed_ate.py --seeds 42 123 456 789 2024
  python common/run_multiseed_ate.py --resume                 # skip finished seeds
  python common/run_multiseed_ate.py --epochs 20 --lr 3e-4   # ATE-specific hyperparams
  python common/run_multiseed_ate.py --model-name t5-base
  python common/run_multiseed_ate.py --no-train               # aggregate only
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
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
from src.inference import predict_aspects_for_records
from src.metrics import evaluate_exact_match

# ─── Paths ────────────────────────────────────────────────────────────────────

DATASET_DIR = ROOT / "dataset"
RUNS_ATE    = ROOT / "runs_ate"
CKPT_BASE   = ROOT / "checkpoints" / "gas_t5_ate"

DEFAULT_SEEDS   = [42, 123, 456]
DEFAULT_MODEL   = "t5-base"
DEFAULT_EPOCHS  = 20
DEFAULT_LR      = 3e-4
DEFAULT_BATCH   = 16
DEFAULT_MAX_IN  = 128
DEFAULT_MAX_TGT = 64

# ─── Seed setup ───────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── One-seed ATE training ────────────────────────────────────────────────────

def train_one_seed_ate(
    seed: int,
    model_name: str,
    epochs: int,
    lr: float,
    batch_size: int,
    max_input_length: int,
    max_target_length: int,
    data_dir: str,
    ckpt_dir: Path,
    pred_csv: Path,
    skip_if_exists: bool = False,
) -> Optional[Dict]:
    """Train ATE for one seed; return metrics dict."""
    meta_path = ckpt_dir / "meta.json"
    if skip_if_exists and pred_csv.is_file() and meta_path.is_file():
        print(f"    [resume] seed={seed} — loading cached metrics")
        return json.loads(meta_path.read_text(encoding="utf-8")).get("metrics")

    set_seed(seed)
    print(f"\n  Seed {seed} — training {model_name} for {epochs} epochs")

    train_loader, dev_loader, test_loader, tokenizer = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
        num_workers=0,
    )
    dev_records  = get_raw_split_records("dev",  data_dir)
    test_records = get_raw_split_records("test", data_dir)

    model = T5AspectExtractor(model_name=model_name)
    if tokenizer is not None:
        model.tokenizer = tokenizer

    trainer = ATETrainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        learning_rate=lr,
        num_epochs=epochs,
        output_dir=str(ckpt_dir),
        dev_records=dev_records,
        test_records=test_records,
    )
    result = trainer.train()

    tm = result.get("test_metrics", {})
    metrics = {
        "seed":            seed,
        "model_name":      model_name,
        "epochs":          epochs,
        "best_dev_f1":     round(float(result.get("best_dev_f1", 0.0)), 4),
        "test_precision":  round(float(tm.get("precision", 0.0)), 4),
        "test_recall":     round(float(tm.get("recall", 0.0)), 4),
        "test_f1":         round(float(tm.get("f1", 0.0)), 4),
        "wall_time_sec":   result.get("wall_time_sec", 0.0),
        "best_checkpoint": result.get("best_checkpoint", ""),
    }
    print(f"    Test P/R/F1 = {metrics['test_precision']:.4f} / "
          f"{metrics['test_recall']:.4f} / {metrics['test_f1']:.4f}")

    # ── Load best checkpoint and generate test predictions ─────────────────────
    best_ckpt = result.get("best_checkpoint") or str(ckpt_dir / "last")
    if Path(best_ckpt).is_dir():
        eval_model = T5AspectExtractor.from_pretrained(
            best_ckpt, device=model.device
        )
    else:
        eval_model = model
        print("    [warn] best checkpoint not found — using last epoch model")

    preds, golds = predict_aspects_for_records(
        eval_model, test_records,
        max_input_length=max_input_length,
    )
    # Verify metrics on the actual prediction output
    final_metrics = evaluate_exact_match(preds, golds)
    metrics["test_precision"] = round(float(final_metrics.get("precision", 0.0)), 4)
    metrics["test_recall"]    = round(float(final_metrics.get("recall", 0.0)), 4)
    metrics["test_f1"]        = round(float(final_metrics.get("f1", 0.0)), 4)

    # ── Save per-seed predictions CSV ─────────────────────────────────────────
    pred_csv.parent.mkdir(parents=True, exist_ok=True)
    rows_to_write = []
    for record, pred_terms, gold_terms in zip(test_records, preds, golds):
        sentence = str(record["input_text"])
        gold_str = "|".join(gold_terms)
        if pred_terms:
            for pt in pred_terms:
                rows_to_write.append({
                    "sentence":       sentence,
                    "predicted_term": pt,
                    "gold_terms":     gold_str,
                })
        else:
            # No prediction → empty row so the sentence still appears in e2e eval
            rows_to_write.append({
                "sentence":       sentence,
                "predicted_term": "",
                "gold_terms":     gold_str,
            })
    with open(pred_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sentence", "predicted_term", "gold_terms"])
        w.writeheader()
        w.writerows(rows_to_write)
    print(f"    Predictions → {pred_csv}  ({len(rows_to_write)} rows)")

    # ── Save per-seed meta ─────────────────────────────────────────────────────
    meta = {
        "seed": seed, "model_name": model_name,
        "epochs": epochs, "lr": lr, "batch_size": batch_size,
        "metrics": metrics,
        "pred_csv": str(pred_csv),
        "best_checkpoint": best_ckpt,
    }
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metrics


# ─── Aggregation ──────────────────────────────────────────────────────────────

def aggregate_ate(rows: List[Dict]) -> Dict:
    """Compute mean±std across seeds."""
    seeds = [r["seed"] for r in rows]
    for col in ["test_precision", "test_recall", "test_f1",
                "best_dev_f1", "wall_time_sec"]:
        vals = np.array([float(r.get(col, 0.0)) for r in rows])
        print(f"  {col:20s}: {vals.mean():.4f} ± {vals.std(ddof=0):.4f}")
    return {
        "n_seeds": len(rows),
        "seeds":   seeds,
        **{f"{col}_mean": round(float(np.array([float(r.get(col, 0.0)) for r in rows]).mean()), 4)
           for col in ["test_precision", "test_recall", "test_f1", "best_dev_f1"]},
        **{f"{col}_std":  round(float(np.array([float(r.get(col, 0.0)) for r in rows]).std(ddof=0)), 4)
           for col in ["test_precision", "test_recall", "test_f1", "best_dev_f1"]},
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-seed ATE training with GAS T5")
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--model-name", default=DEFAULT_MODEL)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--lr", type=float, default=DEFAULT_LR,
                   help=f"Learning rate (default: {DEFAULT_LR})")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    p.add_argument("--max-input-length", type=int, default=DEFAULT_MAX_IN)
    p.add_argument("--max-target-length", type=int, default=DEFAULT_MAX_TGT)
    p.add_argument("--data-dir", default=str(DATASET_DIR))
    p.add_argument("--runs-ate-dir", default=str(RUNS_ATE),
                   help="Output dir for per-seed prediction CSVs")
    p.add_argument("--ckpt-dir", default=str(CKPT_BASE),
                   help="Base checkpoint dir (seed subdirs created here)")
    p.add_argument("--resume", action="store_true",
                   help="Skip seeds that already have pred CSV + meta.json")
    p.add_argument("--no-train", action="store_true",
                   help="Only aggregate existing per-seed results")
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    runs_ate = Path(args.runs_ate_dir)
    ckpt_base = Path(args.ckpt_dir)
    runs_ate.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Model      : {args.model_name}")
    print(f"Seeds      : {args.seeds}")
    print(f"Epochs     : {args.epochs}   LR: {args.lr}   Batch: {args.batch_size}")
    print(f"ATE CSVs → : {runs_ate}/seed_<N>/test_predictions.csv")
    print(f"Checkpoints: {ckpt_base}/seed_<N>/")
    print()

    per_seed_metrics: List[Dict] = []

    for i, seed in enumerate(args.seeds, 1):
        print(f"[{i}/{len(args.seeds)}] Seed {seed}")
        ckpt_dir = ckpt_base / f"seed_{seed}"
        pred_csv = runs_ate / f"seed_{seed}" / "test_predictions.csv"

        if args.no_train:
            meta_path = ckpt_dir / "meta.json"
            if meta_path.is_file():
                m = json.loads(meta_path.read_text(encoding="utf-8")).get("metrics")
                if m:
                    per_seed_metrics.append(m)
                    print(f"  [loaded] P/R/F1 = {m['test_precision']:.4f} / "
                          f"{m['test_recall']:.4f} / {m['test_f1']:.4f}")
            else:
                print(f"  [skip] No meta.json at {meta_path}")
            continue

        m = train_one_seed_ate(
            seed=seed,
            model_name=args.model_name,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            max_input_length=args.max_input_length,
            max_target_length=args.max_target_length,
            data_dir=args.data_dir,
            ckpt_dir=ckpt_dir,
            pred_csv=pred_csv,
            skip_if_exists=args.resume,
        )
        if m:
            per_seed_metrics.append(m)

    if not per_seed_metrics:
        print("[warn] No results to aggregate.")
        return

    # ── Aggregate ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ATE Multi-seed Aggregation")
    print("=" * 60)
    agg = aggregate_ate(per_seed_metrics)

    # ── Save per-seed CSV ─────────────────────────────────────────────────────
    raw_csv = runs_ate / "results_ate_multiseed.csv"
    with open(raw_csv, "w", newline="", encoding="utf-8") as f:
        cols = ["seed", "model_name", "epochs", "best_dev_f1",
                "test_precision", "test_recall", "test_f1", "wall_time_sec",
                "best_checkpoint"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(per_seed_metrics)
    print(f"\nPer-seed metrics → {raw_csv}")

    # ── Save summary ──────────────────────────────────────────────────────────
    summ_path = runs_ate / "results_ate_summary.txt"
    with open(summ_path, "w", encoding="utf-8") as f:
        print("ATE Multi-seed Summary", file=f)
        print(f"Model: {args.model_name}  Epochs: {args.epochs}  LR: {args.lr}", file=f)
        print(f"Seeds: {[m['seed'] for m in per_seed_metrics]}", file=f)
        print("-" * 50, file=f)
        print(f"  Precision : {agg['test_precision_mean']:.4f} ± {agg['test_precision_std']:.4f}", file=f)
        print(f"  Recall    : {agg['test_recall_mean']:.4f} ± {agg['test_recall_std']:.4f}", file=f)
        print(f"  F1        : {agg['test_f1_mean']:.4f} ± {agg['test_f1_std']:.4f}", file=f)
        print(f"  Dev F1    : {agg['best_dev_f1_mean']:.4f} ± {agg['best_dev_f1_std']:.4f}", file=f)
        print("-" * 50, file=f)
        for m in per_seed_metrics:
            print(f"  seed={m['seed']}  P={m['test_precision']:.4f}  "
                  f"R={m['test_recall']:.4f}  F1={m['test_f1']:.4f}", file=f)
        print("\nPer-seed prediction CSVs:", file=f)
        for m in per_seed_metrics:
            csv_p = runs_ate / f"seed_{m['seed']}" / "test_predictions.csv"
            print(f"  seed={m['seed']} → {csv_p}", file=f)

    print(f"Summary       → {summ_path}")
    print(f"\n{'=' * 60}")
    print(f"ATE F1: {agg['test_f1_mean']*100:.2f}% ± {agg['test_f1_std']*100:.2f}%"
          f"  (n={agg['n_seeds']} seeds)")
    print(f"  Precision: {agg['test_precision_mean']*100:.2f}%"
          f"  Recall: {agg['test_recall_mean']*100:.2f}%")
    print(f"\nNext step:")
    print(f"  python common/run_multiseed.py --ate-csv-dir {runs_ate}")
    print(f"  → e2e eval uses seed-paired ATE predictions for each APC run")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
