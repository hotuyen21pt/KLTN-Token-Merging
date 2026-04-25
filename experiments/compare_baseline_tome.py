# -*- coding: utf-8 -*-
"""
Train baseline vs ToMe with the same seed and print Acc, F1, wall-clock time.

Metrics come from ``config.max_test_metrics`` (best Acc/F1 logged during APC
training when the instructor evaluates on the test loader each ``log_step``).

Run from repo root:
  python thesis_apc_baseline/experiments/compare_baseline_tome.py

Optional:
  python thesis_apc_baseline/experiments/compare_baseline_tome.py --seed 42 --num-epoch 3 --data-fraction 1.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thesis_apc_baseline.experiments.experiment_common import run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Same seed for both runs for fair comparison.",
    )
    parser.add_argument("--num-epoch", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--data-fraction",
        type=float,
        default=0.2,
        help="Use this fraction of each local train/test *.apc (1.0 = full). Ignored for built-in PyABSA benchmarks.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Optional APC folder (triple-line $T$ format). Default: thesis_apc_baseline/dataset if train files exist.",
    )
    args = parser.parse_args()

    kw = dict(
        seed=args.seed,
        num_epoch=args.num_epoch,
        batch_size=args.batch_size,
        patience=args.patience,
        dataset_dir=args.dataset_dir,
        data_fraction=args.data_fraction,
    )

    baseline = run_training(False, **kw)
    tome = run_training(True, **kw)

    rows = [
        ("Baseline FAST_LCF_BERT", baseline),
        ("FAST_LCF_BERT_TOME", tome),
    ]

    print(
        "\n=== APC comparison (same seed=%d, num_epoch=%d) ===\n"
        % (args.seed, args.num_epoch)
    )
    print(f"{'Run':<28} {'Acc':>10} {'F1':>10} {'Time (s)':>12}")
    print("-" * 62)
    for name, r in rows:
        print(
            f"{name:<28} {r['max_apc_test_acc']*100:>9.2f}% {r['max_apc_test_f1']*100:>9.2f}% {r['wall_time_sec']:>12.1f}"
        )

    delta_t = tome["wall_time_sec"] - baseline["wall_time_sec"]
    print("\nTime delta (ToMe − Baseline): %.1f s" % delta_t)

    out_dir = ROOT / "thesis_apc_baseline" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "compare_baseline_tome.json"
    payload = {
        "seed": args.seed,
        "num_epoch": args.num_epoch,
        "batch_size": args.batch_size,
        "data_fraction": args.data_fraction,
        "baseline": baseline,
        "tome": tome,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nSaved: %s" % out_path)


if __name__ == "__main__":
    main()
