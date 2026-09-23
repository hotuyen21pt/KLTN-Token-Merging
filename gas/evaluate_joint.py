# -*- coding: utf-8 -*-
"""Single-stage 3-label ABSA evaluation using the GAS T5 model.

The GAS model predicts all three labels in ONE decoder pass:
    sentence -> "(food, FOOD, positive); (service, SERVICE, negative)"

No second model is needed.  Category, sentiment, and aspect term are all
read from the same generated string.

Metrics reported
----------------
Aspect Term   P/R/F1  -- exact match on extracted aspect strings
Category      P/R/F1  -- (aspect_term, category) pair match
Sentiment     P/R/F1  -- (aspect_term, sentiment) pair match
Joint         P/R/F1  -- all 3 labels correct simultaneously

Usage
-----
    python gas/evaluate_joint.py \\
        --gas-checkpoint checkpoints/gas_t5/best \\
        --data-dir       dataset \\
        --split          test
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from gas.dataset import load_split_records
from gas.metrics import evaluate_all, format_report
from gas.model import GasT5Model


# ─── Core evaluation ─────────────────────────────────────────────────────────

def run_evaluation(
    model:     GasT5Model,
    records:   List[Dict],
    normalize: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Run single-stage GAS inference and compute all 4 metrics.

    Parameters
    ----------
    model     : trained GasT5Model
    records   : list of main-data record dicts (from ``load_split_records``)
    normalize : apply Levenshtein n-gram normalization to aspect terms

    Returns
    -------
    Dict with keys ``aspect_term``, ``category``, ``sentiment``, ``joint``.
    """
    pred_triples_list, gold_triples_list = model.predict_for_records(
        records, normalize=normalize
    )
    return evaluate_all(pred_triples_list, gold_triples_list)


# ─── Results I/O ─────────────────────────────────────────────────────────────

def save_json(results: Dict, path: Path, meta: Dict = None) -> None:
    payload = {"metrics": results}
    if meta:
        payload["meta"] = meta
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"JSON saved -> {path}")


def save_csv(results: Dict, path: Path) -> None:
    rows = [
        {
            "label":     label,
            "precision": round(m["precision"], 6),
            "recall":    round(m["recall"],    6),
            "f1":        round(m["f1"],        6),
            "tp":        int(m["tp"]),
            "fp":        int(m["fp"]),
            "fn":        int(m["fn"]),
        }
        for label, m in results.items()
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["label", "precision", "recall", "f1", "tp", "fp", "fn"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved  -> {path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Single-stage 3-label ABSA evaluation (GAS T5 model). "
            "Reports P/R/F1 for aspect term, category, sentiment, and joint."
        )
    )
    parser.add_argument(
        "--gas-checkpoint",
        required=True,
        help=(
            "Path to trained GAS T5 checkpoint directory "
            "(e.g. checkpoints/gas_t5/best)."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(ROOT / "dataset"),
        help="Directory containing train.apc / dev.apc / test.apc",
    )
    parser.add_argument(
        "--split",
        choices=["train", "dev", "test"],
        default="test",
        help="Dataset split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        default=False,
        help="Disable Levenshtein n-gram normalization on aspect terms",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Where to save results JSON + CSV. "
            "Defaults to the checkpoint directory."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    gas_ckpt = Path(args.gas_checkpoint)
    if not gas_ckpt.is_dir():
        print(
            f"[error] Checkpoint not found: {gas_ckpt}\n"
            "  Train the GAS model first:\n"
            "    python gas/train_gas.py --output-dir checkpoints/gas_t5 ...",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading GAS model from: {gas_ckpt}")
    model = GasT5Model.from_pretrained(str(gas_ckpt), device=device)

    # Load evaluation data (main records only, no supplement)
    print(f"Loading {args.split} data from: {args.data_dir}")
    main_records = load_split_records(args.split, args.data_dir)
    print(f"  {len(main_records)} sentences")

    # Run evaluation
    print(f"\nRunning evaluation on {args.split} split ...")
    results = run_evaluation(model, main_records, normalize=not args.no_normalize)

    # Print
    W = 72
    print(f"\n{'=' * W}")
    print("3-LABEL ABSA EVALUATION  (single-stage GAS T5)")
    print(f"  Split      : {args.split}")
    print(f"  Checkpoint : {gas_ckpt}")
    print(f"  Sentences  : {len(main_records)}")
    print(f"{'=' * W}")
    print(format_report(results))
    print("Definitions:")
    print("  Aspect Term  -- exact match on aspect string")
    print("  Category     -- (aspect_term, category) both correct")
    print("  Sentiment    -- (aspect_term, sentiment) both correct")
    print("  Joint        -- all 3 labels simultaneously correct")
    print(f"{'=' * W}\n")

    # Save
    out_dir = Path(args.output_dir) if args.output_dir else gas_ckpt.parent
    meta = {
        "split":          args.split,
        "gas_checkpoint": str(gas_ckpt),
        "n_sentences":    len(main_records),
        "normalize":      not args.no_normalize,
    }
    save_json(results, out_dir / f"eval_{args.split}.json", meta)
    save_csv( results, out_dir / f"eval_{args.split}.csv")


if __name__ == "__main__":
    main()
