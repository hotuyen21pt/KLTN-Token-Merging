# -*- coding: utf-8 -*-
"""Run T5 ATE inference on the test split and save predictions to CSV.

Gold labels come from dataset/test_sentences_id.csv (one aspect_term per row).
A sentence is counted correct if the gold term appears anywhere in the
predicted term list (after normalisation: strip, lower, remove dots).

Output (runs_ate/test_ate_predictions.csv):
    sentence       - full sentence text
    predicted_term - one predicted aspect term per row
                     (multiple rows if sentence has multiple predicted terms)
    gold_terms     - gold aspect term (from test_sentences_id)

Usage (from thesis_apc_baseline/ directory):
    python experiments/run_ate_inference.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import T5AspectExtractor
from src.inference import predict_aspects

# ─── Config ───────────────────────────────────────────────────────────────────

ATE_CKPT  = ROOT / "checkpoints/gas_t5_ate/best"
GOLD_CSV  = ROOT / "dataset" / "test_sentences_id.csv"
OUT_DIR   = ROOT / "runs_ate"
OUT_CSV   = OUT_DIR / "test_ate_predictions.csv"

MAX_INPUT_LENGTH = 128
NORMALIZE        = True

# ─── Helpers ──────────────────────────────────────────────────────────────────

def normalize_term(term: str) -> str:
    return term.strip().lower().replace(".", "")


def print_metrics(label: str, stats: List[Tuple[int, int, int]]) -> None:
    """Print micro and macro P/R/F1 from per-instance (tp, fp, fn) tuples."""
    total_tp = sum(s[0] for s in stats)
    total_fp = sum(s[1] for s in stats)
    total_fn = sum(s[2] for s in stats)

    micro_p  = total_tp / max(total_tp + total_fp, 1)
    micro_r  = total_tp / max(total_tp + total_fn, 1)
    micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-9)

    per_f1 = []
    per_p  = []
    per_r  = []
    for tp_i, fp_i, fn_i in stats:
        p_i  = tp_i / max(tp_i + fp_i, 1) if (tp_i + fp_i) > 0 else 0.0
        r_i  = tp_i / max(tp_i + fn_i, 1) if (tp_i + fn_i) > 0 else 0.0
        f1_i = 2 * p_i * r_i / max(p_i + r_i, 1e-9) if (p_i + r_i) > 0 else 0.0
        per_p.append(p_i)
        per_r.append(r_i)
        per_f1.append(f1_i)

    macro_p  = sum(per_p)  / max(len(per_p),  1)
    macro_r  = sum(per_r)  / max(len(per_r),  1)
    macro_f1 = sum(per_f1) / max(len(per_f1), 1)

    n = len(stats)
    print(f"\n{label} ({n} instances):")
    print(f"  Micro  P: {micro_p*100:.2f}%   R: {micro_r*100:.2f}%   F1: {micro_f1*100:.2f}%")
    print(f"  Macro  P: {macro_p*100:.2f}%   R: {macro_r*100:.2f}%   F1: {macro_f1*100:.2f}%")


def load_gold(path: Path) -> List[Tuple[str, str]]:
    """Return list of (sentence, aspect_term) in id order."""
    records: List[Tuple[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            records.append((row["sentence"].strip(), row["aspect_term"].strip()))
    return records


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    if not ATE_CKPT.is_dir():
        raise FileNotFoundError(f"ATE checkpoint not found: {ATE_CKPT}")
    if not GOLD_CSV.is_file():
        raise FileNotFoundError(f"Gold file not found: {GOLD_CSV}")

    print(f"Loading ATE model from {ATE_CKPT} …")
    ate_model = T5AspectExtractor.from_pretrained(str(ATE_CKPT))
    print(f"  Device: {ate_model.device}")

    print(f"Loading gold labels from {GOLD_CSV} …")
    gold_records = load_gold(GOLD_CSV)
    print(f"  {len(gold_records)} records")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows:  List[Dict[str, str]]       = []
    stats: List[Tuple[int, int, int]] = []   # (tp_i, fp_i, fn_i) per sentence

    for i, (sentence, gold_term) in enumerate(gold_records, 1):
        norm_gold = normalize_term(gold_term)

        pred_terms = predict_aspects(
            ate_model,
            sentence,
            max_input_length=MAX_INPUT_LENGTH,
            normalize=NORMALIZE,
        )

        pred_set_norm = set(normalize_term(t) for t in pred_terms)
        tp_i = 1 if norm_gold in pred_set_norm else 0
        fp_i = len(pred_set_norm) - tp_i
        fn_i = 1 - tp_i
        stats.append((tp_i, fp_i, fn_i))

        if tp_i:
            chosen_term = next(t for t in pred_terms if normalize_term(t) == norm_gold)
        else:
            chosen_term = pred_terms[0] if pred_terms else ""

        rows.append({
            "sentence":       sentence,
            "predicted_term": chosen_term,
            "gold_terms":     gold_term,
        })

        if i % 50 == 0 or i == len(gold_records):
            print(f"  [{i}/{len(gold_records)}] sentence processed")

    # ── Write CSV ──────────────────────────────────────────────────────────────
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sentence", "predicted_term", "gold_terms"])
        writer.writeheader()
        writer.writerows(rows)

    # ── Metrics ───────────────────────────────────────────────────────────────
    print_metrics("ATE results on test set (gold: test_sentences_id.csv)", stats)
    print(f"\nSaved {len(rows)} rows → {OUT_CSV}")


if __name__ == "__main__":
    main()
