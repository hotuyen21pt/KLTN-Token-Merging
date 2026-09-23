# -*- coding: utf-8 -*-
"""Run T5 ATE inference on results.csv units and evaluate against test_sentences_id.csv.

Input:
    results.csv               (entity_id, sentence, unit)
    dataset/test_sentences_id.csv (id, sentence, aspect_term, ...)

Algorithm (per entity_id, processed in test_sentences_id.csv id order):
    1. For each unit in the entity_id group, run ATE inference on the unit text.
    2. If any unit produces a predicted_term that matches the gold aspect_term,
       keep that record (sentence, unit, predicted_term).
    3. Otherwise keep the first unit's record.
    4. Evaluation: entity_id is correct if any unit matched the gold term.

Output (runs_ate/results_ate_predictions.csv):
    entity_id      - group identifier (= test_id)
    sentence       - full sentence text
    unit           - selected unit (best-matching or first)
    predicted_term - predicted aspect term for the selected unit
    gold_term      - gold aspect term from test_sentences_id
    correct        - 1 if any unit matched gold, 0 otherwise

Usage (from thesis_apc_baseline/ directory):
    python experiments/run_ate_inference_from_results.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import T5AspectExtractor
from src.inference import predict_aspects

# ─── Config ───────────────────────────────────────────────────────────────────

ATE_CKPT    = ROOT / "checkpoints/gas_t5_ate/best"
RESULTS_CSV = ROOT / "results.csv"
GOLD_CSV    = ROOT / "dataset" / "test_sentences_id.csv"
OUT_DIR     = ROOT / "runs_ate"
OUT_CSV     = OUT_DIR / "results_ate_predictions.csv"

MAX_INPUT_LENGTH = 128
NORMALIZE        = True

# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_results(path: Path) -> Dict[int, List[Dict[str, str]]]:
    """Load results.csv, return {entity_id: [{sentence, unit}, ...]} in insertion order."""
    groups: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            eid = int(row["entity_id"])
            groups[eid].append({"sentence": row["sentence"], "unit": row["unit"]})
    return groups


def load_gold(path: Path) -> Dict[int, Tuple[str, str]]:
    """Load test_sentences_id.csv, return {id: (sentence, aspect_term)}."""
    gold: Dict[int, Tuple[str, str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gold[int(row["id"])] = (row["sentence"].strip(), row["aspect_term"].strip())
    return gold


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


def find_best_match(
    pred_terms: List[str],
    gold_term: str,
) -> Optional[str]:
    """Return the first predicted term that matches gold (after normalization), or None."""
    norm_gold = normalize_term(gold_term)
    for t in pred_terms:
        if normalize_term(t) == norm_gold:
            return t
    return None


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    if not ATE_CKPT.is_dir():
        raise FileNotFoundError(f"ATE checkpoint not found: {ATE_CKPT}")
    if not RESULTS_CSV.is_file():
        raise FileNotFoundError(f"Results file not found: {RESULTS_CSV}")
    if not GOLD_CSV.is_file():
        raise FileNotFoundError(f"Gold file not found: {GOLD_CSV}")

    print(f"Loading ATE model from {ATE_CKPT} …")
    ate_model = T5AspectExtractor.from_pretrained(str(ATE_CKPT))
    print(f"  Device: {ate_model.device}")

    print(f"Loading results from {RESULTS_CSV} …")
    groups = load_results(RESULTS_CSV)
    print(f"  {len(groups)} entity_ids")

    print(f"Loading gold labels from {GOLD_CSV} …")
    gold_by_id = load_gold(GOLD_CSV)
    print(f"  {len(gold_by_id)} gold records")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total   = 0
    correct = 0
    output_rows: List[Dict[str, str]]       = []
    stats:       List[Tuple[int, int, int]] = []   # (tp_i, fp_i, fn_i) per entity_id

    # Iterate in test_sentences_id order (test_id = entity_id + 1)
    for test_id in sorted(gold_by_id.keys()):
        entity_id = test_id 
        gold_sentence, gold_term = gold_by_id[test_id]

        if entity_id not in groups:
            print(f"  [WARN] entity_id={entity_id} (test_id={test_id}) not in results.csv — skipping")
            continue

        records = groups[entity_id]
        sentence = records[0]["sentence"]

        chosen_unit: str = records[0]["unit"]
        chosen_term: str = ""
        is_correct = False

        # Predict for each unit; cache first unit's result for fallback
        first_pred_term: Optional[str] = None

        for rec in records:
            unit = rec["unit"]
            pred_terms = predict_aspects(
                ate_model,
                unit,
                max_input_length=MAX_INPUT_LENGTH,
                normalize=NORMALIZE,
            )
            pred_term_first = pred_terms[0] if pred_terms else ""

            if first_pred_term is None:
                # Save first unit's prediction as fallback
                first_pred_term = pred_term_first

            match = find_best_match(pred_terms, gold_term)
            if match is not None:
                chosen_unit = unit
                chosen_term = match
                is_correct  = True
                break

        if not is_correct:
            # No unit matched — fall back to first unit
            chosen_unit = records[0]["unit"]
            chosen_term = first_pred_term if first_pred_term is not None else ""

        total += 1
        if is_correct:
            correct += 1

        # per-entity TP/FP/FN: one gold, one chosen prediction
        tp_i = 1 if is_correct else 0
        fp_i = 0 if is_correct else (1 if chosen_term else 0)
        fn_i = 1 - tp_i
        stats.append((tp_i, fp_i, fn_i))

        output_rows.append({
            "entity_id":     str(entity_id),
            "sentence":      sentence,
            "unit":          chosen_unit,
            "predicted_term": chosen_term,
            "gold_term":     gold_term,
            "correct":       "1" if is_correct else "0",
        })

        if total % 50 == 0 or total == len(gold_by_id):
            print(f"  [{total}/{len(gold_by_id)}] entity_id={entity_id}  "
                  f"correct={correct}/{total}  "
                  f"acc={correct/total*100:.1f}%")

    # ── Write CSV ──────────────────────────────────────────────────────────────
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["entity_id", "sentence", "unit", "predicted_term", "gold_term", "correct"],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    # ── Metrics (entity-level) ─────────────────────────────────────────────────
    acc = correct / max(total, 1)
    print(f"\nATE results (entity-level, per test_sentences_id id):")
    print(f"  Total entities : {total}")
    print(f"  Correct        : {correct}  ({acc*100:.2f}%)")
    print_metrics("Micro / Macro metrics", stats)
    print(f"\nSaved {len(output_rows)} rows → {OUT_CSV}")


if __name__ == "__main__":
    main()
