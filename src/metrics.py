# -*- coding: utf-8 -*-
"""Exact-match precision / recall / F1 for generative ATE."""

from __future__ import annotations

import unicodedata
from typing import Dict, Iterable, List, Sequence, Tuple


def _canonical_term(text: str) -> str:
    """Canonical form for evaluation without changing reported surface text."""
    text = unicodedata.normalize("NFC", str(text))
    text = " ".join(text.split()).strip().casefold()
    return text.replace(".", "")


def _as_set(items: Sequence[str]) -> set[str]:
    return {_canonical_term(x) for x in items if x and _canonical_term(x)}


def sentence_exact_match_counts(
    predictions: Sequence[str],
    golds: Sequence[str],
) -> Tuple[int, int, int]:
    """Return (tp, fp, fn) for one sentence using exact string match."""
    pred_set = _as_set(predictions)
    gold_set = _as_set(golds)
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    return tp, fp, fn


def compute_prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_exact_match(
    all_predictions: Iterable[Sequence[str]],
    all_golds: Iterable[Sequence[str]],
) -> Dict[str, float]:
    """Micro-averaged exact-match P/R/F1 over all aspects."""
    total_tp = total_fp = total_fn = 0
    for preds, golds in zip(all_predictions, all_golds):
        tp, fp, fn = sentence_exact_match_counts(list(preds), list(golds))
        total_tp += tp
        total_fp += fp
        total_fn += fn
    metrics = compute_prf(total_tp, total_fp, total_fn)
    metrics.update({"tp": float(total_tp), "fp": float(total_fp), "fn": float(total_fn)})
    return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    return (
        f"P={metrics['precision']:.4f} "
        f"R={metrics['recall']:.4f} "
        f"F1={metrics['f1']:.4f}"
    )
