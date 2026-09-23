# -*- coding: utf-8 -*-
"""Evaluate ALL Bert/ checkpoints on the test set with GOLD aspect terms.

Aspect terms are taken directly from test.apc (ground truth) — no ATE
prediction error.  This measures the upper-bound APC + category quality
of each trained model.

Usage (from repo root):
    python common/eval_bert_gold_aspects.py

Output:
    runs_bert_gold/eval_gold_aspects.csv   ← one row per config
    (also printed to stdout as a summary table)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5EncoderModel
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.dataset_utils import (
    ApcFileDataset,
    build_label_maps_from_apc,
    SENTIMENT_LABELS,
    SENTIMENT_MAP,
)
from models.fast_lcf_bert_multitask import FastLcfBertMultiTask

# ─── Paths ────────────────────────────────────────────────────────────────────

BERT_DIR    = ROOT / "T5"
DATASET_DIR = ROOT / "dataset"
TRAIN_APC   = DATASET_DIR / "train.apc"
DEV_APC     = DATASET_DIR / "dev.apc"
TEST_APC    = DATASET_DIR / "test.apc"
OUT_DIR     = ROOT / "runs_bert_gold"

PRETRAINED_MODEL = "t5-base"
BATCH_SIZE       = 32
MAX_SEQ_LEN      = 128
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Model loader ─────────────────────────────────────────────────────────────

def load_model_from_ckpt(
    ckpt_dir: Path,
    num_aspect_cat: int,
) -> Optional[nn.Module]:
    """Load FastLcfBertMultiTask from a checkpoint directory.

    Returns None if best_model.pt or meta.json is missing.
    """
    model_pt  = ckpt_dir / "best_model.pt"
    meta_path = ckpt_dir / "meta.json"
    if not model_pt.is_file() or not meta_path.is_file():
        return None

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cfg  = meta.get("config", {})

    bert  = T5EncoderModel.from_pretrained(PRETRAINED_MODEL)
    model = FastLcfBertMultiTask(
        bert=bert,
        num_sentiment=3,
        num_aspect_cat=num_aspect_cat,
        use_lcf=cfg.get("use_lcf", False),
        use_cdm=cfg.get("use_cdm", False),
        use_tome=cfg.get("use_tome", False),
        tome_resize=cfg.get("tome_resize", True),
        tome_merge_strategy=cfg.get("merge_strategy", "bipartite"),
        dropout=0.1,
        num_heads=8,
        tome_merge_steps=2,
        srd_threshold=5,
        use_pre_tome=cfg.get("use_pre_tome", False),
        pre_tome_merge_steps=cfg.get("pre_tome_merge_steps", 1),
        pre_tome_merge_strategy=cfg.get("merge_strategy", "bipartite"),
        pre_tome_resize=cfg.get("tome_resize", True),
    )

    state = torch.load(model_pt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

# ─── Evaluation (gold aspect terms via ApcFileDataset) ────────────────────────

def evaluate_gold(
    model: nn.Module,
    test_loader: DataLoader,
    cat_labels_order: List[str],
) -> Dict:
    """Run inference with gold aspect terms; return full metrics dict."""
    sent_pred: List[int] = []
    sent_true: List[int] = []
    cat_pred:  List[int] = []
    cat_true:  List[int] = []

    with torch.no_grad():
        for batch in test_loader:
            ids  = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)
            lcf  = batch["lcf_vec"].to(DEVICE)
            y_s  = batch["sentiment_label"]
            y_c  = batch["aspect_cat_label"]

            out = model(ids, attn, lcf)
            sent_pred += out["sentiment_logits"].argmax(-1).cpu().tolist()
            sent_true += y_s.tolist()
            cat_pred  += out["aspect_cat_logits"].argmax(-1).cpu().tolist()
            cat_true  += y_c.tolist()

    # ── Joint: both sentiment AND category must be correct simultaneously ──────
    joint_true = [f"{s}_{c}" for s, c in zip(sent_true, cat_true)]
    joint_pred = [f"{s}_{c}" for s, c in zip(sent_pred, cat_pred)]

    result: Dict = {
        # Sentiment overall
        "sentiment_acc":      round(accuracy_score(sent_true, sent_pred) * 100, 2),
        "sentiment_f1_micro": round(f1_score(sent_true, sent_pred, average="micro", zero_division=0) * 100, 2),
        # Category overall
        "aspect_cat_acc":     round(accuracy_score(cat_true, cat_pred) * 100, 2),
        "aspect_cat_f1_micro":round(f1_score(cat_true, cat_pred, average="micro", zero_division=0) * 100, 2),
        # Joint micro
        "joint_f1_micro":     round(f1_score(joint_true, joint_pred, average="micro", zero_division=0) * 100, 2),
        "joint_precision":    round(precision_score(joint_true, joint_pred, average="micro", zero_division=0) * 100, 2),
        "joint_recall":       round(recall_score(joint_true, joint_pred, average="micro", zero_division=0) * 100, 2),
        # Joint macro
        "joint_f1_macro":     round(f1_score(joint_true, joint_pred, average="macro", zero_division=0) * 100, 2),
    }

    # ── Per-sentiment F1 ──────────────────────────────────────────────────────
    sent_f1_per = f1_score(
        sent_true, sent_pred, average=None, zero_division=0,
        labels=list(range(len(SENTIMENT_LABELS))),
    ) * 100
    for i, lbl in enumerate(SENTIMENT_LABELS):
        result[f"sent_f1_{lbl}"] = round(float(sent_f1_per[i]), 2)

    # ── Per-category F1 ───────────────────────────────────────────────────────
    cat_f1_per = f1_score(
        cat_true, cat_pred, average=None, zero_division=0,
        labels=list(range(len(cat_labels_order))),
    ) * 100
    for i, lbl in enumerate(cat_labels_order):
        result[f"cat_f1_{lbl}"] = round(float(cat_f1_per[i]), 2)

    return result

# ─── Summary table ────────────────────────────────────────────────────────────

def print_summary(rows: List[Dict], cat_labels_order: List[str]) -> None:
    W = 120
    print(f"\n{'═' * W}")
    print("BERT GOLD-ASPECT EVALUATION  (test set, aspect terms = ground truth)")
    print(f"{'═' * W}")
    print(f"  {'Config':<32} {'SentF1':>7} {'CatF1':>7} {'JntMicF1':>10} {'JntMacF1':>10} "
          f"{'JntP':>7} {'JntR':>7} {'BestDevF1':>10}")
    print(f"{'─' * W}")
    for r in sorted(rows, key=lambda x: x["joint_f1_micro"], reverse=True):
        print(
            f"  {r['config']:<32}"
            f" {r['sentiment_f1_micro']:>6.2f}%"
            f" {r['aspect_cat_f1_micro']:>6.2f}%"
            f" {r['joint_f1_micro']:>9.2f}%"
            f" {r['joint_f1_macro']:>9.2f}%"
            f" {r['joint_precision']:>6.2f}%"
            f" {r['joint_recall']:>6.2f}%"
            f" {r.get('best_dev_f1', 0):>9.2f}%"
        )
    print(f"{'─' * W}")
    # Per-sentiment
    print(f"\n  {'Config':<32}", end="")
    for lbl in SENTIMENT_LABELS:
        print(f" {'F1-' + lbl:>12}", end="")
    print()
    print(f"{'─' * W}")
    for r in sorted(rows, key=lambda x: x["joint_f1_micro"], reverse=True):
        print(f"  {r['config']:<32}", end="")
        for lbl in SENTIMENT_LABELS:
            print(f" {r.get('sent_f1_' + lbl, 0):>11.2f}%", end="")
        print()
    # Per-category
    print(f"\n  {'Config':<32}", end="")
    for lbl in cat_labels_order:
        print(f" {'F1-' + lbl[:7]:>10}", end="")
    print()
    print(f"{'─' * W}")
    for r in sorted(rows, key=lambda x: x["joint_f1_micro"], reverse=True):
        print(f"  {r['config']:<32}", end="")
        for lbl in cat_labels_order:
            print(f" {r.get('cat_f1_' + lbl, 0):>9.2f}%", end="")
        print()
    print(f"\n{'═' * W}")

# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Device : {DEVICE}")
    print(f"Model  : {PRETRAINED_MODEL}")
    print(f"Bert/  : {BERT_DIR}\n")

    # Build label maps from all three splits to ensure consistency
    _, aspect_cat_map = build_label_maps_from_apc(
        str(TRAIN_APC), str(DEV_APC), str(TEST_APC),
    )
    cat_id2label     = {v: k for k, v in aspect_cat_map.items()}
    cat_labels_order = [cat_id2label[i] for i in sorted(cat_id2label)]
    num_aspect_cat   = len(aspect_cat_map)

    print(f"Sentiment classes  : {SENTIMENT_LABELS}")
    print(f"Category  classes  : {cat_labels_order}\n")

    # Build test DataLoader (gold aspect terms from test.apc)
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    test_ds   = ApcFileDataset(str(TEST_APC), tokenizer, aspect_cat_map, MAX_SEQ_LEN)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Test set: {len(test_ds)} samples\n")

    # Scan all Bert/ subdirectories
    ckpt_dirs = sorted(
        [d for d in BERT_DIR.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )
    print(f"Found {len(ckpt_dirs)} checkpoint directories.\n")

    rows: List[Dict] = []

    for ckpt_dir in ckpt_dirs:
        config_name = ckpt_dir.name
        print(f"  [{config_name}] loading …", flush=True)

        model = load_model_from_ckpt(ckpt_dir, num_aspect_cat)
        if model is None:
            print(f"    SKIP (missing best_model.pt or meta.json)")
            continue

        # Read best_dev_f1 from meta.json for reference
        meta = json.loads((ckpt_dir / "meta.json").read_text(encoding="utf-8"))
        best_dev_f1 = meta.get("best_dev_f1", 0.0)

        metrics = evaluate_gold(model, test_loader, cat_labels_order)
        metrics["config"]       = config_name
        metrics["best_dev_f1"]  = best_dev_f1
        rows.append(metrics)

        print(
            f"    OK  SentF1={metrics['sentiment_f1_micro']:.2f}%"
            f"  CatF1={metrics['aspect_cat_f1_micro']:.2f}%"
            f"  JointMicro={metrics['joint_f1_micro']:.2f}%"
            f"  JointMacro={metrics['joint_f1_macro']:.2f}%"
        )

        # Free GPU memory between models
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not rows:
        print("No valid checkpoints found. Exiting.")
        return

    print_summary(rows, cat_labels_order)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "eval_gold_aspects.csv"

    fieldnames = (
        ["config", "best_dev_f1",
         "sentiment_acc", "sentiment_f1_micro",
         "aspect_cat_acc", "aspect_cat_f1_micro",
         "joint_f1_micro", "joint_precision", "joint_recall", "joint_f1_macro"]
        + [f"sent_f1_{lbl}" for lbl in SENTIMENT_LABELS]
        + [f"cat_f1_{lbl}"  for lbl in cat_labels_order]
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults → {csv_path}")


if __name__ == "__main__":
    main()
