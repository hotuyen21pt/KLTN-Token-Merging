# -*- coding: utf-8 -*-
"""Joint multi-task training: sentiment + category trained simultaneously.

Strategy
--------
- Sentiment head : trained on main .apc + supplement data (all samples).
                   Uses weighted CrossEntropyLoss to handle class imbalance.
- Category  head : trained on main .apc samples ONLY.
                   Supplement samples are masked out via is_supplement flag.
- Both losses are summed: loss = sent_loss + cat_loss
- Early stopping monitors dev joint F1 (both sentiment + category correct).

Usage (from thesis_apc_baseline/ directory):
    python experiments/run_joint_experiments.py

Outputs (under runs_joint/):
    experiment_results_joint.txt   ← summary table (printed to screen + file)
    experiment_results_joint.csv   ← one row per config, all metrics
    <short_id>/best_model.pt       ← best checkpoint per config
"""

from __future__ import annotations

import contextlib
import copy
import csv
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ─── Tee: write to stdout AND a file simultaneously ───────────────────────────

class _Tee:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data: str) -> None:
        for s in self._streams:
            s.write(data)
    def flush(self) -> None:
        for s in self._streams:
            s.flush()


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, T5EncoderModel, AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score, classification_report, precision_score, recall_score

from common.dataset_utils import (
    ApcFileDataset,
    build_label_maps_from_apc,
    parse_apc_file,
    SENTIMENT_LABELS,
    SENTIMENT_MAP,
)
from models.fast_lcf_bert_multitask import FastLcfBertMultiTask

# ─── Paths ────────────────────────────────────────────────────────────────────

DATASET_DIR    = ROOT / "dataset"
TRAIN_APC      = DATASET_DIR / "train.apc"
DEV_APC        = DATASET_DIR / "dev.apc"
TEST_APC       = DATASET_DIR / "test.apc"
RUNS_DIR       = ROOT / "runs_joint"
SUPPLEMENT_DIR = DATASET_DIR / "supplement"

SUPPLEMENT_FILES: List[str] = [
    str(SUPPLEMENT_DIR / "negative.tsv"),
    str(SUPPLEMENT_DIR / "neutral.tsv"),
]

# ─── Model selection ──────────────────────────────────────────────────────────
# Change MODEL_TYPE to switch between encoders.  Add new entries to
# _MODEL_CONFIGS to register additional pretrained checkpoints.

MODEL_TYPE = "bert"   # "bert" | "t5"

_MODEL_CONFIGS = {
    "bert": "bert-base-uncased",
    # "t5":   "t5-base",
}

if MODEL_TYPE not in _MODEL_CONFIGS:
    raise ValueError(f"Unknown MODEL_TYPE={MODEL_TYPE!r}. Choose from: {list(_MODEL_CONFIGS)}")

PRETRAINED_MODEL = _MODEL_CONFIGS[MODEL_TYPE]

# ─── Hyperparameters ──────────────────────────────────────────────────────────
SEED             = 42
NUM_EPOCHS       = 15
PATIENCE         = 4
BATCH_SIZE       = 16
LR               = 2e-5
MAX_SEQ_LEN      = 128
DROPOUT          = 0.1
NUM_HEADS        = 8
SRD_THRESHOLD    = 5  # LCF-ATEPC CDW full-weight radius α (paper default)
TOME_MERGE_STEPS     = 2     # Number of TOME merge rounds (post-BERT)
USE_MIXED_PRECISION  = True  # Use torch.cuda.amp when running on GPU
PRE_TOME_MERGE_STEPS = 1   # Conservative: 1 merge step before BERT encoder

# ── Preprocessing config ──────────────────────────────────────────────────────
# Clause splitting mode applied before tokenisation.
#   "none"      — use full sentences (no splitting)
#   "rulebase"  — regex split on comma / semicolon / adversative conjunctions
#                 (but, yet, however, although, though, whereas); fast, no GPU
#   "uos"       — LLM-based UOS segmenter via Ollama (requires ollama serve)
CLAUSE_SPLIT_MODE = "none"

# Set True to mix supplement TSV data into sentiment-head training (recommended).
# Set False to train on main .apc data only.
USE_SUPPLEMENT = False

# Default loss and early stopping weights
DEFAULT_TASK_WEIGHT_SENT = 1.0  # 1.317
DEFAULT_TASK_WEIGHT_CAT  = 1.0
DEFAULT_ES_WEIGHT_SENT   = 0.5  # For early stopping validation metric
DEFAULT_ES_WEIGHT_CAT    = 0.5

# Override weights per config (task_weight_sent loss, task_weight_cat, es_weight_sent, es_weight_cat)
WEIGHT_OVERRIDES = {
    "baseline_sent_focus": (1.2, 0.8, 0.6, 0.4),  # Emphasize sentiment
    "baseline_cat_focus":  (0.8, 1.2, 0.4, 0.6),  # Emphasize category
    "baseline_balanced":   (1.0, 1.0, 0.5, 0.5),  # Balanced
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# (use_lcf, use_cdm, use_tome, tome_resize, merge_strategy, use_pre_tome, display_name, short_id)
CONFIGS: List[Tuple] = [
    # ── 12 methods: 1 baseline + 2 LCF-only + 6 post-BERT + 3 pre-BERT ─────────
    #    1.  baseline_balanced    — anchor (no LCF, no ToMe)
    #    2.  lcf_only_cdm         — LCF, no ToMe, CDM hard masking
    #    3.  lcf_only_cdw         — LCF, no ToMe, CDW soft weighting
    #    4.  lcf_bip_cdm_compact  — post-BERT Bip + CDM
    #    5.  lcf_bip_cdw_compact  — post-BERT Bip + CDW
    #    6.  lcf_seq_cdm_compact  — post-BERT Seq + CDM
    #    7.  lcf_seq_cdw_compact  — post-BERT Seq + CDW
    #    8.  lcf_scm_cdm_compact  — post-BERT SCM + CDM
    #    9.  lcf_scm_cdw_compact  — post-BERT SCM + CDW
    #   10.  lcf_pre_bip          — pre-BERT Bip (no post-BERT merge)
    #   11.  lcf_pre_seq          — pre-BERT Seq (no post-BERT merge)
    #   12.  lcf_pre_scm          — pre-BERT SCM (no post-BERT merge)
    #
    # Baseline
    # (False, False, False, True, "bipartite",          False, "Baseline (Balanced)",    "baseline_balanced"),
    # ── LCF only (no ToMe) ────────────────────────────────────────────────────
    # (True,  True,  False, False, "bipartite",          False, "LCF only CDM",           "lcf_only_cdm"),
    # (True,  False, False, False, "bipartite",          False, "LCF only CDW",           "lcf_only_cdw"),
    # ── Post-BERT ToMe (compact = no sequence resize after merge) ─────────────
    # (True,  True,  True,  False, "bipartite",          False, "LCF+Bip CDM (compact)",  "lcf_bip_cdm_compact"),
    # (True,  False, True,  False, "bipartite",          False, "LCF+Bip CDW (compact)",  "lcf_bip_cdw_compact"),
    # (True,  True,  True,  True, "sequential_local",   False, "LCF+Seq CDM (compact)",  "lcf_seq_cdm_compact"),
    # (True,  False, True,  True, "sequential_local",   False, "LCF+Seq CDW (compact)",  "lcf_seq_cdw_compact"),
    (True,  True,  True,  False, "sequential_cosine", False, "LCF+SCM CDM (compact)", "lcf_scm_cdm_compact"),
    (True,  False, True,  False, "sequential_cosine", False, "LCF+SCM CDW (compact)", "lcf_scm_cdw_compact"),
    (True,  True,  True,  True, "sequential_cosine", False, "LCF+SCM CDM (resize)", "lcf_scm_cdm_resize"),
    (True,  False, True,  True, "sequential_cosine", False, "LCF+SCM CDW (resize)", "lcf_scm_cdw_resize"),
    # ── Pre-BERT ToMe (merge at embedding level BEFORE BERT encoder) ──────────
    # use_tome=False → no post-BERT merge; tome_resize unused when use_tome=False
    # (False,  False,  True, False, "bipartite",          False,  "Bip",             "bip"),
    # (False,  False,  True, False, "sequential_local",   False,  "Seq",             "seq"),
    # (False,  False,  True, False, "sequential_cosine", False,  "SCM",             "scm"),
    # ── Pre-BERT + Post-BERT combined ─────────────────────────────────────────
    # (True,  True,  True,  False, "bipartite",          True,  "LCF+Pre+PostBip",        "lcf_pre_post_bip"),
    # (True,  True,  True,  False, "sequential_local",   True,  "LCF+Pre+PostSeq",        "lcf_pre_post_seq"),
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_encoder(model_type: str, pretrained: str):
    """Load the correct HuggingFace encoder for the given model_type."""
    if model_type == "t5":
        return T5EncoderModel.from_pretrained(pretrained)
    return AutoModel.from_pretrained(pretrained)


def get_weights_for_config(short_id: str) -> Tuple[float, float, float, float]:
    """Get task and ES weights for a config. Returns (task_w_sent, task_w_cat, es_w_sent, es_w_cat)."""
    if short_id in WEIGHT_OVERRIDES:
        return WEIGHT_OVERRIDES[short_id]
    else:
        return (DEFAULT_TASK_WEIGHT_SENT, DEFAULT_TASK_WEIGHT_CAT, 
                DEFAULT_ES_WEIGHT_SENT, DEFAULT_ES_WEIGHT_CAT)


def compute_sentiment_class_weights(dataset: ApcFileDataset) -> torch.Tensor:
    counts = torch.zeros(len(SENTIMENT_MAP))
    for sample in dataset.samples:
        counts[sample["sentiment_label"]] += 1
    total   = counts.sum()
    n_cls   = len(SENTIMENT_MAP)
    weights = total / (n_cls * counts.clamp(min=1))
    print(f"  Sentiment class weights: ", end="")
    for lbl, idx in sorted(SENTIMENT_MAP.items(), key=lambda x: x[1]):
        print(f"{lbl}={weights[idx]:.3f}", end="  ")
    print()
    return weights.to(DEVICE)


def compute_category_class_weights(
    dataset: ApcFileDataset,
    aspect_cat_map: Dict[str, int],
) -> torch.Tensor:
    """Inverse-frequency weights for category classes.
    Only main samples (is_supplement=False) are counted.
    """
    num_cat = len(aspect_cat_map)
    counts  = torch.zeros(num_cat)
    for sample in dataset.samples:
        if not sample["is_supplement"].item():
            counts[sample["aspect_cat_label"]] += 1
    total   = counts.sum()
    weights = total / (num_cat * counts.clamp(min=1))
    id2cat  = {v: k for k, v in aspect_cat_map.items()}
    print(f"  Category  class weights: ", end="")
    for idx in range(num_cat):
        print(f"{id2cat[idx]}={weights[idx]:.3f}", end="  ")
    print()
    return weights.to(DEVICE)





# ─── ATE-based inference helpers ─────────────────────────────────────────────

def tokenize_single(
    text: str,
    aspect_term: str,
    tokenizer,
    max_seq_len: int = 128,
) -> Dict[str, torch.Tensor]:
    """Tokenize one (sentence, aspect_term) pair — same format as ApcFileDataset.

    Returns dict with input_ids, attention_mask, lcf_vec (all 1-D tensors).
    """
    pad_or_unk = tokenizer.pad_token or tokenizer.unk_token or "[PAD]"
    aspect = aspect_term.strip() or pad_or_unk

    enc = tokenizer(
        text,
        aspect,
        max_length=max_seq_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    input_ids      = enc["input_ids"].squeeze(0)
    attention_mask = enc["attention_mask"].squeeze(0)
    token_type_ids = enc.get("token_type_ids", torch.zeros_like(input_ids)).squeeze(0)
    offsets        = enc["offset_mapping"].squeeze(0)

    asp_stripped  = aspect_term.strip()
    asp_cs = text.find(asp_stripped) if asp_stripped else -1
    asp_ce = asp_cs + len(asp_stripped) - 1 if asp_cs >= 0 else -1

    lcf_vec = torch.zeros_like(input_ids, dtype=torch.float32)
    if asp_cs >= 0:
        for k in range(input_ids.size(0)):
            if attention_mask[k].item() == 0:
                continue
            if token_type_ids[k].item() != 0:
                continue
            tok_s = int(offsets[k, 0].item())
            tok_e = int(offsets[k, 1].item())
            if tok_s == 0 and tok_e == 0:
                continue
            if tok_e > asp_cs and tok_s <= asp_ce:
                lcf_vec[k] = 1.0
    if lcf_vec.sum().item() == 0:
        lcf_vec = token_type_ids.float()

    return {"input_ids": input_ids, "attention_mask": attention_mask, "lcf_vec": lcf_vec}


def predict_from_ate_csv(
    model: nn.Module,
    tokenizer,
    ate_csv_path: Path,
    aspect_cat_map: Dict[str, int],
    out_path: Path,
    max_seq_len: int = 128,
    batch_size: int = 32,
) -> None:
    """Run joint model inference on ATE-predicted terms and save to CSV.

    Reads runs_ate/test_ate_predictions.csv (sentence, predicted_term, gold_terms),
    predicts category + sentiment for each row, writes to out_path.

    Output columns: sentence, predicted_term, predicted_category,
                    predicted_sentiment, gold_terms
    """
    if not ate_csv_path.is_file():
        print(f"  [skip] ATE CSV not found: {ate_csv_path}")
        return

    id2cat = {v: k for k, v in aspect_cat_map.items()}

    with open(ate_csv_path, newline="", encoding="utf-8") as f:
        rows_in = list(csv.DictReader(f))

    model.eval()
    rows_out: List[Dict[str, str]] = []

    for start in range(0, len(rows_in), batch_size):
        chunk = rows_in[start : start + batch_size]

        samples = [
            tokenize_single(r["sentence"], r["predicted_term"], tokenizer, max_seq_len)
            for r in chunk
        ]

        ids  = torch.stack([s["input_ids"]      for s in samples]).to(DEVICE)
        attn = torch.stack([s["attention_mask"]  for s in samples]).to(DEVICE)
        lcf  = torch.stack([s["lcf_vec"]         for s in samples]).to(DEVICE)

        with torch.no_grad():
            out = model(ids, attn, lcf)

        sent_preds = out["sentiment_logits"].argmax(-1).cpu().tolist()
        cat_preds  = out["aspect_cat_logits"].argmax(-1).cpu().tolist()

        for row, sp, cp in zip(chunk, sent_preds, cat_preds):
            rows_out.append({
                "sentence":             row["sentence"],
                "predicted_term":       row["predicted_term"],
                "predicted_category":   id2cat.get(cp, str(cp)),
                "predicted_sentiment":  SENTIMENT_LABELS[sp],
                "gold_terms":           row.get("gold_terms", ""),
            })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sentence", "predicted_term",
                        "predicted_category", "predicted_sentiment", "gold_terms"],
        )
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"  ATE-based predictions → {out_path}  ({len(rows_out)} rows)")


# ─── Predictions CSV ──────────────────────────────────────────────────────────

def save_predictions_csv(
    test_apc_path: str,
    sent_pred: List[int],
    cat_pred:  List[int],
    cat_id2label: Dict[str, int],
    out_path: Path,
) -> None:
    """Write per-sample test predictions to a CSV file.

    Columns: sentence, aspect_term, predicted_category, predicted_sentiment,
             gold_category, gold_sentiment
    """
    raw = parse_apc_file(test_apc_path)
    id2cat  = {v: k for k, v in cat_id2label.items()}

    if len(raw) != len(sent_pred):
        print(f"  [warn] save_predictions_csv: {len(raw)} samples in file "
              f"but {len(sent_pred)} predictions — truncating to min")

    n = min(len(raw), len(sent_pred), len(cat_pred))

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sentence", "aspect_term",
                "predicted_category", "predicted_sentiment",
                "gold_category", "gold_sentiment",
            ],
        )
        writer.writeheader()
        for i in range(n):
            writer.writerow({
                "sentence":             raw[i]["text"],
                "aspect_term":          raw[i]["aspect_term"],
                "predicted_category":   id2cat.get(cat_pred[i], str(cat_pred[i])),
                "predicted_sentiment":  SENTIMENT_LABELS[sent_pred[i]],
                "gold_category":        raw[i]["aspect_category"],
                "gold_sentiment":       raw[i]["sentiment"],
            })
    print(f"  Predictions saved → {out_path}  ({n} rows)")


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    sent_criterion: nn.Module,
    cat_criterion:  nn.Module,
) -> Dict:
    model.eval()
    total_loss = 0.0
    sent_pred, sent_true = [], []
    cat_pred,  cat_true  = [], []

    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)
            lcf  = batch["lcf_vec"].to(DEVICE)
            y_s  = batch["sentiment_label"].to(DEVICE)
            y_c  = batch["aspect_cat_label"].to(DEVICE)

            main_mask = ~batch["is_supplement"].to(DEVICE)
            out       = model(ids, attn, lcf)

            sent_loss = sent_criterion(out["sentiment_logits"], y_s)
            cat_loss  = (
                cat_criterion(out["aspect_cat_logits"][main_mask], y_c[main_mask])
                if main_mask.any() else torch.tensor(0.0, device=DEVICE)
            )
            total_loss += (sent_loss + cat_loss).item()

            sent_pred += out["sentiment_logits"].argmax(-1).cpu().tolist()
            sent_true += y_s.cpu().tolist()
            cat_pred  += out["aspect_cat_logits"][main_mask].argmax(-1).cpu().tolist()
            cat_true  += y_c[main_mask].cpu().tolist()

    n = max(len(loader), 1)
    # Joint: both sentiment AND category must be correct simultaneously.
    # dev/test loaders contain no supplement rows, so len(sent_pred) == len(cat_pred).
    min_len    = min(len(sent_pred), len(cat_pred))
    joint_true = [f"{s}_{c}" for s, c in zip(sent_true[:min_len], cat_true[:min_len])]
    joint_pred = [f"{s}_{c}" for s, c in zip(sent_pred[:min_len], cat_pred[:min_len])]
    return {
        "loss":            round(total_loss / n, 4),
        "sentiment_acc":   round(accuracy_score(sent_true, sent_pred) * 100, 2),
        "sentiment_f1":    round(f1_score(sent_true, sent_pred, average="micro",  zero_division=0) * 100, 2),
        "aspect_cat_acc":  round(accuracy_score(cat_true, cat_pred) * 100, 2),
        "aspect_cat_f1":   round(f1_score(cat_true,  cat_pred,  average="micro",  zero_division=0) * 100, 2),
        "joint_acc":          round(accuracy_score(joint_true, joint_pred) * 100, 2),
        "joint_f1":           round(f1_score(joint_true,       joint_pred, average="micro", zero_division=0) * 100, 2),
        "joint_precision":    round(precision_score(joint_true, joint_pred, average="micro", zero_division=0) * 100, 2),
        "joint_recall":       round(recall_score(joint_true,    joint_pred, average="micro", zero_division=0) * 100, 2),
        "joint_f1_macro":     round(f1_score(joint_true,       joint_pred, average="macro", zero_division=0) * 100, 2),
        "joint_precision_macro": round(precision_score(joint_true, joint_pred, average="macro", zero_division=0) * 100, 2),
        "joint_recall_macro":    round(recall_score(joint_true,    joint_pred, average="macro", zero_division=0) * 100, 2),
        "sent_pred": sent_pred, "sent_true": sent_true,
        "cat_pred":  cat_pred,  "cat_true":  cat_true,
    }


# ─── Joint training ───────────────────────────────────────────────────────────

def train_joint(
    use_lcf:          bool,
    use_cdm:          bool,
    use_tome:         bool,
    tome_resize:      bool,
    merge_strategy:   str,
    use_pre_tome:     bool,
    task_weight_sent: float,
    task_weight_cat:  float,
    es_weight_sent:   float,
    es_weight_cat:    float,
    short_id:         str,
    train_ds:         ApcFileDataset,   # main + supplement
    dev_ds:           ApcFileDataset,
    test_ds:          ApcFileDataset,
    num_sentiment:    int,
    num_aspect_cat:   int,
    aspect_cat_map:   Dict[str, int],
    tokenizer=None,
) -> Dict:
    """Train both heads jointly.

    Sentiment loss  : all samples (main + supplement) — fixes class imbalance.
    Category loss   : main samples only (supplement masked via is_supplement).
    Early stopping  : dev joint micro-F1.

    Returns a dict with train_time_sec, per-class F1 for both tasks.
    """
    set_seed(SEED)

    g = torch.Generator(); g.manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  generator=g)
    dev_loader   = DataLoader(dev_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    bert = _load_encoder(MODEL_TYPE, PRETRAINED_MODEL)
    model = FastLcfBertMultiTask(
        bert=bert,
        num_sentiment=num_sentiment,
        num_aspect_cat=num_aspect_cat,
        use_lcf=use_lcf,
        use_cdm=use_cdm,
        use_tome=use_tome,
        tome_resize=tome_resize,
        tome_merge_strategy=merge_strategy,
        dropout=DROPOUT,
        num_heads=NUM_HEADS,
        tome_merge_steps=TOME_MERGE_STEPS,
        srd_threshold=SRD_THRESHOLD,
        use_pre_tome=use_pre_tome,
        pre_tome_merge_steps=PRE_TOME_MERGE_STEPS,
        pre_tome_merge_strategy=merge_strategy,
        pre_tome_resize=tome_resize,
    ).to(DEVICE)

    optimiser = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_MIXED_PRECISION and DEVICE.type == "cuda")

    sent_weights   = compute_sentiment_class_weights(train_ds)
    sent_criterion = nn.CrossEntropyLoss(weight=sent_weights)

    # Category weights: main samples only (supplement has no category label)
    cat_weights   = compute_category_class_weights(train_ds, aspect_cat_map)
    cat_criterion = nn.CrossEntropyLoss(weight=cat_weights)

    ckpt_dir = RUNS_DIR / short_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_dev_f1 = -1.0
    best_epoch  = 0
    no_improve  = 0
    best_state  = None

    t0 = time.perf_counter()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            ids  = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)
            lcf  = batch["lcf_vec"].to(DEVICE)
            y_s  = batch["sentiment_label"].to(DEVICE)
            y_c  = batch["aspect_cat_label"].to(DEVICE)

            with torch.amp.autocast("cuda", enabled=USE_MIXED_PRECISION and DEVICE.type == "cuda"):
                out = model(ids, attn, lcf)

                # Sentiment: ALL samples (main + supplement)
                sent_loss = sent_criterion(out["sentiment_logits"], y_s)

                # Category: main samples ONLY
                main_mask = ~batch["is_supplement"].to(DEVICE)
                cat_loss  = (
                    cat_criterion(out["aspect_cat_logits"][main_mask], y_c[main_mask])
                    if main_mask.any() else torch.tensor(0.0, device=DEVICE)
                )

                # Weighted combination of losses
                loss = task_weight_sent * sent_loss + task_weight_cat * cat_loss

            optimiser.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)
        dev_m    = evaluate(model, dev_loader, sent_criterion, cat_criterion)

        # Early stopping on dev joint F1 (higher is better)
        dev_joint_f1 = dev_m["joint_f1"]

        print(
            f"    [{short_id}] epoch {epoch:2d}/{NUM_EPOCHS}"
            f"  train_loss={avg_loss:.4f}"
            f"  dev_loss={dev_m['loss']:.4f}"
            f"  dev_sent_f1={dev_m['sentiment_f1']:.1f}%"
            f"  dev_cat_f1={dev_m['aspect_cat_f1']:.1f}%"
            f"  dev_joint_f1={dev_joint_f1:.1f}%"
        )

        if dev_joint_f1 > best_dev_f1 + 1e-2:
            best_dev_f1 = dev_joint_f1
            best_epoch  = epoch
            no_improve  = 0
            best_state  = copy.deepcopy(model.state_dict())
            torch.save(best_state, ckpt_dir / "best_model.pt")
            # Save meta so pipeline_inference.py can reconstruct label maps
            import json as _json
            cat_id2label_local = {v: k for k, v in aspect_cat_map.items()}
            _meta = {
                "sentiment_labels":    SENTIMENT_LABELS,
                "category_labels":     [cat_id2label_local[i] for i in sorted(cat_id2label_local)],
                "num_aspect_cat":      num_aspect_cat,
                "best_dev_f1":         round(best_dev_f1, 4),
                "best_epoch":          best_epoch,
                "config": {
                    "use_lcf": use_lcf, "use_cdm": use_cdm,
                    "use_tome": use_tome, "tome_resize": tome_resize,
                    "merge_strategy": merge_strategy,
                    "use_pre_tome": use_pre_tome,
                    "pre_tome_merge_steps": PRE_TOME_MERGE_STEPS,
                    "clause_split_mode": CLAUSE_SPLIT_MODE,
                },
            }
            (ckpt_dir / "meta.json").write_text(
                _json.dumps(_meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"    [{short_id}] early stop at epoch {epoch}")
                break

    train_time = round(time.perf_counter() - t0, 2)

    # Patch train_time_sec into meta.json now that training is complete
    import json as _json
    _meta_path = ckpt_dir / "meta.json"
    if _meta_path.is_file():
        _meta = _json.loads(_meta_path.read_text(encoding="utf-8"))
        _meta["train_time_sec"] = train_time
        _meta_path.write_text(_json.dumps(_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = 1

    test_m = evaluate(model, test_loader, sent_criterion, cat_criterion)

    # ── Save per-sample predictions to CSV ───────────────────────────────────
    save_predictions_csv(
        test_apc_path=str(TEST_APC),
        sent_pred=test_m["sent_pred"],
        cat_pred=test_m["cat_pred"],
        cat_id2label=aspect_cat_map,
        out_path=ckpt_dir / "test_predictions.csv",
    )

    # ── Inference on ATE-predicted terms (end-to-end pipeline) ───────────────
    predict_from_ate_csv(
        model=model,
        tokenizer=tokenizer,
        ate_csv_path=RUNS_DIR.parent / "runs_ate" / "test_ate_predictions.csv",
        aspect_cat_map=aspect_cat_map,
        out_path=ckpt_dir / "test_predictions_from_ate.csv",
        max_seq_len=MAX_SEQ_LEN,
    )

    # ── Per-class F1 ─────────────────────────────────────────────────────────
    sent_f1_per = f1_score(
        test_m["sent_true"], test_m["sent_pred"],
        average=None, zero_division=0, labels=list(range(len(SENTIMENT_LABELS)))
    ) * 100

    cat_id2label     = {v: k for k, v in aspect_cat_map.items()}
    cat_labels_order = [cat_id2label[i] for i in sorted(cat_id2label)]
    cat_f1_per = f1_score(
        test_m["cat_true"], test_m["cat_pred"],
        average=None, zero_division=0, labels=list(range(len(aspect_cat_map)))
    ) * 100

    # ── Per-category × per-sentiment F1 (within each category's sample subset) ─
    cat_sent_f1 = {}
    for ci, cat_lbl in enumerate(cat_labels_order):
        indices = [i for i, c in enumerate(test_m["cat_true"]) if c == ci]
        if not indices:
            for sent_lbl in SENTIMENT_LABELS:
                cat_sent_f1[f"cs_f1_{cat_lbl}_{sent_lbl}"] = 0.0
            continue
        s_true_ci = [test_m["sent_true"][i] for i in indices]
        s_pred_ci = [test_m["sent_pred"][i] for i in indices]
        per_cls = f1_score(
            s_true_ci, s_pred_ci,
            average=None, zero_division=0,
            labels=list(range(len(SENTIMENT_LABELS))),
        ) * 100
        for si, sent_lbl in enumerate(SENTIMENT_LABELS):
            cat_sent_f1[f"cs_f1_{cat_lbl}_{sent_lbl}"] = round(float(per_cls[si]), 2)

    # ── Print reports ─────────────────────────────────────────────────────────
    print(f"\n  ── Test results [{short_id}] ──")
    print("  Sentiment classification report:")
    print(classification_report(
        test_m["sent_true"], test_m["sent_pred"],
        target_names=SENTIMENT_LABELS, zero_division=0,
    ))
    print("  Aspect-category classification report:")
    print(classification_report(
        test_m["cat_true"], test_m["cat_pred"],
        target_names=cat_labels_order, zero_division=0,
    ))

    result = {
        "train_time_sec":  train_time,
        "best_epoch":      best_epoch,
        "best_dev_f1":     round(best_dev_f1, 4),
        # Sentiment overall
        "sentiment_f1":    test_m["sentiment_f1"],
        "sentiment_acc":   test_m["sentiment_acc"],
        # Sentiment per-class
        **{f"sent_f1_{SENTIMENT_LABELS[i]}": round(float(sent_f1_per[i]), 2)
           for i in range(len(SENTIMENT_LABELS))},
        # Category overall
        "aspect_cat_acc":  test_m["aspect_cat_acc"],
        "aspect_cat_f1":   test_m["aspect_cat_f1"],
        # Category per-class
        **{f"cat_f1_{cat_labels_order[i]}": round(float(cat_f1_per[i]), 2)
           for i in range(len(cat_labels_order))},
        # Joint: both sentiment AND category correct simultaneously
        "joint_f1":               test_m["joint_f1"],
        "joint_precision":        test_m["joint_precision"],
        "joint_recall":           test_m["joint_recall"],
        "joint_acc":              test_m["joint_acc"],
        "joint_f1_macro":         test_m["joint_f1_macro"],
        "joint_precision_macro":  test_m.get("joint_precision_macro", 0),
        "joint_recall_macro":     test_m.get("joint_recall_macro", 0),
        # Per-category × per-sentiment F1
        **cat_sent_f1,
    }
    return result


# ─── Summary table ────────────────────────────────────────────────────────────

def print_summary_table(
    results: List[Dict],
    labels:  List[str],
    cat_labels_order: List[str],
) -> None:
    W = 148
    print(f"\n{'═' * W}")
    print("EXPERIMENT SUMMARY — Joint training (Sent: main+supp | Cat: main only)")
    print(f"{'═' * W}")
    print(
        f"  Seed: {SEED}  MaxEpochs: {NUM_EPOCHS}  Patience: {PATIENCE}"
        f"  Batch: {BATCH_SIZE}  LR: {LR}  Device: {DEVICE}"
        f"  UseSupp: True  UseWeights: True"
    )

    # ── Overall table ─────────────────────────────────────────────────────────
    print(f"\n{'─' * W}")
    print(f"  {'Configuration':<26} {'LCF':>4} {'PreToMe':>7} {'Strategy':<16} {'Resize':>6}"
          f" {'Time(s)':>8} {'BestEp':>7}"
          f" {'Sent-F1':>9} {'Cat-F1':>8} {'Micro-F1':>9} {'Macro-F1':>9} {'Joint-P':>8} {'Joint-R':>8} {'SentAcc':>8} {'CatAcc':>8}")
    print(f"{'─' * W}")

    baseline_time = next(
        (r["train_time_sec"] for r in results if not r["use_tome"] and not r.get("use_pre_tome")),
        None,
    )
    for r, label in zip(results, labels):
        lcf_tag      = "Y" if r["use_lcf"] else "N"
        pre_tome_tag = "Y" if r.get("use_pre_tome") else "N"
        any_tome     = r["use_tome"] or r.get("use_pre_tome")
        strategy_tag = r["merge_strategy"] if any_tome else "—"
        resize_tag   = "—" if not any_tome else ("yes" if r["tome_resize"] else "NO")
        speedup = ""
        if r["use_tome"] and not r["tome_resize"] and baseline_time:
            ratio = baseline_time / max(r["train_time_sec"], 1e-6)
            speedup = f"  x{ratio:.2f}"
        print(
            f"  {label:<26} {lcf_tag:>4} {pre_tome_tag:>7} {strategy_tag:<16} {resize_tag:>6}"
            f" {r['train_time_sec']:>8.1f} {r['best_epoch']:>7d}"
            f" {r['sentiment_f1']:>8.2f}% {r['aspect_cat_f1']:>7.2f}%"
            f" {r.get('joint_f1', 0):>8.2f}%"
            f" {r.get('joint_f1_macro', 0):>8.2f}%"
            f" {r.get('joint_precision', 0):>7.2f}%"
            f" {r.get('joint_recall', 0):>7.2f}%"
            f" {r['sentiment_acc']:>8.2f}% {r['aspect_cat_acc']:>8.2f}%{speedup}"
        )
    print(f"  {'Configuration':<26} {'LCF':>4}", end="")
    for lbl in SENTIMENT_LABELS:
        print(f" {('F1-'+lbl):>12}", end="")
    print()
    print(f"{'─' * W}")
    for r, label in zip(results, labels):
        lcf_tag = "Y" if r["use_lcf"] else "N"
        print(f"  {label:<26} {lcf_tag:>4}", end="")
        for lbl in SENTIMENT_LABELS:
            print(f" {r.get(f'sent_f1_{lbl}', 0):>11.2f}%", end="")
        print()

    # ── Category per-class table ──────────────────────────────────────────────
    print(f"\n{'─' * W}")
    print(f"  {'Configuration':<26} {'LCF':>4}", end="")
    for lbl in cat_labels_order:
        print(f" {('F1-'+lbl[:7]):>10}", end="")
    print()
    print(f"{'─' * W}")
    for r, label in zip(results, labels):
        lcf_tag = "Y" if r["use_lcf"] else "N"
        print(f"  {label:<26} {lcf_tag:>4}", end="")
        for lbl in cat_labels_order:
            print(f" {r.get(f'cat_f1_{lbl}', 0):>9.2f}%", end="")
        print()

    # ── Per-category × per-sentiment F1-micro table ──────────────────────────
    print(f"\n{'─' * W}")
    print("  Per-Category × Sentiment F1-micro  (sentiment F1 restricted to samples of each category)")
    for cat_lbl in cat_labels_order:
        print(f"\n  Category: {cat_lbl}")
        print(f"  {'Configuration':<26} {'LCF':>4}", end="")
        for sent_lbl in SENTIMENT_LABELS:
            print(f" {'F1-' + sent_lbl:>12}", end="")
        print()
        print(f"  {'─' * (34 + 13 * len(SENTIMENT_LABELS))}")
        for r, label in zip(results, labels):
            lcf_tag = "Y" if r["use_lcf"] else "N"
            print(f"  {label:<26} {lcf_tag:>4}", end="")
            for sent_lbl in SENTIMENT_LABELS:
                print(f" {r.get(f'cs_f1_{cat_lbl}_{sent_lbl}', 0):>11.2f}%", end="")
            print()

    print(f"\n{'═' * W}")
    print("  Sent training : main .apc + supplement (negative.tsv + neutral.tsv)")
    print("  Cat  training : main .apc only — supplement masked out via is_supplement")
    print("  bipartite → ToMe CVPR 2023 | sequential_local → new neighbour merge")
    print(f"{'═' * W}\n")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    for p in [TRAIN_APC, DEV_APC, TEST_APC]:
        if not p.is_file():
            raise FileNotFoundError(f"Dataset file not found: {p}")

    set_seed(SEED)
    print(f"Device  : {DEVICE}")
    print(f"Model   : {PRETRAINED_MODEL} (type={MODEL_TYPE})")
    print(f"Seed    : {SEED}")
    print(f"Runs dir: {RUNS_DIR}\n")

    if USE_SUPPLEMENT:
        avail_supplements = [p for p in SUPPLEMENT_FILES if Path(p).is_file()]
        missing = [p for p in SUPPLEMENT_FILES if not Path(p).is_file()]
        if missing:
            print(f"[warn] supplement files not found (skipped): {missing}")
        if avail_supplements:
            print(f"Supplement files: {[Path(p).name for p in avail_supplements]}")
    else:
        avail_supplements = []
        print("Supplement: disabled (USE_SUPPLEMENT=False)")

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    sentiment_map, aspect_cat_map = build_label_maps_from_apc(
        str(TRAIN_APC), str(DEV_APC), str(TEST_APC),
    )
    cat_id2label     = {v: k for k, v in aspect_cat_map.items()}
    cat_labels_order = [cat_id2label[i] for i in sorted(cat_id2label)]

    print(f"Sentiment classes  ({len(sentiment_map)}): {sorted(sentiment_map, key=sentiment_map.get)}")
    print(f"Aspect-cat classes ({len(aspect_cat_map)}): {cat_labels_order}")

    # train_ds: main + supplement  (sentiment head uses ALL samples)
    #           is_supplement=True for supplement rows → category head ignores them
    # dev/test: main only
    print(f"\nBuilding datasets … (clause_split_mode=none during training)")
    train_ds = ApcFileDataset(
        str(TRAIN_APC), tokenizer, aspect_cat_map, MAX_SEQ_LEN,
        supplement_paths=avail_supplements or None,
    )
    dev_ds  = ApcFileDataset(str(DEV_APC),  tokenizer, aspect_cat_map, MAX_SEQ_LEN)
    test_ds = ApcFileDataset(str(TEST_APC), tokenizer, aspect_cat_map, MAX_SEQ_LEN)

    from collections import Counter
    cnt   = Counter(int(s["sentiment_label"]) for s in train_ds.samples)
    id2s  = {v: k for k, v in sentiment_map.items()}
    total = sum(cnt.values())
    n_sup = sum(1 for s in train_ds.samples if s["is_supplement"].item())
    print(f"  Train: {len(train_ds)} samples ({len(train_ds)-n_sup} main + {n_sup} supplement)")
    print(f"  Dev: {len(dev_ds)} | Test: {len(test_ds)}")
    print("  Train sentiment distribution:")
    for idx in sorted(cnt):
        n = cnt[idx]
        print(f"    {id2s[idx]:<10}: {n:5d}  ({n/total*100:.1f}%)")

    results: List[Dict] = []
    labels:  List[str]  = []

    print(f"\n{'═' * 70}")
    print("Joint training: Sentiment (main+supp) + Category (main only)")
    print(f"{'═' * 70}")

    for use_lcf, use_cdm, use_tome, tome_resize, merge_strategy, use_pre_tome, label, short_id in CONFIGS:
        # Get weights for this config
        task_weight_sent, task_weight_cat, es_weight_sent, es_weight_cat = get_weights_for_config(short_id)

        strategy_tag  = merge_strategy if (use_tome or use_pre_tome) else "—"
        resize_tag    = ("resize" if tome_resize else "compact") if (use_tome or use_pre_tome) else "—"
        cdm_tag       = " (CDM)" if use_cdm else ""
        pre_tome_tag  = " [PreToMe]" if use_pre_tome else ""
        print(f"\n{'─' * 70}")
        print(f"Config : {label}{cdm_tag}{pre_tome_tag}  "
              f"(lcf={use_lcf}, cdm={use_cdm}, tome={use_tome}, pre_tome={use_pre_tome}, "
              f"strategy={strategy_tag}, resize={resize_tag})")
        print(f"  Task weights: sent={task_weight_sent}, cat={task_weight_cat} | "
              f"ES weights: sent={es_weight_sent}, cat={es_weight_cat}")
        print(f"{'─' * 70}")

        r = train_joint(
            use_lcf=use_lcf,
            use_cdm=use_cdm,
            use_tome=use_tome,
            tome_resize=tome_resize,
            merge_strategy=merge_strategy,
            use_pre_tome=use_pre_tome,
            task_weight_sent=task_weight_sent,
            task_weight_cat=task_weight_cat,
            es_weight_sent=es_weight_sent,
            es_weight_cat=es_weight_cat,
            short_id=short_id,
            train_ds=train_ds,
            dev_ds=dev_ds,
            test_ds=test_ds,
            num_sentiment=len(sentiment_map),
            num_aspect_cat=len(aspect_cat_map),
            aspect_cat_map=aspect_cat_map,
            tokenizer=tokenizer,
        )
        r["label"]            = label
        r["use_lcf"]          = use_lcf
        r["use_cdm"]          = use_cdm
        r["use_tome"]         = use_tome
        r["tome_resize"]      = tome_resize
        r["merge_strategy"]   = merge_strategy
        r["use_pre_tome"]     = use_pre_tome
        r["task_weight_sent"] = task_weight_sent
        r["task_weight_cat"]  = task_weight_cat
        r["es_weight_sent"]   = es_weight_sent
        r["es_weight_cat"]    = es_weight_cat

        results.append(r)
        labels.append(label)

        print(f"\n  → Train time   : {r['train_time_sec']:.1f}s  | best epoch: {r['best_epoch']}  | best dev joint_f1: {r.get('best_dev_f1', 0):.2f}%")
        print(f"  → Sentiment F1 : {r['sentiment_f1']:.2f}%"
              f"  (pos={r.get('sent_f1_positive',0):.1f}%"
              f"  neg={r.get('sent_f1_negative',0):.1f}%"
              f"  neu={r.get('sent_f1_neutral',0):.1f}%)")
        print(f"  → Category Acc: {r['aspect_cat_acc']:.2f}%")
        print(f"  → Category  F1 : {r['aspect_cat_f1']:.2f}%")
        print(f"  → Joint Micro  : {r['joint_f1']:.2f}%  (P={r.get('joint_precision',0):.2f}% R={r.get('joint_recall',0):.2f}%)")
        print(f"  → Joint Macro  : {r['joint_f1_macro']:.2f}%  (P={r.get('joint_precision_macro',0):.2f}% R={r.get('joint_recall_macro',0):.2f}%)")

    # ── Save outputs ───────────────────────────────────────────────────────────
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    txt_path = RUNS_DIR / "experiment_results_joint.txt"
    with open(txt_path, "w", encoding="utf-8") as txt_f:
        tee = _Tee(sys.stdout, txt_f)
        with contextlib.redirect_stdout(tee):
            print_summary_table(results, labels, cat_labels_order)

    csv_path = RUNS_DIR / "experiment_results_joint.csv"
    fieldnames = (
        ["label", "use_lcf", "use_cdm", "use_tome", "tome_resize", "merge_strategy",
         "use_pre_tome",
         "task_weight_sent", "task_weight_cat", "es_weight_sent", "es_weight_cat",
         "train_time_sec", "best_epoch", "best_dev_f1",
         "sentiment_f1", "sentiment_acc", "aspect_cat_acc", "aspect_cat_f1",
         "joint_f1", "joint_precision", "joint_recall", "joint_acc",
         "joint_f1_macro", "joint_precision_macro", "joint_recall_macro"]
        + [f"sent_f1_{l}" for l in SENTIMENT_LABELS]
        + [f"cat_f1_{l}" for l in cat_labels_order]
        + [f"cs_f1_{cat_lbl}_{sent_lbl}"
           for cat_lbl in cat_labels_order
           for sent_lbl in SENTIMENT_LABELS]
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSummary table → {txt_path}")
    print(f"Full metrics   → {csv_path}")
    print(f"Best models    → {RUNS_DIR}/<config>/best_model.pt")


if __name__ == "__main__":
    main()
