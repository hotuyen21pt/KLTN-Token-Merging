# -*- coding: utf-8 -*-
"""Multi-seed experiment runner — reproduces all thesis tables.

Output tables (matching thesis phuong_phap_thuc_nghiem.tex exactly):
  [1] tab:bert_results   — Oracle BERT (12 configs): Train Time | Micro-F1 | Macro-F1
  [2] tab:end2end_results — E2E BERT (5 configs):  ATE-F1 | Micro-F1 | Macro-F1 | Joint-Acc
  [3] tab:t5_results     — E2E T5 (5 configs):    Train Time | Micro-F1 | Macro-F1 | Joint-Acc
  [4] Resize vs Compact Macro-F1 (4 configs)
  [5] Resize vs Compact Training Time (4 configs)
  [6] tab:paper          — Comparison: GAS / TOFA / our methods
  [NOTE] tab:t5_split (clause splitting) — requires separate run with --clause-split

Hyperparameters (thesis tab:hyperparams):
  lr=2e-5  epochs=15  patience=4  batch=16  max_seq=128  dropout=0.1
  heads=8  srd=5  tome_post_steps=2  tome_pre_steps=1

Usage (from repo root):
  python common/run_multiseed.py                          # bert+t5, seeds [42,123,456], 12 resize configs
  python common/run_multiseed.py --seeds 42 123 456 789 2024
  python common/run_multiseed.py --model-types bert
  python common/run_multiseed.py --configs baseline lcf_bip_cdm
  python common/run_multiseed.py --include-compact        # also run 4 compact configs
  python common/run_multiseed.py --ate-csv runs_ate/test_ate_predictions.csv
  python common/run_multiseed.py --ate-f1 79.87           # GAS ATE F1 for table headers
  python common/run_multiseed.py --resume                 # skip finished checkpoints
  python common/run_multiseed.py --no-train               # aggregate only

Outputs (under runs_multiseed/):
  results_raw.csv / results_aggregated.csv / results_summary.txt
  thesis_tables.txt    ← all thesis tables formatted
  <backbone>/<config>/seed_<N>/best_model.pt
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (AutoConfig, AutoModel, AutoTokenizer,
                          MT5EncoderModel, T5EncoderModel)
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
)

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
SUPPLEMENT_DIR = DATASET_DIR / "supplement"
RUNS_DIR       = ROOT / "runs_multiseed"

SUPPLEMENT_FILES = [
    str(SUPPLEMENT_DIR / "negative.tsv"),
    str(SUPPLEMENT_DIR / "neutral.tsv"),
]

# ─── Hyperparameters (thesis tab:hyperparams) ─────────────────────────────────

NUM_EPOCHS           = 15       # Số epoch tối đa
PATIENCE             = 4        # Early stopping patience (Dev Joint Micro-F1)
BATCH_SIZE           = 16       # Batch size
LR                   = 2e-5     # Learning rate
MAX_SEQ_LEN          = 128      # Độ dài chuỗi tối đa
DROPOUT              = 0.1      # Dropout
NUM_HEADS            = 8        # Số attention head (SA)
SRD_THRESHOLD        = 5        # SRD threshold α (LCF)
TOME_MERGE_STEPS     = 2        # Số bước merge ToMe (post-BERT)
PRE_TOME_MERGE_STEPS = 1        # Số bước merge ToMe (pre-BERT)
USE_MIXED_PRECISION  = True
USE_SUPPLEMENT       = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_SEEDS = [42]

MODEL_REGISTRY = {
    "bert": "bert-base-uncased",
    "t5":   "t5-base",
}

# ─── Reference values (from thesis / related work) ────────────────────────────

GAS_ATE_F1          = 79.87   # GAS ATE F1 on test set (tab:ate_results)
GAS_ATE_PRECISION   = 80.26
GAS_ATE_RECALL      = 79.49

# For tab:paper comparison table
REF_METHODS = [
    # (display_name, micro_f1, micro_precision, micro_recall)
    ("GAS",  69.51, 68.54, 70.51),
    ("TOFA", 62.65, 62.65, 62.65),
]

# ─── All 12 resize configurations (thesis: "tất cả sử dụng resize") ──────────
# (use_lcf, use_cdm, use_tome, tome_resize, merge_strategy, use_pre_tome,
#  display_name, short_id)

ALL_CONFIGS: List[Tuple] = [
    # 1. Baseline
    # (False, False, False, True,  "bipartite",          False, "Base",        "baseline"),
    # 2–3. LCF only
    # (True,  True,  False, True,  "bipartite",          False, "CDM",         "lcf_only_cdm"),
    # # (True,  False, False, True,  "bipartite",          False, "CDW",         "lcf_only_cdw"),
    # # 4–6. ToMe only
    # (False, False, True,  True,  "bipartite",          False, "BiToMe",      "bip"),
    # (False, False, True,  True,  "sequential_local",   False, "SLM",         "seq"),
    # (False, False, True,  True,  "sequential_cosine",  False, "SCM",         "scm"),
    # # 7–12. LCF + ToMe (resize)
    (True,  True,  True,  True,  "bipartite",          False, "BiToMe+CDM",  "lcf_bip_cdm"),
    # (True,  False, True,  True,  "bipartite",          False, "BiToMe+CDW",  "lcf_bip_cdw"),
    (True,  True,  True,  True,  "sequential_local",   False, "SLM+CDM",     "lcf_seq_cdm"),
    (True,  False, True,  True,  "sequential_local",   False, "SLM+CDW",     "lcf_seq_cdw"),
    (True,  True,  True,  True,  "sequential_cosine",  False, "SCM+CDM",     "lcf_scm_cdm"),
    (True,  False, True,  True,  "sequential_cosine",  False, "SCM+CDW",     "lcf_scm_cdw"),
]

# ─── 4 compact configs (for tab:compact_vs_resize comparison) ─────────────────
# Only run with --include-compact flag

# COMPACT_CONFIGS: List[Tuple] = [
#     (True,  True,  True,  False, "sequential_cosine",  False, "SCM+CDM(compact)",    "lcf_scm_cdm_compact"),
#     (True,  True,  True,  False, "bipartite",          False, "BiToMe+CDM(compact)", "lcf_bip_cdm_compact"),
#     (True,  True,  True,  False, "sequential_local",   False, "SLM+CDM(compact)",    "lcf_seq_cdm_compact"),
#     (True,  False, True,  False, "sequential_local",   False, "SLM+CDW(compact)",    "lcf_seq_cdw_compact"),
# ]
COMPACT_CONFIGS: List[Tuple] = []

CONFIG_BY_ID = {cfg[7]: cfg for cfg in ALL_CONFIGS + COMPACT_CONFIGS}

# ─── Selected config groups for thesis tables ──────────────────────────────────
# (không dùng trong lần chạy này — chỉ chạy 1 config lcf_scm_cdw để đo inference time)

# tab:end2end_results — 5 BERT configs (display_name in table, short_id in data)
# E2E_BERT_CONFIGS = [
#     ("GAS + Base",       "baseline"),
#     ("GAS + BiTome",     "bip"),
#     ("GAS + SLM + CDM",  "lcf_seq_cdm"),
#     ("GAS + SLM + CDW",  "lcf_seq_cdw"),
#     ("GAS + SCM + CDM",  "lcf_scm_cdm"),
# ]
E2E_BERT_CONFIGS: List[Tuple] = []

# # tab:t5_results — 5 T5 configs
# E2E_T5_CONFIGS = [
#     ("GAS + Base",   "baseline"),
#     ("BipTome + CDM", "lcf_bip_cdm"),
#     ("SCM + CDM",    "lcf_scm_cdm"),
#     ("SLM + CDM",    "lcf_seq_cdm"),
#     ("SLM + CDW",    "lcf_seq_cdw"),
# ]
E2E_T5_CONFIGS: List[Tuple] = []

# # tab:compact_vs_resize — 4 config pairs (resize_id, compact_id, display_name)
# COMPACT_VS_RESIZE_CONFIGS = [
#     ("lcf_scm_cdm", "lcf_scm_cdm_compact", "SCM + CDM"),
#     ("lcf_bip_cdm",  "lcf_bip_cdm_compact",  "BiTome + CDM"),
#     ("lcf_seq_cdm",  "lcf_seq_cdm_compact",  "SLM + CDM"),
#     ("lcf_seq_cdw",  "lcf_seq_cdw_compact",  "SLM + CDW"),
# ]
COMPACT_VS_RESIZE_CONFIGS: List[Tuple] = []

# # tab:paper — our best E2E results to compare with GAS/TOFA
# # (display_name, model_type, config_id)
# PAPER_OUR_CONFIGS = [
#     ("Base (BERT)",              "bert", "baseline"),
#     ("BipTome+CDM Resize (T5)",  "t5",   "lcf_bip_cdm"),
#     ("SLM+CDM Resize (BERT)",    "bert", "lcf_seq_cdm"),
#     ("SLM+CDW Resize (BERT)",    "bert", "lcf_seq_cdw"),
# ]
PAPER_OUR_CONFIGS: List[Tuple] = []

# ─── Helpers ──────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pick_amp_dtype(model_type: str):
    """Chọn dtype cho autocast; None nghĩa là chạy fp32.

    T5/mT5 được pre-train ở bfloat16 và activations của chúng thường vượt dải
    biểu diễn của fp16 (tối đa 65504) -> tràn thành inf -> loss = NaN. Nếu GPU
    hỗ trợ bf16 (Ampere trở lên) thì dùng bf16; còn lại (T4 compute 7.5) buộc
    phải chạy fp32 cho backbone T5. BERT được train ở fp32 nên fp16 vẫn an toàn.
    """
    if not (USE_MIXED_PRECISION and DEVICE.type == "cuda"):
        return None
    if model_type in {"t5", "mt5"}:
        # torch.cuda.is_bf16_supported() trả True cả khi bf16 chỉ được EMULATE
        # (T4 compute 7.5) nên phải xem compute capability: chỉ Ampere trở lên
        # (>= 8) mới có bf16 chạy trên phần cứng.
        major = torch.cuda.get_device_properties(torch.cuda.current_device()).major
        return torch.bfloat16 if major >= 8 else None
    return torch.float16


def _load_encoder(model_type: str, pretrained: str):
    """Nạp encoder theo ĐÚNG lớp mà config.json khai báo.

    ``T5EncoderModel`` có ``config_class = T5Config`` nên nạp một checkpoint
    mt5 bằng nó sinh cảnh báo "You are using a model of type mt5 to
    instantiate a model of type t5". Với mt5-base thì mọi trường kiến trúc
    đều nằm sẵn trong config.json và hai lớp có cùng default cho phần còn
    lại, nên trọng số vẫn đúng — nhưng dựa vào sự trùng hợp đó là mong manh:
    chỉ cần một biến thể mT5 đặt khác default của T5Config là lệch lặng lẽ.

    Phân nhánh theo ``config.model_type`` chứ không theo khoá trong
    MODEL_REGISTRY, vì khoá đó do người dùng tự đặt và có thể không khớp
    checkpoint thật.
    """
    if model_type in {"t5", "mt5"}:
        cfg_type = getattr(AutoConfig.from_pretrained(pretrained), "model_type", "")
        cls = MT5EncoderModel if cfg_type == "mt5" else T5EncoderModel
        print(f"  [encoder] {cls.__name__} <- config.model_type={cfg_type!r}")
        return cls.from_pretrained(pretrained)
    return AutoModel.from_pretrained(pretrained)


def compute_sentiment_class_weights(dataset: ApcFileDataset) -> torch.Tensor:
    counts = torch.zeros(len(SENTIMENT_MAP))
    for s in dataset.samples:
        counts[s["sentiment_label"]] += 1
    total = counts.sum()
    return (total / (len(SENTIMENT_MAP) * counts.clamp(min=1))).to(DEVICE)


def compute_category_class_weights(
    dataset: ApcFileDataset, aspect_cat_map: Dict[str, int]
) -> torch.Tensor:
    n = len(aspect_cat_map)
    counts = torch.zeros(n)
    for s in dataset.samples:
        if not s["is_supplement"].item():
            counts[s["aspect_cat_label"]] += 1
    total = counts.sum()
    return (total / (n * counts.clamp(min=1))).to(DEVICE)


def tokenize_single(text, aspect_term, tokenizer, max_seq_len=128):
    pad_or_unk = tokenizer.pad_token or tokenizer.unk_token or "[PAD]"
    aspect = aspect_term.strip() or pad_or_unk
    enc = tokenizer(
        text, aspect, max_length=max_seq_len, padding="max_length",
        truncation=True, return_tensors="pt", return_offsets_mapping=True,
    )
    ids   = enc["input_ids"].squeeze(0)
    attn  = enc["attention_mask"].squeeze(0)
    ttype = enc.get("token_type_ids", torch.zeros_like(ids)).squeeze(0)
    offs  = enc["offset_mapping"].squeeze(0)
    asp   = aspect_term.strip()
    cs    = text.find(asp) if asp else -1
    ce    = cs + len(asp) - 1 if cs >= 0 else -1
    lcf   = torch.zeros_like(ids, dtype=torch.float32)
    if cs >= 0:
        for k in range(ids.size(0)):
            if attn[k].item() == 0 or ttype[k].item() != 0:
                continue
            ts, te = int(offs[k, 0].item()), int(offs[k, 1].item())
            if ts == 0 and te == 0:
                continue
            if te > cs and ts <= ce:
                lcf[k] = 1.0
    if lcf.sum().item() == 0:
        lcf = ttype.float()
    return {"input_ids": ids, "attention_mask": attn, "lcf_vec": lcf}


# ─── Oracle evaluation ────────────────────────────────────────────────────────

def evaluate(model, loader, sent_crit, cat_crit):
    model.eval()
    total_loss = 0.0
    sp, st, cp, ct = [], [], [], []
    with torch.no_grad():
        for b in loader:
            ids  = b["input_ids"].to(DEVICE)
            attn = b["attention_mask"].to(DEVICE)
            lcf  = b["lcf_vec"].to(DEVICE)
            ys   = b["sentiment_label"].to(DEVICE)
            yc   = b["aspect_cat_label"].to(DEVICE)
            mm   = ~b["is_supplement"].to(DEVICE)
            out  = model(ids, attn, lcf)
            sl   = sent_crit(out["sentiment_logits"], ys)
            cl   = (cat_crit(out["aspect_cat_logits"][mm], yc[mm])
                    if mm.any() else torch.tensor(0.0, device=DEVICE))
            total_loss += (sl + cl).item()
            sp += out["sentiment_logits"].argmax(-1).cpu().tolist()
            st += ys.cpu().tolist()
            cp += out["aspect_cat_logits"][mm].argmax(-1).cpu().tolist()
            ct += yc[mm].cpu().tolist()
    n  = max(len(loader), 1)
    ml = min(len(sp), len(cp))
    jt = [f"{s}_{c}" for s, c in zip(st[:ml], ct[:ml])]
    jp = [f"{s}_{c}" for s, c in zip(sp[:ml], cp[:ml])]
    def pct(fn, *a, **kw): return round(fn(*a, **kw) * 100, 2)
    return {
        "loss":                  round(total_loss / n, 4),
        "sentiment_acc":         pct(accuracy_score, st, sp),
        "sentiment_f1":          pct(f1_score, st, sp, average="micro", zero_division=0),
        "aspect_cat_acc":        pct(accuracy_score, ct, cp),
        "aspect_cat_f1":         pct(f1_score, ct, cp, average="micro", zero_division=0),
        "joint_acc":             pct(accuracy_score, jt, jp),
        "joint_f1":              pct(f1_score, jt, jp, average="micro", zero_division=0),
        "joint_precision":       pct(precision_score, jt, jp, average="micro", zero_division=0),
        "joint_recall":          pct(recall_score, jt, jp, average="micro", zero_division=0),
        "joint_f1_macro":        pct(f1_score, jt, jp, average="macro", zero_division=0),
        "joint_precision_macro": pct(precision_score, jt, jp, average="macro", zero_division=0),
        "joint_recall_macro":    pct(recall_score, jt, jp, average="macro", zero_division=0),
        "sent_pred": sp, "sent_true": st, "cat_pred": cp, "cat_true": ct,
    }


# ─── End-to-end triplet evaluation (ATE → APC) ───────────────────────────────

def _norm(t): return t.strip().lower()


def eval_triplet_e2e(model, tokenizer, ate_csv_path, aspect_cat_map, batch_size=32):
    if not Path(ate_csv_path).is_file():
        return None
    id2cat = {v: k for k, v in aspect_cat_map.items()}
    with open(ate_csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    gold_entries = parse_apc_file(str(TEST_APC))
    gold_by_sent = defaultdict(set)
    for e in gold_entries:
        gold_by_sent[e["text"].strip()].add((
            _norm(e["aspect_term"]),
            e["aspect_category"].strip().upper(),
            e["sentiment"].strip().capitalize(),
        ))
    samples = [(r["sentence"].strip(), r["predicted_term"]) for r in rows]
    pred_by_sent = defaultdict(set)
    model.eval()
    for i in range(0, len(samples), batch_size):
        chunk = samples[i: i + batch_size]
        tok = [tokenize_single(s, t, tokenizer, MAX_SEQ_LEN) for s, t in chunk]
        ids  = torch.stack([x["input_ids"]     for x in tok]).to(DEVICE)
        attn = torch.stack([x["attention_mask"] for x in tok]).to(DEVICE)
        lcf  = torch.stack([x["lcf_vec"]        for x in tok]).to(DEVICE)
        with torch.no_grad():
            out = model(ids, attn, lcf)
        sps = out["sentiment_logits"].argmax(-1).cpu().tolist()
        cps = out["aspect_cat_logits"].argmax(-1).cpu().tolist()
        for (st, pt), s, c in zip(chunk, sps, cps):
            pred_by_sent[st].add((
                _norm(pt),
                id2cat.get(c, str(c)).upper(),
                SENTIMENT_LABELS[s].capitalize(),
            ))
    tp = fp = fn = 0
    cls_tp = defaultdict(int); cls_fp = defaultdict(int); cls_fn = defaultdict(int)
    for sent in set(gold_by_sent) | set(pred_by_sent):
        golds = gold_by_sent.get(sent, set())
        preds = pred_by_sent.get(sent, set())
        for t in preds:
            k = f"{t[1]}_{t[2]}"
            if t in golds: tp += 1; cls_tp[k] += 1
            else:          fp += 1; cls_fp[k] += 1
        for t in golds:
            if t not in preds: fn += 1; cls_fn[f"{t[1]}_{t[2]}"] += 1
    mp  = tp/(tp+fp)*100 if tp+fp else 0.0
    mr  = tp/(tp+fn)*100 if tp+fn else 0.0
    mf1 = 2*mp*mr/(mp+mr) if mp+mr else 0.0
    all_cls = set(cls_tp)|set(cls_fp)|set(cls_fn)
    cps_list, crs_list, cf1s = [], [], []
    for k in all_cls:
        ctp, cfp, cfn = cls_tp[k], cls_fp[k], cls_fn[k]
        cp2 = ctp/(ctp+cfp)*100 if ctp+cfp else 0.0
        cr2 = ctp/(ctp+cfn)*100 if ctp+cfn else 0.0
        cf  = 2*cp2*cr2/(cp2+cr2) if cp2+cr2 else 0.0
        cps_list.append(cp2); crs_list.append(cr2); cf1s.append(cf)
    return {
        "e2e_tp": tp, "e2e_fp": fp, "e2e_fn": fn,
        "e2e_micro_f1":        round(mf1, 2),
        "e2e_micro_precision": round(mp, 2),
        "e2e_micro_recall":    round(mr, 2),
        "e2e_macro_f1":        round(float(np.mean(cf1s))  if cf1s  else 0.0, 2),
        "e2e_macro_precision": round(float(np.mean(cps_list)) if cps_list else 0.0, 2),
        "e2e_macro_recall":    round(float(np.mean(crs_list)) if crs_list else 0.0, 2),
    }


# ─── Inference time ───────────────────────────────────────────────────────────

def measure_inference_time(model, loader, n_warmup=2):
    model.eval()
    done = 0
    with torch.no_grad():
        for b in loader:
            if done >= n_warmup: break
            model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE), b["lcf_vec"].to(DEVICE))
            done += 1
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_samples = 0
    with torch.no_grad():
        for b in loader:
            model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE), b["lcf_vec"].to(DEVICE))
            n_samples += b["input_ids"].size(0)
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "inference_total_ms":      round(total_ms, 2),
        "inference_per_sample_ms": round(total_ms / max(n_samples, 1), 4),
        "n_test_samples":          n_samples,
    }


# ─── Single-seed training ─────────────────────────────────────────────────────

def train_one_seed(
    *, seed, model_type, pretrained, use_lcf, use_cdm, use_tome,
    tome_resize, merge_strategy, use_pre_tome, short_id,
    train_ds, dev_ds, test_ds, num_sentiment, num_aspect_cat,
    aspect_cat_map, tokenizer, ckpt_dir, ate_csv_path=None, skip_if_exists=False,
):
    meta_path  = ckpt_dir / "meta.json"
    model_path = ckpt_dir / "best_model.pt"
    if skip_if_exists and meta_path.is_file() and model_path.is_file():
        print(f"    [resume] {ckpt_dir.name}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return meta.get("result_dict")

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    dev_loader   = DataLoader(dev_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    bert  = _load_encoder(model_type, pretrained)
    model = FastLcfBertMultiTask(
        bert=bert, num_sentiment=num_sentiment, num_aspect_cat=num_aspect_cat,
        use_lcf=use_lcf, use_cdm=use_cdm, use_tome=use_tome,
        tome_resize=tome_resize, tome_merge_strategy=merge_strategy,
        dropout=DROPOUT, num_heads=NUM_HEADS, tome_merge_steps=TOME_MERGE_STEPS,
        srd_threshold=SRD_THRESHOLD, use_pre_tome=use_pre_tome,
        pre_tome_merge_steps=PRE_TOME_MERGE_STEPS,
        pre_tome_merge_strategy=merge_strategy, pre_tome_resize=tome_resize,
    ).to(DEVICE)

    optimiser  = torch.optim.AdamW(model.parameters(), lr=LR)
    amp_dtype  = _pick_amp_dtype(model_type)
    use_amp    = amp_dtype is not None
    # GradScaler chỉ cần cho fp16; bf16 có cùng dải số mũ với fp32.
    scaler     = torch.amp.GradScaler("cuda", enabled=(amp_dtype is torch.float16))
    print(f"      [AMP] dtype = {amp_dtype or 'fp32 (tat autocast)'}")
    sent_crit  = nn.CrossEntropyLoss(weight=compute_sentiment_class_weights(train_ds))
    cat_crit   = nn.CrossEntropyLoss(weight=compute_category_class_weights(train_ds, aspect_cat_map))

    best_dev_f1 = -1.0; best_epoch = 0; no_improve = 0; best_state = None
    t0 = time.perf_counter()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for b in train_loader:
            ids  = b["input_ids"].to(DEVICE)
            attn = b["attention_mask"].to(DEVICE)
            lcf  = b["lcf_vec"].to(DEVICE)
            ys   = b["sentiment_label"].to(DEVICE)
            yc   = b["aspect_cat_label"].to(DEVICE)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                out  = model(ids, attn, lcf)
                sl   = sent_crit(out["sentiment_logits"], ys)
                mm   = ~b["is_supplement"].to(DEVICE)
                cl   = (cat_crit(out["aspect_cat_logits"][mm], yc[mm])
                        if mm.any() else torch.tensor(0.0, device=DEVICE))
                loss = sl + cl
            optimiser.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.step(optimiser); scaler.update()
            total_loss += loss.item()
        avg_loss = total_loss / max(len(train_loader), 1)
        dev_m    = evaluate(model, dev_loader, sent_crit, cat_crit)
        jf1      = dev_m["joint_f1"]
        print(f"      ep {epoch:2d}/{NUM_EPOCHS}  train={avg_loss:.4f}"
              f"  dev_joint_f1={jf1:.1f}%  dev_sent={dev_m['sentiment_f1']:.1f}%"
              f"  dev_cat={dev_m['aspect_cat_f1']:.1f}%")
        if jf1 > best_dev_f1 + 1e-2:
            best_dev_f1 = jf1; best_epoch = epoch; no_improve = 0
            # Giữ best checkpoint trên CPU: tránh chiếm thêm một bản model
            # trên GPU suốt quá trình train.
            best_state  = {k: v.detach().to("cpu", copy=True)
                           for k, v in model.state_dict().items()}
            torch.save(best_state, ckpt_dir / "best_model.pt")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"      early stop at epoch {epoch}"); break

    train_time = round(time.perf_counter() - t0, 2)
    if best_state is not None:
        model.load_state_dict(best_state)

    test_m  = evaluate(model, test_loader, sent_crit, cat_crit)
    infer   = measure_inference_time(model, test_loader)

    cat_id2label     = {v: k for k, v in aspect_cat_map.items()}
    cat_labels_order = [cat_id2label[i] for i in sorted(cat_id2label)]
    sent_f1_per = f1_score(test_m["sent_true"], test_m["sent_pred"],
                            average=None, zero_division=0,
                            labels=list(range(len(SENTIMENT_LABELS)))) * 100
    cat_f1_per  = f1_score(test_m["cat_true"], test_m["cat_pred"],
                            average=None, zero_division=0,
                            labels=list(range(len(aspect_cat_map)))) * 100

    cs_f1: Dict = {}
    for ci, cat_lbl in enumerate(cat_labels_order):
        idxs = [i for i, c in enumerate(test_m["cat_true"]) if c == ci]
        if not idxs:
            for sl in SENTIMENT_LABELS: cs_f1[f"cs_f1_{cat_lbl}_{sl}"] = 0.0
            continue
        per = f1_score([test_m["sent_true"][i] for i in idxs],
                       [test_m["sent_pred"][i] for i in idxs],
                       average=None, zero_division=0,
                       labels=list(range(len(SENTIMENT_LABELS)))) * 100
        for si, sl in enumerate(SENTIMENT_LABELS):
            cs_f1[f"cs_f1_{cat_lbl}_{sl}"] = round(float(per[si]), 2)

    e2e: Dict = {}
    if ate_csv_path is not None:
        r = eval_triplet_e2e(model, tokenizer, ate_csv_path, aspect_cat_map)
        if r:
            e2e = r
            print(f"      [e2e] MicroF1={r['e2e_micro_f1']:.2f}%"
                  f"  MacroF1={r['e2e_macro_f1']:.2f}%"
                  f"  (TP={r['e2e_tp']} FP={r['e2e_fp']} FN={r['e2e_fn']})")

    result = {
        "seed": seed, "model_type": model_type, "config_id": short_id,
        "train_time_sec": train_time, "best_epoch": best_epoch,
        "best_dev_f1": round(best_dev_f1, 4),
        **infer,
        "sentiment_f1": test_m["sentiment_f1"], "sentiment_acc": test_m["sentiment_acc"],
        "aspect_cat_acc": test_m["aspect_cat_acc"], "aspect_cat_f1": test_m["aspect_cat_f1"],
        "joint_f1":              test_m["joint_f1"],
        "joint_precision":       test_m["joint_precision"],
        "joint_recall":          test_m["joint_recall"],
        "joint_acc":             test_m["joint_acc"],
        "joint_f1_macro":        test_m["joint_f1_macro"],
        "joint_precision_macro": test_m["joint_precision_macro"],
        "joint_recall_macro":    test_m["joint_recall_macro"],
        **{f"sent_f1_{SENTIMENT_LABELS[i]}": round(float(sent_f1_per[i]), 2)
           for i in range(len(SENTIMENT_LABELS))},
        **{f"cat_f1_{cat_labels_order[i]}": round(float(cat_f1_per[i]), 2)
           for i in range(len(cat_labels_order))},
        **cs_f1,
        **e2e,
        "use_lcf": use_lcf, "use_cdm": use_cdm, "use_tome": use_tome,
        "tome_resize": tome_resize, "merge_strategy": merge_strategy,
        "use_pre_tome": use_pre_tome,
    }
    meta = {
        "seed": seed, "model_type": model_type, "config_id": short_id,
        "train_time_sec": train_time, "best_epoch": best_epoch,
        "best_dev_f1": round(best_dev_f1, 4),
        "hyperparams": {
            "num_epochs": NUM_EPOCHS, "patience": PATIENCE,
            "batch_size": BATCH_SIZE, "lr": LR, "max_seq_len": MAX_SEQ_LEN,
            "dropout": DROPOUT, "num_heads": NUM_HEADS,
            "srd_threshold": SRD_THRESHOLD,
            "tome_merge_steps": TOME_MERGE_STEPS,
            "pre_tome_merge_steps": PRE_TOME_MERGE_STEPS,
            "early_stopping": "Dev Joint Micro-F1",
        },
        "result_dict": result,
    }
    (ckpt_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


# ─── Aggregation ──────────────────────────────────────────────────────────────

METRIC_COLS = [
    "train_time_sec", "inference_total_ms", "inference_per_sample_ms", "best_epoch",
    "sentiment_f1", "sentiment_acc", "aspect_cat_acc", "aspect_cat_f1",
    "joint_f1", "joint_precision", "joint_recall", "joint_acc",
    "joint_f1_macro", "joint_precision_macro", "joint_recall_macro",
    "e2e_micro_f1", "e2e_micro_precision", "e2e_micro_recall",
    "e2e_macro_f1", "e2e_macro_precision", "e2e_macro_recall",
]


def aggregate_results(rows: List[Dict]) -> List[Dict]:
    groups: Dict[Tuple, List[Dict]] = defaultdict(list)
    for r in rows:
        groups[(r["model_type"], r["config_id"])].append(r)
    agg = []
    for (mt, cid), grp in groups.items():
        first = grp[0]
        per_class_cols = [k for k in first
                          if (k.startswith(("sent_f1_", "cat_f1_", "cs_f1_")))
                          and not k.endswith(("_pred", "_true"))]
        entry: Dict = {
            "model_type": mt, "config_id": cid,
            "display_name": first.get("display_name", cid),
            "n_seeds": len(grp),
            "seeds": ",".join(str(r["seed"]) for r in grp),
            "use_lcf": first.get("use_lcf"), "use_cdm": first.get("use_cdm"),
            "use_tome": first.get("use_tome"), "tome_resize": first.get("tome_resize"),
            "merge_strategy": first.get("merge_strategy"),
            "use_pre_tome": first.get("use_pre_tome"),
        }
        for col in METRIC_COLS + per_class_cols:
            vals = []
            for r in grp:
                v = r.get(col)
                if v is not None:
                    try: vals.append(float(v))
                    except (TypeError, ValueError): pass
            if vals:
                arr = np.array(vals, dtype=float)
                entry[f"{col}_mean"] = round(float(arr.mean()), 4)
                entry[f"{col}_std"]  = round(float(arr.std(ddof=0)), 4)
            else:
                entry[f"{col}_mean"] = None
                entry[f"{col}_std"]  = None
        agg.append(entry)
    return agg


# ─── Thesis table output ──────────────────────────────────────────────────────

def _lookup(agg: List[Dict], model_type: str, config_id: str) -> Optional[Dict]:
    for r in agg:
        if r["model_type"] == model_type and r["config_id"] == config_id:
            return r
    return None


def _fmean(r: Optional[Dict], col: str, decimals: int = 2) -> str:
    """Format mean±std or just mean if std=0/1seed."""
    if r is None:
        return "—"
    m = r.get(f"{col}_mean")
    s = r.get(f"{col}_std")
    if m is None:
        return "—"
    if r.get("n_seeds", 1) > 1 and s is not None and s > 0:
        return f"{m:.{decimals}f}±{s:.{decimals}f}"
    return f"{m:.{decimals}f}"


def print_thesis_tables(agg: List[Dict], ate_f1: float = GAS_ATE_F1,
                         file=sys.stdout) -> None:
    def pr(*a, **kw): print(*a, **kw, file=file)
    sep = "─" * 100

    # ── [1] tab:bert_results — Oracle BERT (all 12 resize configs) ─────────────
    pr("\n" + "═" * 100)
    pr("[Table 1] tab:bert_results — Oracle BERT (Aspect đúng 100%)")
    pr("          Kết quả các cấu hình trên tập test với backbone BERT")
    pr("═" * 100)
    pr(f"  {'Configuration':<20}  {'Train Time (s)':>15}  {'Micro-F1 (%)':>13}  {'Macro-F1 (%)':>13}")
    pr(sep)
    for cfg in ALL_CONFIGS:
        _, _, _, _, _, _, display_name, short_id = cfg
        r = _lookup(agg, "bert", short_id)
        pr(f"  {display_name:<20}  {_fmean(r,'train_time_sec',1):>15}"
           f"  {_fmean(r,'joint_f1'):>13}  {_fmean(r,'joint_f1_macro'):>13}")
    pr(sep)
    pr("  Micro-F1 / Macro-F1 = Joint micro/macro-F1 (oracle, gold aspect terms)")

    # ── [2] tab:end2end_results — E2E pipeline BERT (5 selected configs) ───────
    pr("\n" + "═" * 100)
    pr("[Table 2] tab:end2end_results — Kết quả end-to-end pipeline hai giai đoạn (BERT)")
    pr("═" * 100)
    pr(f"  {'Configuration':<22}  {'ATE F1 (%)':>11}  {'Micro-F1 (%)':>13}"
       f"  {'Macro-F1 (%)':>13}  {'Joint-Acc (%)':>14}")
    pr(sep)
    for display_name, short_id in E2E_BERT_CONFIGS:
        r = _lookup(agg, "bert", short_id)
        ate_str = f"{ate_f1:.2f}"
        e2e_mf1 = _fmean(r, "e2e_micro_f1")
        e2e_mac = _fmean(r, "e2e_macro_f1")
        j_acc   = _fmean(r, "joint_acc")
        pr(f"  {display_name:<22}  {ate_str:>11}  {e2e_mf1:>13}"
           f"  {e2e_mac:>13}  {j_acc:>14}")
    pr(sep)
    pr("  Micro-F1/Macro-F1 = end-to-end triplet F1 | Joint-Acc = oracle (gold aspects)")
    pr(f"  ATE F1 = {ate_f1:.2f}% (GAS T5-base, cố định cho tất cả cấu hình)")

    # ── [3] tab:t5_results — E2E T5 (5 selected configs) ─────────────────────
    pr("\n" + "═" * 100)
    pr("[Table 3] tab:t5_results — Kết quả các cấu hình trên tập test (backbone T5)")
    pr("═" * 100)
    pr(f"  {'Cấu hình':<22}  {'Train Time (s)':>15}  {'Micro-F1 (%)':>13}"
       f"  {'Macro-F1 (%)':>13}  {'Joint-Acc (%)':>14}")
    pr(sep)
    for display_name, short_id in E2E_T5_CONFIGS:
        r = _lookup(agg, "t5", short_id)
        pr(f"  {display_name:<22}  {_fmean(r,'train_time_sec',1):>15}"
           f"  {_fmean(r,'e2e_micro_f1'):>13}"
           f"  {_fmean(r,'e2e_macro_f1'):>13}"
           f"  {_fmean(r,'joint_acc'):>14}")
    pr(sep)
    pr("  Micro-F1/Macro-F1 = end-to-end triplet F1 | Joint-Acc = oracle (gold aspects)")

    # ── [4] Compact vs Resize Macro-F1 ────────────────────────────────────────
    pr("\n" + "═" * 100)
    pr("[Table 4] Compact vs Resize — So sánh Macro-F1 (end-to-end, tab:compact_vs_resize_micro)")
    pr("═" * 100)
    pr(f"  {'Cấu hình':<20}  {'Resize (%)':>12}  {'Compact (%)':>12}  {'Δ (%)':>8}  Nhận xét")
    pr(sep)
    resize_vals, compact_vals = [], []
    for display_name, resize_id, compact_id in [
        (n, r, c) for r, c, n in COMPACT_VS_RESIZE_CONFIGS
    ]:
        rr = _lookup(agg, "bert", resize_id)   # resize results (BERT e2e)
        rc = _lookup(agg, "bert", compact_id)  # compact results
        rv = rr.get("e2e_macro_f1_mean") if rr else None
        cv = rc.get("e2e_macro_f1_mean") if rc else None
        rv_s = f"{rv:.2f}" if rv is not None else "—"
        cv_s = f"{cv:.2f}" if cv is not None else "—"
        if rv is not None and cv is not None:
            delta = rv - cv
            delta_s = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
            note = "Khác biệt lớn" if abs(delta) > 5 else "Resize tốt hơn"
            resize_vals.append(rv); compact_vals.append(cv)
        else:
            delta_s = "—"; note = "(chưa có dữ liệu compact)"
        pr(f"  {display_name:<20}  {rv_s:>12}  {cv_s:>12}  {delta_s:>8}  {note}")
    if resize_vals and compact_vals:
        avg_r = np.mean(resize_vals); avg_c = np.mean(compact_vals)
        pr(sep)
        pr(f"  {'Trung bình':<20}  {avg_r:>12.2f}  {avg_c:>12.2f}  {avg_r-avg_c:>+8.2f}")
    pr(sep)
    pr("  * Chạy với --include-compact để có dữ liệu compact")

    # ── [5] Compact vs Resize Training Time ───────────────────────────────────
    pr("\n" + "═" * 100)
    pr("[Table 5] Compact vs Resize — So sánh Training Time (tab:compact_vs_resize_micro)")
    pr("═" * 100)
    pr(f"  {'Cấu hình':<20}  {'Resize (s)':>12}  {'Compact (s)':>12}  {'Δ (s)':>10}  Nhận xét")
    pr(sep)
    for display_name, resize_id, compact_id in [
        (n, r, c) for r, c, n in COMPACT_VS_RESIZE_CONFIGS
    ]:
        rr = _lookup(agg, "bert", resize_id)
        rc = _lookup(agg, "bert", compact_id)
        rv = rr.get("train_time_sec_mean") if rr else None
        cv = rc.get("train_time_sec_mean") if rc else None
        rv_s = f"{rv:.1f}" if rv is not None else "—"
        cv_s = f"{cv:.1f}" if cv is not None else "—"
        if rv is not None and cv is not None:
            delta = rv - cv
            delta_s = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
            note = "Compact tốt hơn" if delta > 0 else "Resize tốt hơn rất nhiều"
        else:
            delta_s = "—"; note = "(chưa có dữ liệu compact)"
        pr(f"  {display_name:<20}  {rv_s:>12}  {cv_s:>12}  {delta_s:>10}  {note}")
    pr(sep)

    # ── [6] tab:t5_split — Clause splitting (NOTE: separate run needed) ────────
    pr("\n" + "═" * 100)
    pr("[Table 6] tab:t5_split — Kết quả với tách câu bằng LLM (Ollama Qwen3:8B)")
    pr("═" * 100)
    pr(f"  {'Cấu hình':<25}  {'Micro-Recall (no-split)':>24}  {'Micro-Recall (with-split)':>25}")
    pr(sep)
    split_configs = [
        ("bert", "BERT + Base",          "baseline"),
        ("bert", "BERT + BiToMe + CDM",  "lcf_bip_cdm"),
        ("bert", "BERT + SCM + CDM",     "lcf_scm_cdm"),
        ("bert", "BERT + SLM + CDM",     "lcf_seq_cdm"),
        ("bert", "BERT + SLM + CDW",     "lcf_seq_cdw"),
        ("t5",   "T5 + Base",            "baseline"),
        ("t5",   "T5 + BiToMe + CDM",    "lcf_bip_cdm"),
        ("t5",   "T5 + SCM + CDM",       "lcf_scm_cdm"),
        ("t5",   "T5 + SLM + CDM",       "lcf_seq_cdm"),
        ("t5",   "T5 + SLM + CDW",       "lcf_seq_cdw"),
    ]
    for mt, disp, cid in split_configs:
        r = _lookup(agg, mt, cid)
        nosplit = _fmean(r, "e2e_micro_recall")
        split   = _fmean(r, "e2e_micro_recall_split") if r else "—"
        pr(f"  {disp:<25}  {nosplit:>24}  {split:>25}")
    pr(sep)
    pr("  * Cột 'with-split' cần chạy riêng với --clause-split uos (Ollama Qwen3:8B)")
    pr("  * Cột 'no-split' = e2e_micro_recall từ run hiện tại")

    # ── [7] tab:paper — Comparison with GAS / TOFA ────────────────────────────
    pr("\n" + "═" * 100)
    pr("[Table 7] tab:paper — So sánh với GAS và các phương pháp liên quan")
    pr("═" * 100)
    pr(f"  {'Cấu hình':<30}  {'Micro-F1 (%)':>13}  {'Micro-Precision (%)':>20}  {'Micro-Recall (%)':>17}")
    pr(sep)
    for name, mf1, mp, mr in REF_METHODS:
        pr(f"  {name:<30}  {mf1:>13.2f}  {mp:>20.2f}  {mr:>17.2f}")
    for display_name, mt, cid in PAPER_OUR_CONFIGS:
        r = _lookup(agg, mt, cid)
        pr(f"  {display_name:<30}  {_fmean(r,'e2e_micro_f1'):>13}"
           f"  {_fmean(r,'e2e_micro_precision'):>20}"
           f"  {_fmean(r,'e2e_micro_recall'):>17}")
    pr(sep)

    # ── [8] Inference time summary ─────────────────────────────────────────────
    pr("\n" + "═" * 100)
    pr("[Table 8] Inference Time — Thời gian inference (ms/sample, oracle test set)")
    pr("═" * 100)
    pr(f"  {'Config':<20}  {'Backbone':<6}  {'Total (ms)':>12}  {'Per-sample (ms)':>16}  {'N samples':>10}")
    pr(sep)
    for cfg in ALL_CONFIGS + COMPACT_CONFIGS:
        _, _, _, _, _, _, dname, sid = cfg
        for mt in ["bert", "t5"]:
            r = _lookup(agg, mt, sid)
            if r is None:
                continue
            pr(f"  {sid:<20}  {mt:<6}"
               f"  {_fmean(r,'inference_total_ms',1):>12}"
               f"  {_fmean(r,'inference_per_sample_ms',4):>16}"
               f"  {r.get('n_test_samples_mean', '—'):>10}")
    pr(sep)

    pr("\n" + "═" * 100)
    pr("GHI CHÚ:")
    pr(f"  - Oracle  : aspect term lấy từ nhãn chuẩn (gold), không qua ATE")
    pr(f"  - E2E     : aspect term từ GAS (ATE F1 = {ate_f1:.2f}%), rồi qua APC")
    pr(f"  - Compact : chạy với --include-compact (tome_resize=False)")
    pr(f"  - Split   : chạy clause splitting riêng (cần Ollama Qwen3:8B)")
    pr("═" * 100)


# ─── CSV writers ──────────────────────────────────────────────────────────────

_RAW_PRIORITY = [
    "seed", "model_type", "config_id", "display_name", "ate_csv_seed",
    "train_time_sec", "best_epoch", "best_dev_f1",
    "inference_total_ms", "inference_per_sample_ms", "n_test_samples",
    "joint_f1", "joint_precision", "joint_recall", "joint_acc",
    "joint_f1_macro", "joint_precision_macro", "joint_recall_macro",
    "sentiment_f1", "sentiment_acc", "aspect_cat_acc", "aspect_cat_f1",
    "e2e_micro_f1", "e2e_micro_precision", "e2e_micro_recall",
    "e2e_macro_f1", "e2e_macro_precision", "e2e_macro_recall",
    "e2e_tp", "e2e_fp", "e2e_fn",
    "use_lcf", "use_cdm", "use_tome", "tome_resize", "merge_strategy", "use_pre_tome",
]


def _write_raw_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    keys: List[str] = []; seen: Set[str] = set()
    for k in _RAW_PRIORITY:
        if k not in seen: keys.append(k); seen.add(k)
    for r in rows:
        for k in r:
            if k not in seen and not k.endswith(("_pred", "_true")):
                keys.append(k); seen.add(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def _write_agg_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    keys: List[str] = []; seen: Set[str] = set()
    for k in ["model_type", "config_id", "display_name", "n_seeds", "seeds",
               "use_lcf", "use_cdm", "use_tome", "tome_resize", "merge_strategy", "use_pre_tome"]:
        if k not in seen: keys.append(k); seen.add(k)
    for r in rows:
        for k in r:
            if k not in seen: keys.append(k); seen.add(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-seed runner — outputs all thesis tables"
    )
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, metavar="N",
                   help=f"Seeds (default: {DEFAULT_SEEDS})")
    p.add_argument("--model-types", nargs="+", default=["bert"],
                   choices=list(MODEL_REGISTRY),
                   help="Backbones (default: bert)")
    p.add_argument("--configs", nargs="+", default=None, metavar="ID",
                   help=f"Config IDs (default: all 12). Available: {[c[7] for c in ALL_CONFIGS]}")
    p.add_argument("--include-compact", action="store_true",
                   help="Also run 4 compact configs (for resize vs compact tables)")
    p.add_argument("--ate-csv", default=None,
                   help="Single ATE predictions CSV shared across all seeds "
                        "(e.g. runs_ate/test_ate_predictions.csv). "
                        "Overridden by --ate-csv-dir when both are given.")
    p.add_argument("--ate-csv-dir", default=None, metavar="DIR",
                   help="Directory with per-seed ATE CSVs produced by run_multiseed_ate.py. "
                        "For each seed N the script looks for <DIR>/seed_N/test_predictions.csv. "
                        "This ensures stage-1 (ATE) and stage-2 (APC) use the SAME seed. "
                        "Example: --ate-csv-dir runs_ate")
    p.add_argument("--ate-f1", type=float, default=GAS_ATE_F1,
                   help=f"GAS ATE F1 score for table headers (default: {GAS_ATE_F1})")
    p.add_argument("--resume", action="store_true",
                   help="Skip finished (backbone, config, seed) combos")
    p.add_argument("--no-train", action="store_true",
                   help="Skip training, only aggregate existing results")
    p.add_argument("--runs-dir", default=None,
                   help=f"Output dir (default: {RUNS_DIR})")
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    global RUNS_DIR
    if args.runs_dir:
        RUNS_DIR = Path(args.runs_dir)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    seeds       = args.seeds
    model_types = args.model_types

    # Resolve configs to run
    if args.configs:
        unknown = [c for c in args.configs if c not in CONFIG_BY_ID]
        if unknown:
            print(f"[ERROR] Unknown config IDs: {unknown}")
            print(f"        Available: {list(CONFIG_BY_ID)}")
            sys.exit(1)
        configs = [CONFIG_BY_ID[c] for c in args.configs]
    else:
        configs = list(ALL_CONFIGS)
        if args.include_compact:
            configs += COMPACT_CONFIGS

    # ── Resolve ATE CSV strategy ──────────────────────────────────────────────
    # Priority: --ate-csv-dir > --ate-csv > default single CSV
    #
    # --ate-csv-dir  : seed-paired (recommended).  For seed N uses:
    #                    <dir>/seed_N/test_predictions.csv
    #                  Generated by run_multiseed_ate.py.
    # --ate-csv      : single CSV shared across all seeds (backward compat).
    #                  Only use if ATE was trained with a single seed.

    ate_csv_dir: Optional[Path] = None
    ate_csv_path: Optional[Path] = None   # fallback shared CSV

    if args.ate_csv_dir:
        ate_csv_dir = Path(args.ate_csv_dir)
        if not ate_csv_dir.is_dir():
            print(f"  [warn] --ate-csv-dir {ate_csv_dir} is not a directory — ignored")
            ate_csv_dir = None
        else:
            print(f"  ATE CSV dir: {ate_csv_dir}  → seed-paired e2e eval enabled")
            print(f"    Expected layout: {ate_csv_dir}/seed_<N>/test_predictions.csv")

    if ate_csv_dir is None:
        # Fall back to shared CSV
        if args.ate_csv:
            ate_csv_path = Path(args.ate_csv)
        else:
            _default_ate = ROOT / "runs_ate" / "test_ate_predictions.csv"
            ate_csv_path = _default_ate if _default_ate.is_file() else None
        if ate_csv_path and ate_csv_path.is_file():
            print(f"  ATE CSV (shared): {ate_csv_path}  → e2e eval enabled (not seed-paired)")
        else:
            print(f"  ATE CSV not found → e2e eval disabled (run run_multiseed_ate.py first)")
            ate_csv_path = None

    raw_csv      = RUNS_DIR / "results_raw.csv"
    agg_csv      = RUNS_DIR / "results_aggregated.csv"
    summ_txt     = RUNS_DIR / "results_summary.txt"
    thesis_txt   = RUNS_DIR / "thesis_tables.txt"

    # Load existing results
    existing: List[Dict] = []
    if raw_csv.is_file():
        with open(raw_csv, newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
        for r in existing:
            for col in METRIC_COLS + ["seed", "best_epoch", "n_test_samples",
                                       "e2e_tp", "e2e_fp", "e2e_fn"]:
                if col in r:
                    try: r[col] = float(r[col])
                    except (ValueError, TypeError): pass
            for flag in ["use_lcf", "use_cdm", "use_tome", "tome_resize", "use_pre_tome"]:
                if flag in r:
                    r[flag] = r[flag] in ("True", "true", "1", True)

    all_rows: List[Dict] = list(existing)
    done_keys = {
        (str(r["model_type"]), str(r["config_id"]), int(float(r["seed"])))
        for r in existing
    }

    if args.no_train:
        print("[--no-train] Aggregating existing results only.")
    else:
        for model_type in model_types:
            pretrained = MODEL_REGISTRY[model_type]
            print(f"\n{'=' * 70}")
            print(f"Backbone: {model_type.upper()} ({pretrained})")
            print(f"Seeds:    {seeds}")
            print(f"Configs:  {[c[7] for c in configs]}")
            print(f"Device:   {DEVICE}")
            print(f"{'=' * 70}")

            tokenizer = AutoTokenizer.from_pretrained(pretrained)
            _, aspect_cat_map = build_label_maps_from_apc(
                str(TRAIN_APC), str(DEV_APC), str(TEST_APC)
            )
            num_sentiment  = len(SENTIMENT_MAP)
            num_aspect_cat = len(aspect_cat_map)

            avail_supp = (
                [p for p in SUPPLEMENT_FILES if Path(p).is_file()]
                if USE_SUPPLEMENT else []
            )
            train_ds = ApcFileDataset(str(TRAIN_APC), tokenizer, aspect_cat_map,
                                      MAX_SEQ_LEN, supplement_paths=avail_supp or None)
            dev_ds   = ApcFileDataset(str(DEV_APC),   tokenizer, aspect_cat_map, MAX_SEQ_LEN)
            test_ds  = ApcFileDataset(str(TEST_APC),  tokenizer, aspect_cat_map, MAX_SEQ_LEN)
            print(f"  Train={len(train_ds)} Dev={len(dev_ds)} Test={len(test_ds)}")

            total_runs = len(configs) * len(seeds)
            run_no = 0

            for cfg in configs:
                use_lcf, use_cdm, use_tome, tome_resize, merge_strategy, \
                    use_pre_tome, display_name, short_id = cfg

                for seed in seeds:
                    run_no += 1
                    key = (model_type, short_id, seed)
                    if key in done_keys:
                        print(f"\n  [{run_no}/{total_runs}] SKIP {model_type}/{short_id}/seed={seed}")
                        continue
                    print(f"\n  [{run_no}/{total_runs}] {model_type.upper()} | {display_name} | seed={seed}")
                    print(f"    lcf={use_lcf} cdm={use_cdm} tome={use_tome} "
                          f"resize={tome_resize} strategy={merge_strategy}")

                    # ── Select ATE CSV for this seed ─────────────────────────
                    # Seed-paired takes priority: ATE(seed=N) → APC(seed=N)
                    if ate_csv_dir is not None:
                        seed_ate_csv: Optional[Path] = (
                            ate_csv_dir / f"seed_{seed}" / "test_predictions.csv"
                        )
                        if not seed_ate_csv.is_file():
                            print(f"    [warn] Seed-paired ATE CSV not found: {seed_ate_csv}")
                            print(f"           Run: python common/run_multiseed_ate.py --seeds {seed}")
                            seed_ate_csv = None
                        else:
                            print(f"    ATE (seed={seed}): {seed_ate_csv}")
                    else:
                        seed_ate_csv = ate_csv_path  # shared fallback

                    ckpt_dir = RUNS_DIR / model_type / short_id / f"seed_{seed}"
                    t_start  = time.perf_counter()
                    result   = train_one_seed(
                        seed=seed, model_type=model_type, pretrained=pretrained,
                        use_lcf=use_lcf, use_cdm=use_cdm, use_tome=use_tome,
                        tome_resize=tome_resize, merge_strategy=merge_strategy,
                        use_pre_tome=use_pre_tome, short_id=short_id,
                        train_ds=train_ds, dev_ds=dev_ds, test_ds=test_ds,
                        num_sentiment=num_sentiment, num_aspect_cat=num_aspect_cat,
                        aspect_cat_map=aspect_cat_map, tokenizer=tokenizer,
                        ckpt_dir=ckpt_dir, ate_csv_path=seed_ate_csv,
                        skip_if_exists=args.resume,
                    )
                    elapsed = time.perf_counter() - t_start

                    # Trả VRAM về trước khi dựng model của run kế tiếp.
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    if result is None:
                        print(f"    [warn] No result for {short_id}/seed={seed}")
                        continue
                    result["display_name"] = display_name
                    result["ate_csv_seed"] = (
                        str(seed_ate_csv) if seed_ate_csv else ""
                    )
                    all_rows.append(result)
                    done_keys.add(key)

                    e2e_str = (f"  e2e_MicF1={result['e2e_micro_f1']:.2f}%"
                               if "e2e_micro_f1" in result else "")
                    print(f"    Done {elapsed:.0f}s | train={result['train_time_sec']:.1f}s"
                          f"  infer={result['inference_per_sample_ms']:.2f}ms/samp"
                          f"  Oracle MicF1={result['joint_f1']:.2f}%"
                          f"  MacF1={result['joint_f1_macro']:.2f}%{e2e_str}")

                    _write_raw_csv(raw_csv, all_rows)

    if not all_rows:
        print("[warn] No results to aggregate.")
        return

    for r in all_rows:
        if "display_name" not in r:
            cfg = CONFIG_BY_ID.get(str(r["config_id"]))
            r["display_name"] = cfg[6] if cfg else str(r["config_id"])

    agg = aggregate_results(all_rows)
    _write_raw_csv(raw_csv, all_rows)
    _write_agg_csv(agg_csv, agg)

    # ── Print thesis tables ────────────────────────────────────────────────────
    print_thesis_tables(agg, ate_f1=args.ate_f1)
    with open(thesis_txt, "w", encoding="utf-8") as f:
        print_thesis_tables(agg, ate_f1=args.ate_f1, file=f)

    # ── Brief summary to results_summary.txt ──────────────────────────────────
    with open(summ_txt, "w", encoding="utf-8") as f:
        print(f"Multi-seed results — {len(all_rows)} runs", file=f)
        print(f"Seeds: {seeds}  Backbones: {model_types}", file=f)
        print(f"Configs: {[c[7] for c in configs]}", file=f)

    print(f"\nThesis tables → {thesis_txt}")
    print(f"Raw CSV       → {raw_csv}")
    print(f"Aggregated    → {agg_csv}")
    print(f"Checkpoints   → {RUNS_DIR}/<backbone>/<config>/seed_<N>/best_model.pt")


if __name__ == "__main__":
    main()
