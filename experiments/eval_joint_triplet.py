# -*- coding: utf-8 -*-
"""Joint triplet micro F1: (aspect term, category, sentiment) cả 3 phải đúng.

Pipeline:
    ATE predictions  (runs_ate/test_ate_predictions.csv)
        → Joint model (runs_joint/<config>/best_model.pt)
            → predicted (category, sentiment) cho mỗi predicted term
                → so với gold triplets từ dataset/test.apc

Metric (micro, theo chuẩn ABSA triplet evaluation):
    TP  = predicted triplet khớp CHÍNH XÁC với một gold triplet
    FP  = predicted triplet không khớp với bất kỳ gold nào
    FN  = gold triplet không được predict đúng
    P   = TP / (TP + FP)
    R   = TP / (TP + FN)
    F1  = 2PR / (P + R)

Usage (from thesis_apc_baseline/):
    python experiments/eval_joint_triplet.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from transformers import AutoModel, T5EncoderModel, AutoTokenizer

from common.dataset_utils import parse_apc_file, SENTIMENT_LABELS
from models.fast_lcf_bert_multitask import FastLcfBertMultiTask

# ─── Config ───────────────────────────────────────────────────────────────────

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_APC     = ROOT / "dataset" / "test.apc"
ATE_CSV      = ROOT / "runs_ate" / "test_ate_predictions.csv"
RUNS_DIR     = ROOT / "runs_joint"
OUT_CSV      = ROOT / "runs_ate" / "eval_joint_triplet.csv"

# ─── Model selection ──────────────────────────────────────────────────────────
# Change MODEL_TYPE to switch between encoders (must match run_joint_experiments.py).
MODEL_TYPE = "bert"   # "bert" | "t5"

_MODEL_CONFIGS = {
    "bert": "bert-base-uncased",
    # "t5":   "t5-base",
}
if MODEL_TYPE not in _MODEL_CONFIGS:
    raise ValueError(f"Unknown MODEL_TYPE={MODEL_TYPE!r}. Choose from: {list(_MODEL_CONFIGS)}")

PRETRAINED   = _MODEL_CONFIGS[MODEL_TYPE]
MAX_SEQ_LEN  = 128
BATCH_SIZE   = 32


def _load_encoder(model_type: str, pretrained: str):
    """Load the correct HuggingFace encoder for the given model_type."""
    if model_type == "t5":
        return T5EncoderModel.from_pretrained(pretrained)
    return AutoModel.from_pretrained(pretrained)

# Defaults matching run_joint_experiments.py
_DEFAULT_CFG = dict(
    dropout=0.1, num_heads=8, srd_threshold=5,
    tome_merge_steps=2, pre_tome_merge_steps=1,
)

# ─── Term normalisation ───────────────────────────────────────────────────────

def _norm_term(t: str) -> str:
    """Lowercase + strip whitespace and leading/trailing '.' ',' from a term."""
    return t.strip(" \t\n\r.,").lower()


# ─── Gold triplets ────────────────────────────────────────────────────────────

def load_gold_triplets(
    apc_path: Path,
) -> Dict[str, Set[Tuple[str, str, str]]]:
    """Parse test.apc → {sentence_text: {(term, category, sentiment), ...}}"""
    gold: Dict[str, Set[Tuple[str, str, str]]] = defaultdict(set)
    for s in parse_apc_file(str(apc_path)):
        key = s["text"]
        triple = (
            _norm_term(s["aspect_term"]),
            s["aspect_category"].strip().upper(),
            s["sentiment"].strip().lower(),
        )
        gold[key].add(triple)
    return dict(gold)


# ─── ATE predictions ──────────────────────────────────────────────────────────

def load_ate_predictions(
    ate_csv: Path,
    gold_sentences: List[str] = None,
) -> Dict[str, List[str]]:
    """Load runs_ate/test_ate_predictions.csv → {sentence: [predicted_term, ...]}.

    gold_sentences (optional): ordered list of gold sentence texts from test.apc.
        When provided and len matches the CSV row count, each row is keyed by
        the corresponding gold sentence at the SAME INDEX — robust to any text
        discrepancy between the CSV sentence column and test.apc (different word
        order, unicode quote variants, trailing punctuation, etc.).
        When omitted the raw CSV sentence column is used as the key.
    """
    preds: Dict[str, List[str]] = defaultdict(list)
    with open(ate_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    use_index = gold_sentences is not None and len(rows) == len(gold_sentences)
    if gold_sentences is not None and not use_index:
        print(f"[warn] load_ate_predictions: gold_sentences length ({len(gold_sentences)}) "
              f"!= CSV rows ({len(rows)}) — falling back to CSV sentence text as key")

    for idx, row in enumerate(rows):
        term = row["predicted_term"].strip()
        if term:
            key = gold_sentences[idx] if use_index else row["sentence"]
            preds[key].append(term)
    return dict(preds)

# ─── Tokenise one (sentence, term) pair ──────────────────────────────────────

def tokenize_single(
    text: str,
    aspect_term: str,
    tokenizer,
    max_seq_len: int = 128,
) -> Dict[str, torch.Tensor]:
    pad = tokenizer.pad_token or "[PAD]"
    aspect = aspect_term.strip() or pad

    enc = tokenizer(
        text, aspect,
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

    asp = aspect_term.strip()
    cs  = text.find(asp) if asp else -1
    ce  = cs + len(asp) - 1 if cs >= 0 else -1

    lcf = torch.zeros_like(input_ids, dtype=torch.float32)
    if cs >= 0:
        for k in range(input_ids.size(0)):
            if not attention_mask[k]: continue
            if token_type_ids[k]: continue
            ts, te = int(offsets[k, 0]), int(offsets[k, 1])
            if ts == 0 and te == 0: continue
            if te > cs and ts <= ce:
                lcf[k] = 1.0
    if lcf.sum() == 0:
        lcf = token_type_ids.float()

    return {"input_ids": input_ids, "attention_mask": attention_mask, "lcf_vec": lcf}


# ─── Load one joint model ─────────────────────────────────────────────────────

def load_joint_model(
    run_dir: Path,
    tokenizer,
) -> Tuple[nn.Module, Dict, List[str]]:
    """Load best_model.pt + meta.json from a run directory."""
    meta_path  = run_dir / "meta.json"
    ckpt_path  = run_dir / "best_model.pt"
    if not meta_path.is_file() or not ckpt_path.is_file():
        raise FileNotFoundError(run_dir)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cfg  = meta["config"]
    cat_labels: List[str] = meta["category_labels"]
    num_cat = len(cat_labels)
    cat_map = {lbl: i for i, lbl in enumerate(cat_labels)}

    train_time_sec: float = meta.get("train_time_sec", float("nan"))

    bert  = _load_encoder(MODEL_TYPE, PRETRAINED)
    model = FastLcfBertMultiTask(
        bert=bert,
        num_sentiment=len(SENTIMENT_LABELS),
        num_aspect_cat=num_cat,
        use_lcf=cfg.get("use_lcf", True),
        use_cdm=cfg.get("use_cdm", True),
        use_tome=cfg.get("use_tome", False),
        tome_resize=cfg.get("tome_resize", True),
        tome_merge_strategy=cfg.get("merge_strategy", "bipartite"),
        use_pre_tome=cfg.get("use_pre_tome", False),
        pre_tome_merge_steps=cfg.get("pre_tome_merge_steps", _DEFAULT_CFG["pre_tome_merge_steps"]),
        pre_tome_merge_strategy=cfg.get("merge_strategy", "bipartite"),
        pre_tome_resize=cfg.get("tome_resize", True),
        dropout=_DEFAULT_CFG["dropout"],
        num_heads=_DEFAULT_CFG["num_heads"],
        tome_merge_steps=_DEFAULT_CFG["tome_merge_steps"],
        srd_threshold=_DEFAULT_CFG["srd_threshold"],
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model, cat_map, cat_labels, train_time_sec


# ─── Run inference on ATE predictions for one config ─────────────────────────

def predict_triplets(
    model: nn.Module,
    tokenizer,
    ate_preds: Dict[str, List[str]],
    cat_map: Dict[str, int],
    cat_labels: List[str],
) -> Dict[str, Set[Tuple[str, str, str]]]:
    """Return {sentence: {(term, cat, sent), ...}} from joint model predictions.

    End-to-end: dùng TOÀN BỘ ATE predicted term, không lọc theo gold.
    Mỗi predicted term → 1 triplet để so trực tiếp với gold (term sai bị tính FP).
    """
    id2cat   = {i: lbl for lbl, i in cat_map.items()}
    results: Dict[str, Set[Tuple[str, str, str]]] = {}

    sentences = list(ate_preds.keys())
    # Flatten to (sentence_idx, term) pairs for batching
    flat: List[Tuple[int, str]] = []
    for si, sent in enumerate(sentences):
        for term in ate_preds[sent]:
            flat.append((si, term))

    pred_cat_all:  List[int] = []
    pred_sent_all: List[int] = []

    for start in range(0, len(flat), BATCH_SIZE):
        chunk = flat[start : start + BATCH_SIZE]
        samples = [
            tokenize_single(sentences[si], term, tokenizer, MAX_SEQ_LEN)
            for si, term in chunk
        ]
        ids  = torch.stack([s["input_ids"]     for s in samples]).to(DEVICE)
        attn = torch.stack([s["attention_mask"] for s in samples]).to(DEVICE)
        lcf  = torch.stack([s["lcf_vec"]        for s in samples]).to(DEVICE)

        with torch.no_grad():
            out = model(ids, attn, lcf)
        pred_cat_all  += out["aspect_cat_logits"].argmax(-1).cpu().tolist()
        pred_sent_all += out["sentiment_logits"].argmax(-1).cpu().tolist()

    # Re-assemble per sentence
    for si, sent in enumerate(sentences):
        results[sent] = set()

    ptr = 0
    for si, sent in enumerate(sentences):
        for term in ate_preds[sent]:
            cp = pred_cat_all[ptr]
            sp = pred_sent_all[ptr]
            ptr += 1
            results[sent].add((
                _norm_term(term),
                id2cat[cp].upper(),
                SENTIMENT_LABELS[sp].lower(),
            ))

    return results


# ─── Oracle (upper-bound): joint head trên GOLD aspect term ──────────────────

def oracle_metrics(
    model: nn.Module,
    tokenizer,
    apc_path: Path,
    cat_map: Dict[str, int],
    cat_labels: List[str],
) -> Dict[str, float]:
    """Oracle upper-bound: chạy joint head trên GOLD aspect term (bỏ qua ATE).

    Vì aspect term LUÔN đúng, metric này đo riêng chất lượng tầng phân loại
    (category + sentiment), không pha lẫn lỗi của ATE. Đây là cận trên cho
    end-to-end triplet F1 — phục vụ error analysis: bao nhiêu lỗi triplet đến
    từ ATE (term sai) vs từ tầng phân loại.

    Term luôn đúng nên với mỗi gold triplet có đúng 1 prediction → precision =
    recall = accuracy; báo cáo accuracy cho category, sentiment và joint (cả hai).
    """
    id2cat = {i: lbl for lbl, i in cat_map.items()}

    # Mỗi gold triplet = 1 sample (sentence, gold_term, gold_cat, gold_sent)
    flat: List[Tuple[str, str, str, str]] = []
    for s in parse_apc_file(str(apc_path)):
        flat.append((
            s["text"],
            s["aspect_term"],
            s["aspect_category"].strip().upper(),
            s["sentiment"].strip().lower(),
        ))

    n = len(flat)
    cat_ok = sent_ok = both_ok = 0

    for start in range(0, n, BATCH_SIZE):
        chunk = flat[start : start + BATCH_SIZE]
        samples = [
            tokenize_single(text, term, tokenizer, MAX_SEQ_LEN)
            for text, term, _, _ in chunk
        ]
        ids  = torch.stack([s["input_ids"]      for s in samples]).to(DEVICE)
        attn = torch.stack([s["attention_mask"] for s in samples]).to(DEVICE)
        lcf  = torch.stack([s["lcf_vec"]         for s in samples]).to(DEVICE)

        with torch.no_grad():
            out = model(ids, attn, lcf)
        pcat  = out["aspect_cat_logits"].argmax(-1).cpu().tolist()
        psent = out["sentiment_logits"].argmax(-1).cpu().tolist()

        for (_, _, gcat, gsent), cp, sp in zip(chunk, pcat, psent):
            c = (id2cat[cp].upper()          == gcat)
            s = (SENTIMENT_LABELS[sp].lower() == gsent)
            cat_ok  += int(c)
            sent_ok += int(s)
            both_ok += int(c and s)

    return {
        "n":          n,
        "cat_acc":    round(cat_ok  / max(n, 1) * 100, 2),
        "sent_acc":   round(sent_ok / max(n, 1) * 100, 2),
        "joint_acc":  round(both_ok / max(n, 1) * 100, 2),
    }


# ─── Micro P/R/F1 over all sentences ─────────────────────────────────────────

def micro_prf(
    gold_by_sent:  Dict[str, Set[Tuple]],
    pred_by_sent:  Dict[str, Set[Tuple]],
    all_sentences: List[str],
) -> Dict[str, float]:
    total_tp = total_fp = total_fn = 0
    for sent in all_sentences:
        gold = gold_by_sent.get(sent, set())
        pred = pred_by_sent.get(sent, set())
        tp = len(gold & pred)
        total_tp += tp
        total_fp += len(pred) - tp
        total_fn += len(gold) - tp

    p  = total_tp / max(total_tp + total_fp, 1)
    r  = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return {
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "precision": round(p * 100, 2),
        "recall":    round(r * 100, 2),
        "f1":        round(f1 * 100, 2),
    }


# ─── Macro P/R/F1 over categories ────────────────────────────────────────────

def macro_prf(
    gold_by_sent:  Dict[str, Set[Tuple]],
    pred_by_sent:  Dict[str, Set[Tuple]],
    all_sentences: List[str],
) -> Dict[str, float]:
    """Macro F1 over triplet classes (category, sentiment) pairs.

    Mỗi class = 1 unique (category, sentiment) combination trong gold.
    Average F1 across classes — equal weight cho minority classes.
    """
    from collections import defaultdict as _dd

    cls_tp: Dict[Tuple, int] = _dd(int)
    cls_fp: Dict[Tuple, int] = _dd(int)
    cls_fn: Dict[Tuple, int] = _dd(int)

    for sent in all_sentences:
        gold = gold_by_sent.get(sent, set())
        pred = pred_by_sent.get(sent, set())

        # Group by (category, sentiment) — index 1 and 2 of triplet
        gold_by_cls: Dict[Tuple, set] = _dd(set)
        pred_by_cls: Dict[Tuple, set] = _dd(set)
        for t in gold:
            gold_by_cls[(t[1], t[2])].add(t)
        for t in pred:
            pred_by_cls[(t[1], t[2])].add(t)

        for cls in set(gold_by_cls) | set(pred_by_cls):
            tp = len(gold_by_cls[cls] & pred_by_cls[cls])
            cls_tp[cls] += tp
            cls_fp[cls] += len(pred_by_cls[cls]) - tp
            cls_fn[cls] += len(gold_by_cls[cls]) - tp

    # Only average over classes present in gold
    gold_classes = {(t[1], t[2]) for s in all_sentences for t in gold_by_sent.get(s, set())}
    if not gold_classes:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "per_class": {}}

    p_list, r_list, f1_list = [], [], []
    per_class: Dict[str, float] = {}
    for cls in sorted(gold_classes):
        tp = cls_tp[cls]; fp = cls_fp[cls]; fn = cls_fn[cls]
        p  = tp / max(tp + fp, 1)
        r  = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        p_list.append(p); r_list.append(r); f1_list.append(f1)
        per_class[f"{cls[0]}_{cls[1]}"] = round(f1 * 100, 2)

    n = len(gold_classes)
    return {
        "precision": round(sum(p_list)  / n * 100, 2),
        "recall":    round(sum(r_list)  / n * 100, 2),
        "f1":        round(sum(f1_list) / n * 100, 2),
        "per_class": per_class,
    }


# ─── ATE-only metrics (term extraction, ignore cat+sent) ─────────────────────

def ate_metrics(
    gold_by_sent:  Dict[str, Set[Tuple]],
    ate_preds:     Dict[str, List[str]],
    all_sentences: List[str],
) -> Dict[str, float]:
    total_tp = total_fp = total_fn = 0
    visited: set = set()
    for sent in all_sentences:
        visited.add(sent)
        gold_terms = {t for t, _, _ in gold_by_sent.get(sent, set())}
        pred_terms = {_norm_term(t) for t in ate_preds.get(sent, [])}
        tp = len(gold_terms & pred_terms)
        total_tp += tp
        total_fp += len(pred_terms) - tp
        total_fn += len(gold_terms) - tp
    # Count FP for predicted sentences that had no gold entry (key mismatch guard)
    for sent, terms in ate_preds.items():
        if sent not in visited:
            total_fp += len({_norm_term(t) for t in terms if t.strip()})
    p  = total_tp / max(total_tp + total_fp, 1)
    r  = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return {"precision": round(p*100,2), "recall": round(r*100,2), "f1": round(f1*100,2)}


# ─── Entity / sentence hit accuracy ─────────────────────────────────────────

def entity_hit_acc(
    gold_by_key: Dict,
    pred_by_key: Dict,
    all_keys: List,
) -> Dict:
    """% of keys (entity_id or sentence) where ≥1 predicted triplet matches gold.

    Binary recall metric — does NOT penalise extra predictions per key.
    Hit = 1 iff |gold ∩ pred| ≥ 1 for that key.
    """
    hits = sum(
        1 for k in all_keys
        if len(gold_by_key.get(k, set()) & pred_by_key.get(k, set())) > 0
    )
    total = len(all_keys)
    return {
        "hits":     hits,
        "total":    total,
        "accuracy": round(hits / max(total, 1) * 100, 2),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Device : {DEVICE}")
    print(f"Loading gold triplets from {TEST_APC} …")
    gold_by_sent = load_gold_triplets(TEST_APC)
    all_sentences = list(gold_by_sent.keys())
    print(f"  {len(all_sentences)} unique sentences, "
          f"{sum(len(v) for v in gold_by_sent.values())} gold triplets")

    print(f"Loading ATE predictions from {ATE_CSV} …")
    # Pass gold_sentences for index-based alignment: CSV row i → gold sentence i.
    # This is robust to minor text discrepancies in the CSV's 'sentence' column
    # (e.g. different unicode quote variants, word-order issues in 2 edge cases).
    gold_sentences_ordered = [s["text"] for s in parse_apc_file(str(TEST_APC))]
    ate_preds = load_ate_predictions(ATE_CSV, gold_sentences_ordered)
    print(f"  {sum(len(v) for v in ate_preds.values())} predicted terms "
          f"across {len(ate_preds)} sentences")

    # ATE-only baseline
    ate_m = ate_metrics(gold_by_sent, ate_preds, all_sentences)
    print(f"\nATE (term only): P={ate_m['precision']}%  R={ate_m['recall']}%  "
          f"F1={ate_m['f1']}%")

    # Discover configs
    run_dirs = sorted(
        d for d in RUNS_DIR.iterdir()
        if d.is_dir() and (d / "best_model.pt").is_file() and (d / "meta.json").is_file()
    )
    if not run_dirs:
        print(f"\nNo trained configs found in {RUNS_DIR}")
        return

    print(f"\nFound {len(run_dirs)} config(s): {[d.name for d in run_dirs]}")
    print(f"Loading tokenizer: {PRETRAINED}")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED)

    rows: List[Dict] = []

    for run_dir in run_dirs:
        config_name = run_dir.name
        print(f"\n── Config: {config_name} ──")
        try:
            model, cat_map, cat_labels, train_time_sec = load_joint_model(run_dir, tokenizer)
        except Exception as e:
            print(f"  [error] {e}")
            continue

        print(f"  Running inference on {sum(len(v) for v in ate_preds.values())} terms …")
        pred_by_sent = predict_triplets(model, tokenizer, ate_preds, cat_map, cat_labels)

        m      = micro_prf(gold_by_sent, pred_by_sent, all_sentences)
        macro  = macro_prf(gold_by_sent, pred_by_sent, all_sentences)
        print(f"  Micro → P={m['precision']}%  R={m['recall']}%  F1={m['f1']}%"
              f"  (TP={m['tp']} FP={m['fp']} FN={m['fn']})")
        print(f"  Macro → P={macro['precision']}%  R={macro['recall']}%  F1={macro['f1']}%")
        for cls, f1_cls in macro["per_class"].items():
            print(f"    {cls:<22}: {f1_cls:.2f}%")

        oracle = oracle_metrics(model, tokenizer, TEST_APC, cat_map, cat_labels)
        print(f"  Oracle (gold term, upper bound) → "
              f"Cat-Acc={oracle['cat_acc']}%  Sent-Acc={oracle['sent_acc']}%  "
              f"Joint-Acc={oracle['joint_acc']}%  (n={oracle['n']})")

        rows.append({
            "config":            config_name,
            "train_time_sec":    train_time_sec,
            "ate_f1":            ate_m["f1"],
            "ate_precision":     ate_m["precision"],
            "ate_recall":        ate_m["recall"],
            "micro_precision":   m["precision"],
            "micro_recall":      m["recall"],
            "micro_f1":          m["f1"],
            "macro_precision":   macro["precision"],
            "macro_recall":      macro["recall"],
            "macro_f1":          macro["f1"],
            "oracle_cat_acc":    oracle["cat_acc"],
            "oracle_sent_acc":   oracle["sent_acc"],
            "oracle_joint_acc":  oracle["joint_acc"],
            "tp":                m["tp"],
            "fp":                m["fp"],
            "fn":                m["fn"],
            **{f"f1_{cls}": macro["per_class"].get(cls, 0.0)
               for cls in sorted(macro["per_class"])},
        })

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Summary table ──────────────────────────────────────────────────────────
    W = 158
    print(f"\n{'═' * W}")
    print("JOINT TRIPLET EVALUATION  (term ∩ category ∩ sentiment — ALL 3 phải đúng)")
    print(f"{'═' * W}")
    print(f"  {'Config':<26}  {'Time(s)':>8}  {'ATE-F1':>8}"
          f"  {'Micro-P':>8}  {'Micro-R':>8}  {'Micro-F1':>9}"
          f"  {'Macro-P':>8}  {'Macro-R':>8}  {'Macro-F1':>9}"
          f"  {'OrcCat':>8}  {'OrcSent':>8}  {'OrcJoint':>9}"
          f"  {'TP':>5}  {'FP':>5}  {'FN':>5}")
    print(f"{'─' * W}")
    for row in rows:
        t = row["train_time_sec"]
        time_str = f"{t:>7.1f}s" if t == t else "     N/A"
        print(
            f"  {row['config']:<26}  {time_str}"
            f"  {row['ate_f1']:>7.2f}%"
            f"  {row['micro_precision']:>7.2f}%"
            f"  {row['micro_recall']:>7.2f}%"
            f"  {row['micro_f1']:>8.2f}%"
            f"  {row['macro_precision']:>7.2f}%"
            f"  {row['macro_recall']:>7.2f}%"
            f"  {row['macro_f1']:>8.2f}%"
            f"  {row['oracle_cat_acc']:>7.2f}%"
            f"  {row['oracle_sent_acc']:>7.2f}%"
            f"  {row['oracle_joint_acc']:>8.2f}%"
            f"  {row['tp']:>5}  {row['fp']:>5}  {row['fn']:>5}"
        )
    print(f"{'═' * W}")
    print("  ATE-F1    : F1 trích xuất aspect term (T5)")
    print("  Micro-F1  : micro F1 triplet — phản ánh overall performance")
    print("  Macro-F1  : macro F1 triplet theo category — nhạy với minority class")
    print("  Orc*      : ORACLE trên gold term (upper bound) — Cat/Sent/Joint accuracy")
    print("              đo riêng tầng phân loại, KHÔNG tính lỗi của ATE")
    print(f"{'═' * W}")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    # Collect all category keys that appeared across any config
    all_cat_keys = sorted({k for row in rows for k in row if k.startswith("f1_")})
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["config", "train_time_sec", "ate_f1", "ate_precision", "ate_recall",
                        "micro_precision", "micro_recall", "micro_f1",
                        "macro_precision", "macro_recall", "macro_f1",
                        "oracle_cat_acc", "oracle_sent_acc", "oracle_joint_acc",
                        "tp", "fp", "fn"] + all_cat_keys,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved → {OUT_CSV}")


if __name__ == "__main__":
    main()