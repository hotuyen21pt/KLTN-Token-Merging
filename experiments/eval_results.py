# -*- coding: utf-8 -*-
"""Eval joint triplet từ results.csv qua 2 bước:

    Step 1 — ATE: GAS T5 chạy trên từng unit → extract aspect terms
    Step 2 — APC: Joint model (Bert/T5) chạy trên (sentence, term) → category + sentiment

Pipeline:
    results.csv (entity_id, sentence, unit)
        → GAS T5 (unit) → [(term, ...), ...]  — chỉ lấy term
            → Joint model (sentence, term) → (category, sentiment)
                → so triplet (term, cat, sent) với gold từ test_sentences_id.csv

Metric: micro/macro P/R/F1 — giống eval_joint_triplet.py.

Usage (from thesis_apc_baseline/):
    python experiments/eval_results.py
    python experiments/eval_results.py --model-type bert
    python experiments/eval_results.py --model-type t5 --results-csv results.csv \\
        --gas-checkpoint checkpoints_gas
"""

from __future__ import annotations

import argparse
import csv
import importlib.util as _ilu
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─── CLI ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Eval joint triplet từ results.csv")
parser.add_argument(
    "--model-type", choices=["bert", "t5"], default="t5",
    help="Encoder joint model: 'bert' → Bert/, 't5' → T5/ (default: t5)",
)
parser.add_argument(
    "--runs-dir", type=str, default=None,
    help="Thư mục chứa joint model runs (mặc định: ROOT/Bert hoặc ROOT/T5)",
)
parser.add_argument(
    "--results-csv", type=str, default=None,
    help="File results.csv (mặc định: ROOT/results.csv)",
)
parser.add_argument(
    "--gas-checkpoint", type=str, default=None,
    help="Checkpoint GAS T5 ATE (mặc định: ROOT/checkpoints_gas)",
)
parser.add_argument(
    "--gas-batch-size", type=int, default=16,
    help="Batch size cho GAS T5 inference (default: 16)",
)
args = parser.parse_args()

# ─── Resolve paths ────────────────────────────────────────────────────────────

MODEL_TYPE  = args.model_type.lower()
PRETRAINED  = {"bert": "bert-base-uncased", "t5": "t5-base"}[MODEL_TYPE]
RUNS_DIR    = Path(args.runs_dir) if args.runs_dir else ROOT / ("Bert" if MODEL_TYPE == "bert" else "T5")
RESULTS_CSV = Path(args.results_csv) if args.results_csv else ROOT / "results.csv"
GAS_CKPT    = Path(args.gas_checkpoint) if args.gas_checkpoint else ROOT / "checkpoints_gas"
OUT_CSV     = ROOT / "runs_ate" / f"eval_results_{MODEL_TYPE.upper()}.csv"

for p, name in [(RUNS_DIR, "runs-dir"), (RESULTS_CSV, "results-csv"), (GAS_CKPT, "gas-checkpoint")]:
    if not p.exists():
        print(f"[ERROR] Không tìm thấy {name}: {p}")
        sys.exit(1)

# ─── Load eval_joint_triplet để tái dùng functions ───────────────────────────

_spec = _ilu.spec_from_file_location(
    "eval_joint_triplet",
    Path(__file__).parent / "eval_joint_triplet.py",
)
_ej = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_ej)

# tái dùng: tokenize_single, load_joint_model, micro_prf, macro_prf,
#           entity_hit_acc, _norm_term, DEVICE, SENTIMENT_LABELS, BATCH_SIZE, MAX_SEQ_LEN
_ej.MODEL_TYPE = MODEL_TYPE
_ej.PRETRAINED  = PRETRAINED

DEVICE         = _ej.DEVICE
BATCH_SIZE     = _ej.BATCH_SIZE
MAX_SEQ_LEN    = _ej.MAX_SEQ_LEN
_norm_term      = _ej._norm_term
tokenize_single = _ej.tokenize_single
entity_hit_acc  = _ej.entity_hit_acc

# ─── Load gold từ test_sentences_id.csv ──────────────────────────────────────

print(f"Device : {DEVICE}")
print(f"Loading gold từ dataset/test_sentences_id.csv …")

gold_by_id: Dict[int, Set[Tuple[str, str, str]]] = defaultdict(set)
with open(ROOT / "dataset" / "test_sentences_id.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        eid = int(row["id"])
        triple = (
            _norm_term(row["aspect_term"]),
            row["category"].strip().upper(),
            row["sentiment"].strip().lower(),
        )
        gold_by_id[eid].add(triple)

all_ids = sorted(gold_by_id.keys())
print(f"  {len(all_ids)} entity_ids, "
      f"{sum(len(v) for v in gold_by_id.values())} gold triplets")

# ─── Load results.csv ─────────────────────────────────────────────────────────

print(f"\nLoading results.csv từ {RESULTS_CSV} …")
units_by_id: Dict[int, List[str]] = defaultdict(list)
sentence_by_id: Dict[int, str] = {}

with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        eid  = int(row["entity_id"])
        unit = row["unit"].strip()
        sent = row["sentence"].strip()
        if unit:
            units_by_id[eid].append(unit)
        if eid not in sentence_by_id and sent:
            sentence_by_id[eid] = sent

print(f"  {len(units_by_id)} entities, "
      f"{sum(len(v) for v in units_by_id.values())} units")

# ─── Step 1: GAS T5 ATE — extract aspect terms từ units ──────────────────────

print(f"\nStep 1 — GAS T5 ATE inference")
print(f"  Checkpoint : {GAS_CKPT}")

import re
import torch
from gas.model import GasT5Model
from src.normalization import build_ngram_vocabulary, normalize_aspect

gas_model = GasT5Model.from_pretrained(str(GAS_CKPT), device=DEVICE)

# Parser cho ATE output format: "(term)" hoặc "(term1); (term2)"
# Khác với GAS output "(term, CAT, sent)" — checkpoint này chỉ output term
_ATE_RE = re.compile(r"\(\s*(.+?)\s*\)")

def _parse_ate_output(raw: str) -> List[str]:
    """Parse '(term1); (term2)' → ['term1', 'term2'].
    Cũng handle GAS format '(term, CAT, sent)' → lấy field đầu tiên.
    """
    if not raw or raw.strip().lower() == "none":
        return []
    terms = []
    for m in _ATE_RE.finditer(raw):
        content = m.group(1).strip()
        # nếu là GAS format "term, CAT, sent" → lấy phần trước dấu phẩy đầu
        term = content.split(",")[0].strip()
        if term:
            terms.append(term)
    return terms

# thu thập tất cả units duy nhất để batch inference
all_units = list({u for units in units_by_id.values() for u in units})
print(f"  {len(all_units)} unique units …")

t0 = time.perf_counter()
# dùng raw generation thay vì predict_batch (vốn dùng parse_gas_target không hợp)
gas_model.model.eval()
raw_outputs: List[str] = []
for start in range(0, len(all_units), args.gas_batch_size):
    chunk = all_units[start : start + args.gas_batch_size]
    enc = gas_model.tokenizer(
        chunk, max_length=128, padding=True, truncation=True, return_tensors="pt"
    )
    gen_ids = gas_model._generate_ids(enc["input_ids"], enc["attention_mask"])
    decoded = gas_model.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    raw_outputs.extend(decoded)
ate_sec = time.perf_counter() - t0

# map unit → [term, ...]
# normalize term về n-gram trong unit, bỏ term không tìm thấy trong unit
unit_to_terms: Dict[str, List[str]] = {}
for unit, raw in zip(all_units, raw_outputs):
    raw_terms = _parse_ate_output(raw)
    vocab = build_ngram_vocabulary(unit)
    valid_terms = []
    unit_lower = unit.lower()
    for t in raw_terms:
        # ưu tiên exact match trước
        if t.lower() in unit_lower:
            valid_terms.append(t)
            continue
        # fallback: normalize về n-gram gần nhất trong unit
        normed = normalize_aspect(t, vocab)
        if normed and normed.lower() in unit_lower:
            valid_terms.append(normed)
    unit_to_terms[unit] = valid_terms

total_terms = sum(len(v) for v in unit_to_terms.values())
print(f"  ATE time   : {ate_sec:.3f}s — {total_terms} terms extracted "
      f"từ {len(all_units)} units")

# ─── Xây dựng (entity_id, sentence, term) pairs cho joint model ──────────────

# Mỗi entity_id → gold sentence (từ test_sentences_id.csv)
gold_sents = {row["id"]: row["sentence"]
              for row in csv.DictReader(
                  open(ROOT / "dataset" / "test_sentences_id.csv", encoding="utf-8")
              )}

flat_pairs: List[Tuple[int, str, str]] = []   # (entity_id, sentence, term)
for eid in sorted(units_by_id.keys()):
    sentence = gold_sents.get(str(eid), sentence_by_id.get(eid, ""))
    for unit in units_by_id[eid]:
        for term in unit_to_terms.get(unit, []):
            flat_pairs.append((eid, sentence, term))

# nếu unit không cho ra term nào → thêm unit nguyên vào làm fallback term
for eid in sorted(units_by_id.keys()):
    sentence = gold_sents.get(str(eid), sentence_by_id.get(eid, ""))
    for unit in units_by_id[eid]:
        if not unit_to_terms.get(unit):
            flat_pairs.append((eid, sentence, unit))

print(f"\n  {len(flat_pairs)} (entity_id, sentence, term) pairs → joint model")

# ─── ATE F1: GAS T5 extracted terms vs gold terms per entity ─────────────────
# Tập hợp tất cả terms được extract từ tất cả units của mỗi entity.
# Đây là ATE metric tương đương ate_metrics() trong eval_joint_triplet.py
# nhưng keyed by entity_id thay vì sentence text.

pred_terms_by_id: Dict[int, Set[str]] = defaultdict(set)
for eid, _, term in flat_pairs:
    pred_terms_by_id[eid].add(_norm_term(term))


def _ate_f1_by_id(
    gold_map: "Dict[int, Set]",
    pred_map: "Dict[int, Set[str]]",
    ids: "List[int]",
) -> "Dict[str, float]":
    """Term-level micro P/R/F1 cho split-LLM pipeline, keyed by entity_id.

    Gold terms  = {norm(aspect_term) for (term,cat,sent) in gold_map[eid]}
    Pred terms  = union của tất cả terms GAS T5 extract từ mọi unit của entity.
    Exact-match (sau normalisation): tp = |gold_terms ∩ pred_terms|.
    """
    total_tp = total_fp = total_fn = 0
    for eid in ids:
        gold_terms = {_norm_term(t) for t, _, _ in gold_map.get(eid, set())}
        pred_terms = pred_map.get(eid, set())
        tp          = len(gold_terms & pred_terms)
        total_tp   += tp
        total_fp   += len(pred_terms) - tp
        total_fn   += len(gold_terms) - tp
    p  = total_tp / max(total_tp + total_fp, 1)
    r  = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return {
        "precision": round(p  * 100, 2),
        "recall":    round(r  * 100, 2),
        "f1":        round(f1 * 100, 2),
    }


ate_m = _ate_f1_by_id(gold_by_id, pred_terms_by_id, all_ids)
print(f"\nATE (GAS T5 on units, term only): "
      f"P={ate_m['precision']}%  R={ate_m['recall']}%  F1={ate_m['f1']}%")

# ─── Step 2: Joint model — predict category + sentiment ──────────────────────

print(f"\nStep 2 — Joint model ({MODEL_TYPE.upper()}) inference")
print(f"  Runs dir   : {RUNS_DIR}")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED)

run_dirs = sorted(
    d for d in RUNS_DIR.iterdir()
    if d.is_dir() and (d / "best_model.pt").is_file() and (d / "meta.json").is_file()
)
if not run_dirs:
    print(f"[ERROR] Không tìm thấy config nào trong {RUNS_DIR}")
    sys.exit(1)

print(f"  {len(run_dirs)} config(s): {[d.name for d in run_dirs]}")

rows_out: List[Dict] = []

for run_dir in run_dirs:
    config_name = run_dir.name
    print(f"\n── Config: {config_name} ──")

    try:
        model, cat_map, cat_labels, train_time_sec = _ej.load_joint_model(run_dir, tokenizer)
    except Exception as e:
        print(f"  [error] {e}")
        continue

    id2cat = {i: lbl for lbl, i in cat_map.items()}
    pred_by_id: Dict[int, Set[Tuple[str, str, str]]] = defaultdict(set)

    t0 = time.perf_counter()
    for start in range(0, len(flat_pairs), BATCH_SIZE):
        chunk = flat_pairs[start : start + BATCH_SIZE]
        samples = [
            tokenize_single(sentence, term, tokenizer, MAX_SEQ_LEN)
            for _, sentence, term in chunk
        ]
        ids  = torch.stack([s["input_ids"]      for s in samples]).to(DEVICE)
        attn = torch.stack([s["attention_mask"] for s in samples]).to(DEVICE)
        lcf  = torch.stack([s["lcf_vec"]         for s in samples]).to(DEVICE)

        with torch.no_grad():
            out = model(ids, attn, lcf)

        pred_cats  = out["aspect_cat_logits"].argmax(-1).cpu().tolist()
        pred_sents = out["sentiment_logits"].argmax(-1).cpu().tolist()

        for (eid, _, term), cp, sp in zip(chunk, pred_cats, pred_sents):
            pred_by_id[eid].add((
                _norm_term(term),
                id2cat[cp].upper(),
                _ej.SENTIMENT_LABELS[sp].lower(),
            ))

    infer_sec = time.perf_counter() - t0
    print(f"  Inference time : {infer_sec:.3f}s  "
          f"({infer_sec/len(all_ids)*1000:.1f} ms/entity, "
          f"{infer_sec/max(len(flat_pairs),1)*1000:.1f} ms/pair)")

    # ── Metrics ───────────────────────────────────────────────────────────────
    m     = _ej.micro_prf(gold_by_id, pred_by_id, all_ids)
    macro = _ej.macro_prf(gold_by_id, pred_by_id, all_ids)

    print(f"  Micro → P={m['precision']}%  R={m['recall']}%  F1={m['f1']}%"
          f"  (TP={m['tp']} FP={m['fp']} FN={m['fn']})")
    print(f"  Macro → P={macro['precision']}%  R={macro['recall']}%  F1={macro['f1']}%")
    for cls, f1_cls in macro["per_class"].items():
        print(f"    {cls:<22}: {f1_cls:.2f}%")

    hit = entity_hit_acc(gold_by_id, pred_by_id, all_ids)
    print(f"  Hit-Acc   → {hit['hits']}/{hit['total']} entity có ≥1 triplet đúng = {hit['accuracy']}%")

    oracle = _ej.oracle_metrics(model, tokenizer, ROOT / "dataset" / "test.apc", cat_map, cat_labels)
    print(f"  Oracle (gold term, upper bound) → "
          f"Cat-Acc={oracle['cat_acc']}%  Sent-Acc={oracle['sent_acc']}%  "
          f"Joint-Acc={oracle['joint_acc']}%  (n={oracle['n']})")

    rows_out.append({
        "config":            config_name,
        "train_time_sec":    train_time_sec,
        "ate_time_sec":      round(ate_sec, 3),
        "infer_time_sec":    round(infer_sec, 3),
        "num_pairs":         len(flat_pairs),
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
        "hit_acc":           hit["accuracy"],
        "hit_count":         hit["hits"],
        "tp":                m["tp"],
        "fp":                m["fp"],
        "fn":                m["fn"],
        **{f"f1_{cls}": macro["per_class"].get(cls, 0.0)
           for cls in sorted(macro["per_class"])},
    })

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ─── Summary table ────────────────────────────────────────────────────────────

W = 190
print(f"\n{'═' * W}")
print("EVAL RESULTS  (LLM split → GAS T5 ATE → Joint model — term ∩ category ∩ sentiment)")
print(f"{'═' * W}")
print(f"  {'Config':<26}  {'Train(s)':>8}  {'ATE(s)':>7}  {'Infer(s)':>8}  {'Pairs':>6}"
      f"  {'ATE-F1':>8}"
      f"  {'Micro-P':>8}  {'Micro-R':>8}  {'Micro-F1':>9}"
      f"  {'Macro-P':>8}  {'Macro-R':>8}  {'Macro-F1':>9}"
      f"  {'OrcCat':>8}  {'OrcSent':>8}  {'OrcJoint':>9}"
      f"  {'HitAcc':>8}"
      f"  {'TP':>5}  {'FP':>5}  {'FN':>5}")
print(f"{'─' * W}")
for row in rows_out:
    t = row["train_time_sec"]
    time_str = f"{t:>7.1f}s" if t == t else "     N/A"
    print(
        f"  {row['config']:<26}  {time_str}"
        f"  {row['ate_time_sec']:>6.2f}s"
        f"  {row['infer_time_sec']:>7.3f}s"
        f"  {row['num_pairs']:>6}"
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
        f"  {row['hit_acc']:>7.2f}%"
        f"  {row['tp']:>5}  {row['fp']:>5}  {row['fn']:>5}"
    )
print(f"{'═' * W}")
print("  ATE(s)    : GAS T5 inference time trên tất cả units")
print("  Infer(s)  : Joint model inference time")
print("  Pairs     : số (sentence, term) pairs đưa vào joint model")
print("  ATE-F1    : F1 trích xuất term (GAS T5 trên units) — tương đương ate_metrics() trong eval_joint_triplet")
print("  Micro-F1  : micro F1 triplet end-to-end (term ∩ category ∩ sentiment)")
print("  Macro-F1  : macro F1 theo (category, sentiment) class")
print("  Orc*      : ORACLE trên gold term (upper bound) — đo riêng tầng phân loại, KHÔNG tính lỗi ATE")
print("  HitAcc    : % entity có ≥1 predicted triplet khớp gold (không phạt extra pred)")
print(f"{'═' * W}")

# ─── Save CSV ─────────────────────────────────────────────────────────────────

all_cat_keys = sorted({k for row in rows_out for k in row if k.startswith("f1_")})
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["config", "train_time_sec", "ate_time_sec", "infer_time_sec",
                    "num_pairs",
                    "ate_f1", "ate_precision", "ate_recall",
                    "micro_precision", "micro_recall", "micro_f1",
                    "macro_precision", "macro_recall", "macro_f1",
                    "oracle_cat_acc", "oracle_sent_acc", "oracle_joint_acc",
                    "hit_acc", "hit_count",
                    "tp", "fp", "fn"] + all_cat_keys,
        extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(rows_out)
print(f"\nSaved → {OUT_CSV}")
