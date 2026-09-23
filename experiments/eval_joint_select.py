# -*- coding: utf-8 -*-
"""Eval joint triplet cho danh sách folder được chỉ định cụ thể.

Giống eval_joint_triplet_run.py nhưng thay vì eval TẤT CẢ folder trong runs-dir,
chỉ eval các folder được chỉ định qua --folders.

Usage (from thesis_apc_baseline/):
    # Tên folder tương đối (giải quyết theo --runs-dir, mặc định ROOT/Bert)
    python experiments/eval_joint_select.py --model-type bert \\
        --folders "baseline_balanced bert sup" "lcf_bip_cdm bert sup"

    # Đường dẫn tuyệt đối
    python experiments/eval_joint_select.py \\
        --folders "C:/path/folder1" "C:/path/folder2"

    # Chỉ định runs-dir + tên folder
    python experiments/eval_joint_select.py --model-type bert \\
        --runs-dir Bert \\
        --folders "baseline_balanced" "lcf_scm_cdm_resize" "lcf_seq_cdm_resize"

    # Custom ATE file và output CSV
    python experiments/eval_joint_select.py --model-type bert \\
        --ate-csv runs_ate/test_ate_predictions.csv \\
        --out-csv runs_ate/my_eval.csv \\
        --folders "baseline_balanced" "lcf_bip_cdm_resize"
"""

from __future__ import annotations

import argparse
import csv as _csv
import importlib.util as _ilu
import os as _os
import sys
import tempfile as _tmp
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─── CLI ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Eval joint triplet cho các folder được chỉ định cụ thể",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument(
    "--model-type", choices=["bert", "t5"], default="bert",
    help="Encoder backbone: 'bert' hoặc 't5' (default: bert)",
)
parser.add_argument(
    "--runs-dir", type=str, default=None,
    help="Thư mục gốc để giải quyết tên folder tương đối "
         "(mặc định: ROOT/Bert nếu bert, ROOT/T5 nếu t5)",
)
parser.add_argument(
    "--folders", nargs="+", required=True,
    help="Danh sách folder cần eval (tên tương đối hoặc đường dẫn tuyệt đối). "
         "Ví dụ: --folders \"baseline_balanced bert sup\" \"lcf_bip_cdm bert sup\"",
)
parser.add_argument(
    "--ate-csv", type=str, default=None,
    help="File ATE predictions CSV (mặc định: runs_ate/test_ate_predictions.csv)",
)
parser.add_argument(
    "--out-csv", type=str, default=None,
    help="File CSV output (mặc định: runs_ate/eval_select_<MODEL>_<timestamp>.csv)",
)
args = parser.parse_args()

# ─── Resolve config ───────────────────────────────────────────────────────────

MODEL_TYPE = args.model_type.lower()
PRETRAINED = {"bert": "bert-base-uncased", "t5": "t5-base"}[MODEL_TYPE]

DEFAULT_RUNS_DIR = ROOT / ("Bert" if MODEL_TYPE == "bert" else "T5")
RUNS_DIR = Path(args.runs_dir) if args.runs_dir else DEFAULT_RUNS_DIR

# ─── Resolve danh sách folder ─────────────────────────────────────────────────

run_dirs: list[Path] = []
for folder_arg in args.folders:
    p = Path(folder_arg)
    if not p.is_absolute():
        p = RUNS_DIR / folder_arg
    if not p.is_dir():
        print(f"[WARN] Không tìm thấy folder, bỏ qua: {p}")
        continue
    if not (p / "best_model.pt").is_file():
        print(f"[WARN] Thiếu best_model.pt, bỏ qua: {p}")
        continue
    if not (p / "meta.json").is_file():
        print(f"[WARN] Thiếu meta.json, bỏ qua: {p}")
        continue
    run_dirs.append(p)

if not run_dirs:
    print("[ERROR] Không có folder hợp lệ nào. Kiểm tra lại đường dẫn và --model-type.")
    sys.exit(1)

# ─── ATE CSV ──────────────────────────────────────────────────────────────────

ATE_CSV = Path(args.ate_csv) if args.ate_csv else ROOT / "runs_ate" / "test_ate_predictions.csv"
if not ATE_CSV.is_file():
    print(f"[ERROR] Không tìm thấy ATE CSV: {ATE_CSV}")
    sys.exit(1)

# ─── Align câu ATE CSV theo thứ tự dòng với test.apc ─────────────────────────

from common.dataset_utils import parse_apc_file as _parse_apc

_gold_sents = [e["text"] for e in _parse_apc(str(ROOT / "dataset" / "test.apc"))]

_tmp_path = None
with open(ATE_CSV, newline="", encoding="utf-8") as _fin:
    _rows = list(_csv.DictReader(_fin))
    _fieldnames = list(_rows[0].keys()) if _rows else []

if len(_rows) != len(_gold_sents):
    print(f"[WARN] ATE CSV có {len(_rows)} dòng, test.apc có {len(_gold_sents)} entries "
          f"— không align theo index, dùng nguyên file gốc")
else:
    _aligned = [dict(r, sentence=_gold_sents[i]) for i, r in enumerate(_rows)]
    _changed  = sum(1 for o, n in zip(_rows, _aligned) if o["sentence"] != n["sentence"])
    if _changed:
        _fd, _tmp_path = _tmp.mkstemp(suffix=".csv", prefix="ate_align_")
        _os.close(_fd)
        with open(_tmp_path, "w", newline="", encoding="utf-8") as _fout:
            _w = _csv.DictWriter(_fout, fieldnames=_fieldnames)
            _w.writeheader()
            _w.writerows(_aligned)
        ATE_CSV = Path(_tmp_path)
        print(f"[align] {_changed} câu đã được thay bằng sentence từ test.apc theo thứ tự dòng")

# ─── OUT CSV ──────────────────────────────────────────────────────────────────

if args.out_csv:
    OUT_CSV = Path(args.out_csv)
else:
    import time as _time
    _stamp = _time.strftime("%Y%m%d_%H%M%S")
    OUT_CSV = ROOT / "runs_ate" / f"eval_select_{MODEL_TYPE.upper()}_{_stamp}.csv"

# ─── Load eval_joint_triplet module ───────────────────────────────────────────

_spec = _ilu.spec_from_file_location(
    "eval_joint_triplet",
    Path(__file__).parent / "eval_joint_triplet.py",
)
_eval_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_eval_mod)

# Override model config — phải set TRƯỚC khi gọi load_joint_model
_eval_mod.MODEL_TYPE = MODEL_TYPE
_eval_mod.PRETRAINED = PRETRAINED

# ─── Print config ─────────────────────────────────────────────────────────────

print(f"Model type : {MODEL_TYPE.upper()}")
print(f"Pretrained : {PRETRAINED}")
print(f"Runs dir   : {RUNS_DIR}")
print(f"ATE CSV    : {ATE_CSV}")
print(f"Output CSV : {OUT_CSV}")
print(f"Folders ({len(run_dirs)}):")
for p in run_dirs:
    print(f"  - {p}")
print()

# ─── Eval ─────────────────────────────────────────────────────────────────────

import torch
from transformers import AutoTokenizer

try:
    # Gold triplets
    print(f"Loading gold triplets từ {_eval_mod.TEST_APC} …")
    gold_by_sent = _eval_mod.load_gold_triplets(_eval_mod.TEST_APC)
    all_sentences = list(gold_by_sent.keys())
    print(f"  {len(all_sentences)} sentences, "
          f"{sum(len(v) for v in gold_by_sent.values())} gold triplets")

    # ATE predictions
    print(f"Loading ATE predictions từ {ATE_CSV} …")
    gold_sentences_ordered = [s["text"] for s in _parse_apc(str(ROOT / "dataset" / "test.apc"))]
    ate_preds = _eval_mod.load_ate_predictions(ATE_CSV, gold_sentences_ordered)
    print(f"  {sum(len(v) for v in ate_preds.values())} predicted terms "
          f"across {len(ate_preds)} sentences")

    # ATE-only baseline
    ate_m = _eval_mod.ate_metrics(gold_by_sent, ate_preds, all_sentences)
    print(f"\nATE (term only): P={ate_m['precision']}%  R={ate_m['recall']}%  "
          f"F1={ate_m['f1']}%")

    print(f"\nLoading tokenizer: {PRETRAINED}")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED)

    rows: list[dict] = []

    for run_dir in run_dirs:
        config_name = run_dir.name
        print(f"\n── Config: {config_name} ──")
        try:
            model, cat_map, cat_labels, train_time_sec = _eval_mod.load_joint_model(
                run_dir, tokenizer
            )
        except Exception as e:
            print(f"  [error] load_joint_model: {e}")
            continue

        n_terms = sum(len(v) for v in ate_preds.values())
        print(f"  Running inference trên {n_terms} terms …")
        pred_by_sent = _eval_mod.predict_triplets(
            model, tokenizer, ate_preds, cat_map, cat_labels
        )

        m      = _eval_mod.micro_prf(gold_by_sent, pred_by_sent, all_sentences)
        macro  = _eval_mod.macro_prf(gold_by_sent, pred_by_sent, all_sentences)
        oracle = _eval_mod.oracle_metrics(
            model, tokenizer, _eval_mod.TEST_APC, cat_map, cat_labels
        )

        print(f"  Micro → P={m['precision']}%  R={m['recall']}%  F1={m['f1']}%"
              f"  (TP={m['tp']} FP={m['fp']} FN={m['fn']})")
        print(f"  Macro → P={macro['precision']}%  R={macro['recall']}%  "
              f"F1={macro['f1']}%")
        for cls, f1_cls in sorted(macro["per_class"].items()):
            print(f"    {cls:<24}: {f1_cls:.2f}%")
        print(f"  Oracle (gold term) → "
              f"Cat={oracle['cat_acc']}%  Sent={oracle['sent_acc']}%  "
              f"Joint={oracle['joint_acc']}%  (n={oracle['n']})")

        rows.append({
            "config":           config_name,
            "train_time_sec":   train_time_sec,
            "ate_f1":           ate_m["f1"],
            "ate_precision":    ate_m["precision"],
            "ate_recall":       ate_m["recall"],
            "micro_precision":  m["precision"],
            "micro_recall":     m["recall"],
            "micro_f1":         m["f1"],
            "macro_precision":  macro["precision"],
            "macro_recall":     macro["recall"],
            "macro_f1":         macro["f1"],
            "oracle_cat_acc":   oracle["cat_acc"],
            "oracle_sent_acc":  oracle["sent_acc"],
            "oracle_joint_acc": oracle["joint_acc"],
            "tp":               m["tp"],
            "fp":               m["fp"],
            "fn":               m["fn"],
            **{f"f1_{cls}": macro["per_class"].get(cls, 0.0)
               for cls in sorted(macro["per_class"])},
        })

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Summary table ──────────────────────────────────────────────────────────
    if rows:
        W = 130
        print(f"\n{'═' * W}")
        print("JOINT TRIPLET EVAL — folder được chỉ định")
        print(f"{'═' * W}")
        print(f"  {'Config':<30}  {'Time(s)':>8}  {'ATE-F1':>7}"
              f"  {'Micro-P':>8}  {'Micro-R':>8}  {'Micro-F1':>9}"
              f"  {'Macro-P':>8}  {'Macro-R':>8}  {'Macro-F1':>9}"
              f"  {'OrcSent':>8}  {'OrcJoint':>9}")
        print(f"{'─' * W}")
        for row in rows:
            t = row["train_time_sec"]
            ts = f"{t:>7.1f}s" if t == t else "    N/A"
            print(
                f"  {row['config']:<30}  {ts}"
                f"  {row['ate_f1']:>6.2f}%"
                f"  {row['micro_precision']:>7.2f}%"
                f"  {row['micro_recall']:>7.2f}%"
                f"  {row['micro_f1']:>8.2f}%"
                f"  {row['macro_precision']:>7.2f}%"
                f"  {row['macro_recall']:>7.2f}%"
                f"  {row['macro_f1']:>8.2f}%"
                f"  {row['oracle_sent_acc']:>7.2f}%"
                f"  {row['oracle_joint_acc']:>8.2f}%"
            )
        print(f"{'═' * W}")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    if rows:
        all_f1_keys = sorted({k for row in rows for k in row if k.startswith("f1_")})
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = _csv.DictWriter(
                f,
                fieldnames=[
                    "config", "train_time_sec",
                    "ate_f1", "ate_precision", "ate_recall",
                    "micro_precision", "micro_recall", "micro_f1",
                    "macro_precision", "macro_recall", "macro_f1",
                    "oracle_cat_acc", "oracle_sent_acc", "oracle_joint_acc",
                    "tp", "fp", "fn",
                ] + all_f1_keys,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved → {OUT_CSV}")
    else:
        print("\n[WARN] Không có kết quả để lưu.")

finally:
    if _tmp_path and _os.path.exists(_tmp_path):
        _os.remove(_tmp_path)
