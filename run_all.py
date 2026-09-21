#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""run_all.py — Chạy FULL luồng thực nghiệm ABSA Token-Merging và sinh toàn bộ report.

Script này KHÔNG cài đặt lại logic nào cả: nó điều phối (orchestrate) các script
đã có trong repo, inject cấu hình cho những script vốn hard-code hằng số ở
module level, rồi gom mọi kết quả thành một báo cáo duy nhất trong `reports/`.

    ┌─ env ──────► reports/00_environment.txt
    ├─ ate ──────► checkpoints/gas_t5_ate/seed_<N>/best, runs_ate/seed_<N>/…,
    │              runs_ate/results_ate_multiseed.csv | results_ate_summary.txt
    ├─ ate_infer ► runs_ate/test_ate_predictions.csv
    ├─ apc ──────► runs_joint/<config>/best_model.pt  (BERT)
    │              runs_joint_t5/<config>/best_model.pt  (T5)
    │              runs_joint*/experiment_results_joint.{csv,txt}
    ├─ multiseed ► runs_multiseed/{results_raw,results_aggregated}.csv
    │              runs_multiseed/thesis_tables.txt   ← 8 bảng luận văn
    ├─ gold ─────► runs_bert_gold/{BERT,T5}/eval_gold_aspects.csv
    ├─ triplet ──► runs_ate/eval_joint_triplet_{BERT,T5}.csv
    ├─ gas ──────► checkpoints_gas/best, runs_gas/eval_test.{json,csv}
    ├─ results ──► runs_ate/eval_results_{BERT,T5}.csv
    ├─ uos ──────► uos/output/test/{results.jsonl,metrics.json}
    ├─ figures ──► thesis/figures/*.{pdf,png}
    └─ report ───► reports/REPORT.md + reports/manifest.csv + reports/logs/*.log

Mỗi stage chạy trong một tiến trình con riêng → GPU memory được giải phóng giữa
các stage, và một stage hỏng không làm chết cả lượt chạy (trừ khi --fail-fast).

Ví dụ
-----
    python run_all.py                              # full, 3 seeds, bert+t5, 21 biến thể
    python run_all.py --list                       # liệt kê stage + biến thể rồi thoát
    python run_all.py --dry-run                    # in ra lệnh sẽ chạy, không chạy
    python run_all.py --resume                     # bỏ qua combo đã train xong
    python run_all.py --stages ate ate_infer apc   # chỉ chạy vài stage
    python run_all.py --skip uos gas               # chạy tất cả trừ UOS và GAS
    python run_all.py --no-train --stages report   # chỉ gom report từ kết quả cũ
    python run_all.py --seeds 42 --backbones bert  # bản rút gọn cho máy yếu
    python run_all.py --variants resize            # chỉ 12 cấu hình resize
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

# Console Windows mặc định là cp1252 → ép UTF-8 để bảng/tiếng Việt không vỡ.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 — stream không hỗ trợ reconfigure
        pass

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_SEEDS = [42, 123, 456]
DEFAULT_BACKBONES = ["bert", "t5"]
PRETRAINED_BY_TYPE = {"bert": "bert-base-uncased", "t5": "t5-base"}

# Thư mục checkpoint APC theo backbone (BERT giữ nguyên `runs_joint` để tương
# thích với mặc định của server/app.py: APC_CHECKPOINT_DIR=runs_joint/...).
RUNS_JOINT_NAME_BY_TYPE = {"bert": "runs_joint", "t5": "runs_joint_t5"}

GAS_ATE_F1_FALLBACK = 79.87  # dùng cho header bảng khi chưa chạy stage `ate`

# ── Chế độ smoke: chạy thử toàn bộ đường ống trên dữ liệu tí hon ─────────────
SMOKE_DIR = ROOT / "dataset_smoke"
# Mọi output của lượt smoke đi vào đây để KHÔNG đè lên kết quả thật.
SMOKE_OUT_ROOT = ROOT / "smoke_run"
SMOKE_SIZES = {"train": 48, "dev": 16, "test": 16}
SMOKE_SUPPLEMENT_LINES = 24
SMOKE_UOS_ROWS = 5
# 3 biến thể phủ đủ 3 nhóm code path: post-ToMe resize / post-ToMe compact / pre-ToMe
SMOKE_VARIANT_IDS = ["lcf_scm_cdm", "lcf_scm_cdm_compact", "lcf_pre_scm"]
# eval_results.py hard-code ROOT/"dataset" ở module level (dòng 103, 213, 350)
# nên không chuyển sang dataset_smoke được → bỏ qua stage này khi smoke.
SMOKE_SKIP_STAGES = ["results"]


# ══════════════════════════════════════════════════════════════════════════════
# Catalog biến thể
# ══════════════════════════════════════════════════════════════════════════════
# Mỗi biến thể có HAI id vì hai runner dùng hai quy ước đặt tên khác nhau:
#   ms_id    — id của common/run_multiseed.py (các bảng luận văn tham chiếu id này)
#   joint_id — tên thư mục trong runs_joint/ (quy ước của README + server/app.py)
#
# flags = (use_lcf, use_cdm, use_tome, tome_resize, merge_strategy, use_pre_tome)

VARIANTS: List[Dict] = [
    # ── Nhóm "resize" — 12 cấu hình chính của luận văn ────────────────────────
    dict(group="resize", display="Base", ms_id="baseline", joint_id="baseline_balanced",
         flags=(False, False, False, True, "bipartite", False)),
    dict(group="resize", display="CDM", ms_id="lcf_only_cdm", joint_id="lcf_only_cdm",
         flags=(True, True, False, True, "bipartite", False)),
    dict(group="resize", display="CDW", ms_id="lcf_only_cdw", joint_id="lcf_only_cdw",
         flags=(True, False, False, True, "bipartite", False)),
    dict(group="resize", display="BiToMe", ms_id="bip", joint_id="bip_resize",
         flags=(False, False, True, True, "bipartite", False)),
    dict(group="resize", display="SLM", ms_id="seq", joint_id="seq_resize",
         flags=(False, False, True, True, "sequential_local", False)),
    dict(group="resize", display="SCM", ms_id="scm", joint_id="scm_resize",
         flags=(False, False, True, True, "sequential_cosine", False)),
    dict(group="resize", display="BiToMe+CDM", ms_id="lcf_bip_cdm", joint_id="lcf_bip_cdm_resize",
         flags=(True, True, True, True, "bipartite", False)),
    dict(group="resize", display="BiToMe+CDW", ms_id="lcf_bip_cdw", joint_id="lcf_bip_cdw_resize",
         flags=(True, False, True, True, "bipartite", False)),
    dict(group="resize", display="SLM+CDM", ms_id="lcf_seq_cdm", joint_id="lcf_seq_cdm_resize",
         flags=(True, True, True, True, "sequential_local", False)),
    dict(group="resize", display="SLM+CDW", ms_id="lcf_seq_cdw", joint_id="lcf_seq_cdw_resize",
         flags=(True, False, True, True, "sequential_local", False)),
    dict(group="resize", display="SCM+CDM", ms_id="lcf_scm_cdm", joint_id="lcf_scm_cdm_resize",
         flags=(True, True, True, True, "sequential_cosine", False)),
    dict(group="resize", display="SCM+CDW", ms_id="lcf_scm_cdw", joint_id="lcf_scm_cdw_resize",
         flags=(True, False, True, True, "sequential_cosine", False)),

    # ── Nhóm "compact" — 6 cấu hình đối chứng resize vs compact ───────────────
    dict(group="compact", display="BiToMe+CDM(compact)", ms_id="lcf_bip_cdm_compact",
         joint_id="lcf_bip_cdm_compact", flags=(True, True, True, False, "bipartite", False)),
    dict(group="compact", display="BiToMe+CDW(compact)", ms_id="lcf_bip_cdw_compact",
         joint_id="lcf_bip_cdw_compact", flags=(True, False, True, False, "bipartite", False)),
    dict(group="compact", display="SLM+CDM(compact)", ms_id="lcf_seq_cdm_compact",
         joint_id="lcf_seq_cdm_compact", flags=(True, True, True, False, "sequential_local", False)),
    dict(group="compact", display="SLM+CDW(compact)", ms_id="lcf_seq_cdw_compact",
         joint_id="lcf_seq_cdw_compact", flags=(True, False, True, False, "sequential_local", False)),
    dict(group="compact", display="SCM+CDM(compact)", ms_id="lcf_scm_cdm_compact",
         joint_id="lcf_scm_cdm_compact", flags=(True, True, True, False, "sequential_cosine", False)),
    dict(group="compact", display="SCM+CDW(compact)", ms_id="lcf_scm_cdw_compact",
         joint_id="lcf_scm_cdw_compact", flags=(True, False, True, False, "sequential_cosine", False)),

    # ── Nhóm "pretome" — gộp token TRƯỚC encoder (không gộp sau) ──────────────
    dict(group="pretome", display="LCF+PreBip", ms_id="lcf_pre_bip", joint_id="lcf_pre_bip",
         flags=(True, True, False, False, "bipartite", True)),
    dict(group="pretome", display="LCF+PreSLM", ms_id="lcf_pre_seq", joint_id="lcf_pre_seq",
         flags=(True, True, False, False, "sequential_local", True)),
    dict(group="pretome", display="LCF+PreSCM", ms_id="lcf_pre_scm", joint_id="lcf_pre_scm",
         flags=(True, True, False, False, "sequential_cosine", True)),
]

VARIANT_GROUPS = ["resize", "compact", "pretome"]

# Nhóm cấu hình dùng cho các bảng luận văn (tham chiếu ms_id).
E2E_BERT_CONFIGS = [
    ("GAS + Base", "baseline"),
    ("GAS + BiTome", "bip"),
    ("GAS + SLM + CDM", "lcf_seq_cdm"),
    ("GAS + SLM + CDW", "lcf_seq_cdw"),
    ("GAS + SCM + CDM", "lcf_scm_cdm"),
]
E2E_T5_CONFIGS = [
    ("GAS + Base", "baseline"),
    ("BipTome + CDM", "lcf_bip_cdm"),
    ("SCM + CDM", "lcf_scm_cdm"),
    ("SLM + CDM", "lcf_seq_cdm"),
    ("SLM + CDW", "lcf_seq_cdw"),
]
COMPACT_VS_RESIZE_CONFIGS = [
    ("lcf_scm_cdm", "lcf_scm_cdm_compact", "SCM + CDM"),
    ("lcf_bip_cdm", "lcf_bip_cdm_compact", "BiTome + CDM"),
    ("lcf_seq_cdm", "lcf_seq_cdm_compact", "SLM + CDM"),
    ("lcf_seq_cdw", "lcf_seq_cdw_compact", "SLM + CDW"),
]
PAPER_OUR_CONFIGS = [
    ("Base (BERT)", "bert", "baseline"),
    ("BipTome+CDM Resize (T5)", "t5", "lcf_bip_cdm"),
    ("SLM+CDM Resize (BERT)", "bert", "lcf_seq_cdm"),
    ("SLM+CDW Resize (BERT)", "bert", "lcf_seq_cdw"),
]


def selected_variants(groups: Sequence[str]) -> List[Dict]:
    return [v for v in VARIANTS if v["group"] in groups]


def ms_tuple(v: Dict) -> list:
    """Tuple cấu hình theo đúng thứ tự run_multiseed/run_joint_experiments mong đợi."""
    lcf, cdm, tome, resize, strategy, pre = v["flags"]
    return [lcf, cdm, tome, resize, strategy, pre, v["display"], v["ms_id"]]


def joint_tuple(v: Dict) -> list:
    lcf, cdm, tome, resize, strategy, pre = v["flags"]
    return [lcf, cdm, tome, resize, strategy, pre, v["display"], v["joint_id"]]


# ══════════════════════════════════════════════════════════════════════════════
# Tiện ích
# ══════════════════════════════════════════════════════════════════════════════

def log(msg: str = "") -> None:
    print(msg, flush=True)


def banner(title: str, char: str = "=") -> None:
    log("\n" + char * 78)
    log(f"  {title}")
    log(char * 78)


def child_env() -> dict:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    # scripts/visualize_scm*.py gọi plt.show() — với backend GUI sẽ treo tiến
    # trình trên máy headless (Kaggle, Colab, server CI).
    env.setdefault("MPLBACKEND", "Agg")
    return env


def run_cmd(cmd: List[str], log_path: Path, dry_run: bool = False) -> int:
    """Chạy lệnh con, stream stdout ra console và ghi song song vào log_path."""
    printable = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    log(f"$ {printable}")
    if dry_run:
        return 0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(f"\n{'=' * 78}\n$ {printable}\n{'=' * 78}\n")
        lf.flush()
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT), env=child_env(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
        proc.wait()
        lf.write(f"\n[exit code] {proc.returncode}\n")
    return proc.returncode


def run_driver(name: str, payload: dict, log_path: Path, dry_run: bool = False) -> int:
    """Chạy một driver nội bộ bằng cách tự gọi lại chính file này trong tiến trình con."""
    cmd = [sys.executable, str(Path(__file__).resolve()),
           "--_driver", name, "--_payload", json.dumps(payload, ensure_ascii=False)]
    return run_cmd(cmd, log_path, dry_run=dry_run)


def load_module(path: Path, name: str):
    """Import một file .py như module độc lập (không đụng sys.modules của repo)."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Không import được: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def resolve_ate_checkpoint(seeds: Sequence[int],
                           out_root: Optional[Path] = None) -> Optional[Path]:
    """Tìm checkpoint T5 ATE khả dụng, ưu tiên checkpoint vừa train ở stage `ate`.

    out_root khác ROOT (chế độ smoke) thì ưu tiên checkpoint trong sandbox trước,
    rồi mới rơi về checkpoint thật của repo.
    """
    roots = [out_root] if out_root and out_root != ROOT else []
    roots.append(ROOT)
    candidates: List[Path] = []
    for r in roots:
        for s in seeds:
            candidates.append(r / "checkpoints" / "gas_t5_ate" / f"seed_{s}" / "best")
        candidates.append(r / "checkpoints" / "gas_t5_ate" / "best")
    candidates.append(ROOT / "checkpoints" / "best")
    for c in candidates:
        if (c / "config.json").is_file():
            return c
    return None


def read_ate_f1(default: float = GAS_ATE_F1_FALLBACK,
                out_root: Optional[Path] = None) -> float:
    """Đọc ATE F1 trung bình từ runs_ate/results_ate_multiseed.csv nếu có."""
    path = (out_root or ROOT) / "runs_ate" / "results_ate_multiseed.csv"
    if not path.is_file():
        return default
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        key = next((k for k in ("test_f1", "f1", "F1", "ate_f1")
                    if rows and k in rows[0]), None)
        if not key:
            return default
        vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
        if not vals:
            return default
        mean = sum(vals) / len(vals)
        # run_multiseed_ate.py ghi F1 dạng phân số (0–1); bảng luận văn cần %.
        return mean * 100.0 if mean <= 1.0 else mean
    except Exception:
        return default


def _head_apc_blocks(src: Path, dst: Path, n_blocks: int) -> int:
    """Chép n_blocks mẫu đầu tiên của file .apc (mỗi mẫu 4 dòng) sang dst."""
    lines = [l.strip() for l in src.read_text(encoding="utf-8").splitlines()]
    blocks: List[List[str]] = []
    buf: List[str] = []
    for line in lines:
        if not line:
            buf = []
            continue
        buf.append(line)
        if len(buf) == 4:
            blocks.append(buf)
            buf = []
            if len(blocks) >= n_blocks:
                break
    dst.parent.mkdir(parents=True, exist_ok=True)
    body = "\n\n".join("\n".join(b) for b in blocks)
    dst.write_text(body + "\n", encoding="utf-8")
    return len(blocks)


def build_smoke_dataset(src_dir: Path = ROOT / "dataset",
                        dst_dir: Path = SMOKE_DIR) -> Path:
    """Dựng bản dataset tí hon để kiểm tra toàn bộ đường ống chạy được.

    Lấy N mẫu ĐẦU TIÊN của mỗi split (giữ nguyên thứ tự) để test_sentences_id.csv
    vẫn align theo index với test.apc — điều kiện bắt buộc của eval e2e.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}
    for split, n in SMOKE_SIZES.items():
        counts[split] = _head_apc_blocks(src_dir / f"{split}.apc",
                                         dst_dir / f"{split}.apc", n)

    # test_sentences_id.csv — giữ header + n_test dòng đầu (cùng thứ tự test.apc)
    gold_src = src_dir / "test_sentences_id.csv"
    if gold_src.is_file():
        with open(gold_src, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            rows = [r for _, r in zip(range(counts["test"]), reader)]
        with open(dst_dir / "test_sentences_id.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    # supplement — vài dòng đầu là đủ để nhánh USE_SUPPLEMENT chạy được
    sup_src, sup_dst = src_dir / "supplement", dst_dir / "supplement"
    sup_dst.mkdir(parents=True, exist_ok=True)
    for name in ["negative.tsv", "neutral.tsv"]:
        if (sup_src / name).is_file():
            head = (sup_src / name).read_text(encoding="utf-8").splitlines()
            body = "\n".join(head[:SMOKE_SUPPLEMENT_LINES])
            (sup_dst / name).write_text(body + "\n", encoding="utf-8")

    log(f"[smoke] dataset tí hon → {dst_dir}")
    log(f"[smoke]   train={counts['train']}  dev={counts['dev']}  test={counts['test']}")
    return dst_dir


def ollama_available(host: str = "http://localhost:11434") -> bool:
    try:
        import urllib.request
        with urllib.request.urlopen(f"{host}/api/tags", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Drivers — chạy trong tiến trình con, inject hằng số rồi gọi main() của script gốc
# ══════════════════════════════════════════════════════════════════════════════

def _inject_dataset(mod, data_dir: Path, *, supplement: bool = True) -> None:
    """Trỏ các hằng số dataset của một module sang thư mục dữ liệu khác."""
    mod.DATASET_DIR = data_dir
    mod.TRAIN_APC = data_dir / "train.apc"
    mod.DEV_APC = data_dir / "dev.apc"
    mod.TEST_APC = data_dir / "test.apc"
    if supplement and hasattr(mod, "SUPPLEMENT_FILES"):
        mod.SUPPLEMENT_DIR = data_dir / "supplement"
        mod.SUPPLEMENT_FILES = [str(mod.SUPPLEMENT_DIR / "negative.tsv"),
                                str(mod.SUPPLEMENT_DIR / "neutral.tsv")]


def _inject_epochs(mod, epochs) -> None:
    """Giới hạn số epoch (và patience tương ứng) cho chế độ chạy thử."""
    if not epochs:
        return
    mod.NUM_EPOCHS = int(epochs)
    if hasattr(mod, "PATIENCE"):
        mod.PATIENCE = max(1, min(int(mod.PATIENCE), int(epochs)))
    print(f"[driver] Giới hạn epochs={mod.NUM_EPOCHS} patience={getattr(mod, 'PATIENCE', '—')}")


def driver_ate_infer(p: dict) -> None:
    """experiments/run_ate_inference.py với ATE_CKPT / dataset có thể thay đổi."""
    mod = load_module(ROOT / "experiments" / "run_ate_inference.py", "_drv_ate_infer")
    mod.ATE_CKPT = Path(p["ate_ckpt"])
    mod.OUT_CSV = Path(p["out_csv"])
    mod.OUT_DIR = Path(p["out_csv"]).parent
    mod.GOLD_CSV = Path(p["data_dir"]) / "test_sentences_id.csv"
    print(f"[driver] ATE checkpoint : {mod.ATE_CKPT}")
    print(f"[driver] Gold CSV       : {mod.GOLD_CSV}")
    print(f"[driver] Output CSV     : {mod.OUT_CSV}")
    mod.main()


def driver_triplet(p: dict) -> None:
    """experiments/eval_joint_triplet.py — bản driver của eval_joint_triplet_run.py.

    Dùng driver thay vì gọi thẳng wrapper vì wrapper hard-code
    ROOT/"dataset"/"test.apc" ngay ở module level, không đổi sang dataset khác được.
    Logic align theo index được giữ nguyên như wrapper gốc.
    """
    import csv as _csv
    import os as _os
    import tempfile as _tmp

    from common.dataset_utils import parse_apc_file

    test_apc = Path(p["data_dir"]) / "test.apc"
    ate_csv = Path(p["ate_csv"])

    gold_sents = [e["text"] for e in parse_apc_file(str(test_apc))]
    with open(ate_csv, newline="", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    fieldnames = list(rows[0].keys()) if rows else []

    tmp_path = None
    if len(rows) != len(gold_sents):
        print(f"[driver] [WARN] ATE CSV {len(rows)} dòng vs test.apc {len(gold_sents)} "
              f"entries — không align được theo index, dùng nguyên file gốc")
    else:
        aligned = [dict(r, sentence=gold_sents[i]) for i, r in enumerate(rows)]
        changed = sum(1 for o, n in zip(rows, aligned) if o["sentence"] != n["sentence"])
        if changed:
            fd, tmp_path = _tmp.mkstemp(suffix=".csv", prefix="ate_align_")
            _os.close(fd)
            with open(tmp_path, "w", newline="", encoding="utf-8") as f:
                w = _csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(aligned)
            ate_csv = Path(tmp_path)
            print(f"[driver] [align] {changed} câu thay bằng sentence từ test.apc")

    mod = load_module(ROOT / "experiments" / "eval_joint_triplet.py", "_drv_triplet")
    mod.MODEL_TYPE = p["model_type"]
    mod._MODEL_CONFIGS = dict(PRETRAINED_BY_TYPE)
    mod.PRETRAINED = p["pretrained"]
    mod.RUNS_DIR = Path(p["runs_dir"])
    mod.ATE_CSV = ate_csv
    mod.TEST_APC = test_apc
    mod.OUT_CSV = Path(p["out_csv"])

    print(f"[driver] Backbone   : {mod.MODEL_TYPE} ({mod.PRETRAINED})")
    print(f"[driver] Runs dir   : {mod.RUNS_DIR}")
    print(f"[driver] Test .apc  : {mod.TEST_APC}")
    print(f"[driver] Output CSV : {mod.OUT_CSV}")
    try:
        mod.main()
    finally:
        if tmp_path and _os.path.exists(tmp_path):
            _os.remove(tmp_path)


def driver_apc(p: dict) -> None:
    """experiments/run_joint_experiments.py với CONFIGS / backbone / runs-dir injected."""
    mod = load_module(ROOT / "experiments" / "run_joint_experiments.py", "_drv_apc")

    mod.MODEL_TYPE = p["model_type"]
    mod._MODEL_CONFIGS = dict(PRETRAINED_BY_TYPE)
    mod.PRETRAINED_MODEL = p["pretrained"]
    mod.RUNS_DIR = Path(p["runs_dir"])
    mod.SEED = int(p["seed"])
    mod.CONFIGS = [tuple(c) for c in p["configs"]]
    _inject_dataset(mod, Path(p["data_dir"]))
    _inject_epochs(mod, p.get("epochs"))

    if p.get("resume"):
        kept = [c for c in mod.CONFIGS
                if not (mod.RUNS_DIR / c[7] / "best_model.pt").is_file()]
        skipped = len(mod.CONFIGS) - len(kept)
        if skipped:
            print(f"[driver] --resume: bỏ qua {skipped} cấu hình đã có best_model.pt")
            print(f"[driver] LƯU Ý: experiment_results_joint.csv chỉ chứa các cấu hình "
                  f"được train trong lượt này.")
        mod.CONFIGS = kept

    if not mod.CONFIGS:
        print("[driver] Không còn cấu hình nào cần train — bỏ qua.")
        return

    print(f"[driver] Backbone  : {mod.MODEL_TYPE} ({mod.PRETRAINED_MODEL})")
    print(f"[driver] Runs dir  : {mod.RUNS_DIR}")
    print(f"[driver] Seed      : {mod.SEED}")
    print(f"[driver] Cấu hình  : {[c[7] for c in mod.CONFIGS]}")
    mod.main()


def driver_multiseed(p: dict) -> None:
    """common/run_multiseed.py với đầy đủ catalog biến thể + nhóm bảng luận văn."""
    mod = load_module(ROOT / "common" / "run_multiseed.py", "_drv_multiseed")

    all_cfgs = [tuple(c) for c in p["all_configs"]]
    compact_cfgs = [tuple(c) for c in p["compact_configs"]]

    mod.ALL_CONFIGS = all_cfgs
    mod.COMPACT_CONFIGS = compact_cfgs
    mod.CONFIG_BY_ID = {c[7]: c for c in all_cfgs + compact_cfgs}
    mod.E2E_BERT_CONFIGS = [tuple(x) for x in p["e2e_bert"]]
    mod.E2E_T5_CONFIGS = [tuple(x) for x in p["e2e_t5"]]
    mod.COMPACT_VS_RESIZE_CONFIGS = [tuple(x) for x in p["compact_vs_resize"]]
    mod.PAPER_OUR_CONFIGS = [tuple(x) for x in p["paper_our"]]

    _inject_dataset(mod, Path(p["data_dir"]))
    _inject_epochs(mod, p.get("epochs"))

    print(f"[driver] Resize configs  : {[c[7] for c in all_cfgs]}")
    print(f"[driver] Compact configs : {[c[7] for c in compact_cfgs]}")

    sys.argv = ["run_multiseed.py"] + list(p["argv"])
    print(f"[driver] argv: {sys.argv[1:]}")
    mod.main()


def driver_gold(p: dict) -> None:
    """common/eval_bert_gold_aspects.py cho một backbone / thư mục runs bất kỳ."""
    from transformers import AutoModel

    mod = load_module(ROOT / "common" / "eval_bert_gold_aspects.py", "_drv_gold")
    mod.BERT_DIR = Path(p["runs_dir"])
    mod.OUT_DIR = Path(p["out_dir"])
    mod.PRETRAINED_MODEL = p["pretrained"]
    _inject_dataset(mod, Path(p["data_dir"]), supplement=False)
    if p["model_type"] != "t5":
        # Script gốc hard-code T5EncoderModel; với BERT ta thay bằng AutoModel.
        mod.T5EncoderModel = AutoModel

    print(f"[driver] Backbone : {p['model_type']} ({mod.PRETRAINED_MODEL})")
    print(f"[driver] Runs dir : {mod.BERT_DIR}")
    print(f"[driver] Out dir  : {mod.OUT_DIR}")
    mod.main()


def driver_figures(p: dict) -> None:
    """scripts/generate_thesis_figures.py với thư mục vào/ra thay đổi được."""
    import matplotlib
    matplotlib.use("Agg")

    mod = load_module(ROOT / "scripts" / "generate_thesis_figures.py", "_drv_figures")
    mod.FIG_DIR = Path(p["fig_dir"])
    mod.RUNS = Path(p["runs_ate"])
    mod.DATASET = Path(p["data_dir"])
    mod.FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[driver] Fig dir  : {mod.FIG_DIR}")
    print(f"[driver] Runs ate : {mod.RUNS}")
    print(f"[driver] Dataset  : {mod.DATASET}")
    mod.main()


DRIVERS: Dict[str, Callable[[dict], None]] = {
    "ate_infer": driver_ate_infer,
    "apc": driver_apc,
    "multiseed": driver_multiseed,
    "gold": driver_gold,
    "triplet": driver_triplet,
    "figures": driver_figures,
}


# ══════════════════════════════════════════════════════════════════════════════
# Stages
# ══════════════════════════════════════════════════════════════════════════════

class Ctx:
    """Tham số dùng chung cho mọi stage."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.seeds: List[int] = args.seeds
        self.backbones: List[str] = args.backbones
        self.variants: List[Dict] = selected_variants(args.variants)
        self.resume: bool = args.resume
        self.no_train: bool = args.no_train
        self.dry_run: bool = args.dry_run
        self.smoke: bool = args.smoke
        self.data_dir: Path = Path(args.data_dir) if args.data_dir else ROOT / "dataset"
        self.max_epochs: Optional[int] = args.max_epochs

        if self.smoke:
            # Chạy thử: 1 seed, 3 biến thể phủ đủ code path, 1 epoch, data tí hon.
            self.seeds = self.seeds[:1]
            self.variants = [v for v in VARIANTS if v["ms_id"] in SMOKE_VARIANT_IDS]
            if args.data_dir is None:
                self.data_dir = SMOKE_DIR
            if self.max_epochs is None:
                self.max_epochs = 1

        # Sandbox output: smoke ghi vào smoke_run/, lượt thật ghi thẳng vào repo.
        self.out_root: Path = (
            Path(args.out_root) if args.out_root
            else (SMOKE_OUT_ROOT if self.smoke else ROOT)
        )
        self.reports_dir: Path = self.out_root / "reports"
        self.logs_dir: Path = self.reports_dir / "logs"
        self.status_json: Path = self.reports_dir / "run_status.json"

    def out(self, *parts: str) -> Path:
        """Đường dẫn output, tự chuyển sang sandbox khi chạy smoke."""
        p = self.out_root
        for part in parts:
            p = p / part
        return p

    def runs_joint(self, backbone: str) -> Path:
        return self.out(RUNS_JOINT_NAME_BY_TYPE[backbone])

    def ate_ckpt(self) -> Optional[Path]:
        return resolve_ate_checkpoint(self.seeds, self.out_root)

    def log_path(self, stage: str) -> Path:
        return self.logs_dir / f"{stage}.log"


def stage_env(ctx: Ctx) -> None:
    """Preflight: ghi lại môi trường, dữ liệu, checkpoint sẵn có và kế hoạch chạy."""
    out = ctx.reports_dir / "00_environment.txt"
    if ctx.dry_run:
        log(f"(dry-run) sẽ ghi {out}")
        return

    lines: List[str] = []

    def w(s: str = "") -> None:
        lines.append(s)
        log(s)

    w(f"Thời điểm     : {datetime.now().isoformat(timespec='seconds')}")
    w(f"Repo root     : {ROOT}")
    w(f"Python        : {sys.version.split()[0]}  ({sys.executable})")
    w(f"Platform      : {platform.platform()}")
    w("")

    for pkg in ["torch", "transformers", "sklearn", "numpy", "pandas",
                "matplotlib", "seaborn", "Levenshtein", "fastapi", "tqdm"]:
        try:
            m = __import__(pkg)
            w(f"  {pkg:<14}: {getattr(m, '__version__', 'n/a')}")
        except Exception as e:  # noqa: BLE001
            w(f"  {pkg:<14}: KHÔNG CÀI ĐƯỢC ({type(e).__name__})")
    w("")

    try:
        import torch
        w(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            w(f"  GPU         : {torch.cuda.get_device_name(0)}")
            total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            w(f"  VRAM        : {total:.1f} GB")
        else:
            w("  [cảnh báo] Không có CUDA — training sẽ RẤT chậm trên CPU.")
    except Exception as e:  # noqa: BLE001
        w(f"CUDA check lỗi: {e}")
    w("")

    w(f"Dữ liệu (data-dir = {ctx.data_dir}):")
    rel_data = ctx.data_dir.relative_to(ROOT).as_posix()
    for rel in [f"{rel_data}/train.apc", f"{rel_data}/dev.apc", f"{rel_data}/test.apc",
                f"{rel_data}/test_sentences_id.csv",
                f"{rel_data}/supplement/negative.tsv", f"{rel_data}/supplement/neutral.tsv",
                "results.csv"]:
        p = ROOT / rel
        if p.is_file():
            n = sum(1 for _ in p.open(encoding="utf-8", errors="replace"))
            w(f"  [OK]   {rel:<38} {p.stat().st_size / 1024:8.1f} KB  {n} dòng")
        else:
            w(f"  [THIẾU] {rel}")
    w("")

    ate = ctx.ate_ckpt()
    w(f"ATE checkpoint hiện có : {ate if ate else 'KHÔNG CÓ (stage `ate` sẽ train mới)'}")
    gas = ctx.out("checkpoints_gas", "best")
    w(f"GAS checkpoint hiện có : {gas if gas.is_dir() else 'KHÔNG CÓ (stage `gas` sẽ train mới)'}")
    for bb in ctx.backbones:
        d = ctx.runs_joint(bb)
        done = sorted(x.parent.name for x in d.glob("*/best_model.pt")) if d.is_dir() else []
        w(f"APC checkpoint {bb.upper():<4}  : {len(done)} cấu hình đã có → {done}")
    w(f"Ollama (cho UOS)       : {'đang chạy' if ollama_available() else 'KHÔNG chạy → stage uos sẽ bị bỏ qua'}")
    w("")

    w("Kế hoạch:")
    w(f"  Chế độ    : {'SMOKE (chạy thử)' if ctx.smoke else 'đầy đủ'}")
    w(f"  Data dir  : {ctx.data_dir}")
    w(f"  Max epochs: {ctx.max_epochs if ctx.max_epochs else 'theo mặc định của script'}")
    w(f"  Seeds     : {ctx.seeds}")
    w(f"  Backbones : {ctx.backbones}")
    w(f"  Biến thể  : {len(ctx.variants)} ({', '.join(ctx.args.variants)})")
    for v in ctx.variants:
        lcf, cdm, tome, resize, strat, pre = v["flags"]
        w(f"    - {v['ms_id']:<22} {v['display']:<22} "
          f"lcf={int(lcf)} cdm={int(cdm)} tome={int(tome)} resize={int(resize)} "
          f"pre={int(pre)} strategy={strat}")
    n_runs = len(ctx.variants) * len(ctx.seeds) * len(ctx.backbones)
    w(f"  Tổng số lần train APC (stage multiseed): {n_runs}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"\n→ {out}")


def stage_ate(ctx: Ctx) -> None:
    """Train T5 ATE (GAS) đa seed + sinh prediction CSV theo từng seed."""
    cmd = [sys.executable, "common/run_multiseed_ate.py",
           "--seeds", *[str(s) for s in ctx.seeds],
           "--data-dir", str(ctx.data_dir),
           "--runs-ate-dir", str(ctx.out("runs_ate")),
           "--ckpt-dir", str(ctx.out("checkpoints", "gas_t5_ate"))]
    if ctx.max_epochs:
        cmd += ["--epochs", str(ctx.max_epochs)]
    if ctx.resume:
        cmd.append("--resume")
    if ctx.no_train:
        cmd.append("--no-train")
    rc = run_cmd(cmd, ctx.log_path("ate"), ctx.dry_run)
    if rc != 0:
        raise RuntimeError(f"run_multiseed_ate.py thất bại (exit {rc})")


def stage_ate_infer(ctx: Ctx) -> None:
    """Infer ATE trên tập test → runs_ate/test_ate_predictions.csv (dùng cho mọi eval e2e)."""
    ckpt = ctx.ate_ckpt()
    if ckpt is None:
        raise RuntimeError(
            "Không tìm thấy checkpoint T5 ATE. Chạy stage `ate` trước, "
            "hoặc đặt checkpoint vào checkpoints/gas_t5_ate/best/."
        )
    out_csv = ctx.out("runs_ate", "test_ate_predictions.csv")
    if ctx.resume and out_csv.is_file():
        log(f"--resume: đã có {out_csv} — bỏ qua.")
        return
    rc = run_driver("ate_infer",
                    {"ate_ckpt": str(ckpt), "out_csv": str(out_csv),
                     "data_dir": str(ctx.data_dir)},
                    ctx.log_path("ate_infer"), ctx.dry_run)
    if rc != 0:
        raise RuntimeError(f"ATE inference thất bại (exit {rc})")


def stage_apc(ctx: Ctx) -> None:
    """Train mọi biến thể APC (seed chính) → runs_joint*/<config>/ + bảng tổng hợp."""
    if ctx.no_train:
        log("--no-train: bỏ qua stage `apc` (không có chế độ eval-only cho runner này).")
        return
    seed = ctx.seeds[0]
    for bb in ctx.backbones:
        banner(f"APC · backbone={bb.upper()} · seed={seed}", "-")
        payload = {
            "model_type": bb,
            "pretrained": PRETRAINED_BY_TYPE[bb],
            "runs_dir": str(ctx.runs_joint(bb)),
            "seed": seed,
            "configs": [joint_tuple(v) for v in ctx.variants],
            "resume": ctx.resume,
            "data_dir": str(ctx.data_dir),
            "epochs": ctx.max_epochs,
        }
        rc = run_driver("apc", payload, ctx.log_path(f"apc_{bb}"), ctx.dry_run)
        if rc != 0:
            raise RuntimeError(f"Train APC ({bb}) thất bại (exit {rc})")


def stage_multiseed(ctx: Ctx) -> None:
    """Train đa seed + oracle/e2e eval + sinh 8 bảng luận văn."""
    resize_like = [v for v in ctx.variants if v["group"] != "compact"]
    compact = [v for v in ctx.variants if v["group"] == "compact"]

    argv: List[str] = [
        "--seeds", *[str(s) for s in ctx.seeds],
        "--model-types", *ctx.backbones,
        "--ate-f1", f"{read_ate_f1(out_root=ctx.out_root):.2f}",
        "--runs-dir", str(ctx.out("runs_multiseed")),
    ]
    ate_dir = ctx.out("runs_ate")
    if any((ate_dir / f"seed_{s}" / "test_predictions.csv").is_file() for s in ctx.seeds):
        argv += ["--ate-csv-dir", str(ate_dir)]
    else:
        argv += ["--ate-csv", str(ate_dir / "test_ate_predictions.csv")]
    if compact:
        argv.append("--include-compact")
    if ctx.resume:
        argv.append("--resume")
    if ctx.no_train:
        argv.append("--no-train")

    payload = {
        "all_configs": [ms_tuple(v) for v in resize_like],
        "compact_configs": [ms_tuple(v) for v in compact],
        "e2e_bert": E2E_BERT_CONFIGS,
        "e2e_t5": E2E_T5_CONFIGS,
        "compact_vs_resize": COMPACT_VS_RESIZE_CONFIGS,
        "paper_our": PAPER_OUR_CONFIGS,
        "argv": argv,
        "data_dir": str(ctx.data_dir),
        "epochs": ctx.max_epochs,
    }
    rc = run_driver("multiseed", payload, ctx.log_path("multiseed"), ctx.dry_run)
    if rc != 0:
        raise RuntimeError(f"run_multiseed.py thất bại (exit {rc})")


def stage_gold(ctx: Ctx) -> None:
    """Oracle eval: APC với gold aspect term (trần trên của classifier)."""
    for bb in ctx.backbones:
        runs_dir = ctx.runs_joint(bb)
        if not ctx.dry_run and not any(runs_dir.glob("*/best_model.pt")):
            log(f"[bỏ qua] {runs_dir} chưa có checkpoint nào — chạy stage `apc` trước.")
            continue
        banner(f"Gold-aspect eval · {bb.upper()}", "-")
        payload = {
            "model_type": bb,
            "pretrained": PRETRAINED_BY_TYPE[bb],
            "runs_dir": str(runs_dir),
            "out_dir": str(ctx.out("runs_bert_gold", bb.upper())),
            "data_dir": str(ctx.data_dir),
        }
        rc = run_driver("gold", payload, ctx.log_path(f"gold_{bb}"), ctx.dry_run)
        if rc != 0:
            raise RuntimeError(f"Gold-aspect eval ({bb}) thất bại (exit {rc})")


def stage_triplet(ctx: Ctx) -> None:
    """Eval bộ ba end-to-end (ATE dự đoán → APC) cho từng backbone."""
    ate_csv = ctx.out("runs_ate", "test_ate_predictions.csv")
    if not ctx.dry_run and not ate_csv.is_file():
        raise RuntimeError(f"Thiếu {ate_csv} — chạy stage `ate_infer` trước.")
    for bb in ctx.backbones:
        runs_dir = ctx.runs_joint(bb)
        if not ctx.dry_run and not any(runs_dir.glob("*/best_model.pt")):
            log(f"[bỏ qua] {runs_dir} chưa có checkpoint nào — chạy stage `apc` trước.")
            continue
        banner(f"Joint triplet eval · {bb.upper()}", "-")
        payload = {
            "model_type": bb,
            "pretrained": PRETRAINED_BY_TYPE[bb],
            "runs_dir": str(runs_dir),
            "ate_csv": str(ate_csv),
            "out_csv": str(ctx.out("runs_ate", f"eval_joint_triplet_{bb.upper()}.csv")),
            "data_dir": str(ctx.data_dir),
        }
        rc = run_driver("triplet", payload, ctx.log_path(f"triplet_{bb}"), ctx.dry_run)
        if rc != 0:
            raise RuntimeError(f"Joint triplet eval ({bb}) thất bại (exit {rc})")


def stage_gas(ctx: Ctx) -> None:
    """GAS một bước: T5 sinh thẳng bộ ba (aspect, category, sentiment)."""
    ckpt_root = ctx.out("checkpoints_gas")
    best = ckpt_root / "best"
    out_dir = ctx.out("runs_gas")

    if ctx.no_train:
        log("--no-train: bỏ qua bước train GAS.")
    elif ctx.resume and best.is_dir():
        log(f"--resume: đã có {best} — bỏ qua bước train GAS.")
    else:
        cmd = [sys.executable, "gas/train_gas.py",
               "--data-dir", str(ctx.data_dir),
               "--output-dir", str(ckpt_root),
               "--epochs", str(ctx.max_epochs or 20),
               "--seed", str(ctx.seeds[0])]
        if ctx.max_epochs:
            cmd += ["--patience", "1"]
        rc = run_cmd(cmd, ctx.log_path("gas"), ctx.dry_run)
        if rc != 0:
            raise RuntimeError(f"gas/train_gas.py thất bại (exit {rc})")

    if not ctx.dry_run:
        if not best.is_dir():
            log(f"[bỏ qua] Không có {best} → không eval được GAS.")
            return
        out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "gas/evaluate_joint.py",
           "--gas-checkpoint", str(best),
           "--data-dir", str(ctx.data_dir),
           "--split", "test",
           "--output-dir", str(out_dir)]
    rc = run_cmd(cmd, ctx.log_path("gas"), ctx.dry_run)
    if rc != 0:
        raise RuntimeError(f"gas/evaluate_joint.py thất bại (exit {rc})")


def stage_results(ctx: Ctx) -> None:
    """Eval trên results.csv: LLM split → GAS T5 ATE → APC.

    Lưu ý: cờ `--gas-checkpoint` của eval_results.py nhận checkpoint **GAS T5
    ATE** (chỉ sinh aspect term, xem comment tại eval_results.py:148), KHÔNG
    phải model GAS một bước của `gas/`.  Vì vậy stage này dùng đúng checkpoint
    ATE như stage `ate_infer`, và không phụ thuộc vào stage `gas`.
    """
    results_csv = ROOT / "results.csv"
    gas_ckpt = ctx.ate_ckpt()
    if not ctx.dry_run:
        if not results_csv.is_file():
            log(f"[bỏ qua] Không có {results_csv}.")
            return
        if gas_ckpt is None:
            log("[bỏ qua] Không tìm thấy checkpoint T5 ATE — chạy stage `ate` trước.")
            return
    if gas_ckpt is None:
        gas_ckpt = ROOT / "checkpoints" / "best"
    for bb in ctx.backbones:
        runs_dir = ctx.runs_joint(bb)
        if not ctx.dry_run and not any(runs_dir.glob("*/best_model.pt")):
            log(f"[bỏ qua] {runs_dir} chưa có checkpoint nào.")
            continue
        banner(f"eval_results · {bb.upper()}", "-")
        cmd = [sys.executable, "experiments/eval_results.py",
               "--model-type", bb,
               "--runs-dir", str(runs_dir),
               "--results-csv", str(results_csv),
               "--gas-checkpoint", str(gas_ckpt)]
        rc = run_cmd(cmd, ctx.log_path(f"results_{bb}"), ctx.dry_run)
        if rc != 0:
            raise RuntimeError(f"eval_results.py ({bb}) thất bại (exit {rc})")


def stage_uos(ctx: Ctx) -> None:
    """Tách câu UOS bằng LLM (Ollama Qwen3:8B) trên tập test."""
    host = ctx.args.ollama_host
    if not ctx.dry_run and not ollama_available(host):
        log(f"[bỏ qua] Ollama không phản hồi tại {host}. "
            f"Chạy `ollama serve` + `ollama pull qwen3:8b` rồi chạy lại "
            f"`python run_all.py --stages uos report`.")
        return
    out_dir = (ctx.out("uos_output", "test") if ctx.smoke
               else ROOT / "uos" / "output" / "test")
    cmd = [sys.executable, "uos/run_llm_uos_eval.py",
           "--data_path", str(ctx.data_dir / "test.apc"),
           "--output_dir", str(out_dir),
           "--llm_model", ctx.args.uos_model,
           "--ollama_host", host,
           "--resume"]
    if ctx.smoke:
        cmd += ["--max_rows", str(SMOKE_UOS_ROWS)]
    rc = run_cmd(cmd, ctx.log_path("uos"), ctx.dry_run)
    if rc != 0:
        raise RuntimeError(f"uos/run_llm_uos_eval.py thất bại (exit {rc})")


def stage_figures(ctx: Ctx) -> None:
    """Sinh hình cho luận văn.

    Lượt thật ghi vào thesis/figures/; lượt smoke ghi vào sandbox để không đè
    lên các hình đã commit.
    """
    fig_dir = ctx.out("thesis_figures") if ctx.smoke else ROOT / "thesis" / "figures"
    payload = {
        "fig_dir": str(fig_dir),
        "runs_ate": str(ctx.out("runs_ate")),
        "data_dir": str(ctx.data_dir),
    }
    rc = run_driver("figures", payload, ctx.log_path("figures"), ctx.dry_run)
    if rc != 0:
        log(f"[cảnh báo] generate_thesis_figures thất bại (exit {rc}) — chạy tiếp.")

    if ctx.smoke:
        log("[smoke] Bỏ qua visualize_scm*.py (hình minh hoạ thuật toán, "
            "không phụ thuộc dữ liệu, ghi thẳng vào thesis/figures/).")
        return
    for script in ["scripts/visualize_scm.py", "scripts/visualize_scm_example.py"]:
        rc = run_cmd([sys.executable, script], ctx.log_path("figures"), ctx.dry_run)
        if rc != 0:
            log(f"[cảnh báo] {script} thất bại (exit {rc}) — tiếp tục các hình còn lại.")


# ── Stage `report` ────────────────────────────────────────────────────────────

# (nhóm, glob, mô tả, base) — base "out" = theo out_root (sandbox khi smoke),
#                              base "root" = luôn theo thư mục repo
ARTIFACT_SPECS: List[tuple] = [
    ("Môi trường", "reports/00_environment.txt", "Snapshot môi trường + kế hoạch chạy", "out"),
    ("ATE", "runs_ate/results_ate_multiseed.csv", "P/R/F1 ATE theo từng seed", "out"),
    ("ATE", "runs_ate/results_ate_summary.txt", "ATE mean±std qua các seed", "out"),
    ("ATE", "runs_ate/seed_*/test_predictions.csv", "Prediction ATE theo seed (đầu vào e2e)", "out"),
    ("ATE", "runs_ate/test_ate_predictions.csv", "Prediction ATE dùng chung cho eval e2e", "out"),
    ("APC", "runs_joint/experiment_results_joint.csv", "Toàn bộ metric APC — BERT", "out"),
    ("APC", "runs_joint/experiment_results_joint.txt", "Bảng tổng hợp APC — BERT", "out"),
    ("APC", "runs_joint_t5/experiment_results_joint.csv", "Toàn bộ metric APC — T5", "out"),
    ("APC", "runs_joint_t5/experiment_results_joint.txt", "Bảng tổng hợp APC — T5", "out"),
    ("Multi-seed", "runs_multiseed/thesis_tables.txt", "8 bảng luận văn (oracle, e2e, compact, paper)", "out"),
    ("Multi-seed", "runs_multiseed/results_raw.csv", "Kết quả thô từng (backbone, config, seed)", "out"),
    ("Multi-seed", "runs_multiseed/results_aggregated.csv", "Kết quả gộp mean±std", "out"),
    ("Multi-seed", "runs_multiseed/results_summary.txt", "Tóm tắt lượt chạy multi-seed", "out"),
    ("Oracle", "runs_bert_gold/*/eval_gold_aspects.csv", "APC với gold aspect (trần trên)", "out"),
    ("End-to-end", "runs_ate/eval_joint_triplet_*.csv", "Eval bộ ba e2e theo backbone", "out"),
    ("End-to-end", "runs_ate/eval_results_*.csv", "Eval trên results.csv theo backbone", "out"),
    ("GAS", "runs_gas/eval_test.json", "GAS một bước — metric JSON", "out"),
    ("GAS", "runs_gas/eval_test.csv", "GAS một bước — metric CSV", "out"),
    ("UOS", "<uos>/metrics.json", "Thống kê tách câu UOS", "out"),
    ("UOS", "<uos>/results.jsonl", "Kết quả tách câu từng câu", "out"),
    ("Hình", "thesis/figures/*.pdf", "Hình luận văn (PDF)", "root"),
    ("Hình", "thesis/figures/*.png", "Hình luận văn (PNG)", "root"),
    ("Log", "reports/logs/*.log", "Log đầy đủ của từng stage", "out"),
]


def _embed(md: List[str], title: str, path: Path, max_lines: int = 200,
           lang: str = "text") -> None:
    if not path.is_file():
        return
    md.append(f"### {title}")
    md.append("")
    try:
        shown = path.relative_to(ROOT).as_posix()
    except ValueError:
        shown = str(path)
    md.append(f"`{shown}`")
    md.append("")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    clipped = lines[:max_lines]
    md.append(f"```{lang}")
    md.extend(clipped)
    if len(lines) > max_lines:
        md.append(f"… (còn {len(lines) - max_lines} dòng — xem file gốc)")
    md.append("```")
    md.append("")


def stage_report(ctx: Ctx) -> None:
    """Gom mọi artifact thành reports/REPORT.md + reports/manifest.csv."""
    if ctx.dry_run:
        log(f"(dry-run) sẽ ghi {ctx.reports_dir / 'REPORT.md'} và manifest.csv")
        return

    ctx.reports_dir.mkdir(parents=True, exist_ok=True)
    uos_rel = ("uos_output/test" if ctx.smoke else "uos/output/test")
    fig_rel = ("thesis_figures" if ctx.smoke else "thesis/figures")

    # ── manifest ──────────────────────────────────────────────────────────────
    rows: List[Dict[str, str]] = []
    for group, pattern, desc, base in ARTIFACT_SPECS:
        pattern = pattern.replace("<uos>", uos_rel)
        if base == "root" and ctx.smoke and pattern.startswith("thesis/figures/"):
            pattern = pattern.replace("thesis/figures/", fig_rel + "/")
            base = "out"
        base_dir = ROOT if base == "root" else ctx.out_root
        matches = sorted(base_dir.glob(pattern)) if any(c in pattern for c in "*?[") \
            else ([base_dir / pattern] if (base_dir / pattern).exists() else [])
        if not matches:
            rows.append({"group": group, "path": pattern, "status": "THIẾU",
                         "size_kb": "", "modified": "", "description": desc})
            continue
        for m in matches:
            rows.append({
                "group": group,
                "path": m.relative_to(base_dir).as_posix(),
                "status": "OK",
                "size_kb": f"{m.stat().st_size / 1024:.1f}",
                "modified": datetime.fromtimestamp(m.stat().st_mtime).isoformat(timespec="seconds"),
                "description": desc,
            })

    manifest = ctx.reports_dir / "manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["group", "path", "status", "size_kb",
                                          "modified", "description"])
        w.writeheader()
        w.writerows(rows)

    # ── REPORT.md ─────────────────────────────────────────────────────────────
    status: List[Dict] = []
    if ctx.status_json.is_file():
        try:
            status = json.loads(ctx.status_json.read_text(encoding="utf-8")).get("stages", [])
        except Exception:  # noqa: BLE001
            status = []

    md: List[str] = []
    md.append("# Báo cáo thực nghiệm — ABSA Token Merging")
    md.append("")
    md.append(f"Sinh tự động bởi `run_all.py` lúc "
              f"{datetime.now().isoformat(timespec='seconds')}.")
    md.append("")
    if ctx.smoke:
        md.append("> ⚠️ **Lượt CHẠY THỬ (`--smoke`)** — dataset tí hon "
                  f"({SMOKE_SIZES['train']}/{SMOKE_SIZES['dev']}/{SMOKE_SIZES['test']} mẫu), "
                  f"{ctx.max_epochs} epoch. Các con số dưới đây chỉ dùng để xác nhận "
                  "đường ống chạy được, KHÔNG có giá trị khoa học.")
        md.append("")
    md.append(f"- Chế độ: `{'smoke' if ctx.smoke else 'đầy đủ'}`  ·  data-dir: `{ctx.data_dir.name}`")
    md.append(f"- Seeds: `{ctx.seeds}`")
    md.append(f"- Backbones: `{ctx.backbones}`")
    md.append(f"- Biến thể ({len(ctx.variants)}): "
              + ", ".join(f"`{v['ms_id']}`" for v in ctx.variants))
    md.append(f"- ATE F1 dùng cho header bảng: `{read_ate_f1(out_root=ctx.out_root):.2f}`")
    md.append("")

    md.append("## 1. Trạng thái các stage")
    md.append("")
    if status:
        md.append("| Stage | Trạng thái | Thời gian | Ghi chú |")
        md.append("|---|---|---|---|")
        for s in status:
            dur = s.get("duration_sec")
            dur_s = f"{dur / 60:.1f} phút" if isinstance(dur, (int, float)) else "—"
            md.append(f"| `{s.get('stage')}` | {s.get('status')} | {dur_s} | "
                      f"{s.get('note', '') or ''} |")
    else:
        md.append("_Chưa có `reports/run_status.json` — stage `report` được chạy riêng lẻ._")
    md.append("")

    md.append("## 2. Danh mục kết quả (artifact)")
    md.append("")
    md.append("| Nhóm | File | Trạng thái | KB | Mô tả |")
    md.append("|---|---|---|---|---|")
    for r in rows:
        md.append(f"| {r['group']} | `{r['path']}` | {r['status']} | "
                  f"{r['size_kb']} | {r['description']} |")
    md.append("")
    n_ok = sum(1 for r in rows if r["status"] == "OK")
    md.append(f"**{n_ok}/{len(rows)} artifact có mặt.** Bản đầy đủ: `reports/manifest.csv`.")
    md.append("")

    md.append("## 3. Bảng kết quả chính")
    md.append("")
    _embed(md, "3.1 Các bảng luận văn (multi-seed)",
           ctx.out("runs_multiseed", "thesis_tables.txt"), max_lines=400)
    _embed(md, "3.2 Tổng hợp APC — BERT",
           ctx.out("runs_joint", "experiment_results_joint.txt"), max_lines=200)
    _embed(md, "3.3 Tổng hợp APC — T5",
           ctx.out("runs_joint_t5", "experiment_results_joint.txt"), max_lines=200)
    _embed(md, "3.4 ATE đa seed",
           ctx.out("runs_ate", "results_ate_summary.txt"), max_lines=120)
    _embed(md, "3.5 GAS một bước",
           ctx.out("runs_gas", "eval_test.json"), max_lines=120, lang="json")
    _embed(md, "3.6 UOS — thống kê tách câu",
           ctx.out_root / uos_rel / "metrics.json", max_lines=80, lang="json")
    _embed(md, "3.7 Môi trường chạy",
           ctx.reports_dir / "00_environment.txt", max_lines=120)

    md.append("## 4. Tái tạo")
    md.append("")
    md.append("```bash")
    md.append("python run_all.py --list          # xem stage + biến thể")
    md.append("python run_all.py --dry-run       # xem lệnh sẽ chạy")
    md.append("python run_all.py --resume        # chạy tiếp, bỏ qua phần đã xong")
    md.append("python run_all.py --stages report # chỉ sinh lại báo cáo này")
    md.append("```")
    md.append("")
    md.append("Log chi tiết từng stage: `reports/logs/<stage>.log`.")
    md.append("")

    report = ctx.reports_dir / "REPORT.md"
    report.write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"\n→ {report}")
    log(f"→ {manifest}")
    log(f"   {n_ok}/{len(rows)} artifact có mặt.")


# ── Đăng ký stage (thứ tự chạy) ───────────────────────────────────────────────

STAGES: List[tuple] = [
    ("env", stage_env, "Preflight: môi trường, dữ liệu, checkpoint, kế hoạch"),
    ("ate", stage_ate, "Train T5 ATE (GAS) đa seed + prediction theo seed"),
    ("ate_infer", stage_ate_infer, "Infer ATE trên test → test_ate_predictions.csv"),
    ("apc", stage_apc, "Train mọi biến thể APC (seed chính) → runs_joint*/"),
    ("multiseed", stage_multiseed, "Train đa seed + oracle/e2e eval + 8 bảng luận văn"),
    ("gold", stage_gold, "Oracle eval với gold aspect term"),
    ("triplet", stage_triplet, "Eval bộ ba end-to-end (ATE → APC)"),
    ("gas", stage_gas, "Train + eval GAS một bước"),
    ("results", stage_results, "Eval trên results.csv (LLM split → GAS T5 ATE → APC)"),
    ("uos", stage_uos, "Tách câu UOS bằng LLM qua Ollama"),
    ("figures", stage_figures, "Sinh hình luận văn → thesis/figures/"),
    ("report", stage_report, "Gom tất cả → reports/REPORT.md + manifest.csv"),
]

STAGE_NAMES = [s[0] for s in STAGES]
STAGE_FN = {name: fn for name, fn, _ in STAGES}
STAGE_DESC = {name: desc for name, _, desc in STAGES}


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Chạy full luồng thực nghiệm ABSA Token-Merging và sinh mọi report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Ví dụ\n-----\n")[-1] if "Ví dụ" in __doc__ else None,
    )
    p.add_argument("--stages", nargs="+", default=None, metavar="STAGE",
                   choices=STAGE_NAMES,
                   help=f"Chỉ chạy các stage này (mặc định: tất cả). Có: {STAGE_NAMES}")
    p.add_argument("--skip", nargs="+", default=[], metavar="STAGE",
                   choices=STAGE_NAMES, help="Bỏ qua các stage này")
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, metavar="N",
                   help=f"Seeds (mặc định: {DEFAULT_SEEDS}). Seed đầu tiên dùng cho stage `apc`/`gas`.")
    p.add_argument("--backbones", nargs="+", default=DEFAULT_BACKBONES,
                   choices=list(PRETRAINED_BY_TYPE), help="Backbone APC")
    p.add_argument("--variants", nargs="+", default=VARIANT_GROUPS,
                   choices=VARIANT_GROUPS,
                   help=f"Nhóm biến thể (mặc định: tất cả = {VARIANT_GROUPS})")
    p.add_argument("--smoke", action="store_true",
                   help="CHẠY THỬ: dựng dataset tí hon (48/16/16 mẫu), 1 epoch, "
                        "1 seed, 3 biến thể — để kiểm tra toàn bộ đường ống chạy "
                        "được trước khi tốn GPU cho lượt đầy đủ")
    p.add_argument("--out-root", default=None, metavar="DIR",
                   help="Thư mục gốc cho MỌI output (mặc định: repo; --smoke dùng smoke_run/)")
    p.add_argument("--data-dir", default=None, metavar="DIR",
                   help="Thư mục dữ liệu (mặc định: dataset/; --smoke dùng dataset_smoke/)")
    p.add_argument("--max-epochs", type=int, default=None, metavar="N",
                   help="Giới hạn số epoch cho mọi bước train (mặc định: theo từng script)")
    p.add_argument("--resume", action="store_true",
                   help="Bỏ qua các combo đã có checkpoint / kết quả")
    p.add_argument("--no-train", action="store_true",
                   help="Không train — chỉ inference, eval và gom báo cáo từ kết quả có sẵn")
    p.add_argument("--fail-fast", action="store_true",
                   help="Dừng ngay khi một stage lỗi (mặc định: ghi lỗi rồi chạy tiếp)")
    p.add_argument("--dry-run", action="store_true",
                   help="Chỉ in ra các lệnh sẽ chạy, không thực thi")
    p.add_argument("--list", action="store_true",
                   help="Liệt kê stage và biến thể rồi thoát")
    p.add_argument("--ollama-host", default="http://localhost:11434",
                   help="Host Ollama cho stage `uos`")
    p.add_argument("--uos-model", default="qwen3:8b", help="Model LLM cho stage `uos`")

    # Dùng nội bộ: tự gọi lại chính file này để chạy driver trong tiến trình con.
    p.add_argument("--_driver", default=None, help=argparse.SUPPRESS)
    p.add_argument("--_payload", default=None, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def print_listing(args: argparse.Namespace) -> None:
    banner("STAGES")
    for name, _, desc in STAGES:
        mark = " " if (args.stages is None or name in args.stages) and name not in args.skip else "x"
        log(f"  [{mark}] {name:<12} {desc}")
    log("\n  [x] = bị bỏ qua với tham số hiện tại")

    banner("BIẾN THỂ")
    log(f"  {'ms_id':<24} {'runs_joint/':<26} {'display':<22} flags")
    log("  " + "-" * 104)
    for v in VARIANTS:
        sel = "*" if v["group"] in args.variants else " "
        lcf, cdm, tome, resize, strat, pre = v["flags"]
        log(f" {sel}{v['ms_id']:<24} {v['joint_id']:<26} {v['display']:<22} "
            f"lcf={int(lcf)} cdm={int(cdm)} tome={int(tome)} resize={int(resize)} "
            f"pre={int(pre)} {strat}")
    log("\n  '*' = được chọn. Nhóm: " + ", ".join(
        f"{g} ({sum(1 for v in VARIANTS if v['group'] == g)})" for g in VARIANT_GROUPS))

    if args.smoke:
        log(f"\n  --smoke: chỉ chạy {SMOKE_VARIANT_IDS}, 1 seed, "
            f"{SMOKE_SIZES} mẫu, 1 epoch; bỏ qua stage {SMOKE_SKIP_STAGES}.")
    n = len(selected_variants(args.variants)) * len(args.seeds) * len(args.backbones)
    log(f"\n  Stage `multiseed` sẽ train {n} lần "
        f"({len(selected_variants(args.variants))} biến thể × {len(args.seeds)} seed "
        f"× {len(args.backbones)} backbone).")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    # ── Chế độ driver (tiến trình con) ────────────────────────────────────────
    if args._driver:
        payload = json.loads(args._payload or "{}")
        DRIVERS[args._driver](payload)
        return 0

    if args.list:
        print_listing(args)
        return 0

    ctx = Ctx(args)
    if not ctx.dry_run:
        ctx.reports_dir.mkdir(parents=True, exist_ok=True)
        ctx.logs_dir.mkdir(parents=True, exist_ok=True)

    todo = [n for n in (args.stages or STAGE_NAMES) if n not in args.skip]
    if ctx.smoke:
        dropped = [n for n in todo if n in SMOKE_SKIP_STAGES]
        todo = [n for n in todo if n not in SMOKE_SKIP_STAGES]
        if dropped:
            log(f"[smoke] Bỏ qua stage không chuyển được sang dataset nhỏ: {dropped}")
        if not args.dry_run:
            build_smoke_dataset(dst_dir=ctx.data_dir)
    if not todo:
        log("Không có stage nào để chạy.")
        return 0

    banner("RUN ALL — ABSA Token Merging")
    log(f"  Stages    : {todo}")
    log(f"  Seeds     : {ctx.seeds}")
    log(f"  Backbones : {ctx.backbones}")
    log(f"  Biến thể  : {len(ctx.variants)} "
        + (", ".join(v["ms_id"] for v in ctx.variants) if ctx.smoke
           else "(" + ", ".join(args.variants) + ")"))
    log(f"  Data dir  : {ctx.data_dir}")
    log(f"  Max epochs: {ctx.max_epochs if ctx.max_epochs else '(mặc định của script)'}")
    log(f"  smoke={ctx.smoke}  resume={args.resume}  no_train={args.no_train}  "
        f"dry_run={args.dry_run}")
    log(f"  Out root  : {ctx.out_root}")
    log(f"  Log       : {ctx.logs_dir}")

    results: List[Dict] = []
    t0 = time.perf_counter()

    for name in todo:
        banner(f"STAGE: {name} — {STAGE_DESC[name]}")
        t = time.perf_counter()
        entry: Dict = {"stage": name, "started": datetime.now().isoformat(timespec="seconds")}
        try:
            STAGE_FN[name](ctx)
            entry["status"] = "OK"
        except KeyboardInterrupt:
            entry["status"] = "HỦY"
            entry["duration_sec"] = time.perf_counter() - t
            results.append(entry)
            log("\n[Ctrl-C] Dừng theo yêu cầu.")
            break
        except Exception as e:  # noqa: BLE001
            entry["status"] = "LỖI"
            entry["note"] = f"{type(e).__name__}: {e}"
            log(f"\n[LỖI] stage `{name}`: {type(e).__name__}: {e}")
            if args.fail_fast:
                entry["duration_sec"] = time.perf_counter() - t
                results.append(entry)
                _save_status(results, time.perf_counter() - t0, ctx)
                return 1
        entry.setdefault("duration_sec", time.perf_counter() - t)
        results.append(entry)
        _save_status(results, time.perf_counter() - t0, ctx)
        log(f"\n[{entry['status']}] stage `{name}` — {entry['duration_sec'] / 60:.1f} phút")

    total = time.perf_counter() - t0
    _save_status(results, total, ctx)

    banner("TỔNG KẾT")
    for r in results:
        dur = r.get("duration_sec", 0) / 60
        log(f"  {r['status']:<5} {r['stage']:<12} {dur:6.1f} phút  {r.get('note', '')}")
    log(f"\n  Tổng thời gian: {total / 60:.1f} phút")
    log(f"  Báo cáo       : {ctx.reports_dir / 'REPORT.md'}")

    return 0 if all(r["status"] == "OK" for r in results) else 1


def _save_status(results: List[Dict], total: float, ctx: Ctx) -> None:
    if ctx.dry_run:
        return
    ctx.status_json.parent.mkdir(parents=True, exist_ok=True)
    ctx.status_json.write_text(json.dumps({
        "generated": datetime.now().isoformat(timespec="seconds"),
        "total_sec": total,
        "seeds": ctx.seeds,
        "backbones": ctx.backbones,
        "variant_groups": ctx.args.variants,
        "variants": [v["ms_id"] for v in ctx.variants],
        "resume": ctx.resume,
        "no_train": ctx.no_train,
        "smoke": ctx.smoke,
        "out_root": str(ctx.out_root),
        "data_dir": str(ctx.data_dir),
        "max_epochs": ctx.max_epochs,
        "stages": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
