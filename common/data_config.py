# -*- coding: utf-8 -*-
"""Cấu hình đường dẫn dữ liệu train / dev / test — một nguồn sự thật duy nhất.

Mọi script trong repo (run_all.py, common/*, gas/*, src/*) đều lấy đường dẫn
.apc qua module này, nên chỉ cần khai báo ở MỘT chỗ là cả pipeline dùng theo.

Thứ tự ưu tiên khi tìm một split (train/dev/test):

    1. Tham số truyền thẳng   (--train-file / train=...)
    2. Biến môi trường        (KLTN_TRAIN_FILE ...)
    3. <data_dir>/<tên chuẩn> (train.apc / dev.apc / test.apc)

và `data_dir` cũng theo đúng thứ tự đó:

    1. --data-dir
    2. KLTN_DATA_DIR
    3. tự dò: thư mục Kaggle Dataset trong /kaggle/input có đủ 3 file .apc
    4. <repo>/dataset

Nhờ bước 3, notebook Kaggle chạy được cả khi dữ liệu nằm ở
/kaggle/input/<ten-dataset>/ lẫn khi chỉ có dataset/ đi kèm repo sau git clone.

Dùng trong notebook / script
----------------------------
    from common.data_config import resolve_data_paths

    paths = resolve_data_paths()                      # tự dò
    paths = resolve_data_paths(data_dir="/kaggle/input/absa-rest14")
    paths = resolve_data_paths(train="/kaggle/input/x/train_v2.apc")
    paths.export_env()        # để mọi tiến trình con dùng chung cấu hình
    print(paths.describe())
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# ─── Hằng số ──────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "dataset"

SPLIT_FILES: Dict[str, str] = {
    "train": "train.apc",
    "dev": "dev.apc",
    "test": "test.apc",
}
GOLD_CSV_NAME = "test_sentences_id.csv"
SUPPLEMENT_DIR_NAME = "supplement"

# Biến môi trường — run_all.py export những biến này trước khi gọi tiến trình
# con, nên driver và script con không cần thêm cờ CLI nào.
ENV_DATA_DIR = "KLTN_DATA_DIR"
ENV_SPLIT = {
    "train": "KLTN_TRAIN_FILE",
    "dev": "KLTN_DEV_FILE",
    "test": "KLTN_TEST_FILE",
}
ENV_GOLD_CSV = "KLTN_GOLD_CSV"
ENV_SUPPLEMENT_DIR = "KLTN_SUPPLEMENT_DIR"

# Nơi tự dò dữ liệu khi không khai báo gì (Kaggle mount dataset read-only ở đây).
KAGGLE_INPUT = Path("/kaggle/input")


# ─── Tự dò thư mục dữ liệu ────────────────────────────────────────────────────

def _has_all_splits(d: Path) -> bool:
    return d.is_dir() and all((d / name).is_file() for name in SPLIT_FILES.values())


def autodetect_data_dir() -> Optional[Path]:
    """Tìm thư mục có đủ train.apc + dev.apc + test.apc trong /kaggle/input.

    Quét cả thư mục con một cấp vì Kaggle Dataset thường giữ nguyên cấu trúc
    thư mục lúc upload (ví dụ /kaggle/input/kltn-absa/dataset/train.apc).
    Trả về None khi không tìm thấy → caller rơi về dataset/ của repo.
    """
    if not KAGGLE_INPUT.is_dir():
        return None
    try:
        candidates = sorted(p for p in KAGGLE_INPUT.iterdir() if p.is_dir())
    except OSError:
        return None
    for base in candidates:
        if _has_all_splits(base):
            return base
        try:
            subs = sorted(p for p in base.iterdir() if p.is_dir())
        except OSError:
            continue
        for sub in subs:
            if _has_all_splits(sub):
                return sub
    return None


def resolve_data_dir(data_dir: Optional[str | Path] = None,
                     *, autodetect: bool = True) -> Path:
    """Thư mục dữ liệu theo thứ tự: tham số → env → tự dò Kaggle → dataset/."""
    if data_dir:
        return Path(data_dir).expanduser()
    env = os.environ.get(ENV_DATA_DIR)
    if env:
        return Path(env).expanduser()
    if autodetect:
        found = autodetect_data_dir()
        if found is not None:
            return found
    return DEFAULT_DATA_DIR


# ─── Đường dẫn từng split ─────────────────────────────────────────────────────

def resolve_split_path(split: str,
                       data_dir: Optional[str | Path] = None,
                       override: Optional[str | Path] = None,
                       *, autodetect: bool = True, use_env: bool = True) -> Path:
    """Đường dẫn file .apc của một split, có tính cả override và env.

    Đây là hàm mà các loader (common/ate_dataset_utils.py, gas/dataset.py) gọi,
    nên chỉ cần set env một lần là mọi loader đi theo.

    use_env=False khi caller đã chỉ định data_dir tường minh: giá trị truyền
    thẳng luôn thắng biến môi trường, tránh env cũ đè lên --data-dir mới.
    """
    if split not in SPLIT_FILES:
        raise ValueError(f"Split không hợp lệ: {split!r} — phải là một trong {list(SPLIT_FILES)}")
    if override:
        return Path(override).expanduser()
    if use_env:
        env = os.environ.get(ENV_SPLIT[split])
        if env:
            return Path(env).expanduser()
    return resolve_data_dir(data_dir, autodetect=autodetect) / SPLIT_FILES[split]


@dataclass
class DataPaths:
    """Bộ đường dẫn dữ liệu đã chốt cho một lượt chạy."""

    data_dir: Path
    train: Path
    dev: Path
    test: Path
    gold_csv: Path
    supplement_dir: Path

    # ── Truy cập ────────────────────────────────────────────────────────────
    def split(self, name: str) -> Path:
        return {"train": self.train, "dev": self.dev, "test": self.test}[name]

    def as_dict(self) -> Dict[str, str]:
        """Dạng JSON-safe để nhét vào payload của driver."""
        return {
            "data_dir": str(self.data_dir),
            "train": str(self.train),
            "dev": str(self.dev),
            "test": str(self.test),
            "gold_csv": str(self.gold_csv),
            "supplement_dir": str(self.supplement_dir),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> "DataPaths":
        return cls(
            data_dir=Path(d["data_dir"]),
            train=Path(d["train"]),
            dev=Path(d["dev"]),
            test=Path(d["test"]),
            gold_csv=Path(d["gold_csv"]),
            supplement_dir=Path(d["supplement_dir"]),
        )

    # ── Lan truyền sang tiến trình con ──────────────────────────────────────
    def env(self) -> Dict[str, str]:
        return {
            ENV_DATA_DIR: str(self.data_dir),
            ENV_SPLIT["train"]: str(self.train),
            ENV_SPLIT["dev"]: str(self.dev),
            ENV_SPLIT["test"]: str(self.test),
            ENV_GOLD_CSV: str(self.gold_csv),
            ENV_SUPPLEMENT_DIR: str(self.supplement_dir),
        }

    def export_env(self) -> None:
        """Ghi vào os.environ → mọi tiến trình con kế thừa đúng cấu hình này."""
        os.environ.update(self.env())

    # ── Kiểm tra / hiển thị ─────────────────────────────────────────────────
    def missing(self, *, require_gold: bool = False,
                require_supplement: bool = False) -> List[Path]:
        """Các file bắt buộc còn thiếu (danh sách rỗng = đủ)."""
        out = [p for p in (self.train, self.dev, self.test) if not p.is_file()]
        if require_gold and not self.gold_csv.is_file():
            out.append(self.gold_csv)
        if require_supplement and not self.supplement_dir.is_dir():
            out.append(self.supplement_dir)
        return out

    def describe(self) -> str:
        rows = [
            ("data_dir", self.data_dir, self.data_dir.is_dir()),
            ("train", self.train, self.train.is_file()),
            ("dev", self.dev, self.dev.is_file()),
            ("test", self.test, self.test.is_file()),
            ("gold_csv", self.gold_csv, self.gold_csv.is_file()),
            ("supplement", self.supplement_dir, self.supplement_dir.is_dir()),
        ]
        return "\n".join(
            f"  {'[OK]  ' if ok else '[THIẾU]'} {name:<11}: {path}"
            for name, path, ok in rows
        )


def resolve_data_paths(data_dir: Optional[str | Path] = None,
                       train: Optional[str | Path] = None,
                       dev: Optional[str | Path] = None,
                       test: Optional[str | Path] = None,
                       gold_csv: Optional[str | Path] = None,
                       supplement_dir: Optional[str | Path] = None,
                       *, autodetect: bool = True) -> DataPaths:
    """Chốt toàn bộ đường dẫn dữ liệu cho một lượt chạy.

    Mọi tham số đều tuỳ chọn; cái nào bỏ trống thì rơi về env rồi về
    <data_dir>/<tên chuẩn>.
    """
    root = resolve_data_dir(data_dir, autodetect=autodetect)
    # data_dir truyền thẳng thì nó là nguồn sự thật: bỏ qua env cho các split
    # chưa được override, nếu không env cũ sẽ đè lên --data-dir vừa đưa vào.
    use_env = not bool(data_dir)

    def _aux(explicit, env_key: str, default_name: str) -> Path:
        if explicit:
            return Path(explicit).expanduser()
        env = os.environ.get(env_key) if use_env else None
        return Path(env) if env else root / default_name

    return DataPaths(
        data_dir=root,
        train=resolve_split_path("train", root, train, autodetect=False, use_env=use_env),
        dev=resolve_split_path("dev", root, dev, autodetect=False, use_env=use_env),
        test=resolve_split_path("test", root, test, autodetect=False, use_env=use_env),
        gold_csv=_aux(gold_csv, ENV_GOLD_CSV, GOLD_CSV_NAME),
        supplement_dir=_aux(supplement_dir, ENV_SUPPLEMENT_DIR, SUPPLEMENT_DIR_NAME),
    )


def apply_cli_data_config(args, *, autodetect: bool = True,
                          announce: bool = True) -> DataPaths:
    """Chốt đường dẫn từ cờ CLI, export env rồi in ra — dùng đầu mỗi main()."""
    paths = paths_from_args(args, autodetect=autodetect)
    paths.export_env()
    if announce:
        print("[data] đường dẫn dữ liệu:")
        print(paths.describe())
    return paths


def add_data_args(parser, *, include_gold: bool = False) -> None:
    """Gắn nhóm cờ dữ liệu chuẩn vào một argparse.ArgumentParser."""
    g = parser.add_argument_group("dữ liệu")
    g.add_argument("--data-dir", default=None, metavar="DIR",
                   help=f"Thư mục dữ liệu (mặc định: ${ENV_DATA_DIR} → tự dò "
                        f"/kaggle/input → {DEFAULT_DATA_DIR})")
    g.add_argument("--train-file", default=None, metavar="FILE",
                   help="Ghi đè đường dẫn train.apc")
    g.add_argument("--dev-file", default=None, metavar="FILE",
                   help="Ghi đè đường dẫn dev.apc")
    g.add_argument("--test-file", default=None, metavar="FILE",
                   help="Ghi đè đường dẫn test.apc")
    if include_gold:
        g.add_argument("--gold-csv", default=None, metavar="FILE",
                       help=f"Ghi đè đường dẫn {GOLD_CSV_NAME}")


def paths_from_args(args, *, autodetect: bool = True) -> DataPaths:
    """DataPaths từ namespace của add_data_args()."""
    return resolve_data_paths(
        data_dir=getattr(args, "data_dir", None),
        train=getattr(args, "train_file", None),
        dev=getattr(args, "dev_file", None),
        test=getattr(args, "test_file", None),
        gold_csv=getattr(args, "gold_csv", None),
        autodetect=autodetect,
    )


if __name__ == "__main__":  # python -m common.data_config → xem cấu hình đang có
    p = resolve_data_paths()
    print("Cấu hình đường dẫn dữ liệu đang áp dụng:")
    print(p.describe())
    miss = p.missing(require_gold=True)
    print("\nThiếu file bắt buộc: " + (", ".join(str(m) for m in miss) if miss else "không"))
