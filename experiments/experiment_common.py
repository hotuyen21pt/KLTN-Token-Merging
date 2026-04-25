# -*- coding: utf-8 -*-
"""Shared APC training config + timing/metrics helpers for baseline vs ToMe."""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from pyabsa import (
    AspectPolarityClassification as APC,
    ModelSaveOption,
    DeviceTypeOption,
)

# APC raw format: blocks of 3 lines (sentence with single $T$, aspect term, polarity).
# PyABSA detects files whose path contains ``train`` + ``APC`` (+ ``.apc``); SemEval ``*.xml.seg`` names are skipped.
LOCAL_DATASET_DIR = ROOT / "thesis_apc_baseline" / "dataset"


def prepare_local_apc_dataset(root: Optional[Path] = None) -> Path:
    """Copy ``*.xml.seg`` → ``train.APC.*.apc`` / ``test.APC.*.apc`` if targets missing."""
    root = Path(root) if root is not None else LOCAL_DATASET_DIR
    pairs = [
        ("Restaurants_Train.xml.seg", "train.APC.restaurants.apc"),
        ("Restaurants_Test_Gold.xml.seg", "test.APC.restaurants.apc"),
    ]
    root.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in pairs:
        src = root / src_name
        dst = root / dst_name
        if dst.exists():
            continue
        if src.is_file():
            shutil.copyfile(src, dst)
    return root


def resolve_apc_dataset(dataset_dir: Optional[Union[str, Path]] = None) -> Union[str, Any]:
    """Prefer ``thesis_apc_baseline/dataset`` when present; else Restaurant14 benchmark."""
    root = Path(dataset_dir) if dataset_dir is not None else LOCAL_DATASET_DIR
    marker_seg = root / "Restaurants_Train.xml.seg"
    marker_apc = root / "train.APC.restaurants.apc"
    if root.is_dir() and (marker_seg.is_file() or marker_apc.is_file()):
        prepare_local_apc_dataset(root)
        return str(root)
    return APC.APCDatasetList.Restaurant14


def _read_apc_triplets(path: Path) -> List[Tuple[str, str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    out: List[Tuple[str, str, str]] = []
    for i in range(0, len(lines) - 2, 3):
        out.append((lines[i], lines[i + 1], lines[i + 2]))
    return out


def _write_apc_triplets(path: Path, triplets: List[Tuple[str, str, str]]) -> None:
    flat: List[str] = []
    for a, b, c in triplets:
        flat.extend([a, b, c])
    path.write_text("\n".join(flat) + ("\n" if flat else ""), encoding="utf-8")


def materialize_fraction_apc_dataset(
    dataset_root: Path,
    fraction: float,
    seed: int,
) -> Path:
    """Write train/test ``*.apc`` triplets subsampled to ``fraction`` into a child folder.

    Only ``*.apc`` whose names contain both (``train``+``APC``) or (``test``+``APC``) are
    reduced; other files in ``dataset_root`` are ignored (PyABSA uses the APC files).
    """
    if fraction >= 1.0 - 1e-12:
        return dataset_root

    out_root = dataset_root / "_subsample" / ("p%d_s%d" % (int(round(fraction * 100)), seed))
    out_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    wrote = False

    for apc in sorted(dataset_root.glob("*.apc")):
        low = apc.name.lower()
        if ("train" in low and "apc" in low) or ("test" in low and "apc" in low):
            triplets = _read_apc_triplets(apc)
            n = len(triplets)
            if n == 0:
                continue
            k = max(1, int(n * fraction))
            order = list(range(n))
            rng.shuffle(order)
            order = sorted(order[:k])
            picked = [triplets[i] for i in order]
            _write_apc_triplets(out_root / apc.name, picked)
            wrote = True

    return out_root if wrote else dataset_root


def maybe_subsample_dataset(
    dataset: Union[str, Any],
    fraction: float,
    seed: int,
) -> Union[str, Any]:
    """If ``dataset`` is a local directory with APC files, optionally shrink train/test."""
    if fraction >= 1.0 - 1e-12:
        return dataset
    if not isinstance(dataset, str):
        return dataset
    root = Path(dataset)
    if not root.is_dir():
        return dataset
    prepare_local_apc_dataset(root)
    has_apc = any(
        f.suffix.lower() == ".apc"
        and "apc" in f.name.lower()
        and ("train" in f.name.lower() or "test" in f.name.lower())
        for f in root.glob("*.apc")
    )
    if not has_apc:
        return dataset
    return str(materialize_fraction_apc_dataset(root, fraction, seed))


def build_apc_config(
    use_tome: bool,
    seed: int,
    *,
    num_epoch: int = 3,
    batch_size: int = 16,
    patience: int = 5,
    max_seq_len: int = 80,
) -> Any:
    """Return APC config; set ``use_tome`` / model class before trainer."""
    config = APC.APCConfigManager.get_apc_config_english()
    if use_tome:
        from thesis_apc_baseline.experiments.register_model import (
            register_fast_lcf_bert_tome,
        )

        register_fast_lcf_bert_tome()
        config.model = APC.APCModelList.FAST_LCF_BERT_TOME
        config.use_tome = True
        config.tome_merge_steps = 2
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        config.pad_token_id = int(tok.pad_token_id)
    else:
        config.model = APC.APCModelList.FAST_LCF_BERT

    config.pretrained_bert = "bert-base-uncased"
    config.evaluate_begin = 0
    config.batch_size = batch_size
    config.max_seq_len = max_seq_len
    config.num_epoch = num_epoch
    config.log_step = -1
    config.patience = patience
    config.dropout = 0.5
    config.learning_rate = 2e-5
    config.use_bert_spc = True
    config.lsa = True
    config.cache_dataset = False
    config.seed = [seed]
    return config


def run_training(
    use_tome: bool,
    seed: Optional[int] = None,
    *,
    num_epoch: int = 3,
    batch_size: int = 16,
    patience: int = 5,
    dataset_dir: Optional[Union[str, Path]] = None,
    data_fraction: float = 0.2,
    path_suffix: Optional[str] = None,
    load_inference_model: bool = True,
) -> Dict[str, Any]:
    """Train one run; returns wall time, best logged Acc/F1 (see PyABSA instructor), checkpoint path."""
    if seed is None:
        seed = random.randint(0, 10000)

    config = build_apc_config(
        use_tome,
        seed,
        num_epoch=num_epoch,
        batch_size=batch_size,
        patience=patience,
    )
    dataset = resolve_apc_dataset(dataset_dir)
    dataset = maybe_subsample_dataset(dataset, data_fraction, seed)

    sub = path_suffix or (
        "baseline_fast_lcf_bert" if not use_tome else "fast_lcf_bert_tome"
    )
    save_dir = ROOT / "thesis_apc_baseline" / "checkpoints" / sub

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    trainer = APC.APCTrainer(
        config=config,
        dataset=dataset,
        checkpoint_save_mode=ModelSaveOption.SAVE_MODEL_STATE_DICT,
        path_to_save=str(save_dir),
        auto_device=DeviceTypeOption.AUTO,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    metrics = getattr(trainer.config, "max_test_metrics", {}) or {}
    acc = float(metrics.get("max_apc_test_acc", float("nan")))
    f1 = float(metrics.get("max_apc_test_f1", float("nan")))

    result = {
        "model": config.model.__name__,
        "use_tome": use_tome,
        "seed": seed,
        "data_fraction": data_fraction,
        "dataset": dataset,
        "wall_time_sec": round(t1 - t0, 3),
        "max_apc_test_acc": acc,
        "max_apc_test_f1": f1,
        "checkpoint": trainer.inference_model,
    }

    if load_inference_model:
        trainer.load_trained_model()

    return result

