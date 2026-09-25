# -*- coding: utf-8 -*-
"""ATE dataset loader for generative T5 training (GAS extraction-style).

Delegates all parsing and dataset logic to common/ate_dataset_utils.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.ate_dataset_utils import (  # noqa: E402
    ATEDataset,
    DEFAULT_DATA_DIR,
    aspects_to_target,
    build_tokenizer,
    create_ate_dataloaders,
    load_ate_records,
    load_split_records,
)

__all__ = [
    "ATEDataset",
    "DEFAULT_DATA_DIR",
    "aspects_to_target",
    "build_tokenizer",
    "create_ate_dataloaders",
    "load_ate_records",
    "load_split_records",
    "get_raw_split_records",
    # backward-compat alias
    "create_dataloaders",
]


def create_dataloaders(
    data_dir: Optional[str | Path] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    model_name: Optional[str] = None,
    batch_size: int = 16,
    max_input_length: int = 128,
    max_target_length: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, PreTrainedTokenizer]:
    """Backward-compatible alias for create_ate_dataloaders."""
    return create_ate_dataloaders(
        data_dir=data_dir,
        tokenizer=tokenizer,
        model_name=model_name,
        batch_size=batch_size,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
        num_workers=num_workers,
    )


def get_raw_split_records(
    split: str,
    data_dir: Optional[str | Path] = None,
) -> List[Dict[str, object]]:
    """Return raw records (input_text / target_text / aspects) for a split."""
    return load_split_records(split, data_dir)
