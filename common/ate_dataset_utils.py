# -*- coding: utf-8 -*-
"""Dataset utilities for Aspect Term Extraction (ATE) using a generative T5 model.

Input format (.apc, 4 lines per sample):
    Line 1: sentence with $T$ placeholder
    Line 2: aspect_term  (replaces $T$)
    Line 3: aspect_category  (not used for ATE)
    Line 4: sentiment        (not used for ATE)

One sentence may appear multiple times (with different $T$ terms).
All aspect terms for the same sentence are merged into one target string:
    "(aspect1); (aspect2); ..."
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer, T5Tokenizer

# ─── Constants ────────────────────────────────────────────────────────────────

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "dataset"
SPLIT_FILES = {
    "train": "train.apc",
    "dev":   "dev.apc",
    "test":  "test.apc",
}


# ─── .apc parser (ATE) ────────────────────────────────────────────────────────

def parse_apc_file_for_ate(path: str | Path) -> List[Dict[str, str]]:
    """Parse a 4-line .apc file and return one dict per raw sample.

    The aspect_char_start / aspect_char_end are derived from the position of
    ``$T$`` in the original sentence, NOT from text.find() after replacement.
    This avoids wrong spans when the same aspect term appears multiple times
    in the sentence.

    Returns:
        List of dicts with keys:
            text            - sentence with $T$ replaced by aspect_term
            aspect_term     - the aspect term (may be multi-word)
            aspect_char_start / aspect_char_end - char span in ``text``
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    samples: List[Dict[str, str]] = []
    i = 0
    while i + 3 < len(lines):
        sentence    = lines[i].strip()
        aspect_term = lines[i + 1].strip()
        # lines[i+2] = category, lines[i+3] = sentiment — not needed for ATE
        i += 4

        if not sentence:
            continue

        aspect_stripped = aspect_term.strip()

        # Derive span from $T$ position before replacement.
        t_pos = sentence.find("$T$")
        text  = sentence.replace("$T$", aspect_term).strip()

        if aspect_stripped and t_pos >= 0:
            aspect_char_start = t_pos
            aspect_char_end   = t_pos + len(aspect_stripped) - 1
        else:
            aspect_char_start = -1
            aspect_char_end   = -1

        samples.append(
            {
                "text":             text,
                "aspect_term":      aspect_term,
                "aspect_char_start": aspect_char_start,
                "aspect_char_end":   aspect_char_end,
            }
        )
    return samples


# ─── Grouping & target formatting ─────────────────────────────────────────────

def aspects_to_target(aspects: Sequence[str]) -> str:
    """Convert a list of aspect terms to a GAS extraction target string.

    Example: ["pizza", "service quality"] -> "(pizza); (service quality)"
    Empty list -> "none"
    """
    cleaned = [a.strip() for a in aspects if a and a.strip()]
    if not cleaned:
        return "none"
    return "; ".join(f"({a})" for a in cleaned)


def load_ate_records(path: str | Path) -> List[Dict[str, object]]:
    """Load ATE records from one .apc file.

    Groups raw samples by sentence text so that one sentence with multiple
    aspect terms produces a single record with all terms in the target string.

    Returns:
        List of dicts with keys:
            input_text  - the sentence (with aspects inlined)
            target_text - GAS-format aspect sequence, e.g. "(room); (staff)"
            aspects     - sorted list of raw aspect strings (gold labels)
    """
    grouped: Dict[str, set] = {}
    for sample in parse_apc_file_for_ate(path):
        text   = sample["text"]
        aspect = sample["aspect_term"]
        if text not in grouped:
            grouped[text] = set()
        if aspect:
            grouped[text].add(aspect)

    records: List[Dict[str, object]] = []
    for text in sorted(grouped):
        aspects = sorted(grouped[text])
        records.append(
            {
                "input_text":  text,
                "target_text": aspects_to_target(aspects),
                "aspects":     aspects,
            }
        )
    return records


def load_split_records(
    split: str,
    data_dir: Optional[str | Path] = None,
) -> List[Dict[str, object]]:
    """Load train / dev / test ATE records from the standard .apc files."""
    if split not in SPLIT_FILES:
        raise ValueError(f"Unknown split {split!r}; expected one of {list(SPLIT_FILES)}")
    root     = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    apc_path = root / SPLIT_FILES[split]
    if not apc_path.is_file():
        raise FileNotFoundError(f"Missing split file: {apc_path}")
    records = load_ate_records(apc_path)
    print(f"[ATE] {split}: {len(records)} sentences  ({apc_path.name})")
    return records


# ─── PyTorch Dataset ──────────────────────────────────────────────────────────

class ATEDataset(Dataset):
    """Tokenised sentence → aspect-sequence pairs for T5 seq2seq training."""

    def __init__(
        self,
        records: Sequence[Dict[str, object]],
        tokenizer: PreTrainedTokenizer,
        max_input_length: int = 128,
        max_target_length: int = 64,
    ) -> None:
        self.tokenizer         = tokenizer
        self.max_input_length  = max_input_length
        self.max_target_length = max_target_length
        self.records           = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row         = self.records[idx]
        input_text  = str(row["input_text"])
        target_text = str(row["target_text"])

        model_inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids      = model_inputs["input_ids"].squeeze(0)
        attention_mask = model_inputs["attention_mask"].squeeze(0)
        label_ids      = labels["input_ids"].squeeze(0)
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         label_ids,
        }


# ─── Tokenizer helper ─────────────────────────────────────────────────────────

def build_tokenizer(model_name: str = "t5-base") -> T5Tokenizer:
    return T5Tokenizer.from_pretrained(model_name)


# ─── DataLoader builder ───────────────────────────────────────────────────────

def create_ate_dataloaders(
    data_dir: Optional[str | Path] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    batch_size: int = 16,
    max_input_length: int = 128,
    max_target_length: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, PreTrainedTokenizer]:
    """Build train / dev / test DataLoaders for ATE.

    Returns:
        (train_loader, dev_loader, test_loader, tokenizer)
    """
    tok = tokenizer or build_tokenizer()

    train_records = load_split_records("train", data_dir)
    dev_records   = load_split_records("dev",   data_dir)
    test_records  = load_split_records("test",  data_dir)

    def _loader(records: Sequence[Dict[str, object]], shuffle: bool) -> DataLoader:
        ds = ATEDataset(
            records,
            tok,
            max_input_length=max_input_length,
            max_target_length=max_target_length,
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return (
        _loader(train_records, shuffle=True),
        _loader(dev_records,   shuffle=False),
        _loader(test_records,  shuffle=False),
        tok,
    )
