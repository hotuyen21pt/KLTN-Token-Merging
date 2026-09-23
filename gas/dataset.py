# -*- coding: utf-8 -*-
"""Dataset utilities for single-stage GAS (Generative Aspect Sentiment) training.

The model learns to generate ALL THREE labels in one pass:

    Input  : raw sentence
    Output : "(food, FOOD, positive); (service, SERVICE, negative)"

Data source: 4-line .apc files only.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer, T5Tokenizer

# ─── Constants ────────────────────────────────────────────────────────────────

DEFAULT_DATA_DIR      = Path(__file__).resolve().parents[1] / "dataset"
SPLIT_FILES           = {"train": "train.apc", "dev": "dev.apc", "test": "test.apc"}
VALID_SENTIMENTS: Set[str] = {"positive", "negative", "neutral"}

# Matches "(aspect_term, CATEGORY, sentiment)" — sentiment must be a fixed token.
_GAS_RE = re.compile(
    r"\(\s*(.+?),\s*(.+?),\s*(positive|negative|neutral)\s*\)",
    re.IGNORECASE,
)


# ─── Target string helpers ────────────────────────────────────────────────────

def triples_to_target(triples: List[Tuple[str, str, str]]) -> str:
    """Convert [(aspect, category, sentiment), ...] → GAS target string.

    Example:
        [("food", "FOOD", "positive"), ("service", "SERVICE", "negative")]
        -> "(food, FOOD, positive); (service, SERVICE, negative)"
    """
    cleaned = [
        (a.strip(), c.strip(), s.strip().lower())
        for a, c, s in triples
        if a and a.strip() and c and c.strip()
        and s and s.strip().lower() in VALID_SENTIMENTS
    ]
    if not cleaned:
        return "none"
    return "; ".join(f"({a}, {c}, {s})" for a, c, s in cleaned)


def parse_gas_target(text: str) -> List[Tuple[str, str, str]]:
    """Parse a GAS target string → [(aspect_term, category, sentiment), ...].

    Returns [] for "none" or unrecognised format.
    """
    stripped = text.strip()
    if not stripped or stripped.lower() == "none":
        return []
    return [
        (m.group(1).strip(), m.group(2).strip(), m.group(3).strip().lower())
        for m in _GAS_RE.finditer(stripped)
    ]


# ─── .apc file parser ─────────────────────────────────────────────────────────

def parse_apc_file_for_gas(path: str | Path) -> List[Dict[str, str]]:
    """Parse a 4-line .apc file into raw sample dicts."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    samples: List[Dict[str, str]] = []
    i = 0
    while i + 3 < len(lines):
        sentence    = lines[i].strip()
        aspect_term = lines[i + 1].strip()
        aspect_cat  = lines[i + 2].strip()
        sentiment   = lines[i + 3].strip().lower()
        i += 4

        if not sentence or not aspect_term or sentiment not in VALID_SENTIMENTS:
            continue

        text = sentence.replace("$T$", aspect_term).strip()
        samples.append(
            {
                "text":            text,
                "aspect_term":     aspect_term,
                "aspect_category": aspect_cat or "UNKNOWN",
                "sentiment":       sentiment,
            }
        )
    return samples


# ─── Record builder ───────────────────────────────────────────────────────────

def load_gas_records(path: str | Path) -> List[Dict]:
    """Build GAS records from a .apc file.

    Samples are grouped by sentence (same text after $T$ substitution)
    so each unique sentence produces one record covering all its aspects.

    Record schema
    -------------
    input_text   : str
    target_text  : str  — "(food, FOOD, positive); (service, SERVICE, negative)"
    aspects      : List[str]               — aspect terms (for ATE metric)
    gold_triples : List[Tuple[str,str,str]] — (aspect, category, sentiment)
    """
    raw = parse_apc_file_for_gas(path)

    grouped: Dict[str, List] = {}
    for s in raw:
        grouped.setdefault(s["text"], []).append(s)

    records: List[Dict] = []
    for text in sorted(grouped):
        samples = grouped[text]

        seen: Set[Tuple[str, str, str]] = set()
        unique_triples: List[Tuple[str, str, str]] = []
        for s in samples:
            t = (s["aspect_term"], s["aspect_category"], s["sentiment"])
            if t not in seen:
                seen.add(t)
                unique_triples.append(t)

        records.append(
            {
                "input_text":  text,
                "target_text": triples_to_target(unique_triples),
                "aspects":     [t[0] for t in unique_triples],
                "gold_triples": unique_triples,
            }
        )

    print(f"[GAS] {Path(path).name}: {len(records)} sentences")
    return records


def load_split_records(
    split: str,
    data_dir: Optional[str | Path] = None,
) -> List[Dict]:
    """Load GAS records for train / dev / test split."""
    if split not in SPLIT_FILES:
        raise ValueError(f"Unknown split {split!r}; expected one of {list(SPLIT_FILES)}")
    root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    path = root / SPLIT_FILES[split]
    if not path.is_file():
        raise FileNotFoundError(f"Missing split file: {path}")
    return load_gas_records(path)


# ─── PyTorch Dataset ──────────────────────────────────────────────────────────

class GASDataset(Dataset):
    """Tokenised (sentence → 3-label sequence) pairs for T5 seq2seq training."""

    def __init__(
        self,
        records: Sequence[Dict],
        tokenizer: PreTrainedTokenizer,
        max_input_length:  int = 128,
        max_target_length: int = 128,
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
        label_ids = labels["input_ids"].squeeze(0)
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      model_inputs["input_ids"].squeeze(0),
            "attention_mask": model_inputs["attention_mask"].squeeze(0),
            "labels":         label_ids,
        }


# ─── Tokenizer & DataLoader builders ─────────────────────────────────────────

def build_tokenizer(model_name: str = "t5-base") -> T5Tokenizer:
    return T5Tokenizer.from_pretrained(model_name)


def create_gas_dataloaders(
    data_dir:          Optional[str | Path] = None,
    tokenizer:         Optional[PreTrainedTokenizer] = None,
    batch_size:        int = 16,
    max_input_length:  int = 128,
    max_target_length: int = 128,
    num_workers:       int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, PreTrainedTokenizer, List[Dict], List[Dict]]:
    """Build train / dev / test DataLoaders for single-stage GAS.

    Returns
    -------
    train_loader, dev_loader, test_loader, tokenizer, dev_records, test_records
    """
    tok = tokenizer or build_tokenizer()

    train_records = load_split_records("train", data_dir)
    dev_records   = load_split_records("dev",   data_dir)
    test_records  = load_split_records("test",  data_dir)

    def _loader(records: Sequence[Dict], shuffle: bool) -> DataLoader:
        ds = GASDataset(records, tok, max_input_length, max_target_length)
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
        dev_records,
        test_records,
    )
