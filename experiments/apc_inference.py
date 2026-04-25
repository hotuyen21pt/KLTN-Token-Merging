# -*- coding: utf-8 -*-
"""APC inference helpers: one sample in train-style (sentence + ``$T$`` + aspect) → PyABSA ``predict``."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple, Union

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyabsa import AspectPolarityClassification as APC

from thesis_apc_baseline.experiments.register_model import register_fast_lcf_bert_tome

TrainStyleItem = Union[
    Mapping[str, str],
    Sequence[str],
    Tuple[str, str],
]


def train_style_to_pyabsa_text(sentence_with_t: str, aspect: str) -> str:
    """Turn train-like APC lines (1–2) into the ``[ASP]`` string expected by ``SentimentClassifier.predict``.

    Train APC line 1 uses a single ``$T$`` where the aspect phrase sits; line 2 is the aspect string.
    """
    if "$T$" not in sentence_with_t:
        raise ValueError(
            "Train-style sentence must contain exactly one '$T$' placeholder for the aspect span."
        )
    parts = sentence_with_t.split("$T$")
    if len(parts) != 2:
        raise ValueError("Train-style sentence must split into exactly two parts around '$T$'.")
    left, right = parts[0], parts[1]
    asp = aspect.strip()
    if not asp:
        raise ValueError("aspect must be non-empty.")
    return "{} [ASP] {} [ASP] {}".format(left.rstrip(), asp, right.lstrip()).strip()


def normalize_train_style_item(item: TrainStyleItem) -> Tuple[str, str]:
    """Return ``(sentence_with_$T$, aspect)`` from a dict or a pair."""
    if isinstance(item, (list, tuple)) and len(item) == 2 and not isinstance(item, (str, bytes)):
        return str(item[0]), str(item[1])
    if isinstance(item, Mapping):
        s = item.get("sentence") or item.get("text")
        a = item.get("aspect")
        if s is None or a is None:
            raise KeyError("item must include keys 'sentence' or 'text', and 'aspect'.")
        return str(s), str(a)
    raise TypeError("item must be a mapping with sentence|text+aspect, or a (sentence, aspect) pair.")


def load_apc_sentiment_classifier(checkpoint_dir: Union[str, Path], **kwargs: Any) -> APC.SentimentClassifier:
    """Load ``SentimentClassifier`` from a PyABSA checkpoint folder.

    Registers ``FAST_LCF_BERT_TOME`` so ToMe checkpoints unpickle correctly.
    """
    register_fast_lcf_bert_tome()
    return APC.SentimentClassifier(str(checkpoint_dir), **kwargs)


def infer_train_style_item(
    classifier: APC.SentimentClassifier,
    item: TrainStyleItem,
    *,
    print_result: bool = False,
) -> Any:
    """Run ``predict`` on one train-style sample; returns PyABSA result dict (or list for multi-aspect)."""
    sentence, aspect = normalize_train_style_item(item)
    line = train_style_to_pyabsa_text(sentence, aspect)
    return classifier.predict(line, print_result=print_result, ignore_error=True)


def parse_item_json(s: str) -> TrainStyleItem:
    obj = json.loads(s)
    if not isinstance(obj, dict):
        raise ValueError("JSON must be one object, e.g. {\"sentence\": \"...\", \"aspect\": \"...\"}")
    return obj


def to_json_serializable(obj: Any) -> Any:
    """Turn PyABSA / NumPy values into JSON-friendly Python types."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_serializable(x) for x in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)
