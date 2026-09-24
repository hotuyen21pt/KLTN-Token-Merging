# -*- coding: utf-8 -*-
"""Inference utilities for generative T5 Aspect Term Extraction."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch

from src.model import T5AspectExtractor
from src.normalization import decode_and_normalize, decode_target_text


def generate_target_text(
    model: T5AspectExtractor,
    sentence: str,
    *,
    max_input_length: int = 128,
) -> str:
    """Run beam search generation and return raw decoded target text."""
    encoded = model.tokenizer(
        sentence,
        max_length=max_input_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    generated_ids = model.generate(
        encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
    )
    return model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)


def predict_aspects(
    model: T5AspectExtractor,
    sentence: str,
    *,
    max_input_length: int = 128,
    normalize: bool = True,
) -> List[str]:
    """Predict aspect terms for one sentence."""
    raw_text = generate_target_text(
        model,
        sentence,
        max_input_length=max_input_length,
    )
    if normalize:
        return decode_and_normalize(raw_text, sentence)
    return decode_target_text(raw_text)


def predict_aspects_for_records(
    model: T5AspectExtractor,
    records: Sequence[Dict[str, object]],
    *,
    max_input_length: int = 128,
    normalize: bool = True,
    batch_size: int = 32,
) -> Tuple[List[List[str]], List[List[str]]]:
    """Batch inference over dataset records.

    Sinh chuỗi theo batch thay vì từng câu một. Vẫn dùng
    ``padding="max_length"`` như đường đi một-câu-một-lần nên attention mask
    giống hệt và kết quả tương đương về mặt toán học; chỉ nhanh hơn hàng chục
    lần vì beam search được chạy song song thay vì tuần tự.
    """
    sentences = [str(row["input_text"]) for row in records]
    golds = [list(row.get("aspects") or []) for row in records]
    predictions: List[List[str]] = []

    for start in range(0, len(sentences), batch_size):
        chunk = sentences[start : start + batch_size]
        encoded = model.tokenizer(
            chunk,
            max_length=max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        generated_ids = model.generate(
            encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )
        decoded = model.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        for sentence, text in zip(chunk, decoded):
            predictions.append(
                decode_and_normalize(text, sentence) if normalize
                else decode_target_text(text)
            )

    return predictions, golds


def predict_batch(
    model: T5AspectExtractor,
    sentences: Sequence[str],
    *,
    batch_size: int = 16,
    max_input_length: int = 128,
    normalize: bool = True,
) -> List[List[str]]:
    """Predict aspects for a list of sentences with simple batching."""
    all_preds: List[List[str]] = []
    model.model.eval()

    for start in range(0, len(sentences), batch_size):
        chunk = sentences[start : start + batch_size]
        encoded = model.tokenizer(
            list(chunk),
            max_length=max_input_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            generated_ids = model.generate(
                encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )
        decoded = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for sentence, text in zip(chunk, decoded):
            all_preds.append(
                decode_and_normalize(text, sentence) if normalize
                else decode_target_text(text)
            )

    return all_preds
