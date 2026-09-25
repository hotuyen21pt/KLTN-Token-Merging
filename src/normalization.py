# -*- coding: utf-8 -*-
"""Post-processing normalization for GAS-style ATE outputs."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List, Sequence, Set

_ASPECT_PATTERN = re.compile(r"\(([^)]*)\)")
# Từ = chuỗi ký tự chữ/số, cho phép dấu nháy bên trong ("restaurant's").
_WORD_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


def levenshtein_distance(left: str, right: str) -> int:
    """Khoảng cách soạn thảo giữa hai chuỗi, không cần gói ngoài.

    Cùng định nghĩa với ``Levenshtein.distance`` của gói ``python-Levenshtein``
    mà repo dùng trước đây: chi phí 1 cho chèn / xoá / thay thế, KHÔNG tính
    hoán vị (Damerau). Bỏ gói ngoài vì Kaggle chạy offline không pip install
    được, và đây là chỗ duy nhất repo cần tới nó.

    Quy hoạch động hai hàng: ``previous`` là hàng i-1, ``current`` là hàng i,
    nên bộ nhớ là O(len(right)) thay vì O(len(left) * len(right)).
    """
    if left == right:
        return 0
    # Đối xứng, nên đổi chỗ để hàng DP luôn là chuỗi ngắn hơn.
    if len(right) > len(left):
        left, right = right, left
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, 1):
        current = [row]
        for column, right_char in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


def trim_punctuation(text: str) -> str:
    """Cắt dấu câu ở HAI ĐẦU, giữ nguyên dấu bên trong.

    Giữ nguyên bên trong là bắt buộc: "restaurant's space" hay "Wi-Fi" phải
    còn nguyên, chỉ bỏ phần dính vào rìa như "bathroom," hay "(bữa ăn".
    """
    s = unicodedata.normalize("NFC", text)
    s = " ".join(s.split())
    while s and unicodedata.category(s[0]).startswith("P"):
        s = s[1:]
    while s and unicodedata.category(s[-1]).startswith("P"):
        s = s[:-1]
    return s.strip()


def build_ngram_vocabulary(sentence: str) -> Set[str]:
    """Build candidate vocabulary V from all word n-grams in the sentence.

    Gồm ba nguồn, và hai nguồn sau là bắt buộc để gold có thể khớp lại được:

    1. n-gram tách bằng khoảng trắng (bản gốc).
    2. Bản đã cắt dấu câu ở hai đầu của từng n-gram. Nếu thiếu, câu chứa
       "a clean pool, nice" sẽ chỉ sinh ra "pool," chứ không có "pool", nên
       aspect gold "pool" bị Levenshtein ánh xạ sang "pool," và mất điểm —
       đo được là mất 14-16% F1, tức trần cứng mà model không thể vượt.
    3. n-gram theo biên từ, lấy nguyên văn theo offset. Giúp tách "staff"
       ra khỏi token "staff...anywhere" mà vẫn giữ đúng dạng bề mặt.
    """
    sentence = unicodedata.normalize("NFC", sentence)
    candidates: Set[str] = set()

    words = sentence.split()
    for start in range(len(words)):
        for end in range(start + 1, len(words) + 1):
            span = " ".join(words[start:end])
            if span:
                candidates.add(span)
                trimmed = trim_punctuation(span)
                if trimmed:
                    candidates.add(trimmed)

    spans = [(m.start(), m.end()) for m in _WORD_PATTERN.finditer(sentence)]
    for i in range(len(spans)):
        for j in range(i, len(spans)):
            seg = sentence[spans[i][0]:spans[j][1]]
            if seg:
                candidates.add(seg)

    return candidates


def normalize_aspect(term: str, vocabulary: Set[str]) -> str:
    """Map an aspect to the closest n-gram in V (identity if already present)."""
    cleaned = trim_punctuation(term)
    if not cleaned or not vocabulary:
        return cleaned
    if cleaned in vocabulary:
        return cleaned

    best = cleaned
    best_dist = None
    # Duyệt theo thứ tự đã sắp: kết quả không phụ thuộc thứ tự băm của set,
    # nên hai lần chạy cho ra cùng một đáp án.
    for candidate in sorted(vocabulary):
        # Levenshtein >= chênh lệch độ dài, nên ứng viên có chênh lệch >=
        # khoảng cách tốt nhất hiện tại không thể tốt hơn. Bỏ qua cho nhanh,
        # kết quả không đổi.
        if best_dist is not None and abs(len(candidate) - len(cleaned)) >= best_dist:
            continue
        dist = levenshtein_distance(cleaned, candidate)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = candidate
    return best


def normalize_aspects(aspects: Sequence[str], sentence: str) -> List[str]:
    """Normalize each predicted aspect against sentence n-grams."""
    sentence = unicodedata.normalize("NFC", sentence)
    vocab = build_ngram_vocabulary(sentence)
    return [normalize_aspect(a, vocab) for a in aspects if a and a.strip()]


def decode_target_text(text: str) -> List[str]:
    """Decode GAS extraction-style output into aspect strings.

    Examples:
        "(pizza); (service)" -> ["pizza", "service"]
        "(pizza)"            -> ["pizza"]
        "none"               -> []
        "pizza" (no parens)  -> []
    """
    stripped = text.strip()
    if not stripped or stripped.lower() == "none":
        return []

    matches = _ASPECT_PATTERN.findall(stripped)
    if not matches:
        return []

    aspects: List[str] = []
    for match in matches:
        aspect = match.strip()
        if aspect:
            aspects.append(aspect)
    return aspects


def decode_and_normalize(text: str, sentence: str) -> List[str]:
    """Decode model output and apply Levenshtein n-gram normalization."""
    raw = decode_target_text(text)
    return normalize_aspects(raw, sentence)


def format_aspects_for_display(aspects: Iterable[str]) -> str:
    cleaned = [a.strip() for a in aspects if a and a.strip()]
    if not cleaned:
        return "none"
    return "; ".join(f"({a})" for a in cleaned)
