from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

DEFAULT_INPUT_DIR = Path(r"C:\Users\NGUYEN HO TUYEN\Desktop\work\nckh\prism\data\raw")
SPLITS = ("train", "dev", "test")
VALID_SENTIMENTS = {"positive", "negative", "neutral"}


def read_jsonl(path: Path):
    records = []
    errors = 0
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[warn] {path.name}:{line_no}: {exc}")
                errors += 1
                continue
            if isinstance(record, dict):
                records.append(record)
            else:
                print(f"[warn] {path.name}:{line_no}: record is not an object")
                errors += 1
    return records, errors


def replace_aspect(text: str, aspect: str):
    start = text.find(aspect)
    if start >= 0:
        return text[:start] + "$T$" + text[start + len(aspect):]
    match = re.search(re.escape(aspect), text, flags=re.IGNORECASE)
    if match:
        return text[:match.start()] + "$T$" + text[match.end():]
    return None


def convert(input_dir: Path, output_dir: Path, split: str):
    records, json_errors = read_jsonl(input_dir / f"{split}.jsonl")
    stats = Counter(records_read=len(records), json_errors=json_errors)
    samples = []
    seen = set()

    for record in records:
        text = str(record.get("text") or "").strip()
        if not text:
            stats["empty_text"] += 1
            continue
        for quad in record.get("quads") or []:
            stats["quads_total"] += 1
            aspect = str(quad.get("aspect_term") or "").strip()
            if not aspect:
                stats["implicit_aspect_skipped"] += 1
                continue
            sentence = replace_aspect(text, aspect)
            if sentence is None:
                stats["aspect_not_found_skipped"] += 1
                continue
            category = str(quad.get("aspect_category") or "UNKNOWN").strip().upper()
            sentiment = str(quad.get("sentiment") or "").strip().lower()
            sentiment = {"pos": "positive", "neg": "negative", "neu": "neutral"}.get(sentiment, sentiment)
            if sentiment not in VALID_SENTIMENTS:
                stats["unknown_sentiment_skipped"] += 1
                continue
            sample = (sentence, aspect, category, sentiment)
            if sample in seen:
                stats["duplicate_skipped"] += 1
                continue
            seen.add(sample)
            samples.append(sample)
            stats["samples_written"] += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{split}.apc"
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for sentence, aspect, category, sentiment in samples:
            handle.write(f"{sentence}\n{aspect}\n{category}\n{sentiment}\n")
    return output, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir / "converted_apc").resolve()
    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")
    for split in SPLITS:
        output, stats = convert(input_dir, output_dir, split)
        print(f"\n{split}: {output}")
        for key, value in sorted(stats.items()):
            print(f"  {key}: {value}")
    print(f"\nDATA_INPUT_DIR={output_dir}")


if __name__ == "__main__":
    main()
