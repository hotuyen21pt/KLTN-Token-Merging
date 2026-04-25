# -*- coding: utf-8 -*-
"""
Load a trained APC checkpoint and predict polarity for **one** train-style sample:

  - ``sentence``: raw sentence with exactly one ``$T$`` (same convention as ``*.apc`` line 1)
  - ``aspect``: aspect span text (same as ``*.apc`` line 2)

Run from repo root (parent of ``thesis_apc_baseline``), e.g. ``kltn``:

  python thesis_apc_baseline/experiments/infer_one.py ^
    --checkpoint thesis_apc_baseline/checkpoints/baseline_fast_lcf_bert/<run_folder> ^
    --sentence "The $T$ was cold ." ^
    --aspect "pizza"

Or pass one JSON object:

  python thesis_apc_baseline/experiments/infer_one.py ^
    --checkpoint <path> ^
    --json-item "{\"sentence\": \"The $T$ was cold .\", \"aspect\": \"pizza\"}"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thesis_apc_baseline.experiments.apc_inference import (
    infer_train_style_item,
    load_apc_sentiment_classifier,
    normalize_train_style_item,
    parse_item_json,
    to_json_serializable,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="APC inference: one train-style sentence + aspect.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Folder that contains PyABSA .state_dict / .config / .tokenizer (inner run directory).",
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--json-item",
        type=str,
        default=None,
        help='One JSON object, e.g. {"sentence": "The $T$ was cold.", "aspect": "pizza"}',
    )
    g.add_argument("--sentence", type=str, default=None, help="Sentence with a single $T$ placeholder.")
    parser.add_argument(
        "--aspect",
        type=str,
        default=None,
        help="Aspect text (required with --sentence).",
    )
    args = parser.parse_args()

    if args.sentence is not None:
        if not args.aspect:
            parser.error("--aspect is required when using --sentence")
        item = {"sentence": args.sentence, "aspect": args.aspect}
    else:
        item = parse_item_json(args.json_item)

    normalize_train_style_item(item)

    clf = load_apc_sentiment_classifier(args.checkpoint, verbose=False)
    out = infer_train_style_item(clf, item, print_result=False)
    print(json.dumps(to_json_serializable(out), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
