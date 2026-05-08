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
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thesis_apc_baseline.experiments.apc_inference import (
    infer_train_style_item,
    load_apc_sentiment_classifier,
    normalize_train_style_item,
    parse_item_json,
    train_style_to_pyabsa_text,
    to_json_serializable,
)


def _attach_tome_trace_hook(classifier: Any) -> Dict[str, Any]:
    """Hook ToMe forward_with_trace to capture real merge pairs during inference."""
    cache: Dict[str, Any] = {"trace": None, "hooked": False}
    try:
        model_ens = getattr(classifier, "model", None)
        models = getattr(model_ens, "models", None)
        base = models[0] if models and len(models) > 0 else None
        tome = getattr(base, "tome", None)
        if tome is None or not hasattr(tome, "forward_with_trace"):
            return cache

        original = tome.forward_with_trace

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            trace, merged_h, merged_lcf = original(*args, **kwargs)
            cache["trace"] = trace
            return trace, merged_h, merged_lcf

        tome.forward_with_trace = _wrapped  # type: ignore[assignment]
        cache["hooked"] = True
    except Exception:
        return cache
    return cache


def _print_real_tome_trace(trace: Any) -> None:
    if not isinstance(trace, list) or not trace:
        print("[12] real_tome_trace: unavailable")
        return
    print("[12] real_tome_trace_by_step:")
    for seq in trace:
        bidx = seq.get("batch_index", 0)
        print(f"  - batch_index={bidx}, length_in={seq.get('length_in')}")
        for s in seq.get("steps", []):
            step = s.get("step")
            if s.get("skipped", False):
                print(f"    step {step}: skipped ({s.get('reason')})")
                continue
            lb = s.get("length_before")
            la = s.get("length_after")
            pairs = s.get("pairs", [])
            print(f"    step {step}: len {lb} -> {la}, merged_pairs={len(pairs)}")
            print(f"      pairs(src,dst): {pairs}")


def _estimate_tome_lengths(valid_tokens: int, merge_steps: int, protect_tokens: int = 2) -> List[Dict[str, int]]:
    """Estimate token count shrink per ToMe step (before resize back to fixed length)."""
    rows: List[Dict[str, int]] = []
    cur = max(0, int(valid_tokens))
    for step in range(int(merge_steps)):
        interior = max(0, cur - protect_tokens)
        merged_pairs = interior // 2
        nxt = cur - merged_pairs
        rows.append(
            {
                "step": step,
                "length_before": cur,
                "merged_pairs": merged_pairs,
                "length_after": nxt,
            }
        )
        if merged_pairs == 0:
            break
        cur = nxt
    return rows


def _debug_print_pipeline(
    item: Dict[str, str],
    classifier: Any,
    raw_payload: Dict[str, Any],
) -> None:
    sentence_with_t, aspect = normalize_train_style_item(item)
    pyabsa_text = train_style_to_pyabsa_text(sentence_with_t, aspect)
    sentence_reconstructed = sentence_with_t.replace("$T$", aspect)

    print("\n=== INPUT TRACE ===")
    print("[0] raw_payload_from_cli/json:")
    print(json.dumps(raw_payload, indent=2, ensure_ascii=False))
    print(f"[1] normalized_sentence_with_$T$: {sentence_with_t}")
    print(f"[2] normalized_aspect: {aspect}")
    print(f"[3] reconstructed_raw_sentence: {sentence_reconstructed}")
    print(f"[4] pyabsa_text_for_predict: {pyabsa_text}")

    tokenizer = getattr(classifier, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "__call__"):
        max_len = int(getattr(classifier.config, "max_seq_len", 0) or 0)
        enc = tokenizer(
            pyabsa_text,
            truncation=True if max_len > 0 else False,
            max_length=max_len if max_len > 0 else None,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids = enc["input_ids"][0].tolist()
        attn = enc.get("attention_mask")
        valid = int(attn[0].sum().item()) if attn is not None else len(ids)
        print("\n=== TOKEN TRACE ===")
        print(f"[5] token_count_input_ids: {len(ids)}")
        print(f"[6] token_count_valid_by_attention_mask: {valid}")
        print(f"[7] first_32_token_ids: {ids[:32]}")
    else:
        valid = -1
        print("\n=== TOKEN TRACE ===")
        print("[5] tokenizer unavailable for token counting.")

    use_tome = bool(getattr(classifier.config, "use_tome", False))
    merge_steps = int(getattr(classifier.config, "tome_merge_steps", 0) or 0)
    print("\n=== MERGE TRACE (ESTIMATION) ===")
    print(f"[8] use_tome: {use_tome}")
    print(f"[9] tome_merge_steps: {merge_steps}")
    if use_tome and valid >= 0:
        sim = _estimate_tome_lengths(valid, merge_steps, protect_tokens=2)
        print("[10] merge_simulation_before_resize:")
        print(json.dumps(sim, indent=2, ensure_ascii=False))
        print(
            f"[11] note: model resizes merged sequence back to fixed length={valid} "
            "for downstream layers."
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
    parser.add_argument(
        "--show-steps",
        action="store_true",
        help="Print intermediate transformations and token merge shrink estimation.",
    )
    args = parser.parse_args()

    if args.sentence is not None:
        if not args.aspect:
            parser.error("--aspect is required when using --sentence")
        raw_payload: Dict[str, Any] = {"sentence": args.sentence, "aspect": args.aspect}
        item = dict(raw_payload)
    else:
        item = parse_item_json(args.json_item)
        raw_payload = dict(item) if isinstance(item, dict) else {"json_item": args.json_item}

    sentence_with_t, aspect = normalize_train_style_item(item)
    item = {"sentence": sentence_with_t, "aspect": aspect}

    clf = load_apc_sentiment_classifier(args.checkpoint, verbose=False)
    trace_cache: Dict[str, Any] = {}
    if args.show_steps:
        _debug_print_pipeline(item, clf, raw_payload=raw_payload)
        trace_cache = _attach_tome_trace_hook(clf)
    out = infer_train_style_item(clf, item, print_result=False)
    if args.show_steps and trace_cache.get("hooked"):
        print("\n=== MERGE TRACE (REAL) ===")
        _print_real_tome_trace(trace_cache.get("trace"))
    print(json.dumps(to_json_serializable(out), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
