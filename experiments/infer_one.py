# -*- coding: utf-8 -*-
"""
Load a trained APC checkpoint and predict polarity for **one** train-style sample:

  - ``sentence``: raw sentence with exactly one ``$T$`` (same convention as ``*.apc`` line 1)
  - ``aspect``: aspect span text (same as ``*.apc`` line 2)

JSON output includes ``thesis_visualization`` with ``token_texts`` (valid subwords).
If the checkpoint uses ToMe (``use_tome``), the same block also includes
``tome_token_evolution``: per-step ``token_texts_before`` / ``token_texts_after`` and ``pairs_text``.

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
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thesis_apc_baseline.experiments.apc_inference import (
    infer_train_style_item,
    load_apc_sentiment_classifier,
    normalize_train_style_item,
    parse_item_json,
    thesis_visualization_for_train_style_item,
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


def _merge_token_display(a: str, b: str) -> str:
    """Readable label after ToMe averaging (embedding space), for thesis diagrams."""
    return f"({a}∥{b})"


def _enrich_tome_trace_with_token_texts(
    trace: Any,
    token_texts: List[str],
) -> Optional[List[Dict[str, Any]]]:
    """Map merge indices → subword strings; per step: before / after sequences and pair captions."""
    if not isinstance(trace, list) or not trace or not token_texts:
        return None

    out_batches: List[Dict[str, Any]] = []

    for seq in trace:
        bidx = int(seq.get("batch_index", 0))
        steps_in = seq.get("steps") or []
        labels = list(token_texts)
        enriched_steps: List[Dict[str, Any]] = []

        for s in steps_in:
            if s.get("skipped"):
                enriched_steps.append(
                    {
                        "step": s.get("step"),
                        "skipped": True,
                        "reason": s.get("reason"),
                        "token_texts_state": list(labels),
                    }
                )
                continue

            want_before = int(s["length_before"])
            if len(labels) != want_before:
                if len(labels) > want_before:
                    labels = labels[:want_before]
                else:
                    labels = labels + ["<?>"] * (want_before - len(labels))

            token_texts_before = list(labels)
            pairs = s.get("pairs") or []
            pairs_text: List[Dict[str, Any]] = []
            for p in pairs:
                if len(p) < 2:
                    continue
                si, di = int(p[0]), int(p[1])
                pairs_text.append(
                    {
                        "src_idx": si,
                        "dst_idx": di,
                        "src": token_texts_before[si] if 0 <= si < len(token_texts_before) else "?",
                        "dst": token_texts_before[di] if 0 <= di < len(token_texts_before) else "?",
                        "merged_display": _merge_token_display(
                            token_texts_before[si] if 0 <= si < len(token_texts_before) else "?",
                            token_texts_before[di] if 0 <= di < len(token_texts_before) else "?",
                        ),
                    }
                )

            keep = s.get("keep_mask") or []
            labels_merged = list(token_texts_before)
            for p in pairs:
                if len(p) < 2:
                    continue
                si, di = int(p[0]), int(p[1])
                if 0 <= si < len(labels_merged) and 0 <= di < len(labels_merged):
                    labels_merged[si] = _merge_token_display(labels_merged[si], labels_merged[di])

            token_texts_after = [
                labels_merged[i]
                for i in range(len(labels_merged))
                if i < len(keep) and keep[i]
            ]
            labels = token_texts_after

            enriched_steps.append(
                {
                    "step": s.get("step"),
                    "skipped": False,
                    "length_before": s.get("length_before"),
                    "length_after": s.get("length_after"),
                    "token_texts_before": token_texts_before,
                    "token_texts_after": token_texts_after,
                    "pairs": pairs,
                    "pairs_text": pairs_text,
                }
            )

        out_batches.append(
            {
                "batch_index": bidx,
                "length_in": seq.get("length_in"),
                "length_after_resize": seq.get("length_after_resize"),
                "token_texts_initial": list(token_texts),
                "steps": enriched_steps,
                "token_texts_after_merges_pre_resize": list(labels),
            }
        )

    return out_batches


def _print_real_tome_trace(trace: Any) -> None:
    if not isinstance(trace, list) or not trace:
        print("[12] real_tome_trace: unavailable")
        return

    print("[12] real_tome_trace_by_step:")

    for seq in trace:
        bidx = seq.get("batch_index", 0)

        print(f"\n=== batch {bidx} ===")
        print(f"length_in = {seq.get('length_in')}")

        for s in seq.get("steps", []):
            step = s.get("step")

            if s.get("skipped", False):
                print(f"\nstep {step}: skipped")
                continue

            print(f"\nstep {step}")
            print(
                f"len {s['length_before']} -> {s['length_after']}"
            )

            print("pairs:")
            for p in s["pairs"]:
                print(f"  src={p[0]} dst={p[1]}")

            print("keep_mask:")
            print(s["keep_mask"])

            print("x_before_preview:")
            for row in s["x_before_preview"][:5]:
                print(row)

            print("x_after_preview:")
            for row in s["x_after_preview"][:5]:
                print(row)

            print("lcf_before:")
            print(s["lcf_before"])

            print("lcf_after:")
            print(s["lcf_after"])


def _print_tome_token_evolution(evolution: List[Dict[str, Any]]) -> None:
    """Human-readable token-string state per ToMe step (for --show-steps)."""
    print("\n[13] tome_token_evolution (subword labels):")
    for batch in evolution:
        print(f"\n--- batch {batch.get('batch_index', 0)} ---")
        print(f"token_texts_initial: {batch.get('token_texts_initial')}")
        for st in batch.get("steps", []):
            if st.get("skipped"):
                print(f"  step {st.get('step')}: skipped ({st.get('reason')})")
                continue
            print(f"  step {st.get('step')}: len {st.get('length_before')} -> {st.get('length_after')}")
            print(f"    before: {st.get('token_texts_before')}")
            print(f"    after:  {st.get('token_texts_after')}")
            for pt in st.get("pairs_text") or []:
                print(
                    f"    pair ({pt.get('src_idx')},{pt.get('dst_idx')}): "
                    f"{pt.get('src')} + {pt.get('dst')} -> {pt.get('merged_display')}"
                )
        print(f"  after all merges (pre resize): {batch.get('token_texts_after_merges_pre_resize')}")


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
    use_tome = bool(getattr(clf.config, "use_tome", False))
    if args.show_steps:
        _debug_print_pipeline(item, clf, raw_payload=raw_payload)
    if args.show_steps or use_tome:
        trace_cache = _attach_tome_trace_hook(clf)

    out = infer_train_style_item(clf, item, print_result=False)

    viz = thesis_visualization_for_train_style_item(clf, item)
    raw_trace = trace_cache.get("trace")
    enriched = _enrich_tome_trace_with_token_texts(raw_trace, viz.get("token_texts") or [])
    if enriched is not None:
        viz = {**viz, "tome_token_evolution": enriched}

    if args.show_steps and trace_cache.get("hooked"):
        print("\n=== MERGE TRACE (REAL) ===")
        _print_real_tome_trace(raw_trace)
        if enriched:
            _print_tome_token_evolution(enriched)
    ser = to_json_serializable(out)
    if isinstance(ser, dict):
        ser_with_viz = {**ser, "thesis_visualization": viz}
    else:
        ser_with_viz = {"result": ser, "thesis_visualization": viz}
    print(json.dumps(to_json_serializable(ser_with_viz), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
