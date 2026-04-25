
# -*- coding: utf-8 -*-
import torch
from transformers import BertTokenizer, BertModel
import pprint
import argparse
import json

from thesis_apc_baseline.token_merging.tome_1d import ToMeSequenceMerger

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 32

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)

tome = ToMeSequenceMerger(
    num_merge_steps=2,
    protect_cls=True,
    protect_sep=True
)

# =========================
# PIPELINE
# =========================
def run_pipeline(input_json):
    sentence = input_json["sentence"]
    aspect = input_json["aspect"]

    print("\n================ INPUT ================")
    print("Sentence:", sentence)
    print("Aspect  :", aspect)

    # =========================
    # STEP 1: TOKENIZE
    # =========================
    encoded = tokenizer(
        sentence,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN
    )

    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    print("\n=== TOKENS ===")
    for i, tok in enumerate(tokens):
        print(f"{i:2d}: {tok}")

    # =========================
    # STEP 2: BERT
    # =========================
    with torch.no_grad():
        hidden = bert(input_ids).last_hidden_state

    print("\n=== BERT OUTPUT ===")
    print("shape:", hidden.shape)

    # =========================
    # STEP 3: BUILD LCF
    # =========================
    lcf_vec = torch.zeros_like(input_ids).float()
    aspect_tokens = tokenizer.tokenize(aspect)

    for i, tok in enumerate(tokens):
        if tok in aspect_tokens:
            lcf_vec[0, i] = 1

    print("\n=== LCF VECTOR ===")
    print(lcf_vec[0])

    # =========================
    # STEP 4: APPLY TOME
    # =========================
    trace, merged_h, merged_lcf = tome.forward_with_trace(
        hidden,
        lcf_vec,
        attention_mask
    )

    print("\n=== AFTER TOME ===")
    print("hidden shape:", merged_h.shape)
    print("lcf shape   :", merged_lcf.shape)

    # =========================
    # STEP 5: TRACE
    # =========================
    print("\n================ TRACE ================")
    pprint.pprint(trace[0])

    # =========================
    # STEP 6: MERGE VIEW
    # =========================
    print("\n================ MERGE PAIRS ================")

    for step in trace[0]["steps"]:
        if step.get("skipped"):
            continue

        print(f"\nStep {step['step']}:")
        for src, dst in step["pairs"]:
            print(f"  {tokens[src]} <-- merged with --> {tokens[dst]}")

    return merged_h, merged_lcf


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        type=str,
        required=True,
        help='Input JSON: {"sentence": "...", "aspect": "..."}'
    )

    args = parser.parse_args()

    # Parse JSON string
    input_json = json.loads(args.json)

    run_pipeline(input_json)

