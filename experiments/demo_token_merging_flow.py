# -*- coding: utf-8 -*-
"""Print token-merging trace on random embeddings (shape matches a short APC sequence).

Shows length before/after each bipartite step and final resize — useful for thesis diagrams.

Run from repo root:
  python thesis_apc_baseline/experiments/demo_token_merging_flow.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from thesis_apc_baseline.token_merging.tome_1d import ToMeSequenceMerger


def main():
    torch.manual_seed(0)
    B, L, D = 2, 32, 768
    hidden = torch.randn(B, L, D)
    lcf = torch.zeros(B, L)
    lcf[:, 8:14] = 1.0
    mask = torch.ones(B, L)
    mask[:, 24:] = 0

    merger = ToMeSequenceMerger(num_merge_steps=3, protect_cls=True, protect_sep=True)
    trace, h_out, l_out = merger.forward_with_trace(hidden, lcf, mask)

    def serializable(tr):
        out = []
        for item in tr:
            row = dict(item)
            row["steps"] = item["steps"]
            out.append(row)
        return out

    print("=== Token merging trace (batch_size=%d, seq_len=%d, dim=%d) ===" % (B, L, D))
    print(json.dumps(serializable(trace), indent=2))
    print("output hidden:", tuple(h_out.shape), "lcf:", tuple(l_out.shape))


if __name__ == "__main__":
    main()
