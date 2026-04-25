# -*- coding: utf-8 -*-
"""
Same setup as train_baseline.py but FAST_LCF_BERT_TOME + token merging enabled.

Run from repo root:
  python thesis_apc_baseline/experiments/train_apc_tome.py
"""

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thesis_apc_baseline.experiments.experiment_common import run_training


def main():
    seed = random.randint(0, 10000)
    r = run_training(True, seed=seed)
    print(
        "\nToMe | seed=%s | Acc=%.4f | F1=%.4f | wall_time_sec=%s"
        % (seed, r["max_apc_test_acc"], r["max_apc_test_f1"], r["wall_time_sec"])
    )


if __name__ == "__main__":
    main()
