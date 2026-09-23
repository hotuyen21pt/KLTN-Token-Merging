# -*- coding: utf-8 -*-
"""CLI entry point for training the single-stage GAS T5 model.

The model learns to generate all three labels in one decoder pass:
    sentence -> "(food, FOOD, positive); (service, SERVICE, negative)"

Usage
-----
    python gas/train_gas.py \\
        --data-dir   dataset \\
        --output-dir checkpoints/gas_t5 \\
        --epochs     20

After training, run evaluation:
    python gas/evaluate_joint.py \\
        --gas-checkpoint checkpoints/gas_t5/best \\
        --data-dir       dataset
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from gas.dataset import DEFAULT_DATA_DIR, create_gas_dataloaders
from gas.model import GasT5Model
from gas.trainer import GasTrainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train single-stage GAS T5 model: generates "
            "(aspect_term, category, sentiment) triples in one decoder pass."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing train.apc / dev.apc / test.apc",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "checkpoints" / "gas_t5"),
        help="Checkpoint output directory",
    )
    parser.add_argument("--model-name",        type=str,   default="t5-base")
    parser.add_argument("--batch-size",         type=int,   default=16)
    parser.add_argument("--learning-rate",      type=float, default=3e-4)
    parser.add_argument("--epochs",             type=int,   default=20)
    parser.add_argument("--max-input-length",   type=int,   default=128)
    parser.add_argument("--max-target-length",  type=int,   default=128)
    parser.add_argument("--patience",           type=int,   default=5)
    parser.add_argument("--seed",               type=int,   default=42)
    parser.add_argument("--num-workers",        type=int,   default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print(f"Data dir      : {args.data_dir}")
    print(f"Output dir    : {args.output_dir}")
    print(f"Model         : {args.model_name}")
    print(f"Device        : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("Target format : (aspect_term, CATEGORY, sentiment)")

    (
        train_loader, dev_loader, test_loader,
        tokenizer, dev_records, test_records,
    ) = create_gas_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        num_workers=args.num_workers,
    )

    model = GasT5Model(model_name=args.model_name)
    model.tokenizer = tokenizer

    trainer = GasTrainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        patience=args.patience,
        output_dir=args.output_dir,
        dev_records=dev_records,
        test_records=test_records,
    )

    result = trainer.train()

    print("\n=== Training complete ===")
    print(f"Best dev joint-F1 : {result['best_dev_f1']:.4f}")
    print(f"Best checkpoint   : {result['best_checkpoint']}")
    print(f"Wall time (sec)   : {result['wall_time_sec']}")

    tm = result.get("test_metrics", {})
    if tm:
        for key, lbl in [
            ("aspect_term", "ATE   "),
            ("category",    "Cat   "),
            ("sentiment",   "Sent  "),
            ("joint",       "Joint "),
        ]:
            m = tm.get(key, {})
            if m:
                print(
                    f"Test {lbl} P/R/F1 : "
                    f"{m.get('precision', 0):.4f} / "
                    f"{m.get('recall',    0):.4f} / "
                    f"{m.get('f1',        0):.4f}"
                )

    best_ckpt = result.get("best_checkpoint") or f"{args.output_dir}/last"
    print(
        f"\nNext step:\n"
        f"  python gas/evaluate_joint.py \\\n"
        f"    --gas-checkpoint {best_ckpt} \\\n"
        f"    --data-dir       {args.data_dir}"
    )


if __name__ == "__main__":
    main()
