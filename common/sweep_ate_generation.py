# -*- coding: utf-8 -*-
"""Quét tham số beam search cho ATE trên checkpoint ĐÃ train — không train lại.

Vì sao cần: chẩn đoán trên test cho thấy model sinh thiếu aspect chứ không
sinh sai — trung bình 1.19 aspect/câu trong khi gold là 1.51, và recall tụt
theo số aspect trong câu (1 gold: 60.8%, 2 gold: 42.0%, 4 gold: 18.1%).
Đó là dấu hiệu kinh điển của beam search với ``length_penalty = 1.0``: điểm
của một chuỗi là ``logprob / len ** length_penalty`` nên chuỗi ngắn gần như
luôn thắng, model dừng sau aspect đầu tiên.

Script này nạp checkpoint một lần rồi chạy lại inference cho từng tổ hợp
(num_beams, length_penalty), in P/R/F1 kèm số aspect sinh ra mỗi câu và
recall tách theo số gold — để thấy rõ tổ hợp nào thật sự chữa được lỗi sinh
thiếu chứ không phải chỉ đổi điểm số.

Chọn trên DEV, không phải test.

Usage:
  python common/sweep_ate_generation.py --ckpt checkpoints/gas_t5_ate/seed_42/best
  python common/sweep_ate_generation.py --ckpt <dir> --split dev \
      --beams 1 4 8 --length-penalty 1.0 1.2 1.5 2.0
  python common/sweep_ate_generation.py --ckpt <dir> --limit 300   # quét nhanh
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.dataset import get_raw_split_records  # noqa: E402
from src.inference import predict_aspects_for_records  # noqa: E402
from src.metrics import evaluate_exact_match  # noqa: E402
from src.model import GenerationConfig, T5AspectExtractor  # noqa: E402


def recall_by_gold_count(preds, golds, canon) -> Dict[int, float]:
    """Recall tách theo số aspect gold trong câu — chỗ lỗi sinh thiếu lộ ra."""
    hit = defaultdict(int)
    tot = defaultdict(int)
    for p, g in zip(preds, golds):
        k = min(len(g), 4)  # gộp 4+ vì ít câu
        ps = {canon(x) for x in p}
        gs = {canon(x) for x in g}
        hit[k] += len(ps & gs)
        tot[k] += len(gs)
    return {k: hit[k] / tot[k] * 100 for k in sorted(tot) if tot[k]}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quét num_beams / length_penalty cho ATE trên checkpoint đã train")
    p.add_argument("--ckpt", required=True,
                   help="Thư mục checkpoint (vd: .../seed_42/best)")
    p.add_argument("--data-dir", default=str(ROOT / "dataset"))
    p.add_argument("--split", default="dev", choices=["dev", "test"],
                   help="Chọn tham số trên dev; test chỉ để báo cáo cuối")
    p.add_argument("--beams", type=int, nargs="+", default=[4],
                   help="Các giá trị num_beams cần thử (mặc định: 4)")
    p.add_argument("--length-penalty", type=float, nargs="+",
                   default=[1.0, 1.2, 1.5, 2.0],
                   help="Các giá trị length_penalty cần thử")
    p.add_argument("--max-input-length", type=int, default=128)
    p.add_argument("--max-target-length", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit", type=int, default=0,
                   help="Chỉ dùng N câu đầu để quét nhanh (0 = toàn bộ)")
    p.add_argument("--no-normalize", action="store_true",
                   help="Tắt chuẩn hoá n-gram (để đo riêng ảnh hưởng của decode)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = Path(args.ckpt)
    if not ckpt.is_dir():
        raise FileNotFoundError(f"Không thấy checkpoint: {ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = get_raw_split_records(args.split, args.data_dir)
    if args.limit:
        records = records[: args.limit]

    golds = [[str(a) for a in r["aspects"]] for r in records]
    n_gold = sum(len(g) for g in golds)
    print(f"Checkpoint : {ckpt}")
    print(f"Split      : {args.split}  ({len(records)} câu, {n_gold} aspect gold, "
          f"TB {n_gold / max(len(records), 1):.2f}/câu)")
    print(f"Device     : {device}")
    print()

    # Nạp model MỘT lần; mỗi tổ hợp chỉ thay GenerationConfig.
    model = T5AspectExtractor.from_pretrained(str(ckpt), device=device)

    from src.metrics import _canonical_term as canon

    rows: List[Dict] = []
    header = (f"{'beams':>5} {'len_pen':>8} | {'P':>6} {'R':>6} {'F1':>6} | "
              f"{'pred/câu':>8} | recall theo số gold trong câu")
    print(header)
    print("-" * len(header))

    for beams in args.beams:
        # Greedy (beams=1) không dùng length_penalty — chạy 4 lần cùng một
        # cấu hình chỉ tốn GPU mà ra 4 dòng giống hệt nhau.
        lps = args.length_penalty if beams > 1 else args.length_penalty[:1]
        for lp in lps:
            model.generation = GenerationConfig(
                max_length=args.max_target_length,
                num_beams=beams,
                length_penalty=lp,
            )
            preds, _ = predict_aspects_for_records(
                model, records,
                max_input_length=args.max_input_length,
                normalize=not args.no_normalize,
                batch_size=args.batch_size,
            )
            m = evaluate_exact_match(preds, golds)
            per_sent = sum(len(p) for p in preds) / max(len(preds), 1)
            rbg = recall_by_gold_count(preds, golds, canon)
            rbg_txt = "  ".join(f"{k}{'+' if k == 4 else ''}:{v:.0f}%"
                                for k, v in rbg.items())
            print(f"{beams:>5} {lp:>8.2f} | {m['precision']*100:>6.2f} "
                  f"{m['recall']*100:>6.2f} {m['f1']*100:>6.2f} | "
                  f"{per_sent:>8.2f} | {rbg_txt}")
            rows.append({"num_beams": beams, "length_penalty": lp,
                         "f1": m["f1"], "precision": m["precision"],
                         "recall": m["recall"], "per_sent": per_sent})

    best = max(rows, key=lambda r: r["f1"])
    base = next((r for r in rows
                 if r["num_beams"] == 4 and abs(r["length_penalty"] - 1.0) < 1e-9), None)
    print()
    print("=" * 70)
    print(f"TỐT NHẤT trên {args.split}: beams={best['num_beams']}  "
          f"length_penalty={best['length_penalty']}  F1={best['f1']*100:.2f}%")
    if base is not None:
        delta = (best["f1"] - base["f1"]) * 100
        print(f"So với mặc định (beams=4, lp=1.0, F1={base['f1']*100:.2f}%): "
              f"{delta:+.2f} điểm F1, {best['per_sent'] - base['per_sent']:+.2f} aspect/câu")
        if abs(delta) < 0.3:
            print("Chênh lệch không đáng kể -> lỗi sinh thiếu KHÔNG do beam search;")
            print("nghi tiếp: dung lượng model / số epoch / phân bố số aspect trong train.")
    print()
    print("Dùng lại khi train:")
    print(f"  python common/run_multiseed_ate.py --num-beams {best['num_beams']} "
          f"--length-penalty {best['length_penalty']} ...")
    print("=" * 70)


if __name__ == "__main__":
    main()
