#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate thesis figures for LaTeX (Overleaf).

Usage (from repo root):
    pip install matplotlib pandas seaborn
    python scripts/generate_thesis_figures.py

Output: thesis/figures/*.pdf (and .png)
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "thesis" / "figures"
RUNS = ROOT / "runs_ate"
DATASET = ROOT / "dataset"

FIG_DIR.mkdir(exist_ok=True)

# Thesis-friendly style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def parse_apc_labels(apc_path: Path) -> list[tuple[str, str]]:
    """Parse .apc file -> list of (category, sentiment)."""
    lines = apc_path.read_text(encoding="utf-8").splitlines()
    records = []
    i = 0
    while i < len(lines):
        block = []
        while i < len(lines) and lines[i].strip():
            block.append(lines[i].strip())
            i += 1
        i += 1  # skip blank
        if len(block) >= 4:
            cat, sent = block[2], block[3]
            records.append((cat, sent.lower()))
    return records


def fig_label_distribution():
    """Bar chart: joint label counts on train set."""
    train = DATASET / "train.apc"
    if not train.exists():
        print(f"[skip] {train} not found")
        return
    recs = parse_apc_labels(train)
    labels = [f"{s}_{c}" for c, s in recs]
    counts = Counter(labels)
    df = pd.DataFrame({"label": list(counts.keys()), "count": list(counts.values())})
    df = df.sort_values("count", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.22)))
    sns.barplot(data=df, y="label", x="count", ax=ax, color="#2E75B6")
    ax.set_xlabel("Số mẫu (train)")
    ax.set_ylabel("Nhãn ghép sentiment_category")
    ax.set_title("Phân bố nhãn trên tập huấn luyện")
    fig.savefig(FIG_DIR / "fig_label_distribution.pdf")
    fig.savefig(FIG_DIR / "fig_label_distribution.png")
    plt.close(fig)
    print("  -> fig_label_distribution.pdf")


def load_joint_csv(name: str) -> pd.DataFrame:
    path = RUNS / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def fig_micro_f1_comparison():
    """Grouped bar: Micro F1 for 12 configs, BERT vs T5."""
    bert = load_joint_csv("eval_joint_triplet_BERT.csv")
    t5 = load_joint_csv("eval_joint_triplet_T5.csv")
    order = sorted(bert["config"].tolist(), key=lambda x: bert.loc[bert["config"] == x, "micro_f1"].values[0], reverse=True)

    x = range(len(order))
    w = 0.35
    bert_f1 = [bert.loc[bert["config"] == c, "micro_f1"].values[0] for c in order]
    t5_f1 = [t5.loc[t5["config"] == c, "micro_f1"].values[0] for c in order]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar([i - w / 2 for i in x], bert_f1, width=w, label="BERT", color="#1F4E79")
    ax.bar([i + w / 2 for i in x], t5_f1, width=w, label="T5", color="#2E75B6")
    ax.set_xticks(list(x))
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Micro F1 (%)")
    ax.set_title("So sánh Micro F1 Joint — 12 cấu hình")
    ax.legend()
    ax.set_ylim(65, 73)
    ax.axhline(70.53, color="gray", ls="--", lw=0.8, label="BERT baseline")
    fig.savefig(FIG_DIR / "fig_micro_f1_comparison.pdf")
    fig.savefig(FIG_DIR / "fig_micro_f1_comparison.png")
    plt.close(fig)
    print("  -> fig_micro_f1_comparison.pdf")


def fig_train_time_vs_f1():
    """Scatter: train time vs micro F1 (BERT Joint)."""
    df = load_joint_csv("eval_joint_triplet_BERT.csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        df["train_time_sec"],
        df["micro_f1"],
        c=df["macro_f1"],
        cmap="viridis",
        s=80,
        edgecolors="k",
        linewidths=0.5,
    )
    for _, row in df.iterrows():
        if row["micro_f1"] >= 71 or row["config"] == "baseline_balanced":
            ax.annotate(
                row["config"].replace("_resize", "").replace("lcf_", "")[:12],
                (row["train_time_sec"], row["micro_f1"]),
                fontsize=7,
                alpha=0.85,
            )
    plt.colorbar(sc, ax=ax, label="Macro F1")
    ax.set_xlabel("Thời gian huấn luyện (s)")
    ax.set_ylabel("Micro F1 (%)")
    ax.set_title("BERT Joint: Train time vs Micro F1")
    fig.savefig(FIG_DIR / "fig_train_time_vs_f1.pdf")
    fig.savefig(FIG_DIR / "fig_train_time_vs_f1.png")
    plt.close(fig)
    print("  -> fig_train_time_vs_f1.pdf")


def fig_per_label_f1_heatmap():
    """Heatmap per-label F1 for best BERT config."""
    df = load_joint_csv("eval_joint_triplet_BERT.csv")
    row = df.loc[df["config"] == "lcf_seq_cdm_resize"].iloc[0]
    f1_cols = [c for c in df.columns if c.startswith("f1_")]
    labels = [c.replace("f1_", "") for c in f1_cols]
    values = [row[c] for c in f1_cols]

    # reshape: sentiment x category (rough)
    cats = ["AMENITY", "BRANDING", "EXPERIENCE", "FACILITY", "LOYALTY", "SERVICE"]
    sents = ["negative", "neutral", "positive"]
    mat = [[float("nan")] * len(cats) for _ in sents]
    for lab, val in zip(labels, values):
        parts = lab.rsplit("_", 1)
        if len(parts) == 2:
            cat, sent = parts[0], parts[1]
            if cat in cats and sent in sents:
                mat[sents.index(sent)][cats.index(cat)] = val

    fig, ax = plt.subplots(figsize=(9, 3.5))
    sns.heatmap(
        mat,
        annot=True,
        fmt=".1f",
        xticklabels=cats,
        yticklabels=sents,
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        ax=ax,
        cbar_kws={"label": "F1 (%)"},
    )
    ax.set_title("F1 theo nhãn — lcf_seq_cdm_resize (BERT Joint)")
    fig.savefig(FIG_DIR / "fig_per_label_f1_heatmap.pdf")
    fig.savefig(FIG_DIR / "fig_per_label_f1_heatmap.png")
    plt.close(fig)
    print("  -> fig_per_label_f1_heatmap.pdf")


def fig_pipeline_architecture():
    """Simple 4-stage pipeline diagram (matplotlib fallback if no TikZ)."""
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.axis("off")
    stages = [
        (1.0, "Stage 1\nTiền xử lý SPC"),
        (3.5, "Stage 2\nATE (T5)"),
        (6.0, "Stage 3\nAPC (LCF+ToMe)"),
        (8.5, "Stage 4\nBộ ba"),
    ]
    for i, (x, txt) in enumerate(stages):
        rect = plt.Rectangle((x - 0.9, 0.25), 1.8, 0.5, fc="#E8F0FE", ec="#1F4E79", lw=1.5)
        ax.add_patch(rect)
        ax.text(x, 0.5, txt, ha="center", va="center", fontsize=8)
        if i < len(stages) - 1:
            ax.annotate("", xy=(stages[i + 1][0] - 0.95, 0.5), xytext=(x + 0.95, 0.5),
                        arrowprops=dict(arrowstyle="->", color="#333"))
    ax.set_title("Kiến trúc pipeline bốn stage")
    fig.savefig(FIG_DIR / "fig_pipeline_architecture.pdf")
    fig.savefig(FIG_DIR / "fig_pipeline_architecture.png")
    plt.close(fig)
    print("  -> fig_pipeline_architecture.pdf")


def fig_tome_strategies():
    """Sơ đồ minh họa ba chiến lược gộp token (Step 3 Chapter 3)."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    titles = ["Bipartite\n(ToMe CVPR 2023)", "Sequential Local", "Sequential Cosine (SCM)"]
    descs = [
        "Pool: loại CLS, SEP, aspect\n"
        "Chia xen kẽ A = {p0,p2,...}, B = {p1,p3,...}\n"
        "Mỗi a ∈ A chọn b ∈ B cosine lớn nhất\n"
        "Hợp nhất: x_a ← (x_a + x_b) / 2",
        "Quét trái → phải\n"
        "Tại token i: tính sim_left, sim_right\n"
        "sim_left > sim_right → i gập vào trái\n"
        "Giữ ranh giới LCF (mid_sep không gộp)",
        "Quét trái → phải theo vị trí\n"
        "Chọn token trái nhất chưa bị bảo vệ\n"
        "Bảo vệ: CLS, SEP, aspect\n"
        "Gộp với hàng xóm cosine cao nhất (toàn chuỗi)",
    ]
    colors = ["#E8F0FE", "#FFF3CD", "#D4EDDA"]
    borders = ["#1F4E79", "#856404", "#155724"]
    for ax, title, desc, fc, ec in zip(axes, titles, descs, colors, borders):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.text(0.5, 0.90, title, ha="center", va="top", fontsize=11,
                fontweight="bold", transform=ax.transAxes, color=ec)
        ax.text(0.5, 0.72, desc, ha="center", va="top", fontsize=8.5,
                transform=ax.transAxes, linespacing=1.6,
                bbox=dict(boxstyle="round,pad=0.5", fc=fc, ec=ec, lw=1.2))
    fig.suptitle("Ba chiến lược gộp token trong ToMeSequenceMerger\n"
                 "(token_merging/tome_1d.py)", fontsize=12, y=1.04)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_tome_strategies.pdf")
    fig.savefig(FIG_DIR / "fig_tome_strategies.png")
    plt.close(fig)
    print("  -> fig_tome_strategies.pdf")


def fig_resize_modes():
    """Bar chart so sánh ba chế độ ToMe: resize / compact / pre-BERT.

    Đọc runs_ate/eval_resize_modes.csv nếu có; nếu không, dùng giá trị
    placeholder để giữ cấu trúc hình.
    """
    csv_path = RUNS / "eval_resize_modes.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        modes   = df["mode"].tolist()
        micro   = df["micro_f1"].tolist()
        infer   = df["infer_ms"].tolist()
    else:
        # Placeholder — điền sau khi chạy thực nghiệm compact/pre-BERT
        modes  = ["Baseline\n(no ToMe)", "Resize\n(hiện tại)", "Compact\n(TBD)", "Pre-BERT\n(TBD)"]
        micro  = [68.92, 71.18, None, None]
        infer  = [10.0,  10.2,  None, None]

    x = range(len(modes))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    bar_colors = ["#ADB5BD" if v is None else "#2E75B6" for v in micro]
    micro_plot = [v if v is not None else 0 for v in micro]
    ax1.bar(x, micro_plot, color=bar_colors, edgecolor="k", linewidth=0.6)
    ax1.set_xticks(list(x)); ax1.set_xticklabels(modes, fontsize=9)
    ax1.set_ylabel("Micro F1 (%)")
    ax1.set_title("Chất lượng biểu diễn (Micro F1)")
    ax1.set_ylim(60, 74)
    for i, v in enumerate(micro):
        if v is not None:
            ax1.text(i, v + 0.1, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        else:
            ax1.text(i, 61, "TBD", ha="center", va="bottom", fontsize=8, color="gray")

    infer_colors = ["#ADB5BD" if v is None else "#1F4E79" for v in infer]
    infer_plot   = [v if v is not None else 0 for v in infer]
    ax2.bar(x, infer_plot, color=infer_colors, edgecolor="k", linewidth=0.6)
    ax2.set_xticks(list(x)); ax2.set_xticklabels(modes, fontsize=9)
    ax2.set_ylabel("Infer time (ms / câu)")
    ax2.set_title("Tốc độ suy luận (ms / câu)")
    for i, v in enumerate(infer):
        if v is not None:
            ax2.text(i, v + 0.05, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        else:
            ax2.text(i, 0.5, "TBD", ha="center", va="bottom", fontsize=8, color="gray")

    fig.suptitle("So sánh ba chế độ ToMe: Resize / Compact / Pre-BERT\n"
                 "(config: lcf_seq_cdm, BERT, Joint, tome_steps=2)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_resize_vs_compact.pdf")
    fig.savefig(FIG_DIR / "fig_resize_vs_compact.png")
    plt.close(fig)
    print("  -> fig_resize_vs_compact.pdf")


def fig_cdm_vs_cdw():
    """Bar chart so sánh CDM vs CDW: Micro F1 và Macro F1 (Joint, BERT)."""
    try:
        df = load_joint_csv("eval_joint_triplet_BERT.csv")
    except FileNotFoundError:
        print("[skip] eval_joint_triplet_BERT.csv not found")
        return

    cdw_configs = [c for c in df["config"].tolist() if "cdw" in c]
    cdm_configs = [c for c in df["config"].tolist() if "cdm" in c]
    if not cdw_configs or not cdm_configs:
        print("[skip] CDW/CDM configs not found in CSV")
        return

    labels, cdw_micro, cdm_micro, cdw_macro, cdm_macro = [], [], [], [], []
    for cdm_c in cdm_configs:
        base = cdm_c.replace("_cdm", "")
        cdw_c = base + "_cdw"
        cdw_c_alt = cdm_c.replace("cdm", "cdw")
        cdw_match = cdw_c if cdw_c in cdw_configs else (cdw_c_alt if cdw_c_alt in cdw_configs else None)
        if cdw_match is None:
            continue
        short = base.replace("lcf_", "").replace("_resize", "")
        labels.append(short)
        cdm_row = df.loc[df["config"] == cdm_c].iloc[0]
        cdw_row = df.loc[df["config"] == cdw_match].iloc[0]
        cdm_micro.append(cdm_row["micro_f1"]); cdw_micro.append(cdw_row["micro_f1"])
        cdm_macro.append(cdm_row["macro_f1"]); cdw_macro.append(cdw_row["macro_f1"])

    if not labels:
        print("[skip] no matching CDM/CDW pairs")
        return

    x = range(len(labels)); w = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.bar([i - w/2 for i in x], cdm_micro, w, label="CDM", color="#1F4E79")
    ax1.bar([i + w/2 for i in x], cdw_micro, w, label="CDW", color="#9DC3E6")
    ax1.set_xticks(list(x)); ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax1.set_ylabel("Micro F1 (%)"); ax1.set_title("Micro F1: CDM vs CDW")
    ax1.legend(); ax1.set_ylim(65, 74)

    ax2.bar([i - w/2 for i in x], cdm_macro, w, label="CDM", color="#843C0C")
    ax2.bar([i + w/2 for i in x], cdw_macro, w, label="CDW", color="#F4B183")
    ax2.set_xticks(list(x)); ax2.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax2.set_ylabel("Macro F1 (%)"); ax2.set_title("Macro F1: CDM vs CDW (nhãn thiểu số)")
    ax2.legend()

    fig.suptitle("CDM vs CDW: Micro F1 và Macro F1 (Joint, BERT)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_cdm_vs_cdw.pdf")
    fig.savefig(FIG_DIR / "fig_cdm_vs_cdw.png")
    plt.close(fig)
    print("  -> fig_cdm_vs_cdw.pdf")


def main():
    print("Generating thesis figures ->", FIG_DIR)
    fig_label_distribution()
    fig_micro_f1_comparison()
    fig_train_time_vs_f1()
    fig_per_label_f1_heatmap()
    fig_pipeline_architecture()
    fig_tome_strategies()
    fig_resize_modes()
    fig_cdm_vs_cdw()
    print("Done.")


if __name__ == "__main__":
    main()
