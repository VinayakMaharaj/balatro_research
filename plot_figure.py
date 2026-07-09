"""
plot_figure.py
Produces a publication-ready 2-panel figure for the IEEE ICTMOD 2026 paper.

Panel A (left):  Box-and-whisker plot of final ante distribution by agent
Panel B (right): Grouped bar chart of qualitative LLM metrics

Output:
    figures/figure1.pdf  — vector, for Overleaf
    figures/figure1.png  — 300 DPI raster, for preview

Usage:
    python plot_figure.py --results results_final.csv --output figures/
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

# ---------------------------------------------------------------------------
# IEEE figure dimensions
# Two-column IEEE paper: max width 7.16 inches
# Single-column: 3.5 inches
# Using double-column width for the 2-panel figure
# ---------------------------------------------------------------------------

FIG_WIDTH   = 7.16   # inches — IEEE double column
FIG_HEIGHT  = 2.8    # inches
DPI         = 300

# Colorblind-safe palette (Wong 2011)
COLORS = {
    "flush_bot":    "#56B4E9",  # sky blue
    "meta_bot":     "#009E73",  # green
    "rag_llm_bot":  "#CC79A7",  # pink/purple
    "llm_bot":      "#E69F00",  # orange
    "rag_meta_bot": "#0072B2",  # dark blue
}

BOT_LABELS = {
    "flush_bot":    "FlushBot",
    "meta_bot":     "MetaBot",
    "rag_llm_bot":  "RAGBot\n(rules)",
    "llm_bot":      "LLMBot",
    "rag_meta_bot": "RAGBot\n(strategy)",
}

BOT_ORDER = ["flush_bot", "meta_bot", "rag_llm_bot", "llm_bot", "rag_meta_bot"]

LLM_BOTS       = ["rag_llm_bot", "llm_bot", "rag_meta_bot"]
LLM_BOT_LABELS = ["RAGBot\n(rules)", "LLMBot", "RAGBot\n(strategy)"]

# ---------------------------------------------------------------------------
# IEEE matplotlib style
# ---------------------------------------------------------------------------

def set_ieee_style():
    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size":          8,
        "axes.titlesize":     8,
        "axes.labelsize":     8,
        "xtick.labelsize":    7,
        "ytick.labelsize":    7,
        "legend.fontsize":    7,
        "figure.dpi":         DPI,
        "axes.linewidth":     0.5,
        "xtick.major.width":  0.5,
        "ytick.major.width":  0.5,
        "xtick.major.size":   2,
        "ytick.major.size":   2,
        "lines.linewidth":    0.8,
        "patch.linewidth":    0.5,
        "grid.linewidth":     0.3,
        "grid.alpha":         0.4,
        "axes.grid":          True,
        "axes.grid.axis":     "y",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "savefig.dpi":        DPI,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.02,
    })


# ---------------------------------------------------------------------------
# Panel A: box-and-whisker of final ante
# ---------------------------------------------------------------------------

def plot_boxplot(ax: plt.Axes, df: pd.DataFrame):
    bots    = [b for b in BOT_ORDER if b in df["bot_type"].unique()]
    data    = [df[df["bot_type"] == b]["final_ante"].values for b in bots]
    labels  = [BOT_LABELS[b] for b in bots]
    colors  = [COLORS[b] for b in bots]

    bp = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="black", linewidth=1.0),
        whiskerprops=dict(linewidth=0.6),
        capprops=dict(linewidth=0.6),
        flierprops=dict(marker="o", markersize=2, linestyle="none",
                        markerfacecolor="gray", alpha=0.5),
        boxprops=dict(linewidth=0.5),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Overlay mean as white diamond
    for i, (bot, d) in enumerate(zip(bots, data)):
        if len(d) > 0:
            ax.plot(i + 1, np.mean(d), marker="D", color="white",
                    markersize=3.5, markeredgecolor="black",
                    markeredgewidth=0.5, zorder=5)

    ax.set_ylabel("Final ante reached")
    ax.set_ylim(0.5, 5.5)
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.set_title("(a) Performance distribution", loc="left", pad=3)

    # Mean diamond legend entry
    diamond = mpatches.Patch(facecolor="white", edgecolor="black",
                              linewidth=0.5, label="Mean")
    ax.legend(handles=[diamond], loc="upper left", frameon=False,
              markerscale=0.8, handlelength=1)

    ax.tick_params(axis="x", which="both", bottom=False)


# ---------------------------------------------------------------------------
# Panel B: qualitative LLM metrics
# ---------------------------------------------------------------------------

def plot_qualitative(ax: plt.Axes):
    """
    Three grouped bars per LLM agent:
      - Self-corrections per hand (scaled x10 for visibility on same axis)
      - Ante-1 blind skips (normalised to per-50-seeds)
      - Parse failure rate (%)
    """

    # Qualitative metrics from analyze_results.py output
    # self_corr: avg self-corrections per hand decision
    # ante1_skips: total ante-1 blind skips across 50 seeds
    # parse_fail: parse failure rate as percentage (0-100 scale / 10 for display)
    metrics = {
        "rag_llm_bot":  {"self_corr": 0.95, "ante1_skips": 78,  "parse_fail_pct": 3.5},
        "llm_bot":      {"self_corr": 0.52, "ante1_skips": 28,  "parse_fail_pct": 0.4},
        "rag_meta_bot": {"self_corr": 0.44, "ante1_skips": 0,   "parse_fail_pct": 0.5},
    }

    width = 0.22
    x     = np.arange(len(LLM_BOTS))

    # Normalise all metrics to same scale (0-1) for grouped bars
    # self_corr already 0-1 range
    # ante1_skips: normalise by max (78) -> 0-1
    # parse_fail_pct: already small %, divide by 10 to bring to ~0-1 range
    max_skips = 78.0

    self_corr  = [metrics[b]["self_corr"] for b in LLM_BOTS]
    skips_norm = [metrics[b]["ante1_skips"] / max_skips for b in LLM_BOTS]
    parse_norm = [metrics[b]["parse_fail_pct"] / 10.0 for b in LLM_BOTS]

    bar_colors = ["#D55E00", "#F0E442", "#0072B2"]

    b1 = ax.bar(x - width, self_corr,  width, label="Self-corrections / hand",
                color=bar_colors[0], alpha=0.85, linewidth=0.3, edgecolor="black")
    b2 = ax.bar(x,          skips_norm, width, label=f"Ante-1 skips (÷{int(max_skips)})",
                color=bar_colors[1], alpha=0.85, linewidth=0.3, edgecolor="black")
    b3 = ax.bar(x + width,  parse_norm, width, label="Parse fail rate (÷10%)",
                color=bar_colors[2], alpha=0.85, linewidth=0.3, edgecolor="black")

    # Raw value labels above each bar
    raw_labels = [
        [f"{metrics[b]['self_corr']:.2f}" for b in LLM_BOTS],
        [str(metrics[b]["ante1_skips"]) for b in LLM_BOTS],
        [f"{metrics[b]['parse_fail_pct']:.1f}%" for b in LLM_BOTS],
    ]
    for bars, labels in zip([b1, b2, b3], raw_labels):
        for bar, label in zip(bars, labels):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                    label, ha="center", va="bottom", fontsize=5.5)

    ax.set_xticks(x)
    ax.set_xticklabels(LLM_BOT_LABELS)
    ax.set_ylabel("Normalised value")
    ax.set_ylim(0, 1.35)
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.set_title("(b) Decision quality (LLM agents)", loc="left", pad=3)
    ax.tick_params(axis="x", which="both", bottom=False)
    ax.legend(loc="upper right", frameon=False, ncol=1,
              handlelength=1.0, handletextpad=0.4, labelspacing=0.2,
              fontsize=6)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(results_path: str, output_dir: str):
    set_ieee_style()

    df      = pd.read_csv(results_path)
    df      = df[df["outcome"].isin(["won","lost"])].copy()
    numeric = ["final_ante","peak_ante","final_round","hands_played",
               "discards_used","blinds_skipped","jokers_bought"]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        gridspec_kw={"width_ratios": [1.1, 1]},
    )
    fig.subplots_adjust(wspace=0.35, left=0.08, right=0.98, top=0.93, bottom=0.15)

    plot_boxplot(ax_a, df)
    plot_qualitative(ax_b)

    # Save
    pdf_path = out / "figure1.pdf"
    png_path = out / "figure1.png"
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, format="png", dpi=DPI)
    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results_final.csv")
    parser.add_argument("--output",  default="figures")
    args = parser.parse_args()
    main(args.results, args.output)