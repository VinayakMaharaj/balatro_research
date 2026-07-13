"""
plot_figure.py
Produces two separate publication-ready figures for the IEEE ICTMOD 2026 paper.

Figure 1 (figure1_performance.pdf/.svg):
    Box-and-whisker plot of final ante distribution — all 5 agents
    IEEE single-column width: 3.5"

Figure 2 (figure2_qualitative.pdf/.svg):
    Grouped bar chart of decision quality metrics — 3 LLM agents
    IEEE single-column width: 3.5"

Vector formats (PDF + SVG) scale losslessly at any zoom level.

Usage:
    python plot_figure.py --results results_final.csv --output figures/
"""

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
# IEEE dimensions — vector output, no DPI limit applies to PDF/SVG
# ---------------------------------------------------------------------------

FIG_WIDTH_1COL = 3.5    # IEEE single column
FIG_WIDTH_2COL = 7.16   # IEEE double column
DPI            = 300    # for PNG preview only

# ---------------------------------------------------------------------------
# Colorblind-safe palette (Wong 2011) — consistent across both figures
# ---------------------------------------------------------------------------

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

# Display order for figure 1 — all 5 agents, logical left-to-right
BOT_ORDER = ["flush_bot", "meta_bot", "rag_llm_bot", "llm_bot", "rag_meta_bot"]

# LLM agents for figure 2
LLM_BOTS       = ["rag_llm_bot", "llm_bot", "rag_meta_bot"]
LLM_BOT_LABELS = ["RAGBot\n(rules)", "LLMBot", "RAGBot\n(strategy)"]

# Qualitative metrics from analyze_results.py output
QUALITATIVE = {
    "rag_llm_bot":  {"self_corr": 0.95, "ante1_skips": 78,  "parse_fail_pct": 3.5},
    "llm_bot":      {"self_corr": 0.52, "ante1_skips": 28,  "parse_fail_pct": 0.4},
    "rag_meta_bot": {"self_corr": 0.44, "ante1_skips": 0,   "parse_fail_pct": 0.5},
}


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
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.03,
    })


def save_figure(fig, out: Path, stem: str):
    """Save as PDF, SVG, and PNG — PDF/SVG are vector (lossless at any zoom)."""
    for fmt in ["pdf", "svg", "png"]:
        path = out / f"{stem}.{fmt}"
        fig.savefig(path, format=fmt, dpi=DPI if fmt == "png" else None)
        print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 1: Box-and-whisker — all 5 agents
# ---------------------------------------------------------------------------

def plot_figure1(df: pd.DataFrame, out: Path):
    bots   = [b for b in BOT_ORDER if b in df["bot_type"].unique()]
    data   = [df[df["bot_type"] == b]["final_ante"].values for b in bots]
    labels = [BOT_LABELS[b] for b in bots]
    colors = [COLORS[b] for b in bots]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_1COL, 3.2))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.93, bottom=0.18)

    bp = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        widths=0.5,
        showfliers=True,
        medianprops=dict(color="black", linewidth=1.0),
        whiskerprops=dict(linewidth=0.6, linestyle="--"),
        capprops=dict(linewidth=0.6),
        flierprops=dict(marker="o", markersize=2, linestyle="none",
                        markerfacecolor="#888888", markeredgecolor="#888888", alpha=0.5),
        boxprops=dict(linewidth=0.5),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Mean diamond
    for i, d in enumerate(data):
        if len(d) > 0:
            ax.plot(i + 1, np.mean(d), marker="D", color="white",
                    markersize=3.5, markeredgecolor="black",
                    markeredgewidth=0.5, zorder=5)

    # Annotate means above boxes
    for i, d in enumerate(data):
        if len(d) > 0:
            ax.text(i + 1, np.mean(d) + 0.22, f"{np.mean(d):.2f}",
                    ha="center", va="bottom", fontsize=5.5, color="#333333")

    ax.set_ylabel("Final ante reached")
    ax.set_ylim(0.5, 5.8)
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.set_title("Performance Distribution by Agent", pad=4)
    ax.tick_params(axis="x", which="both", bottom=False)

    diamond = mpatches.Patch(facecolor="white", edgecolor="black",
                              linewidth=0.5, label="Mean")
    ax.legend(handles=[diamond], loc="upper left", frameon=False, handlelength=1)

    save_figure(fig, out, "figure1_performance")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Decision quality — 3 LLM agents, normalised grouped bars
# ---------------------------------------------------------------------------

def plot_figure2(out: Path):
    x     = np.arange(len(LLM_BOTS))
    width = 0.22

    max_skips = 78.0  # normalisation denominator

    self_corr  = [QUALITATIVE[b]["self_corr"] for b in LLM_BOTS]
    skips_norm = [QUALITATIVE[b]["ante1_skips"] / max_skips for b in LLM_BOTS]
    parse_norm = [QUALITATIVE[b]["parse_fail_pct"] / 10.0 for b in LLM_BOTS]

    raw_self   = [f"{QUALITATIVE[b]['self_corr']:.2f}" for b in LLM_BOTS]
    raw_skips  = [str(QUALITATIVE[b]["ante1_skips"]) for b in LLM_BOTS]
    raw_parse  = [f"{QUALITATIVE[b]['parse_fail_pct']:.1f}%" for b in LLM_BOTS]

    bar_colors = ["#D55E00", "#F0E442", "#0072B2"]  # colorblind safe

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_1COL, 3.0))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.88, bottom=0.18)

    b1 = ax.bar(x - width, self_corr,  width,
                label="Self-corrections / hand",
                color=bar_colors[0], alpha=0.85, linewidth=0.3, edgecolor="black")
    b2 = ax.bar(x,          skips_norm, width,
                label=f"Ante-1 blind skips (÷{int(max_skips)})",
                color=bar_colors[1], alpha=0.85, linewidth=0.3, edgecolor="black")
    b3 = ax.bar(x + width,  parse_norm, width,
                label="Parse failure rate (÷10%)",
                color=bar_colors[2], alpha=0.85, linewidth=0.3, edgecolor="black")

    # Raw value labels above each bar
    for bars, raw in zip([b1, b2, b3], [raw_self, raw_skips, raw_parse]):
        for bar, label in zip(bars, raw):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + 0.02, label,
                    ha="center", va="bottom", fontsize=5.5)

    ax.set_xticks(x)
    ax.set_xticklabels(LLM_BOT_LABELS)
    ax.set_ylabel("Normalised value")
    ax.set_ylim(0, 1.35)
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.set_title("Decision Quality: LLM Agents", pad=4)
    ax.tick_params(axis="x", which="both", bottom=False)
    ax.legend(loc="upper right", frameon=False, ncol=1,
              handlelength=1.0, handletextpad=0.4, labelspacing=0.25,
              fontsize=6)

    save_figure(fig, out, "figure2_qualitative")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(results_path: str, output_dir: str):
    set_ieee_style()

    df = pd.read_csv(results_path)
    df = df[df["outcome"].isin(["won","lost"])].copy()
    for col in ["final_ante","peak_ante","final_round","hands_played",
                "discards_used","blinds_skipped","jokers_bought"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    print("Generating Figure 1: Performance distribution (all 5 agents)...")
    plot_figure1(df, out)

    print("Generating Figure 2: Decision quality (LLM agents)...")
    plot_figure2(out)

    print("\nDone. Files saved:")
    for f in sorted(out.glob("figure*")):
        print(f"  {f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results_final.csv")
    parser.add_argument("--output",  default="figures")
    args = parser.parse_args()
    main(args.results, args.output)