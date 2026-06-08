"""
analyze_results.py
Statistical analysis of Balatro bot experiment results.
Produces tables and figures for the paper.

Usage:
    python analyze_results.py --results results.csv --output figures/

Output:
    - summary_table.csv        : mean ± std per bot for all metrics
    - ante_progression.png     : boxplot of final_ante by bot
    - round_progression.png    : boxplot of final_round by bot
    - economy_comparison.png   : jokers_bought, blinds_skipped, discards_used
    - hand_type_distribution.png : stacked bar of hand types per bot
    - llm_cost_summary.csv     : token/cost breakdown for LLM bots
    - qualitative_summary.csv  : LLM reasoning patterns from llm_decisions.jsonl
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# Optional matplotlib — skip plot generation if not available
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not installed — skipping plot generation. pip install matplotlib")


# ---------------------------------------------------------------------------
# Bot display order and colors
# ---------------------------------------------------------------------------

BOT_ORDER = ["flush_bot", "meta_bot", "llm_bot", "rag_llm_bot", "rl_bot"]
BOT_COLORS = {
    "flush_bot":   "#636EFA",
    "meta_bot":    "#EF553B",
    "llm_bot":     "#00CC96",
    "rag_llm_bot": "#AB63FA",
    "rl_bot":      "#FFA15A",
}
BOT_LABELS = {
    "flush_bot":   "FlushBot",
    "meta_bot":    "MetaBot",
    "llm_bot":     "LLMBot",
    "rag_llm_bot": "RAG-LLMBot",
    "rl_bot":      "RLBot (PPO)",
}


# ---------------------------------------------------------------------------
# Load and clean data
# ---------------------------------------------------------------------------

def load_results(results_path: str) -> pd.DataFrame:
    df = pd.read_csv(results_path)

    # Drop error/incomplete rows for statistical analysis
    df_clean = df[df["outcome"].isin(["won", "lost"])].copy()

    # Fix column types
    numeric_cols = [
        "final_ante", "peak_ante", "final_round", "hands_played",
        "discards_used", "blinds_skipped", "final_dollars",
        "jokers_bought", "rerolls_used", "llm_calls",
        "tokens_used", "estimated_cost_usd", "duration_seconds"
    ]
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").fillna(0)

    # peak_ante fallback to final_ante if column missing
    if "peak_ante" not in df_clean.columns:
        df_clean["peak_ante"] = df_clean["final_ante"]

    df_clean["won"] = (df_clean["outcome"] == "won").astype(int)

    print(f"Loaded {len(df)} total rows, {len(df_clean)} valid games")
    print(f"Bots present: {sorted(df_clean['bot_type'].unique())}")
    print(f"Error/incomplete rows dropped: {len(df) - len(df_clean)}\n")

    return df_clean


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "final_ante", "peak_ante", "final_round",
        "hands_played", "discards_used", "blinds_skipped",
        "final_dollars", "jokers_bought", "won",
    ]

    rows = []
    bots_present = [b for b in BOT_ORDER if b in df["bot_type"].unique()]

    for bot in bots_present:
        sub = df[df["bot_type"] == bot]
        row = {"bot_type": bot, "n_games": len(sub)}
        for m in metrics:
            if m in sub.columns:
                row[f"{m}_mean"] = round(sub[m].mean(), 3)
                row[f"{m}_std"]  = round(sub[m].std(), 3)
                row[f"{m}_median"] = round(sub[m].median(), 3)
        rows.append(row)

    return pd.DataFrame(rows)


def print_summary_table(summary: pd.DataFrame):
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    key_cols = [
        "bot_type", "n_games",
        "final_ante_mean", "final_ante_std",
        "peak_ante_mean",
        "final_round_mean", "final_round_std",
        "jokers_bought_mean",
        "blinds_skipped_mean",
        "won_mean",
    ]
    cols = [c for c in key_cols if c in summary.columns]
    print(summary[cols].to_string(index=False))
    print()


def significance_tests(df: pd.DataFrame, metric: str = "final_ante"):
    """Pairwise Mann-Whitney U tests between bots for a given metric."""
    bots = [b for b in BOT_ORDER if b in df["bot_type"].unique()]
    print(f"Pairwise Mann-Whitney U tests: {metric}")
    print("-" * 50)
    for i, b1 in enumerate(bots):
        for b2 in bots[i+1:]:
            g1 = df[df["bot_type"] == b1][metric].dropna()
            g2 = df[df["bot_type"] == b2][metric].dropna()
            if len(g1) < 3 or len(g2) < 3:
                print(f"  {b1} vs {b2}: insufficient data")
                continue
            u, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {BOT_LABELS.get(b1, b1):12} vs {BOT_LABELS.get(b2, b2):12}: "
                  f"U={u:.0f}, p={p:.4f} {sig}")
    print()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_ante_boxplot(df: pd.DataFrame, output_dir: Path):
    if not HAS_MATPLOTLIB:
        return
    bots = [b for b in BOT_ORDER if b in df["bot_type"].unique()]
    data = [df[df["bot_type"] == b]["final_ante"].values for b in bots]
    labels = [BOT_LABELS.get(b, b) for b in bots]
    colors = [BOT_COLORS.get(b, "#888") for b in bots]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, patch_artist=True, labels=labels)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Final Ante Reached")
    ax.set_title("Agent Performance: Final Ante Distribution")
    ax.axhline(y=8, color="gold", linestyle="--", alpha=0.5, label="Win condition (Ante 8)")
    ax.legend()
    plt.tight_layout()
    path = output_dir / "ante_progression.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_round_boxplot(df: pd.DataFrame, output_dir: Path):
    if not HAS_MATPLOTLIB:
        return
    bots = [b for b in BOT_ORDER if b in df["bot_type"].unique()]
    data = [df[df["bot_type"] == b]["final_round"].values for b in bots]
    labels = [BOT_LABELS.get(b, b) for b in bots]
    colors = [BOT_COLORS.get(b, "#888") for b in bots]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, patch_artist=True, labels=labels)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Final Round Reached")
    ax.set_title("Agent Performance: Final Round Distribution")
    plt.tight_layout()
    path = output_dir / "round_progression.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_economy(df: pd.DataFrame, output_dir: Path):
    if not HAS_MATPLOTLIB:
        return
    bots = [b for b in BOT_ORDER if b in df["bot_type"].unique()]
    labels = [BOT_LABELS.get(b, b) for b in bots]
    metrics = ["jokers_bought", "blinds_skipped", "discards_used"]
    metric_labels = ["Jokers Bought", "Blinds Skipped", "Discards Used"]

    x = np.arange(len(bots))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (m, ml) in enumerate(zip(metrics, metric_labels)):
        if m not in df.columns:
            continue
        means = [df[df["bot_type"] == b][m].mean() for b in bots]
        ax.bar(x + i * width, means, width, label=ml, alpha=0.8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Average per Game")
    ax.set_title("Economy Metrics by Agent")
    ax.legend()
    plt.tight_layout()
    path = output_dir / "economy_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_hand_types(df: pd.DataFrame, output_dir: Path):
    if not HAS_MATPLOTLIB:
        return
    if "hands_by_type" not in df.columns:
        return

    bots = [b for b in BOT_ORDER if b in df["bot_type"].unique()]
    hand_types = ["flush", "straight", "four_of_a_kind", "full_house",
                  "three_of_a_kind", "two_pair", "pair", "high_card"]
    ht_colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728",
                 "#9467bd","#8c564b","#e377c2","#7f7f7f"]

    bot_ht_data = {}
    for bot in bots:
        sub = df[df["bot_type"] == bot]
        totals = {ht: 0 for ht in hand_types}
        for _, row in sub.iterrows():
            try:
                ht_dict = json.loads(row["hands_by_type"]) if row["hands_by_type"] else {}
                for ht, count in ht_dict.items():
                    if ht in totals:
                        totals[ht] += count
            except Exception:
                pass
        total_hands = sum(totals.values()) or 1
        bot_ht_data[bot] = {ht: totals[ht] / total_hands for ht in hand_types}

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(bots))
    bottoms = np.zeros(len(bots))
    labels = [BOT_LABELS.get(b, b) for b in bots]

    for ht, color in zip(hand_types, ht_colors):
        vals = [bot_ht_data[b][ht] for b in bots]
        ax.bar(x, vals, bottom=bottoms, label=ht.replace("_", " ").title(),
               color=color, alpha=0.85)
        bottoms += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fraction of Hands Played")
    ax.set_title("Hand Type Distribution by Agent")
    ax.legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    path = output_dir / "hand_type_distribution.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# LLM cost summary
# ---------------------------------------------------------------------------

def llm_cost_summary(df: pd.DataFrame, output_dir: Path):
    llm_bots = ["llm_bot", "rag_llm_bot"]
    llm_df = df[df["bot_type"].isin(llm_bots)]
    if llm_df.empty:
        return

    rows = []
    for bot in llm_bots:
        sub = llm_df[llm_df["bot_type"] == bot]
        if sub.empty:
            continue
        rows.append({
            "bot_type": bot,
            "n_games": len(sub),
            "avg_llm_calls": round(sub["llm_calls"].mean(), 1),
            "avg_tokens_per_game": round(sub["tokens_used"].mean(), 0),
            "avg_cost_per_game_usd": round(sub["estimated_cost_usd"].mean(), 5),
            "total_cost_usd": round(sub["estimated_cost_usd"].sum(), 4),
            "projected_100game_cost_usd": round(sub["estimated_cost_usd"].mean() * 100, 2),
        })

    cost_df = pd.DataFrame(rows)
    path = output_dir / "llm_cost_summary.csv"
    cost_df.to_csv(path, index=False)
    print(f"Saved: {path}")
    print(cost_df.to_string(index=False))
    print()


# ---------------------------------------------------------------------------
# Qualitative analysis from llm_decisions.jsonl
# ---------------------------------------------------------------------------

def qualitative_summary(decisions_path: str, output_dir: Path):
    p = Path(decisions_path)
    if not p.exists():
        print(f"No qualitative log found at {decisions_path}, skipping")
        return

    records = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except Exception:
                pass

    if not records:
        return

    df = pd.DataFrame(records)
    print(f"Qualitative log: {len(df)} decisions loaded")

    rows = []
    for bot in df["bot_type"].unique():
        sub = df[df["bot_type"] == bot]
        for state in ["SELECTING_HAND", "SHOP", "BLIND_SELECT"]:
            state_sub = sub[sub["state"] == state]
            if state_sub.empty:
                continue

            # Action distribution
            if state == "SELECTING_HAND":
                actions = state_sub["parsed_action"].apply(
                    lambda x: x.get("action", "unknown") if isinstance(x, dict) else "unknown"
                )
                action_counts = actions.value_counts().to_dict()
            elif state == "BLIND_SELECT":
                actions = state_sub["parsed_action"].apply(
                    lambda x: x.get("action", "unknown") if isinstance(x, dict) else "unknown"
                )
                action_counts = actions.value_counts().to_dict()
            else:
                action_counts = {}

            rows.append({
                "bot_type": bot,
                "state": state,
                "n_decisions": len(state_sub),
                "action_distribution": json.dumps(action_counts),
                "avg_reasoning_length": round(
                    state_sub["reasoning"].apply(lambda x: len(str(x))).mean(), 1
                ),
            })

    qual_df = pd.DataFrame(rows)
    path = output_dir / "qualitative_summary.csv"
    qual_df.to_csv(path, index=False)
    print(f"Saved: {path}")
    print(qual_df.to_string(index=False))
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Balatro bot results")
    parser.add_argument("--results", default="results.csv")
    parser.add_argument("--decisions", default="llm_decisions.jsonl")
    parser.add_argument("--output", default="figures")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    df = load_results(args.results)

    summary = compute_summary(df)
    print_summary_table(summary)
    summary.to_csv(output_dir / "summary_table.csv", index=False)
    print(f"Saved: {output_dir / 'summary_table.csv'}\n")

    significance_tests(df, "final_ante")
    significance_tests(df, "final_round")

    plot_ante_boxplot(df, output_dir)
    plot_round_boxplot(df, output_dir)
    plot_economy(df, output_dir)
    plot_hand_types(df, output_dir)

    llm_cost_summary(df, output_dir)
    qualitative_summary(args.decisions, output_dir)

    print("Analysis complete.")
