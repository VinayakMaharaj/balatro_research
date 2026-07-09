"""
analyze_results.py
Statistical analysis of Balatro bot experiment results.
Produces tables and figures for the paper.

Usage:
    python analyze_results.py --results results_final.csv --output figures/

Output:
    - summary_table.csv          : mean +/- std per bot for all metrics
    - ante_progression.png       : boxplot of final_ante by bot
    - round_progression.png      : boxplot of final_round by bot
    - economy_comparison.png     : jokers_bought, blinds_skipped, discards_used
    - hand_type_distribution.png : stacked bar of hand types per bot
    - llm_cost_summary.csv       : token and cost breakdown for LLM bots
    - qualitative_summary.csv    : LLM reasoning patterns from decision logs
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not installed — skipping plot generation.")


# ---------------------------------------------------------------------------
# Bot display order, colors, labels
# ---------------------------------------------------------------------------

BOT_ORDER = [
    "meta_bot",
    "llm_bot",
    "rag_meta_bot",
    "flush_bot",
    "rag_llm_bot",
]

PRIMARY_BOTS = ["meta_bot", "llm_bot", "rag_meta_bot"]

BOT_COLORS = {
    "flush_bot":   "#636EFA",
    "meta_bot":    "#EF553B",
    "llm_bot":     "#00CC96",
    "rag_llm_bot": "#AB63FA",
    "rag_meta_bot":"#FFA15A",
}

BOT_LABELS = {
    "flush_bot":   "FlushBot",
    "meta_bot":    "MetaBot",
    "llm_bot":     "LLMBot",
    "rag_llm_bot": "RAGBot (rules)",
    "rag_meta_bot":"RAGBot (strategy)",
}

# Decision log files per bot
DECISION_LOGS = {
    "llm_bot":     "llm_decisions.jsonl",
    "rag_llm_bot": "rag_decisions.jsonl",
    "rag_meta_bot":"rag_meta_decisions.jsonl",
}

# Filter timestamps — only use clean runs from these dates onward
CLEAN_FROM = {
    "llm_bot":     "2026-06-29",
    "rag_llm_bot": "2026-07-04",
    "rag_meta_bot":"2026-06-30",
}


# ---------------------------------------------------------------------------
# Load and clean data
# ---------------------------------------------------------------------------

def load_results(results_path: str) -> pd.DataFrame:
    df = pd.read_csv(results_path)
    df_clean = df[df["outcome"].isin(["won","lost"])].copy()

    numeric_cols = [
        "final_ante","peak_ante","final_round","hands_played",
        "discards_used","blinds_skipped","final_dollars",
        "jokers_bought","rerolls_used","llm_calls",
        "tokens_used","estimated_cost_usd","duration_seconds",
    ]
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").fillna(0)

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
        "final_ante","peak_ante","final_round",
        "hands_played","discards_used","blinds_skipped",
        "final_dollars","jokers_bought","won",
    ]
    rows = []
    bots_present = [b for b in BOT_ORDER if b in df["bot_type"].unique()]

    for bot in bots_present:
        sub = df[df["bot_type"] == bot]
        row = {"bot_type": bot, "n_games": len(sub)}
        for m in metrics:
            if m in sub.columns:
                row[f"{m}_mean"]   = round(sub[m].mean(), 3)
                row[f"{m}_std"]    = round(sub[m].std(),  3)
                row[f"{m}_median"] = round(sub[m].median(), 3)
        rows.append(row)

    return pd.DataFrame(rows)


def print_summary_table(summary: pd.DataFrame):
    key_cols = [
        "bot_type","n_games",
        "final_ante_mean","final_ante_std",
        "peak_ante_mean",
        "final_round_mean","final_round_std",
        "jokers_bought_mean",
        "blinds_skipped_mean",
        "won_mean",
    ]
    cols = [c for c in key_cols if c in summary.columns]

    print("=" * 70)
    print("PRIMARY RESULTS (MetaBot vs LLMBot vs RAGBot strategy)")
    print("=" * 70)
    primary = summary[summary["bot_type"].isin(PRIMARY_BOTS)]
    print(primary[cols].to_string(index=False))
    print()

    print("=" * 70)
    print("SECONDARY / ABLATION AGENTS")
    print("=" * 70)
    secondary = summary[~summary["bot_type"].isin(PRIMARY_BOTS)]
    if not secondary.empty:
        print(secondary[cols].to_string(index=False))
    print()


def significance_tests(df: pd.DataFrame, metric: str = "final_ante"):
    primary  = [b for b in PRIMARY_BOTS if b in df["bot_type"].unique()]
    all_bots = [b for b in BOT_ORDER if b in df["bot_type"].unique()]
    print(f"Pairwise Mann-Whitney U tests (primary agents): {metric}")
    print("-" * 50)
    bots = primary
    for i, b1 in enumerate(bots):
        for b2 in bots[i+1:]:
            g1 = df[df["bot_type"] == b1][metric].dropna()
            g2 = df[df["bot_type"] == b2][metric].dropna()
            if len(g1) < 3 or len(g2) < 3:
                print(f"  {BOT_LABELS.get(b1,b1)} vs {BOT_LABELS.get(b2,b2)}: insufficient data")
                continue
            u, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {BOT_LABELS.get(b1,b1):20} vs {BOT_LABELS.get(b2,b2):20}: "
                  f"U={u:.0f}, p={p:.4f} {sig}")
    print()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_ante_boxplot(df: pd.DataFrame, output_dir: Path):
    if not HAS_MATPLOTLIB: return
    bots   = [b for b in BOT_ORDER if b in df["bot_type"].unique()]
    data   = [df[df["bot_type"] == b]["final_ante"].values for b in bots]
    labels = [BOT_LABELS.get(b, b) for b in bots]
    colors = [BOT_COLORS.get(b, "#888") for b in bots]

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(data, patch_artist=True, tick_labels=labels)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)

    ax.set_ylabel("Final Ante Reached")
    ax.set_title("Agent Performance: Final Ante Distribution")
    ax.axhline(y=8, color="gold", linestyle="--", alpha=0.5, label="Win condition (Ante 8)")
    ax.legend()
    plt.tight_layout()
    path = output_dir / "ante_progression.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


def plot_round_boxplot(df: pd.DataFrame, output_dir: Path):
    if not HAS_MATPLOTLIB: return
    bots   = [b for b in BOT_ORDER if b in df["bot_type"].unique()]
    data   = [df[df["bot_type"] == b]["final_round"].values for b in bots]
    labels = [BOT_LABELS.get(b, b) for b in bots]
    colors = [BOT_COLORS.get(b, "#888") for b in bots]

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(data, patch_artist=True, tick_labels=labels)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)

    ax.set_ylabel("Final Round Reached")
    ax.set_title("Agent Performance: Final Round Distribution")
    plt.tight_layout()
    path = output_dir / "round_progression.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


def plot_economy(df: pd.DataFrame, output_dir: Path):
    if not HAS_MATPLOTLIB: return
    bots    = [b for b in BOT_ORDER if b in df["bot_type"].unique()]
    labels  = [BOT_LABELS.get(b, b) for b in bots]
    metrics = ["jokers_bought","blinds_skipped","discards_used"]
    mlabels = ["Jokers Bought","Blinds Skipped","Discards Used"]

    x     = np.arange(len(bots))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (m, ml) in enumerate(zip(metrics, mlabels)):
        if m not in df.columns: continue
        means = [df[df["bot_type"] == b][m].mean() for b in bots]
        ax.bar(x + i * width, means, width, label=ml, alpha=0.8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Average per Game")
    ax.set_title("Economy Metrics by Agent")
    ax.legend()
    plt.tight_layout()
    path = output_dir / "economy_comparison.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


def plot_hand_types(df: pd.DataFrame, output_dir: Path):
    if not HAS_MATPLOTLIB: return
    if "hands_by_type" not in df.columns: return

    bots       = [b for b in BOT_ORDER if b in df["bot_type"].unique()]
    hand_types = [
        "straight_flush","four_of_a_kind","full_house","flush",
        "straight","three_of_a_kind","two_pair","pair","high_card",
    ]
    ht_colors = [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
        "#8c564b","#e377c2","#7f7f7f","#bcbd22",
    ]

    bot_ht_data = {}
    for bot in bots:
        sub    = df[df["bot_type"] == bot]
        totals = {ht: 0 for ht in hand_types}
        for _, row in sub.iterrows():
            try:
                ht_dict = json.loads(row["hands_by_type"]) if row["hands_by_type"] else {}
                for ht, count in ht_dict.items():
                    if ht in totals: totals[ht] += count
            except Exception:
                pass
        total_hands = sum(totals.values()) or 1
        bot_ht_data[bot] = {ht: totals[ht] / total_hands for ht in hand_types}

    fig, ax = plt.subplots(figsize=(11, 5))
    x       = np.arange(len(bots))
    bottoms = np.zeros(len(bots))
    labels  = [BOT_LABELS.get(b, b) for b in bots]

    for ht, color in zip(hand_types, ht_colors):
        vals = [bot_ht_data[b][ht] for b in bots]
        ax.bar(x, vals, bottom=bottoms,
               label=ht.replace("_"," ").title(), color=color, alpha=0.85)
        bottoms += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fraction of Hands Played")
    ax.set_title("Hand Type Distribution by Agent")
    ax.legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    path = output_dir / "hand_type_distribution.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# LLM cost summary — includes all three LLM bots
# ---------------------------------------------------------------------------

def llm_cost_summary(df: pd.DataFrame, output_dir: Path):
    llm_bots = ["llm_bot","rag_llm_bot","rag_meta_bot"]
    llm_df   = df[df["bot_type"].isin(llm_bots)]
    if llm_df.empty: return

    rows = []
    for bot in llm_bots:
        sub = llm_df[llm_df["bot_type"] == bot]
        if sub.empty: continue
        rows.append({
            "bot_type":                  BOT_LABELS.get(bot, bot),
            "n_games":                   len(sub),
            "avg_llm_calls":             round(sub["llm_calls"].mean(), 1),
            "avg_tokens_per_game":       round(sub["tokens_used"].mean(), 0),
            "avg_cost_per_game_usd":     round(sub["estimated_cost_usd"].mean(), 5),
            "total_cost_usd":            round(sub["estimated_cost_usd"].sum(), 4),
            "projected_100game_cost_usd":round(sub["estimated_cost_usd"].mean() * 100, 2),
        })

    cost_df = pd.DataFrame(rows)
    path    = output_dir / "llm_cost_summary.csv"
    cost_df.to_csv(path, index=False)
    print(f"Saved: {path}")
    print(cost_df.to_string(index=False))
    print()


# ---------------------------------------------------------------------------
# Qualitative analysis — loads all three decision logs, filters by clean date
# ---------------------------------------------------------------------------

def qualitative_summary(output_dir: Path):
    """
    Loads llm_decisions.jsonl, rag_decisions.jsonl, rag_meta_decisions.jsonl.
    Filters to clean runs only (by timestamp).
    Produces per-bot per-state breakdown and a top-level summary for the paper.
    """
    all_records = []

    print("Loading qualitative decision logs...")
    for bot, log_path in DECISION_LOGS.items():
        p = Path(log_path)
        if not p.exists():
            print(f"  No log found at {log_path}, skipping {bot}")
            continue

        clean_from = CLEAN_FROM.get(bot, "2026-01-01")
        count = 0
        with p.open(encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    if record.get("timestamp","") >= clean_from:
                        record["bot_type"] = bot
                        all_records.append(record)
                        count += 1
                except Exception:
                    pass
        print(f"  {BOT_LABELS.get(bot,bot)}: {count} decisions loaded (from {clean_from})")

    if not all_records:
        print("No qualitative records found.")
        return

    df = pd.DataFrame(all_records)

    # Fix parsed_action if stored as string
    def safe_get_action(x):
        if isinstance(x, dict):
            return x.get("action","unknown")
        if isinstance(x, str):
            try:
                return json.loads(x).get("action","unknown")
            except Exception:
                pass
        return "unknown"

    # ---------------------------------------------------------------------------
    # Per-bot per-state summary
    # ---------------------------------------------------------------------------
    state_rows = []
    for bot in ["llm_bot","rag_llm_bot","rag_meta_bot"]:
        if bot not in df["bot_type"].unique(): continue
        sub = df[df["bot_type"] == bot]

        for state in ["SELECTING_HAND","SHOP","BLIND_SELECT"]:
            state_sub = sub[sub["state"] == state]
            if state_sub.empty: continue

            action_counts = {}
            if state in ("SELECTING_HAND","BLIND_SELECT"):
                actions = state_sub["parsed_action"].apply(safe_get_action)
                action_counts = actions.value_counts().to_dict()

            avg_corrections = round(state_sub["self_corrections"].mean(), 2) \
                if "self_corrections" in state_sub.columns else 0.0
            parse_rate = round(state_sub["parse_success"].mean(), 3) \
                if "parse_success" in state_sub.columns else 1.0
            avg_reasoning = round(
                state_sub["reasoning"].apply(lambda x: len(str(x))).mean(), 1
            ) if "reasoning" in state_sub.columns else 0.0

            state_rows.append({
                "bot_type":             BOT_LABELS.get(bot, bot),
                "state":                state,
                "n_decisions":          len(state_sub),
                "action_distribution":  json.dumps(action_counts),
                "avg_reasoning_length": avg_reasoning,
                "avg_self_corrections": avg_corrections,
                "parse_success_rate":   parse_rate,
            })

    qual_df = pd.DataFrame(state_rows)
    path    = output_dir / "qualitative_summary.csv"
    qual_df.to_csv(path, index=False)
    print(f"\nSaved: {path}")
    print(qual_df.to_string(index=False))

    # ---------------------------------------------------------------------------
    # Top-level paper metrics summary
    # ---------------------------------------------------------------------------
    print("\n" + "="*60)
    print("QUALITATIVE METRICS FOR PAPER")
    print("="*60)

    paper_rows = []
    for bot in ["llm_bot","rag_llm_bot","rag_meta_bot"]:
        if bot not in df["bot_type"].unique(): continue
        sub = df[df["bot_type"] == bot]

        hand_sub  = sub[sub["state"] == "SELECTING_HAND"]
        blind_sub = sub[sub["state"] == "BLIND_SELECT"]
        shop_sub  = sub[sub["state"] == "SHOP"]

        # Parse success
        parse_ok    = hand_sub["parse_success"].sum() if "parse_success" in hand_sub.columns else len(hand_sub)
        parse_total = len(hand_sub)

        # Self corrections total
        total_corrections = int(sub["self_corrections"].sum()) if "self_corrections" in sub.columns else 0

        # Ante 1 blind skips
        ante1_skips = int(blind_sub[
            (blind_sub["parsed_action"].apply(safe_get_action) == "skip") &
            (blind_sub["ante"].apply(lambda x: int(x) if str(x).isdigit() else 0) <= 1)
        ].shape[0]) if not blind_sub.empty else 0

        # Hand recognition accuracy (played best available hand)
        if "best_available" in hand_sub.columns and "action_chosen" in hand_sub.columns:
            play_sub = hand_sub[hand_sub["action_chosen"] == "play"]
            if not play_sub.empty and "available_hands" in play_sub.columns:
                hits = 0
                for _, row in play_sub.iterrows():
                    avail = row.get("available_hands")
                    if isinstance(avail, list) and avail:
                        best = avail[0]
                    elif isinstance(avail, str):
                        try:
                            avail_list = json.loads(avail)
                            best = avail_list[0] if avail_list else None
                        except Exception:
                            best = None
                    else:
                        best = None
                    if best and row.get("best_available") == best:
                        hits += 1
                hand_recog = f"{hits}/{len(play_sub)}"
            else:
                hand_recog = "n/a"
        else:
            hand_recog = "n/a"

        # Play vs discard ratio
        if not hand_sub.empty:
            play_count    = int((hand_sub["parsed_action"].apply(safe_get_action) == "play").sum())
            discard_count = int((hand_sub["parsed_action"].apply(safe_get_action) == "discard").sum())
            ratio = f"{play_count} play / {discard_count} discard"
        else:
            ratio = "n/a"

        # Boss adaptations
        boss_adaptations = int(sub["boss_adaptations"].sum()) if "boss_adaptations" in sub.columns else 0

        row = {
            "agent":                 BOT_LABELS.get(bot, bot),
            "total_decisions":       len(sub),
            "hand_parse_success":    f"{int(parse_ok)}/{parse_total} ({int(parse_ok)/max(parse_total,1)*100:.1f}%)",
            "total_self_corrections":total_corrections,
            "ante1_blind_skips":     ante1_skips,
            "hand_recog_accuracy":   hand_recog,
            "play_discard_ratio":    ratio,
            "boss_adaptations":      boss_adaptations,
        }
        paper_rows.append(row)

        print(f"\n{BOT_LABELS.get(bot,bot)}:")
        for k, v in row.items():
            if k != "agent":
                print(f"  {k:<28} {v}")

    paper_df = pd.DataFrame(paper_rows)
    path2    = output_dir / "qualitative_paper_metrics.csv"
    paper_df.to_csv(path2, index=False)
    print(f"\nSaved: {path2}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Balatro bot results")
    parser.add_argument("--results", default="results_final.csv")
    parser.add_argument("--output",  default="figures")
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

    print("Loading qualitative decision logs...")
    qualitative_summary(output_dir)

    print("Analysis complete.")