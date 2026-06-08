"""
run_all.py
Master experiment runner — runs all bots sequentially with configurable
runs per bot. Designed for the 3-hour pre-meeting window.

Usage:
    # Full 100-game run (overnight)
    python run_all.py --mode full

    # Pre-meeting run (~2.5 hours, skips rag_llm_bot)
    python run_all.py --mode meeting

    # Quick test (1 run per seed, all bots)
    python run_all.py --mode test

    # Custom
    python run_all.py --flush-runs 20 --meta-runs 20 --llm-runs 5 --rag-runs 0 --rl-runs 20

    # Skip specific bots
    python run_all.py --mode meeting --skip rag_llm_bot llm_bot
"""

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SEEDS = [f"SEED{str(i).zfill(3)}" for i in range(1, 101)]
RESULTS_FILE = "results.csv"
PYTHON = sys.executable

# ---------------------------------------------------------------------------
# Run profiles
# ---------------------------------------------------------------------------

PROFILES = {
    "full": {
        # 100 games each = 20 runs x 5 seeds
        "flush_bot":   20,
        "meta_bot":    20,
        "llm_bot":     20,
        "rag_llm_bot": 20,
        "rl_bot":      20,
    },
    "meeting": {
        # ~2.5 hours: skip rag, reduce llm
        "flush_bot":   20,
        "meta_bot":    20,
        "llm_bot":     5,
        "rag_llm_bot": 0,
        "rl_bot":      20,
    },
    "test": {
        # 1 run per seed per bot — just verify everything works
        "flush_bot":   1,
        "meta_bot":    1,
        "llm_bot":     1,
        "rag_llm_bot": 1,
        "rl_bot":      1,
    },
}

# Rough time estimates per game (seconds)
TIME_ESTIMATES = {
    "flush_bot":   20,
    "meta_bot":    25,
    "llm_bot":     90,
    "rag_llm_bot": 120,
    "rl_bot":      25,
}


def estimate_time(runs: dict) -> str:
    total_seconds = sum(
        runs.get(bot, 0) * len(SEEDS) * TIME_ESTIMATES[bot]
        for bot in TIME_ESTIMATES
    )
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    return f"~{hours}h {minutes}m"


# ---------------------------------------------------------------------------
# Bot runners
# ---------------------------------------------------------------------------

def run_heuristic(bot: str, runs: int, results: str, port: int) -> bool:
    if runs == 0:
        logger.info(f"Skipping {bot} (0 runs)")
        return True

    bot_flag = "flush" if bot == "flush_bot" else "meta"
    cmd = [
        PYTHON, "heuristic_bots.py",
        "--bot", bot_flag,
        "--seeds", *SEEDS,
        "--runs-per-seed", str(runs),
        "--results", results,
        "--port", str(port),
    ]
    return _run_cmd(cmd, bot, runs)


def run_llm(runs: int, results: str, port: int) -> bool:
    if runs == 0:
        logger.info("Skipping llm_bot (0 runs)")
        return True

    cmd = [
        PYTHON, "llm_bot.py",
        "--seeds", *SEEDS,
        "--runs-per-seed", str(runs),
        "--results", results,
        "--port", str(port),
    ]
    return _run_cmd(cmd, "llm_bot", runs)


def run_rag(runs: int, results: str, port: int) -> bool:
    if runs == 0:
        logger.info("Skipping rag_llm_bot (0 runs)")
        return True

    cmd = [
        PYTHON, "rag_pipeline.py",
        "--seeds", *SEEDS,
        "--runs-per-seed", str(runs),
        "--results", results,
        "--port", str(port),
    ]
    return _run_cmd(cmd, "rag_llm_bot", runs)


def run_rl(runs: int, results: str, port: int, model_path: str) -> bool:
    if runs == 0:
        logger.info("Skipping rl_bot (0 runs)")
        return True

    model_zip = Path(f"{model_path}.zip")
    if not model_zip.exists():
        logger.error(f"No trained RL model at {model_zip}. Run training first:")
        logger.error("  python rl_bot.py --train --mock --timesteps 500000")
        return False

    cmd = [
        PYTHON, "rl_bot.py",
        "--run",
        "--seeds", *SEEDS,
        "--runs-per-seed", str(runs),
        "--results", results,
        "--port", str(port),
        "--model-path", model_path,
    ]
    return _run_cmd(cmd, "rl_bot", runs)


def _run_cmd(cmd: list, bot_name: str, runs: int) -> bool:
    n_games = runs * len(SEEDS)
    logger.info(f"{'='*60}")
    logger.info(f"Starting {bot_name}: {n_games} games ({runs} runs x {len(SEEDS)} seeds)")
    logger.info(f"Estimated time: {TIME_ESTIMATES.get(bot_name, 30) * n_games // 60} min")
    logger.info(f"{'='*60}")

    start = time.time()
    try:
        result = subprocess.run(cmd, check=True)
        elapsed = round(time.time() - start, 1)
        logger.info(f"{bot_name} complete in {elapsed}s")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"{bot_name} failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        logger.warning(f"{bot_name} interrupted by user")
        return False


# ---------------------------------------------------------------------------
# Results cleanup
# ---------------------------------------------------------------------------

def clean_results_csv(results_path: str):
    """Remove duplicate headers from results.csv."""
    p = Path(results_path)
    if not p.exists():
        return

    with p.open("r") as f:
        lines = f.readlines()

    if not lines:
        return

    header = lines[0]
    cleaned = [header] + [l for l in lines[1:] if l.strip() != header.strip()]

    if len(cleaned) < len(lines):
        with p.open("w") as f:
            f.writelines(cleaned)
        logger.info(f"Cleaned {len(lines) - len(cleaned)} duplicate headers from {results_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all Balatro bots")
    parser.add_argument("--mode", choices=["full", "meeting", "test"],
                        help="Preset run profile")
    parser.add_argument("--flush-runs",  type=int, help="Runs per seed for flush_bot")
    parser.add_argument("--meta-runs",   type=int, help="Runs per seed for meta_bot")
    parser.add_argument("--llm-runs",    type=int, help="Runs per seed for llm_bot")
    parser.add_argument("--rag-runs",    type=int, help="Runs per seed for rag_llm_bot")
    parser.add_argument("--rl-runs",     type=int, help="Runs per seed for rl_bot")
    parser.add_argument("--skip",        nargs="+", default=[],
                        choices=["flush_bot", "meta_bot", "llm_bot", "rag_llm_bot", "rl_bot"],
                        help="Bots to skip")
    parser.add_argument("--results",     default=RESULTS_FILE)
    parser.add_argument("--port",        type=int, default=12346)
    parser.add_argument("--model-path",  default="rl_model/ppo_balatro")
    parser.add_argument("--analyze",     action="store_true",
                        help="Run analyze_results.py after all bots finish")
    args = parser.parse_args()

    # Build runs dict from mode or individual args
    if args.mode:
        runs = dict(PROFILES[args.mode])
    else:
        runs = dict(PROFILES["meeting"])  # default

    # Override with individual args
    if args.flush_runs is not None: runs["flush_bot"]   = args.flush_runs
    if args.meta_runs  is not None: runs["meta_bot"]    = args.meta_runs
    if args.llm_runs   is not None: runs["llm_bot"]     = args.llm_runs
    if args.rag_runs   is not None: runs["rag_llm_bot"] = args.rag_runs
    if args.rl_runs    is not None: runs["rl_bot"]      = args.rl_runs

    # Apply skips
    for bot in args.skip:
        runs[bot] = 0

    # Print plan
    print(f"\n{'='*60}")
    print(f"BALATRO EXPERIMENT RUN — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")
    for bot, n in runs.items():
        if n > 0:
            games = n * len(SEEDS)
            print(f"  {bot:20s}: {n} runs x {len(SEEDS)} seeds = {games} games")
        else:
            print(f"  {bot:20s}: SKIPPED")
    print(f"\n  Estimated total time: {estimate_time(runs)}")
    print(f"  Results file: {args.results}")
    print(f"{'='*60}\n")

    # Clean existing CSV
    clean_results_csv(args.results)

    # Run bots
    overall_start = time.time()
    results_log = {}

    if runs["flush_bot"] > 0 and "flush_bot" not in args.skip:
        results_log["flush_bot"] = run_heuristic("flush_bot", runs["flush_bot"], args.results, args.port)

    if runs["meta_bot"] > 0 and "meta_bot" not in args.skip:
        results_log["meta_bot"] = run_heuristic("meta_bot", runs["meta_bot"], args.results, args.port)

    if runs["llm_bot"] > 0 and "llm_bot" not in args.skip:
        results_log["llm_bot"] = run_llm(runs["llm_bot"], args.results, args.port)

    if runs["rag_llm_bot"] > 0 and "rag_llm_bot" not in args.skip:
        results_log["rag_llm_bot"] = run_rag(runs["rag_llm_bot"], args.results, args.port)

    if runs["rl_bot"] > 0 and "rl_bot" not in args.skip:
        results_log["rl_bot"] = run_rl(runs["rl_bot"], args.results, args.port, args.model_path)

    # Summary
    total_elapsed = round(time.time() - overall_start, 1)
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE — {total_elapsed}s elapsed")
    print(f"{'='*60}")
    for bot, success in results_log.items():
        status = "✓" if success else "✗ FAILED"
        print(f"  {bot:20s}: {status}")

    # Run analysis
    if args.analyze:
        print(f"\nRunning analysis...")
        subprocess.run([PYTHON, "analyze_results.py", "--results", args.results], check=False)

    print(f"\nResults saved to: {args.results}")
    print(f"Run analysis: python analyze_results.py --results {args.results}")