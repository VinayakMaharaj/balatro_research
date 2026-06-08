# Can an LLM Play Balatro Without Ever Practicing?

Benchmarking agent architectures on long-horizon sparse-reward decision problems using Balatro as a testbed.

**University of Toronto** | Supervised research project | Target venue: IEEE Conference on Games 2027

---

## Overview

This project compares four agent architectures on the roguelike deckbuilder Balatro:

| Agent | Description |
|-------|-------------|
| `FlushBot` | Heuristic baseline — always plays flushes, never buys |
| `MetaBot` | Meta-informed heuristic — tiered joker priority, blind skipping, flush fishing |
| `LLMBot` | Zero-shot Claude (Haiku) — formats game state as prompt, parses JSON response |
| `RAGLLMBot` | LLM + Pinecone RAG — retrieves relevant Balatro rules before each decision |
| `RLBot` | PPO (Stable-Baselines3) — trained on mock env, evaluated on real game |

---

## Project Structure

```
balatro_research/
├── balatro_client.py      # HTTP JSON-RPC 2.0 client for Balatrobot API
├── base_bot.py            # Base class: game loop, CSV logging, W&B tracking
├── heuristic_bots.py      # FlushBot + MetaBot
├── llm_bot.py             # LLMBot (zero-shot Claude)
├── rag_pipeline.py        # RAGLLMBot + Pinecone index builder
├── balatro_env.py         # Gymnasium environment (real game)
├── balatro_mock_env.py    # Gymnasium environment (fast simulation)
├── rl_bot.py              # RLBot (PPO via Stable-Baselines3)
├── run_all.py             # Master experiment runner
├── analyze_results.py     # Statistical analysis + figure generation
├── data/
│   └── jokers.json        # 150 jokers with descriptions
├── results.csv            # Experiment results (appended per run)
├── llm_decisions.jsonl    # Per-decision LLM qualitative log
├── rag_decisions.jsonl    # Per-decision RAG-LLM qualitative log
├── rl_model/
│   └── ppo_balatro.zip    # Trained PPO model
└── figures/               # Generated plots and tables
```

---

## Setup

**Requirements:** Python 3.10, Balatro with [Balatrobot mod](https://github.com/coder/balatrobot) (v1.5.0)

```bash
pip install -r requirements.txt --break-system-packages
```

**Environment variables:**
```powershell
$env:ANTHROPIC_API_KEY="your_key"
$env:PINECONE_API_KEY="your_key"
```

---

## Running Experiments

### 1. Train the RL agent (do this first)
```bash
# Fast mock training (~25 min, recommended)
python rl_bot.py --train --mock --timesteps 500000

# Real env training (very slow, ~1 it/s)
python rl_bot.py --train --timesteps 1000
```

### 2. Run all bots

```bash
# Pre-meeting run (~2.5 hours, skips rag_llm_bot)
python run_all.py --mode meeting --analyze

# Full 100-game run per bot (overnight)
python run_all.py --mode full --analyze

# Custom
python run_all.py --flush-runs 20 --meta-runs 20 --llm-runs 10 --rag-runs 3 --rl-runs 20 --analyze

# Quick test (1 run per seed, all bots)
python run_all.py --mode test
```

### 3. Run individual bots
```bash
python heuristic_bots.py --bot flush --runs-per-seed 20
python heuristic_bots.py --bot meta  --runs-per-seed 20
python llm_bot.py        --runs-per-seed 5
python rag_pipeline.py   --runs-per-seed 3
python rl_bot.py         --run --runs-per-seed 20
```

### 4. Build RAG index (one-time)
```bash
python rag_pipeline.py --build-index
```

### 5. Analyze results
```bash
python analyze_results.py --results results.csv --output figures/
```

Produces:
- `figures/summary_table.csv` — mean ± std per bot for all metrics
- `figures/ante_progression.png` — boxplot of final ante by agent
- `figures/round_progression.png` — boxplot of final round by agent
- `figures/economy_comparison.png` — jokers bought, blinds skipped, discards
- `figures/hand_type_distribution.png` — hand type breakdown per agent
- `figures/llm_cost_summary.csv` — token and cost breakdown
- `figures/qualitative_summary.csv` — LLM reasoning patterns

---

## Experiment Protocol

- **Seeds:** `AAAAAAA`, `BBBBBBB`, `CCCCCCC`, `DDDDDDD`, `EEEEEEE` (fixed across all agents)
- **Target:** 20 runs × 5 seeds = 100 games per agent
- **Deck:** Red Deck (standard, +1 discard per round)
- **Stake:** White (baseline difficulty)
- **LLM:** Claude Haiku (development) → Claude Sonnet (final paper runs)
- **RL training:** PPO, 500k steps on mock env, MlpPolicy
- **Tracking:** Weights & Biases (`balatro-research` project)

---

## Key Design Decisions

**Why mock env for RL training?**
The live Balatrobot API runs at ~1 it/s via HTTP. Training 500k steps on the real env would take ~140 hours. The mock environment approximates core mechanics (hand level scaling, tiered joker effects, interest, boss debuffs) and allows sim-to-real transfer — a standard approach in game AI research.

**Why Haiku for development?**
Cost. Haiku is ~10x cheaper than Sonnet. Development runs use Haiku; final paper runs switch to `claude-sonnet-4-6` for maximum reasoning quality.

**Why Pinecone over local vector store?**
Resume/recruiter visibility. ChromaDB is fine technically but Pinecone is industry-standard.

---

## Results

Tracked at: [wandb.ai/vinayakcpa-university-of-toronto/balatro-research](https://wandb.ai/vinayakcpa-university-of-toronto/balatro-research)

| Agent | Avg Final Ante | Avg Final Round | Win Rate |
|-------|---------------|-----------------|----------|
| FlushBot | 1.0 | 1.4 | 0% |
| MetaBot | 1.8 | 3.8 | 0% |
| LLMBot | 1.2 | 2.2 | 0% |
| RAGLLMBot | 1.0 | 1.6 | 0% |
| RLBot | TBD | TBD | TBD |

*Preliminary results from 5-seed baseline. Full 100-game results pending.*

---

## Citation

```bibtex
@inproceedings{maharaj2027balatro,
  title={Can an LLM Play Balatro Without Ever Practicing?},
  author={Maharaj, Vinayak},
  booktitle={IEEE Conference on Games},
  year={2027}
}
```