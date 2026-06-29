# Can an LLM Play Balatro Without Ever Practicing?

**Benchmarking agent architectures on long-horizon, sparse-reward decision problems**

> University of Toronto · Supervised Research · Target venue: IEEE Conference on Games 2027  
> Supervised by Prof. Patrick Hosein (TTLab, University of the West Indies)

---

## Overview

This project uses **Balatro** — a poker-based roguelike card game — as a testbed to compare how different AI paradigms handle long-horizon strategic decisions with sparse, delayed rewards. The central research question is whether a zero-shot large language model can outperform a trained reinforcement learning agent on a task it has never explicitly practiced.

Five agents are evaluated under controlled conditions (identical seeds, deck, stake):

| Agent | Paradigm | Description |
|---|---|---|
| **FlushBot** | Rule-based (floor) | Always plays the best flush available, never buys anything. Absolute baseline. |
| **MetaBot** | Rule-based (meta) | Tiered joker priority, dominant hand tracking, interest-aware economy, conservative blind skipping. |
| **LLMBot** | Zero-shot LLM | Formats game state as a structured prompt, calls Claude Sonnet 4.6, parses JSON response. No strategy hints — pure reasoning from scratch. |
| **RAGLLMBot** | LLM + RAG | Same as LLMBot but retrieves relevant rules from a Pinecone vector index before each decision. Simulates a player who read the rulebook. |
| **RLBot** | Reinforcement Learning | MaskablePPO (Stable-Baselines3) trained on a custom mock environment, evaluated on the real game via HTTP API. |

---

## Key Findings (Preliminary)

| Agent | Avg Final Ante | Avg Final Round | Notes |
|---|---|---|---|
| FlushBot | 1.03 | 1.6 | Floor baseline |
| MetaBot | ~2.2 | ~5.2 | Ceiling — full heuristic |
| RAGLLMBot | ~1.75 | ~2.75 | Rules context boosts LLM reasoning |
| LLMBot | ~1.50 | ~3.75 | Zero-shot surprisingly competitive |
| RLBot | ~1.40 | ~3.20 | Limited by sim-to-real gap |

**Core finding:** Zero-shot Claude Sonnet 4.6 outperforms a 120M-step MaskablePPO agent. The RL agent's failure is traceable to two causes: a persistent sim-to-real reward signal gap (cash hoarding learned in mock env) and the fundamental limitation of a memoryless MLP policy for long-horizon game strategy.

Full results tracked at [wandb.ai/vinayakcpa-university-of-toronto/balatro-research](https://wandb.ai/vinayakcpa-university-of-toronto/balatro-research).

---

## Technical Stack

- **Language:** Python 3.10
- **RL:** MaskablePPO via `stable-baselines3` + `sb3-contrib`, GPU training (CUDA)
- **LLM:** Anthropic Claude Sonnet 4.6 (`claude-sonnet-4-6`)
- **Vector store:** Pinecone (`balatro-rules` index, 26 rule vectors, `all-MiniLM-L6-v2` embeddings)
- **Game API:** [coder/balatrobot](https://github.com/coder/balatrobot) HTTP JSON-RPC mod (v1.5.0, port 12346)
- **Experiment tracking:** Weights & Biases
- **Analysis:** pandas, scipy, matplotlib

---

## Project Structure

```
balatro_research/
├── base_bot.py            # Base class: game loop, CSV logging, W&B tracking
├── balatro_client.py      # HTTP JSON-RPC 2.0 client for Balatrobot API
├── heuristic_bots.py      # FlushBot + MetaBot implementations
├── llm_bot.py             # LLMBot — zero-shot Claude Sonnet 4.6
├── rag_pipeline.py        # RAGLLMBot — LLM + Pinecone retrieval
├── rl_bot.py              # RLBot — MaskablePPO policy, diagnostic tracker
├── balatro_mock_env.py    # Custom mock Gymnasium env (OBS_DIM=148, 120M step training)
├── balatro_env.py         # Real-game Gymnasium env (wraps live API)
├── run_all.py             # Master experiment runner
├── analyze_results.py     # Statistical analysis + figure generation
├── data/
│   ├── rules.json         # 26 official Balatro rules (RAG corpus)
│   └── jokers.json        # 150 joker descriptions
├── results_final.csv      # Canonical experiment results
├── llm_decisions.jsonl    # Per-decision LLM reasoning log (qualitative analysis)
├── rag_decisions.jsonl    # Per-decision RAG-LLM reasoning log
├── diagnostics.json       # RLBot per-game death reason classification
└── rl_model/
    └── ppo_balatro.zip    # Trained MaskablePPO model (120M steps, 16 envs)
```

---

## Setup

### Requirements

- Python 3.10
- Balatro (Steam) with [coder/balatrobot mod](https://github.com/coder/balatrobot) v1.5.0 installed
- CUDA-capable GPU (recommended for RL training)
- Anthropic API key (for LLMBot/RAGLLMBot)
- Pinecone API key (for RAGLLMBot)

### Install dependencies

```bash
pip install -r requirements.txt --break-system-packages
```

### Set API keys (Windows PowerShell — persists across sessions)

```powershell
[System.Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY", "your_key", "User")
[System.Environment]::SetEnvironmentVariable("PINECONE_API_KEY", "your_key", "User")
```

### Start Balatro

Launch Balatro with the balatrobot mod loaded. The HTTP API runs on `127.0.0.1:12346` by default.

---

## Running Experiments

### Run all agents (sequential)

```bash
# Full run — 100 seeds each heuristic, 50 seeds each LLM/RAG/RL
python run_all.py --mode full --results results_final.csv

# Quick test — 5 seeds per bot to verify everything works
python run_all.py --mode test --results results_test.csv
```

### Run individual agents

```bash
# Heuristic bots
python heuristic_bots.py --bot flush --results results_final.csv
python heuristic_bots.py --bot meta  --results results_final.csv

# LLM / RAG
python llm_bot.py      --seeds SEED001 SEED002 ... SEED050 --results results_final.csv
python rag_pipeline.py --seeds SEED001 SEED002 ... SEED050 --results results_final.csv

# RL bot (requires trained model)
python rl_bot.py --run --results results_final.csv
```

### Train the RL agent from scratch

```bash
# Mock environment training — ~18 hours on GPU, 120M steps, 16 parallel envs
python rl_bot.py --train --mock --timesteps 120000000

# Quick smoke test (500k steps)
python rl_bot.py --train --mock --timesteps 500000
```

### Build the RAG index (one-time setup)

```bash
python rag_pipeline.py --build-index
```

### Analyze results

```bash
python analyze_results.py --results results_final.csv --output figures/
```

Produces:
- `figures/summary_table.csv` — mean ± std per agent for all metrics
- `figures/ante_progression.png` — boxplot of final ante by agent
- `figures/economy_comparison.png` — jokers bought, blinds skipped, discards used
- `figures/hand_type_distribution.png` — hand type breakdown per agent
- `figures/llm_cost_summary.csv` — token and cost breakdown for LLM agents
- `figures/qualitative_summary.csv` — LLM reasoning patterns from decision logs

---

## Experiment Protocol

| Parameter | Value |
|---|---|
| Seeds | `SEED001`–`SEED100` (heuristic), `SEED001`–`SEED050` (LLM/RL) |
| Deck | Red Deck (+1 discard per round) |
| Stake | White (baseline difficulty) |
| Runs per seed | 1 |
| LLM model | `claude-sonnet-4-6` |
| RL algorithm | MaskablePPO, net_arch=[256,256,128] |
| RL training steps | 120,000,000 |
| RL parallel envs | 16 |
| Obs dimension | 148 |
| Curriculum | Ante 1→2→3→4→5→6 unlocked at 0/40M/60M/80M/100M/120M steps |

---

## Design Decisions

### Why a mock environment for RL?

The live Balatrobot API runs at ~1 it/s over HTTP. Training 120M steps on the real environment would take ~33 days. The mock environment (`balatro_mock_env.py`) approximates core mechanics — hand scoring, joker effects, interest system, boss blind debuffs, curriculum learning — enabling GPU-accelerated training at ~1,900 it/s.

The sim-to-real transfer gap is itself a key research finding: the agent learned to hoard cash for interest income in the mock env, a behavior that transfers poorly to the real game where early joker purchases compound multiplicatively.

### Why zero-shot (no few-shot examples)?

The research question is whether LLMs can reason about novel game mechanics from scratch. Few-shot examples would constitute implicit strategy transfer. The LLMBot prompt contains only mechanical game state — current hand, chips needed, joker descriptions, available actions — with no strategy hints or example decisions.

### Why Pinecone over a local vector store?

Practical demonstration of production RAG infrastructure. ChromaDB would work equally well technically.

### RLBot shop policy

After 120M training steps the RL agent's shop policy failed to converge — it learned to hoard cash for interest income rather than buy jokers, a reward signal failure in the mock environment. The final RLBot uses MetaBot's rule-based shop policy to isolate the RL contribution to hand selection and blind decisions only. This creates a clean ablation: MetaBot (rule-based everything) vs RLBot (learned hand selection, rule-based shop).

---

## Qualitative LLM Analysis

Every LLM and RAG decision is logged to `llm_decisions.jsonl` and `rag_decisions.jsonl` with:

- Full prompt and raw response
- Parsed action and reasoning text
- Parse success/failure and retry flag
- Input/output token counts
- Available hands vs hands chosen (hand recognition accuracy)
- Self-correction count (how often Claude said "wait" or "actually" mid-reasoning)
- Boss blind adaptation flag
- Blind skip tracking (ante 1 skips specifically)

Filter to clean runs: `timestamp >= "2026-06-29"`.

---

## Citation

```bibtex
@inproceedings{maharaj2027balatro,
  title     = {Can an LLM Play Balatro Without Ever Practicing?},
  author    = {Maharaj, Vinayak},
  booktitle = {Proceedings of the IEEE Conference on Games},
  year      = {2027}
}
```

---

## Author

**Vinayak Maharaj** · [vinayakmaharaj.dev](https://vinayakmaharaj.dev) · University of Toronto (BSc CS + Statistics, 2025)

Research at TTLab under Prof. Patrick Hosein · Target: IEEE CoG 2027, AAAI 2027