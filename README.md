# Can Generative AI be used to win at Balatro?

**Comparing retrieval-augmented and zero-shot LLM agents against a hand-coded heuristic on a long-horizon sequential decision problem**

> Vinayak Maharaj, University of Toronto  
> Prof. Patrick Hosein, TTLab, University of the West Indies  
> Accepted at IEEE ICTMOD 2026

---

## Overview

This repository contains the full experimental code for the paper *"Can Generative AI be used to win at Balatro?"*, which uses **Balatro** — a poker-based roguelike card game — as a testbed for evaluating how different AI approaches handle long-horizon decision-making under uncertainty and sparse rewards.

The central research question: given identical domain knowledge, does an LLM augmented with retrieval outperform the same LLM operating zero-shot, and does either outperform a deterministic system built from that same knowledge?

---

## Key Results

347 valid games across five agents.

| Agent | n | Avg Final Ante | SD | Avg Final Round | SD | Win Rate |
|---|---|---|---|---|---|---|
| **RAGBot (strategy)** | 48 | **2.31** | 0.97 | **5.94** | 2.39 | 0% |
| LLMBot | 50 | 2.08 | 0.94 | 4.94 | 2.59 | 0% |
| MetaBot | 99 | 2.06 | 0.78 | 5.02 | 1.90 | 0% |
| RAGBot (rules) | 50 | 1.88 | 0.92 | 3.16 | 1.75 | 0% |
| FlushBot | 100 | 1.14 | 0.35 | 2.35 | 0.96 | 0% |

**Finding 1:** RAGBot (strategy) achieves the highest average final Ante of any agent (2.31) and significantly outperforms MetaBot on final Round (p=0.041, uncorrected), though the Ante-level advantage does not reach significance at this sample size (p=0.148). This is evidence that LLM reasoning over expert strategy knowledge can outperform deterministic execution of the same knowledge, strongest on the finer-grained Round metric.

**Finding 2:** Zero-shot LLMBot matches MetaBot on both metrics, with no significant difference on Ante (p=0.861) or Round (p=0.456) — a clean null result showing Claude Sonnet 4.6's pretraining already encodes sufficient Balatro domain knowledge to compete with a hand-coded heuristic built from the same sources.

**Finding 3 (negative result):** Rules-only RAG measurably underperforms zero-shot LLM on final Round (p=0.0002, survives Bonferroni correction), though the Ante-level difference is not significant (p=0.231). Retrieved mechanics add noise rather than signal, since Claude already encodes Balatro's rules from pretraining. This is supported qualitatively: RAGBot (rules) self-corrects 0.95 times per hand-selection decision (vs 0.44 for RAGBot strategy, 0.52 for LLMBot) and skips the Ante-1 Small/Big Blind 78 times (vs 28 for LLMBot and 0 for RAGBot strategy) — a policy neither MetaBot nor RAGBot (strategy) permits at all.

**Overall:** none of the five agents came close to winning (0% win rate across all 347 games). The differences reported above are differences in how far each agent gets, not in whether the run succeeds.

---

## Agents

| Agent | File | Paradigm |
|---|---|---|
| FlushBot | `heuristic_bots.py` | Rule-based floor — always plays best flush, never buys |
| MetaBot | `heuristic_bots.py` | Hand-coded heuristic with tiered joker priority, dominant hand tracking, economy rules |
| LLMBot | `llm_bot.py` | Zero-shot Claude Sonnet 4.6, mechanical game state prompt only |
| RAGBot (rules) | `rag_pipeline.py` | LLM + Pinecone rules corpus (27 vectors, official game mechanics) |
| RAGBot (strategy) | `rag_meta_pipeline.py` | LLM + Pinecone strategy corpus mirroring MetaBot's decision logic |

---

## Technical Stack

- **Language:** Python 3.10
- **Game interface:** [coder/balatrobot](https://github.com/coder/balatrobot) HTTP JSON-RPC mod (v1.5.0)
- **LLM:** Anthropic Claude Sonnet 4.6 (`claude-sonnet-4-6`)
- **Vector store:** Pinecone (`balatro-rules` and `balatro-strategy` indexes, `all-MiniLM-L6-v2` embeddings)
- **Experiment tracking:** Weights & Biases
- **Analysis:** pandas, scipy, matplotlib

---

## Repository Structure
balatro_research/
├── base_bot.py # Base class: game loop, CSV logging, W&B tracking
├── balatro_client.py # HTTP JSON-RPC client for balatrobot API
├── heuristic_bots.py # FlushBot and MetaBot
├── llm_bot.py # LLMBot — zero-shot Claude Sonnet 4.6
├── rag_pipeline.py # RAGBot (rules) — LLM + rules-only Pinecone index
├── rag_meta_pipeline.py # RAGBot (strategy) — LLM + strategy Pinecone index
├── analyze_results.py # Statistical analysis and figure generation
├── plot_figure.py # IEEE-formatted publication figures (PDF + SVG)
├── data/
│ ├── rules.json # Official game rules (RAG corpus)
│ └── strategy.json # Strategy chunks mirroring MetaBot's logic
├── results_final.csv # Full experimental results (347 valid games)
├── llm_decisions.jsonl # Per-decision LLM reasoning log
├── rag_decisions.jsonl # Per-decision RAGBot (rules) reasoning log
├── rag_meta_decisions.jsonl # Per-decision RAGBot (strategy) reasoning log
└── figures/
├── figure1_performance.pdf # Box-and-whisker: final ante by agent
├── figure2_qualitative.pdf # Decision quality metrics: LLM agents
└── summary_table.csv # Mean ± std per agent for all metrics


---

## Setup

### Requirements

- Python 3.10+
- Balatro (Steam) with [coder/balatrobot](https://github.com/coder/balatrobot) mod v1.5.0
- Anthropic API key
- Pinecone API key (for RAG agents)

### Install

```bash
pip install -r requirements.txt
```

### Set API keys (Windows — persists across sessions)

```powershell
[System.Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY", "your_key", "User")
[System.Environment]::SetEnvironmentVariable("PINECONE_API_KEY", "your_key", "User")
```

### Build Pinecone indexes (one-time)

```bash
# Rules index for RAGBot (rules)
python rag_pipeline.py --build-index

# Strategy index for RAGBot (strategy)
python rag_meta_pipeline.py --build-index
```

---

## Running Experiments

### Heuristic agents (no API required)

```bash
python heuristic_bots.py --bot flush --results results_final.csv
python heuristic_bots.py --bot meta  --results results_final.csv
```

### LLM agents

```bash
# Zero-shot
python llm_bot.py --seeds SEED001 ... SEED050 --results results_final.csv

# RAG (rules only)
python rag_pipeline.py --seeds SEED001 ... SEED050 --results results_final.csv

# RAG (strategy)
python rag_meta_pipeline.py --seeds SEED001 ... SEED050 --results results_final.csv
```

PowerShell seed expansion:

```powershell
$seeds = 1..50 | ForEach-Object { "SEED{0:D3}" -f $_ }
python llm_bot.py --seeds $seeds --results results_final.csv
```

---

## Reproducing Paper Results

```bash
# 1. Run all agents (assumes Balatro is running with balatrobot mod)
python heuristic_bots.py --bot flush --results results_final.csv
python heuristic_bots.py --bot meta  --results results_final.csv

$seeds = 1..50 | ForEach-Object { "SEED{0:D3}" -f $_ }
python llm_bot.py            --seeds $seeds --results results_final.csv
python rag_pipeline.py       --seeds $seeds --results results_final.csv
python rag_meta_pipeline.py  --seeds $seeds --results results_final.csv

# 2. Analyse results
python analyze_results.py --results results_final.csv --output figures/

# 3. Generate publication figures
python plot_figure.py --results results_final.csv --output figures/
```

All agents use **Red Deck, White Stake**. Heuristic agents ran on 100 seeds (99 valid for MetaBot); LLM-driven agents ran on 50 seeds each (48 valid for RAGBot strategy), reflecting the higher per-game API cost of LLM inference relative to deterministic code execution.

---

## Experiment Protocol

| Parameter | Value |
|---|---|
| Seeds (heuristic) | SEED001–SEED100 |
| Seeds (LLM/RAG) | SEED001–SEED050 |
| Deck | Red Deck (+1 discard per round) |
| Stake | White (baseline difficulty) |
| Runs per seed | 1 |
| LLM model | `claude-sonnet-4-6` |
| Embedding model | `all-MiniLM-L6-v2` |
| RAG top-k | 5 |
| Rules index vectors | 27 |

---

## Design Notes

### Why Balatro?

Balatro requires multi-step planning across 3 blinds per ante, economy management across shops, and adaptation to 23 distinct Boss Blind effects. It is stochastic (no fixed sequence of moves to memorize), has sparse terminal rewards, and forces long-horizon tradeoffs — a Joker bought at Ante 1 can determine whether Ante 6 is survivable.

### Why zero-shot (no few-shot examples)?

The research question is whether LLMs can reason about novel game mechanics from scratch. Adding few-shot examples would constitute implicit strategy transfer and confound the comparison with MetaBot.

### RAGBot (rules) negative result

Rules-only retrieval underperforms zero-shot LLM on final Round because Claude's pretraining data includes Balatro wikis, Reddit posts, and strategy guides — the rules corpus adds redundant context that interferes with the model's own reasoning rather than supplying new information. This is evidenced by significantly higher self-correction rates (0.95 vs 0.44 per hand) and 78 Ante-1 blind skips vs 0 for RAGBot (strategy). Self-corrections concentrate in hand-composition arithmetic (the model losing count of its own hand mid-reasoning) rather than genuine rule misapplication.

### Strategy corpus design

`data/strategy.json` encodes exactly the decision logic implemented in MetaBot's code — joker tier priorities, interest floor thresholds, dominant hand tracking, blind skip rules, and boss blind adaptations — as retrievable text chunks. Nothing in the corpus exceeds what MetaBot's code executes, ensuring a fair knowledge-matched comparison.

### Why not reinforcement learning

A MaskablePPO agent was trained alongside the five reported agents (310M timesteps, curriculum learning against a mock environment), but its shop policy converged on hoarding cash rather than buying Jokers — a reward-misspecification failure only visible once training had converged. A hybrid version using MetaBot's shop heuristic performed only marginally above FlushBot. The mock environment's fidelity around shop economy, not the RL algorithm, was the limiting factor; this direction was not pursued further for the paper.

---

## Citation

```bibtex
@inproceedings{maharaj2026balatro,
  title     = {Can Generative {AI} be used to win at {Balatro}?},
  author    = {Maharaj, Vinayak and Hosein, Patrick},
  booktitle = {Proceedings of the 8th IEEE International Conference on
               Technology Management, Operations and Decisions (ICTMOD)},
  year      = {2026},
}
```

---

## Contact

**Vinayak Maharaj** · University of Toronto (BSc CS + Statistics)  
**Prof. Patrick Hosein** · TTLab, University of the West Indies
