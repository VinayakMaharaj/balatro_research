# Can an LLM Play Balatro Without Ever Practicing?

**Comparing retrieval-augmented and zero-shot LLM agents against a hand-coded heuristic on a long-horizon sequential decision problem**

> Vinayak Maharaj, University of Toronto  
> Prof. Patrick Hosein, TTLab, University of the West Indies  
> Published at IEEE ICTMOD 2026

---

## Overview

This repository contains the full experimental code for the paper *"Can an LLM Play Balatro Without Ever Practicing?"*, which uses **Balatro** — a poker-based roguelike card game — as a testbed for evaluating how different AI approaches handle long-horizon decision-making under uncertainty and sparse rewards.

The central research question: can a zero-shot large language model, with no prior training on the task, match or exceed a carefully hand-coded heuristic agent? And does retrieval-augmented generation (RAG) with expert strategy knowledge close the gap further?

---

## Key Results

| Agent | Avg Final Ante | n | Description |
|---|---|---|---|
| **RAGBot (strategy)** | **2.24** | 48 | LLM + expert strategy via RAG |
| **LLMBot** | **2.08** | 50 | Zero-shot Claude Sonnet 4.6 |
| **MetaBot** | **2.06** | 99 | Hand-coded heuristic baseline |
| RAGBot (rules) | 1.88 | 50 | LLM + rules-only RAG (ablation) |
| FlushBot | 1.14 | 100 | Rule-based floor baseline |

**Finding 1:** RAGBot (strategy) outperforms MetaBot (2.24 vs 2.06), demonstrating that LLM reasoning over expert strategy knowledge can exceed deterministic execution of the same knowledge.

**Finding 2:** Zero-shot LLMBot matches MetaBot (2.08 vs 2.06, p=ns), showing that Claude Sonnet's pretraining encodes sufficient game knowledge to compete with a hand-coded heuristic.

**Finding 3 (negative result):** Rules-only RAG underperforms zero-shot LLM (1.88 vs 2.08). Retrieved rules add noise rather than signal because Claude already encodes Balatro's mechanics from pretraining data. This is supported qualitatively — RAGBot (rules) produced 0.95 self-corrections per hand decision vs 0.44 for RAGBot (strategy).

---

## Agents

| Agent | File | Paradigm |
|---|---|---|
| FlushBot | `heuristic_bots.py` | Rule-based floor — always plays best flush, never buys |
| MetaBot | `heuristic_bots.py` | Hand-coded heuristic with tiered joker priority, dominant hand tracking, economy rules |
| LLMBot | `llm_bot.py` | Zero-shot Claude Sonnet 4.6, mechanical game state prompt only |
| RAGBot (rules) | `rag_pipeline.py` | LLM + Pinecone rules corpus (27 vectors, official game mechanics) |
| RAGBot (strategy) | `rag_meta_pipeline.py` | LLM + Pinecone strategy corpus (13 chunks mirroring MetaBot's decision logic) |

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

```
balatro_research/
├── base_bot.py              # Base class: game loop, CSV logging, W&B tracking
├── balatro_client.py        # HTTP JSON-RPC client for balatrobot API
├── heuristic_bots.py        # FlushBot and MetaBot
├── llm_bot.py               # LLMBot — zero-shot Claude Sonnet 4.6
├── rag_pipeline.py          # RAGBot (rules) — LLM + rules-only Pinecone index
├── rag_meta_pipeline.py     # RAGBot (strategy) — LLM + strategy Pinecone index
├── analyze_results.py       # Statistical analysis and figure generation
├── plot_figure.py           # IEEE-formatted publication figures (PDF + SVG)
├── data/
│   ├── rules.json           # 27 official game rules (RAG corpus)
│   └── strategy.json        # 13 strategy chunks mirroring MetaBot's logic
├── results_final.csv        # Full experimental results (347 valid games)
├── llm_decisions.jsonl      # Per-decision LLM reasoning log
├── rag_decisions.jsonl      # Per-decision RAGBot (rules) reasoning log
├── rag_meta_decisions.jsonl # Per-decision RAGBot (strategy) reasoning log
└── figures/
    ├── figure1_performance.pdf   # Box-and-whisker: final ante by agent
    ├── figure2_qualitative.pdf   # Decision quality metrics: LLM agents
    └── summary_table.csv         # Mean ± std per agent for all metrics
```

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

All agents use **Red Deck, White Stake, seeds SEED001–SEED100** (heuristic) or **SEED001–SEED050** (LLM/RAG).

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
| Strategy index vectors | 13 |

---

## Design Notes

### Why Balatro?

Balatro requires multi-step planning across 3 blinds per ante, economy management across shops, and adaptation to random boss blind effects. It has sparse rewards (win/lose per blind), stochastic card draws, and a large action space — making it a challenging and realistic testbed for agent comparison.

### Why zero-shot (no few-shot examples)?

The research question is whether LLMs can reason about novel game mechanics from scratch. Adding few-shot examples would constitute implicit strategy transfer and confound the comparison with MetaBot.

### RAGBot (rules) negative result

Rules-only retrieval underperforms zero-shot LLM because Claude's pretraining data includes Balatro wikis, Reddit posts, and strategy guides — the rules corpus adds redundant context that interferes with the model's own reasoning rather than supplying new information. This is evidenced by significantly higher self-correction rates (0.95 vs 0.44 per hand) and 78 ante-1 blind skips vs 0 for RAGBot (strategy).

### Strategy corpus design

`data/strategy.json` encodes exactly the decision logic implemented in MetaBot's code — joker tier priorities, interest floor thresholds, dominant hand tracking, blind skip rules, and boss blind adaptations — as 13 retrievable text chunks. Nothing in the corpus exceeds what MetaBot's code executes, ensuring a fair knowledge-matched comparison.

---

## Citation

```bibtex
@inproceedings{maharaj2026balatro,
  title     = {Can an {LLM} Play {Balatro} Without Ever Practicing?},
  author    = {Maharaj, Vinayak and Hosein, Patrick},
  booktitle = {Proceedings of the 8th IEEE International Conference on
               Technology Management, Operations and Decisions (ICTMOD)},
  year      = {2026},
  address   = {Paris, France},
}
```

---

## Contact

**Vinayak Maharaj** · University of Toronto (BSc CS + Statistics)  
**Prof. Patrick Hosein** · TTLab, University of the West Indies