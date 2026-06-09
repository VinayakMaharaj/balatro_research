"""
rag_pipeline.py
RAG pipeline for Balatro using Pinecone + sentence-transformers.

Rules corpus: official game rules only (what a new player reads before playing).
No strategy guides, no tier lists, no community meta.

Two parts:
    1. build_index()  - embeds rules corpus, uploads to Pinecone
    2. RAGLLMBot      - inherits LLMBot, retrieves relevant rules before each LLM call

Run index building once:
    python rag_pipeline.py --build-index

Then run the RAG bot:
    python rag_pipeline.py --runs-per-seed 1
"""

import os
import re
import json
import time
import logging
import argparse
import unicodedata

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from llm_bot import (
    LLMBot, format_state_for_llm, parse_llm_response,
    MODEL, MAX_TOKENS, QualitativeLogger
)
from base_bot import (
    get_hand_cards, get_discards_left, get_hands_left,
    get_shop_cards, get_shop_packs, get_money, get_blind_type, get_ante
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PINECONE_INDEX = "balatro-rules"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

# ---------------------------------------------------------------------------
# Corpus loader — official rules only
# ---------------------------------------------------------------------------

def build_corpus() -> list[dict]:
    """
    Load only the official rules corpus from data/rules.json.
    No joker descriptions, no strategy guides, no tier lists.
    This simulates a player who read the rulebook before playing.
    """
    rules_path = "data/rules.json"
    chunks = []

    try:
        with open(rules_path, 'r', encoding='utf-8') as f:
            rules = json.load(f)

        for rule in rules:
            name = rule.get("name", "")
            desc = rule.get("description", "")
            if name and desc:
                chunk_id = f"rule_{name.lower().replace(' ', '_')[:40]}"
                chunk_id = unicodedata.normalize('NFKD', chunk_id).encode('ascii', 'ignore').decode('ascii')
                chunks.append({
                    "id": chunk_id,
                    "category": "rules",
                    "title": name,
                    "text": f"{name}: {desc}",
                })

        logger.info(f"Loaded {len(chunks)} rules from {rules_path}")

    except FileNotFoundError:
        logger.error(f"Rules file not found: {rules_path}")
        logger.error("Run with --build-index after creating data/rules.json")
    except Exception as e:
        logger.error(f"Failed to load rules: {e}")

    logger.info(f"Total corpus size: {len(chunks)} chunks")
    return chunks


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def build_index(pinecone_api_key: str) -> None:
    logger.info("Loading embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL)

    logger.info("Building rules corpus...")
    chunks = build_corpus()

    if not chunks:
        logger.error("No chunks to index. Check data/rules.json exists.")
        return

    logger.info(f"Connecting to Pinecone index: {PINECONE_INDEX}")
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(PINECONE_INDEX)

    # Clear existing vectors first
    logger.info("Clearing existing index vectors...")
    try:
        index.delete(delete_all=True)
        logger.info("Index cleared")
    except Exception as e:
        logger.warning(f"Could not clear index: {e}")

    logger.info(f"Embedding and uploading {len(chunks)} chunks...")
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        embeddings = embedder.encode(texts, show_progress_bar=False)

        vectors = []
        for chunk, embedding in zip(batch, embeddings):
            vectors.append({
                "id": chunk["id"],
                "values": embedding.tolist(),
                "metadata": {
                    "category": chunk["category"],
                    "title": chunk["title"],
                    "text": chunk["text"],
                }
            })

        index.upsert(vectors=vectors)
        logger.info(f"Uploaded batch {i // batch_size + 1}/{(len(chunks) + batch_size - 1) // batch_size}")

    stats = index.describe_index_stats()
    logger.info(f"Index built. Total vectors: {stats.total_vector_count}")


# ---------------------------------------------------------------------------
# RAG retriever
# ---------------------------------------------------------------------------

class BalatroRAG:
    def __init__(self, pinecone_api_key: str):
        logger.info("Loading embedding model for RAG...")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        pc = Pinecone(api_key=pinecone_api_key)
        self.index = pc.Index(PINECONE_INDEX)
        logger.info("RAG retriever ready")

    def retrieve(self, query: str, top_k: int = TOP_K) -> str:
        embedding = self.embedder.encode(query).tolist()
        results = self.index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
        )

        if not results.matches:
            return ""

        chunks = []
        for match in results.matches:
            metadata = match.metadata
            chunks.append(f"- {metadata.get('text', '')}")

        return "\n".join(chunks)

    def retrieve_for_state(self, state: dict) -> str:
        state_name = state.get("state", "")

        # Build query from current game state — focus on rules relevant to current phase
        query_parts = []

        if state_name == "SELECTING_HAND":
            query_parts.append("poker hand scoring chips multiplier")
            # Add boss blind effect if present
            blinds = state.get("blinds", {})
            for bk in ["small", "big", "boss"]:
                b = blinds.get(bk, {})
                if b.get("status") in ("SELECT", "CURRENT"):
                    effect = b.get("effect", "")
                    if effect:
                        query_parts.append(effect)
                    break
            query_parts.append("discard strategy hand")

        elif state_name == "SHOP":
            query_parts.append("shop joker planet tarot interest economy")
            query_parts.append("money interest mechanic")

        elif state_name == "BLIND_SELECT":
            query_parts.append("blind skip tag reward boss blind")

        elif state_name == "SMODS_BOOSTER_OPENED":
            query_parts.append("booster pack tarot planet spectral choose cards")

        query = " ".join(filter(None, query_parts))
        return self.retrieve(query)


# ---------------------------------------------------------------------------
# RAG LLM Bot
# ---------------------------------------------------------------------------

class RAGLLMBot(LLMBot):
    """
    LLM agent augmented with RAG retrieval from Pinecone.
    Retrieves only official Balatro rules — simulates a player who
    read the rulebook but has no prior strategy knowledge.
    """

    BOT_TYPE = "rag_llm_bot"
    WANDB_TAGS = ["rag_llm_bot", "rag", "zero_shot"]

    def __init__(self, pinecone_api_key: str | None = None, **kwargs):
        super().__init__(**kwargs)
        key = pinecone_api_key or os.environ.get("PINECONE_API_KEY")
        if not key:
            raise ValueError(
                "Pinecone API key required. "
                "Set $env:PINECONE_API_KEY='your_key' or pass --pinecone-api-key"
            )
        self.rag = BalatroRAG(pinecone_api_key=key)
        self._qual_logger = QualitativeLogger("rag_decisions.jsonl")

    def _build_rag_prompt(self, state: dict) -> str:
        base_prompt = format_state_for_llm(state)
        relevant_rules = self.rag.retrieve_for_state(state)

        if relevant_rules:
            rag_section = f"""RELEVANT RULES (from official Balatro rulebook):
{relevant_rules}

"""
            # Insert rules after the first line (game status line)
            lines = base_prompt.split("\n", 2)
            if len(lines) >= 2:
                return lines[0] + "\n" + lines[1] + "\n\n" + rag_section + ("\n".join(lines[2:]) if len(lines) > 2 else "")

        return base_prompt

    def select_hand_action(self, state: dict) -> tuple[str, list[int]]:
        prompt = self._build_rag_prompt(state)
        response = self._call_llm(prompt)
        parsed = parse_llm_response(response, "SELECTING_HAND")

        action = parsed.get("action", "play")
        cards = parsed.get("cards", [0, 1, 2, 3, 4])
        reasoning = parsed.get("reasoning", "")

        if reasoning:
            logger.info(f"RAG-LLM: {reasoning}")

        self._qual_logger.log(
            bot_type=self.BOT_TYPE,
            seed=self._current_seed,
            ante=get_ante(state),
            round_num=state.get("round_num", 0),
            state_name="SELECTING_HAND",
            prompt=prompt,
            response=response,
            parsed_action=parsed,
        )

        hand = get_hand_cards(state)
        cards = [c for c in cards if 0 <= c < len(hand)]
        if not cards:
            cards = list(range(min(5, len(hand))))

        if action == "discard" and get_discards_left(state) <= 0:
            action = "play"
            cards = list(range(min(5, len(hand))))

        return action, cards

    def select_shop_action(self, state: dict) -> list[dict]:
        prompt = self._build_rag_prompt(state)
        response = self._call_llm(prompt)
        parsed = parse_llm_response(response, "SHOP")
        actions = parsed.get("actions", [{"action": "end_shop"}])

        self._qual_logger.log(
            bot_type=self.BOT_TYPE,
            seed=self._current_seed,
            ante=get_ante(state),
            round_num=state.get("round_num", 0),
            state_name="SHOP",
            prompt=prompt,
            response=response,
            parsed_action=parsed,
        )

        valid_actions = []
        shop_cards = get_shop_cards(state)
        packs = get_shop_packs(state)
        money = get_money(state)

        for action_dict in actions:
            action = action_dict.get("action")

            if action == "buy_card":
                idx = action_dict.get("index", 0)
                if idx < len(shop_cards):
                    cost = shop_cards[idx].get("cost", {}).get("buy", 999)
                    if cost <= money:
                        valid_actions.append({"action": "buy_card", "index": idx})
                        money -= cost
                    else:
                        logger.warning(f"Cannot afford card {idx} (cost={cost}, money={money})")

            elif action == "buy_pack":
                idx = action_dict.get("index", 0)
                if idx < len(packs):
                    cost = packs[idx].get("cost", {}).get("buy", 999)
                    if cost <= money:
                        valid_actions.append({"action": "buy_pack", "index": idx})
                        money -= cost

            elif action == "reroll":
                reroll_cost = state.get("round", {}).get("reroll_cost", 5)
                if money >= reroll_cost:
                    valid_actions.append({"action": "reroll"})
                    money -= reroll_cost

            elif action == "end_shop":
                valid_actions.append({"action": "end_shop"})
                break

        if not valid_actions or valid_actions[-1].get("action") != "end_shop":
            valid_actions.append({"action": "end_shop"})

        return valid_actions

    def select_blind_action(self, state: dict) -> str:
        prompt = self._build_rag_prompt(state)
        response = self._call_llm(prompt)
        parsed = parse_llm_response(response, "BLIND_SELECT")

        action = parsed.get("action", "select")
        reasoning = parsed.get("reasoning", "")

        if reasoning:
            logger.info(f"RAG-LLM: {reasoning}")

        self._qual_logger.log(
            bot_type=self.BOT_TYPE,
            seed=self._current_seed,
            ante=get_ante(state),
            round_num=state.get("round_num", 0),
            state_name="BLIND_SELECT",
            prompt=prompt,
            response=response,
            parsed_action=parsed,
        )

        blind_type = get_blind_type(state)
        if blind_type == "boss" and action == "skip":
            logger.warning("RAG-LLM tried to skip boss blind, overriding to select")
            action = "select"

        return action

    def select_pack_action(self, state: dict) -> dict:
        prompt = self._build_rag_prompt(state)
        response = self._call_llm(prompt)

        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                parsed = {"action": "skip", "cards": [], "reasoning": "fallback"}
        except json.JSONDecodeError:
            parsed = {"action": "skip", "cards": [], "reasoning": "fallback"}

        reasoning = parsed.get("reasoning", "")
        if reasoning:
            logger.info(f"RAG-LLM pack: {reasoning}")

        self._qual_logger.log(
            bot_type=self.BOT_TYPE,
            seed=self._current_seed,
            ante=get_ante(state),
            round_num=state.get("round_num", 0),
            state_name="SMODS_BOOSTER_OPENED",
            prompt=prompt,
            response=response,
            parsed_action=parsed,
        )

        return parsed


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

BENCHMARK_SEEDS = [f"SEED{str(i).zfill(3)}" for i in range(1, 101)]
RUNS_PER_SEED = 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG pipeline for Balatro")
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--api-key", help="Anthropic API key")
    parser.add_argument("--pinecone-api-key", help="Pinecone API key")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--seeds", nargs="+", default=BENCHMARK_SEEDS)
    parser.add_argument("--runs-per-seed", type=int, default=RUNS_PER_SEED)
    parser.add_argument("--results", default="results.csv")
    parser.add_argument("--port", type=int, default=12346)
    args = parser.parse_args()

    pinecone_key = args.pinecone_api_key or os.environ.get("PINECONE_API_KEY")
    if not pinecone_key:
        print("ERROR: Pinecone API key required.")
        exit(1)

    if args.build_index:
        build_index(pinecone_key)
        exit(0)

    bot = RAGLLMBot(
        api_key=args.api_key,
        pinecone_api_key=pinecone_key,
        model=args.model,
        port=args.port,
        results_path=args.results,
    )

    if not bot.client.health():
        print("ERROR: Cannot connect to Balatro.")
        exit(1)

    print(f"Running RAGLLMBot on {len(args.seeds)} seeds x {args.runs_per_seed} runs")

    results = bot.run_experiment(seeds=args.seeds, runs_per_seed=args.runs_per_seed)

    completed = [r for r in results if r["outcome"] in ("won", "lost")]
    if completed:
        avg_round = sum(r["final_round"] for r in completed) / len(completed)
        avg_ante = sum(r["final_ante"] for r in completed) / len(completed)
        wins = sum(1 for r in results if r["outcome"] == "won")
        print(f"\nSummary:")
        print(f"  Completed: {len(completed)}/{len(results)}")
        print(f"  Wins: {wins}")
        print(f"  Avg round: {avg_round:.2f}")
        print(f"  Avg ante: {avg_ante:.2f}")
        print(f"  LLM calls: {bot._total_llm_calls}")
        print(f"  Tokens used: {bot._total_tokens_used}")
        print(f"  Decisions logged to: rag_decisions.jsonl")