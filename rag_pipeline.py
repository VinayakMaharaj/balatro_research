"""
rag_pipeline.py
RAG pipeline for Balatro rules using Pinecone + sentence-transformers.
 
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
# Fallback hardcoded rules
# ---------------------------------------------------------------------------
 
FALLBACK_RULES = [
    {
        "id": "poker_hands_ranking",
        "category": "poker_hands",
        "title": "Poker Hand Rankings",
        "text": "Poker hands from best to worst: Royal Flush, Straight Flush, Four of a Kind, Full House, Flush (5 same suit), Straight (5 consecutive ranks), Three of a Kind, Two Pair, Pair, High Card. Flush requires exactly 5 cards of the same suit. Straight requires 5 consecutive ranks.",
    },
    {
        "id": "scoring_basics",
        "category": "mechanics",
        "title": "Scoring Basics",
        "text": "Score = (Base Chips + Card Chips) * Multiplier. Base chips and mult come from the poker hand type. Card chips come from individual card ranks. Jokers add flat chips, flat mult, or multiplier (Xmult) bonuses.",
    },
    {
        "id": "shop_basics",
        "category": "mechanics",
        "title": "Shop Basics",
        "text": "The shop appears after each round. You can buy Jokers, Tarot cards, Planet cards, Vouchers, and Booster Packs. Jokers persist between rounds and multiply scoring. Buy jokers early to scale damage. Interest: earn $1 for every $5 held at end of round, up to $5 per round.",
    },
    {
        "id": "blind_basics",
        "category": "blinds",
        "title": "Blind Structure",
        "text": "Each Ante has 3 blinds: Small Blind, Big Blind, Boss Blind. Small and Big blinds can be skipped for a Tag reward. Boss blinds cannot be skipped and have special negative effects. Chip requirements scale each ante. Defeating all 8 antes wins the run.",
    },
    {
        "id": "interest_mechanic",
        "category": "mechanics",
        "title": "Interest and Economy",
        "text": "At end of each round, earn $1 interest for every $5 you hold, up to a maximum of $5 interest per round (requires $25). Saving money compounds over time. Do not spend all money if you can maintain interest income.",
    },
    {
        "id": "skip_blind_tags",
        "category": "mechanics",
        "title": "Skipping Blinds for Tags",
        "text": "Skipping Small or Big blind grants a Tag reward shown in the blind select screen. Tags provide powerful one-time bonuses like free jokers, money, or hand upgrades. Boss blinds cannot be skipped. Skipping is free and you still advance.",
    },
    {
        "id": "joker_blueprint",
        "category": "jokers",
        "title": "Blueprint",
        "text": "Blueprint: Copies the ability of the Joker to the right. If no Joker is to the right, Blueprint does nothing.",
    },
    {
        "id": "joker_cavendish",
        "category": "jokers",
        "title": "Cavendish",
        "text": "Cavendish: X3 Mult. Has a 1 in 1000 chance to be destroyed at end of round. Very powerful multiplier joker.",
    },
    {
        "id": "joker_ride_the_bus",
        "category": "jokers",
        "title": "Ride the Bus",
        "text": "Ride the Bus: Gains +1 Mult per consecutive hand played without a scoring face card. Resets to 0 when a face card scores.",
    },
    {
        "id": "joker_hologram",
        "category": "jokers",
        "title": "Hologram",
        "text": "Hologram: Gains X0.25 Mult every time a playing card is added to your deck.",
    },
    {
        "id": "deck_red",
        "category": "decks",
        "title": "Red Deck",
        "text": "Red Deck: +1 Discard every round. Gives one extra discard per round compared to default.",
    },
    {
        "id": "edition_polychrome",
        "category": "editions",
        "title": "Polychrome Edition",
        "text": "Polychrome Edition: X1.5 Mult when the card scores. Extremely powerful on jokers.",
    },
    {
        "id": "seal_red",
        "category": "seals",
        "title": "Red Seal",
        "text": "Red Seal: Retrigger this card 1 additional time when scored. Doubles the effect of the card including joker bonuses.",
    },
]
 
 
# ---------------------------------------------------------------------------
# Corpus builder
# ---------------------------------------------------------------------------
 
def load_json_data(filepath: str, category: str) -> list[dict]:
    chunks = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
 
        items = data if isinstance(data, list) else data.get(category, data.get('items', []))
 
        for item in items:
            name = item.get('name', item.get('Name', ''))
            desc = item.get('description', item.get('Description', item.get('effect', '')))
 
            if len(name) > 2 and len(desc) > 20:
                chunk_id = f"{category}_{name.lower().replace(' ', '_')[:30]}"
                chunk_id = unicodedata.normalize('NFKD', chunk_id).encode('ascii', 'ignore').decode('ascii')
                chunks.append({
                    "id": chunk_id,
                    "category": category,
                    "title": name,
                    "text": f"{name}: {desc}",
                })
 
        logger.info(f"Loaded {len(chunks)} chunks from {filepath}")
    except FileNotFoundError:
        logger.warning(f"JSON file not found: {filepath}")
    except Exception as e:
        logger.warning(f"Failed to load {filepath}: {e}")
 
    return chunks
 
 
def build_corpus() -> list[dict]:
    all_chunks = []
 
    JSON_FILES = {
        "jokers":       "data/jokers.json",
        "tarots":       "data/tarots.json",
        "planets":      "data/planets.json",
        "spectrals":    "data/spectrals.json",
        "poker_hands":  "data/poker_hands.json",
        "blinds":       "data/blinds.json",
        "vouchers":     "data/vouchers.json",
        "decks":        "data/decks.json",
        "enhancements": "data/enhancements.json",
        "editions":     "data/editions.json",
        "seals":        "data/seals.json",
        "tags":         "data/tags.json",
    }
 
    for category, filepath in JSON_FILES.items():
        chunks = load_json_data(filepath, category)
        all_chunks.extend(chunks)
 
    existing_ids = {c["id"] for c in all_chunks}
    for rule in FALLBACK_RULES:
        if rule["id"] not in existing_ids:
            all_chunks.append(rule)
 
    logger.info(f"Total corpus size: {len(all_chunks)} chunks")
    return all_chunks
 
 
# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------
 
def build_index(pinecone_api_key: str) -> None:
    logger.info("Loading embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL)
 
    logger.info("Building rules corpus...")
    chunks = build_corpus()
 
    logger.info(f"Connecting to Pinecone index: {PINECONE_INDEX}")
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(PINECONE_INDEX)
 
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
        jokers = state.get("jokers", {}).get("cards", [])
        joker_names = [j.get("label", "") for j in jokers]
 
        query_parts = []
 
        if state_name == "SELECTING_HAND":
            query_parts.append("poker hand scoring chips multiplier")
            if joker_names:
                query_parts.extend(joker_names[:3])
            blinds = state.get("blinds", {})
            for bk in ["small", "big", "boss"]:
                b = blinds.get(bk, {})
                if b.get("status") in ("SELECT", "CURRENT"):
                    blind_name = b.get("name", "")
                    if blind_name and blind_name not in ("Small Blind", "Big Blind"):
                        query_parts.append(blind_name)
                    break
 
        elif state_name == "SHOP":
            query_parts.append("joker buy shop economy interest")
            shop_cards = state.get("shop", {}).get("cards", [])
            for card in shop_cards[:3]:
                query_parts.append(card.get("label", ""))
 
        elif state_name == "BLIND_SELECT":
            query_parts.append("blind skip tag reward")
            blinds = state.get("blinds", {})
            boss = blinds.get("boss", {})
            if boss.get("name"):
                query_parts.append(boss["name"])
 
        query = " ".join(filter(None, query_parts))
        return self.retrieve(query)
 
 
# ---------------------------------------------------------------------------
# RAG LLM Bot
# ---------------------------------------------------------------------------
 
class RAGLLMBot(LLMBot):
    """
    LLM agent augmented with RAG retrieval from Pinecone.
    Retrieves relevant Balatro rules at each decision point.
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
        # Separate qualitative log for RAG bot
        self._qual_logger = QualitativeLogger("rag_decisions.jsonl")
 
    def _build_rag_prompt(self, state: dict) -> str:
        base_prompt = format_state_for_llm(state)
        relevant_rules = self.rag.retrieve_for_state(state)
 
        if relevant_rules:
            rag_section = f"""RELEVANT RULES (retrieved from Balatro knowledge base):
{relevant_rules}
 
"""
            lines = base_prompt.split("\n", 3)
            if len(lines) >= 3:
                return lines[0] + "\n" + lines[1] + "\n\n" + rag_section + "\n".join(lines[2:])
 
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
 
        # Fix: guard boss blind skip
        blind_type = get_blind_type(state)
        if blind_type == "boss" and action == "skip":
            logger.warning("RAG-LLM tried to skip boss blind, overriding to select")
            action = "select"
 
        return action
 
 
# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------
 
BENCHMARK_SEEDS = ["AAAAAAA", "BBBBBBB", "CCCCCCC", "DDDDDDD", "EEEEEEE"]
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
 