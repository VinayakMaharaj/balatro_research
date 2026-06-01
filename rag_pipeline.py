"""
rag_pipeline.py
RAG pipeline for Balatro rules using Pinecone + sentence-transformers + LangChain.

Two parts:
    1. build_index()  - scrapes Balatro wiki, embeds rules, uploads to Pinecone
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
import requests
import unicodedata

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from llm_bot import LLMBot, format_state_for_llm, MODEL, MAX_TOKENS
from base_bot import get_hand_cards, get_discards_left, get_hands_left

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PINECONE_INDEX = "balatro-rules"
EMBED_MODEL = "all-MiniLM-L6-v2"  # 384 dimensions, fast, free
TOP_K = 5  # number of rule chunks to retrieve per query

# Balatro wiki pages to scrape
WIKI_PAGES = [
    ("jokers",       "https://balatrogame.fandom.com/wiki/Jokers"),
    ("tarots",       "https://balatrogame.fandom.com/wiki/Tarot_Cards"),
    ("planets",      "https://balatrogame.fandom.com/wiki/Planet_Cards"),
    ("spectrals",    "https://balatrogame.fandom.com/wiki/Spectral_Cards"),
    ("poker_hands",  "https://balatrogame.fandom.com/wiki/Poker_Hands"),
    ("blinds",       "https://balatrogame.fandom.com/wiki/Blinds"),
    ("vouchers",     "https://balatrogame.fandom.com/wiki/Vouchers"),
    ("decks",        "https://balatrogame.fandom.com/wiki/Decks"),
    ("enhancements", "https://balatrogame.fandom.com/wiki/Card_Enhancements"),
    ("editions",     "https://balatrogame.fandom.com/wiki/Editions"),
    ("seals",        "https://balatrogame.fandom.com/wiki/Seals"),
    ("tags",         "https://balatrogame.fandom.com/wiki/Tags"),
]

# Fallback hardcoded rules if wiki scraping fails
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
        "id": "joker_mad",
        "category": "jokers",
        "title": "Mad Joker",
        "text": "Mad Joker: +10 Mult if played hand contains a Two Pair. Only activates on Two Pair hands specifically, not other hand types.",
    },
    {
        "id": "joker_scary_face",
        "category": "jokers",
        "title": "Scary Face",
        "text": "Scary Face: Played face cards give +30 Chips when scored. Face cards are Jacks, Queens, and Kings. Each face card in a played hand gives +30 chips.",
    },
    {
        "id": "joker_odd_todd",
        "category": "jokers",
        "title": "Odd Todd",
        "text": "Odd Todd: Played cards with odd rank give +31 Chips when scored. Odd ranks are: Ace (A), 9, 7, 5, 3. Each odd-ranked card in a played hand gives +31 chips.",
    },
    {
        "id": "joker_blueprint",
        "category": "jokers",
        "title": "Blueprint",
        "text": "Blueprint: Copies the ability of the Joker to the right. If no Joker is to the right, Blueprint does nothing.",
    },
    {
        "id": "joker_ride_the_bus",
        "category": "jokers",
        "title": "Ride the Bus",
        "text": "Ride the Bus: Gains +1 Mult per consecutive hand played without a scoring face card. Resets to 0 when a face card scores. Build up by playing hands without face cards.",
    },
    {
        "id": "joker_hologram",
        "category": "jokers",
        "title": "Hologram",
        "text": "Hologram: Gains X0.25 Mult every time a playing card is added to your deck. Scales with deck additions from Standard Packs, Tarot cards like The Fool, etc.",
    },
    {
        "id": "joker_cavendish",
        "category": "jokers",
        "title": "Cavendish",
        "text": "Cavendish: X3 Mult. Has a 1 in 1000 chance to be destroyed at end of round. Very powerful multiplier joker.",
    },
    {
        "id": "joker_blueprint_brainstorm",
        "category": "jokers",
        "title": "Brainstorm",
        "text": "Brainstorm: Copies the ability of the leftmost Joker. Combine with powerful jokers for double effect.",
    },
    {
        "id": "tarot_strength",
        "category": "tarots",
        "title": "Strength Tarot",
        "text": "The Strength: Increases the rank of up to 2 selected cards by 1. For example, a 9 becomes a 10, a Jack becomes a Queen.",
    },
    {
        "id": "tarot_death",
        "category": "tarots",
        "title": "Death Tarot",
        "text": "Death: Select 2 cards, converts the left card into the right card. Used to create pairs or same-suit cards.",
    },
    {
        "id": "planet_jupiter",
        "category": "planets",
        "title": "Jupiter Planet Card",
        "text": "Jupiter: Upgrades Flush hand level by +1, giving +2 Mult and +15 Chips to all future Flush hands.",
    },
    {
        "id": "deck_red",
        "category": "decks",
        "title": "Red Deck",
        "text": "Red Deck: +1 Discard every round. Gives one extra discard per round compared to default, allowing more card cycling.",
    },
    {
        "id": "boss_the_arm",
        "category": "blinds",
        "title": "The Arm Boss Blind",
        "text": "The Arm: Decreases the level of played poker hand by 1 after each play. Avoid leveling up hands you plan to play against The Arm, or play varied hands to minimize level loss impact.",
    },
    {
        "id": "boss_the_hook",
        "category": "blinds",
        "title": "The Hook Boss Blind",
        "text": "The Hook: Discards 2 random cards from your hand at the start of each play. Plan for reduced hand size when facing The Hook.",
    },
    {
        "id": "boss_the_manacle",
        "category": "blinds",
        "title": "The Manacle Boss Blind",
        "text": "The Manacle: -1 Hand Size. Your hand is reduced by 1 card for the entire blind, making it harder to form strong hands.",
    },
    {
        "id": "edition_foil",
        "category": "editions",
        "title": "Foil Edition",
        "text": "Foil Edition: +50 Chips when the card scores. Applies to both playing cards and jokers.",
    },
    {
        "id": "edition_holographic",
        "category": "editions",
        "title": "Holographic Edition",
        "text": "Holographic Edition: +10 Mult when the card scores. Applies to both playing cards and jokers.",
    },
    {
        "id": "edition_polychrome",
        "category": "editions",
        "title": "Polychrome Edition",
        "text": "Polychrome Edition: X1.5 Mult when the card scores. Extremely powerful on jokers.",
    },
    {
        "id": "seal_gold",
        "category": "seals",
        "title": "Gold Seal",
        "text": "Gold Seal: Earn $3 when this card is played and scores. Good for economy building.",
    },
    {
        "id": "seal_red",
        "category": "seals",
        "title": "Red Seal",
        "text": "Red Seal: Retrigger this card 1 additional time when scored. Doubles the effect of the card including joker bonuses.",
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
]


# ---------------------------------------------------------------------------
# Wiki scraper
# ---------------------------------------------------------------------------

def scrape_wiki_page(url: str, category: str) -> list[dict]:
    """
    Scrape a Balatro wiki page and extract card/mechanic descriptions.
    Returns list of chunks with id, category, title, text.
    """
    headers = {"User-Agent": "BalatroResearchBot/1.0 (academic research)"}
    chunks = []

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            logger.warning(f"Failed to fetch {url}: {response.status_code}")
            return chunks

        # Simple text extraction - find table rows with card names and descriptions
        text = response.text

        # Extract content between wikitable rows
        # Look for patterns like: CardName | Description
        rows = re.findall(
            r'<td[^>]*>([^<]{3,50})</td>\s*<td[^>]*>(.*?)</td>',
            text,
            re.DOTALL
        )

        for name, desc in rows:
            name = re.sub(r'<[^>]+>', '', name).strip()
            desc = re.sub(r'<[^>]+>', '', desc).strip()
            desc = re.sub(r'\s+', ' ', desc)

            if len(name) > 2 and len(desc) > 20:
                chunk_id = f"{category}_{name.lower().replace(' ', '_')[:30]}"
                # Normalize to ASCII
                chunk_id = unicodedata.normalize('NFKD', chunk_id).encode('ascii', 'ignore').decode('ascii')
                chunks.append({
                    "id": chunk_id,
                    "category": category,
                    "title": name,
                    "text": f"{name}: {desc}",
                })

        logger.info(f"Scraped {len(chunks)} chunks from {category}")

    except Exception as e:
        logger.warning(f"Failed to scrape {url}: {e}")

    return chunks


def load_json_data(filepath: str, category: str) -> list[dict]:
    """Load card data from a local JSON file."""
    chunks = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        items = data if isinstance(data, list) else data.get(category, data.get('items', []))
        
        for item in items:
            name = item.get('name', item.get('Name', ''))
            desc = item.get('description', item.get('Description', item.get('effect', '')))
            
            if len(name) > 2 and len(desc) > 20:
                chunk_id = f"{category}_{name.lower().replace(' ', '_')[:30]}"
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
    """
    Build the full rules corpus from local JSON files.
    Falls back to hardcoded rules if files are missing.
    """
    all_chunks = []

    # Map category names to local JSON files
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

    # Always include fallback rules
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
    """
    Build the Pinecone index from the Balatro rules corpus.
    Only needs to be run once.
    """
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

    # Verify
    stats = index.describe_index_stats()
    logger.info(f"Index built successfully. Total vectors: {stats.total_vector_count}")


# ---------------------------------------------------------------------------
# RAG retriever
# ---------------------------------------------------------------------------

class BalatroRAG:
    """
    Retrieves relevant Balatro rules from Pinecone given a query.
    """

    def __init__(self, pinecone_api_key: str):
        logger.info("Loading embedding model for RAG...")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        pc = Pinecone(api_key=pinecone_api_key)
        self.index = pc.Index(PINECONE_INDEX)
        logger.info("RAG retriever ready")

    def retrieve(self, query: str, top_k: int = TOP_K) -> str:
        """
        Retrieve the top-k most relevant rule chunks for a query.
        Returns formatted string ready to inject into a prompt.
        """
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
        """
        Build a query from the current game state and retrieve relevant rules.
        """
        state_name = state.get("state", "")
        jokers = state.get("jokers", {}).get("cards", [])
        joker_names = [j.get("label", "") for j in jokers]

        # Build query based on state
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

    At each decision point, retrieves the most relevant Balatro rules
    from the vector database and injects them into the prompt before
    calling Claude.

    This is the fifth ablation condition and the most sophisticated
    LLM variant in the paper.
    """

    BOT_TYPE = "rag_llm_bot"

    def __init__(self, pinecone_api_key: str | None = None, **kwargs):
        super().__init__(**kwargs)
        key = pinecone_api_key or os.environ.get("PINECONE_API_KEY")
        if not key:
            raise ValueError(
                "Pinecone API key required. "
                "Set $env:PINECONE_API_KEY='your_key' or pass --pinecone-api-key"
            )
        self.rag = BalatroRAG(pinecone_api_key=key)

    def _build_rag_prompt(self, state: dict) -> str:
        """
        Build prompt with RAG context injected.
        Retrieves relevant rules and prepends them to the standard prompt.
        """
        base_prompt = format_state_for_llm(state)
        relevant_rules = self.rag.retrieve_for_state(state)

        if relevant_rules:
            rag_section = f"""RELEVANT RULES (retrieved from Balatro knowledge base):
{relevant_rules}

"""
            # Insert RAG section at the start of the prompt after the header
            lines = base_prompt.split("\n", 3)
            if len(lines) >= 3:
                return lines[0] + "\n" + lines[1] + "\n\n" + rag_section + "\n".join(lines[2:])

        return base_prompt

    def select_hand_action(self, state: dict) -> tuple[str, list[int]]:
        """Ask Claude what to play or discard, with RAG context."""
        from llm_bot import parse_llm_response
        prompt = self._build_rag_prompt(state)
        response = self._call_llm(prompt)

        parsed = parse_llm_response(response, "SELECTING_HAND")
        action = parsed.get("action", "play")
        cards = parsed.get("cards", [0, 1, 2, 3, 4])
        reasoning = parsed.get("reasoning", "")

        if reasoning:
            logger.info(f"RAG-LLM: {reasoning}")

        hand = get_hand_cards(state)
        cards = [c for c in cards if 0 <= c < len(hand)]
        if not cards:
            cards = list(range(min(5, len(hand))))

        if action == "discard" and get_discards_left(state) <= 0:
            action = "play"
            cards = list(range(min(5, len(hand))))

        return action, cards

    def select_shop_action(self, state: dict) -> list[dict]:
        """Ask Claude what to buy, with RAG context."""
        from llm_bot import parse_llm_response
        from base_bot import get_shop_cards, get_shop_packs, get_money
        prompt = self._build_rag_prompt(state)
        response = self._call_llm(prompt)

        parsed = parse_llm_response(response, "SHOP")
        actions = parsed.get("actions", [{"action": "end_shop"}])

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
        """Ask Claude whether to select or skip, with RAG context."""
        from llm_bot import parse_llm_response
        from base_bot import get_blind_type
        prompt = self._build_rag_prompt(state)
        response = self._call_llm(prompt)

        parsed = parse_llm_response(response, "BLIND_SELECT")
        action = parsed.get("action", "select")
        reasoning = parsed.get("reasoning", "")

        if reasoning:
            logger.info(f"RAG-LLM: {reasoning}")

        blind_type = get_blind_type(state)
        if blind_type == "boss" and action == "skip":
            action = "select"

        return action


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

BENCHMARK_SEEDS = ["AAAAAAA", "BBBBBBB", "CCCCCCC", "DDDDDDD", "EEEEEEE"]
RUNS_PER_SEED = 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG pipeline for Balatro")
    parser.add_argument("--build-index", action="store_true", help="Build Pinecone index from wiki")
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
        print("ERROR: Pinecone API key required. Set $env:PINECONE_API_KEY or pass --pinecone-api-key")
        exit(1)

    if args.build_index:
        logger.info("Building Pinecone index...")
        build_index(pinecone_key)
        logger.info("Index built. Run without --build-index to start the bot.")
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
        cost = (bot._total_tokens_used / 1_000_000) * 0.80
        print(f"  Estimated cost: ${cost:.4f}")