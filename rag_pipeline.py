"""
rag_pipeline.py
RAG pipeline for Balatro using Pinecone + sentence-transformers.

Rules corpus: official game rules only (what a new player reads before playing).
No strategy guides, no tier lists, no joker descriptions, no community meta.
No strategy bias — RAG only prevents Claude from breaking game rules.

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
    LLMBot, parse_llm_response,
    MODEL, MAX_TOKENS, QualitativeLogger,
    build_hand_prompt, build_shop_prompt, build_blind_prompt, build_pack_prompt,
    detect_available_hands,
)
from base_bot import (
    get_hand_cards, get_discards_left, get_hands_left,
    get_shop_cards, get_shop_packs, get_money, get_blind_type, get_ante,
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
EMBED_MODEL    = "all-MiniLM-L6-v2"
TOP_K          = 5

# ---------------------------------------------------------------------------
# Corpus loader — official rules only, no strategy
# ---------------------------------------------------------------------------

# Rules entries that contain strategy bias — excluded from RAG index
EXCLUDED_RULE_NAMES = {
    "Discarding Strategy",  # tells Claude HOW to discard, not just what discard does
}

def build_corpus() -> list[dict]:
    """
    Load only the official rules corpus from data/rules.json.
    Excludes any entries with strategy bias.
    No joker descriptions, no strategy guides, no tier lists.
    Simulates a player who read the rulebook but has no prior strategy knowledge.
    """
    rules_path = "data/rules.json"
    chunks     = []

    try:
        with open(rules_path, "r", encoding="utf-8") as f:
            rules = json.load(f)

        excluded = 0
        for rule in rules:
            name = rule.get("name", "")
            desc = rule.get("description", "")

            if name in EXCLUDED_RULE_NAMES:
                logger.info(f"Excluding strategy-biased rule: {name}")
                excluded += 1
                continue

            if name and desc:
                chunk_id = f"rule_{name.lower().replace(' ', '_')[:40]}"
                chunk_id = unicodedata.normalize("NFKD", chunk_id).encode("ascii","ignore").decode("ascii")
                chunks.append({
                    "id":       chunk_id,
                    "category": "rules",
                    "title":    name,
                    "text":     f"{name}: {desc}",
                })

        logger.info(f"Loaded {len(chunks)} rules from {rules_path} ({excluded} excluded for strategy bias)")

    except FileNotFoundError:
        logger.error(f"Rules file not found: {rules_path}")
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
    pc    = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(PINECONE_INDEX)

    logger.info("Clearing existing index vectors...")
    try:
        index.delete(delete_all=True)
        logger.info("Index cleared")
    except Exception as e:
        logger.warning(f"Could not clear index: {e}")

    logger.info(f"Embedding and uploading {len(chunks)} chunks...")
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch      = chunks[i:i+batch_size]
        texts      = [c["text"] for c in batch]
        embeddings = embedder.encode(texts, show_progress_bar=False)

        vectors = []
        for chunk, embedding in zip(batch, embeddings):
            vectors.append({
                "id":     chunk["id"],
                "values": embedding.tolist(),
                "metadata": {
                    "category": chunk["category"],
                    "title":    chunk["title"],
                    "text":     chunk["text"],
                },
            })

        index.upsert(vectors=vectors)
        logger.info(f"Uploaded batch {i//batch_size+1}/{(len(chunks)+batch_size-1)//batch_size}")

    stats = index.describe_index_stats()
    logger.info(f"Index built. Total vectors: {stats.total_vector_count}")


# ---------------------------------------------------------------------------
# RAG retriever
# ---------------------------------------------------------------------------

class BalatroRAG:
    def __init__(self, pinecone_api_key: str):
        logger.info("Loading embedding model for RAG...")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        pc            = Pinecone(api_key=pinecone_api_key)
        self.index    = pc.Index(PINECONE_INDEX)
        logger.info("RAG retriever ready")

    def retrieve(self, query: str, top_k: int = TOP_K) -> str:
        embedding = self.embedder.encode(query).tolist()
        results   = self.index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
        )
        if not results.matches:
            return ""
        return "\n".join(
            f"- {m.metadata.get('text','')}" for m in results.matches
        )

    def retrieve_for_state(self, state: dict) -> str:
        """Build query from game state phase — retrieve relevant rules only."""
        state_name  = state.get("state","")
        query_parts = []

        if state_name == "SELECTING_HAND":
            query_parts.append("poker hand scoring chips multiplier")
            # Include boss blind effect text so rules about that effect are retrieved
            blinds = state.get("blinds",{})
            for bk in ["small","big","boss"]:
                b = blinds.get(bk,{})
                if b.get("status") in ("SELECT","CURRENT"):
                    effect = b.get("effect","")
                    if effect:
                        query_parts.append(effect)
                    break
            query_parts.append("hand discard rules")

        elif state_name == "SHOP":
            query_parts.append("shop joker planet tarot voucher interest economy money")

        elif state_name == "BLIND_SELECT":
            query_parts.append("blind skip tag reward boss blind select")

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
    read the rulebook before playing but has no strategy knowledge.
    RAG only prevents Claude from breaking game rules, not guide strategy.
    """

    BOT_TYPE   = "rag_llm_bot"
    WANDB_TAGS = ["rag_llm_bot","rag","zero_shot"]

    def __init__(self, pinecone_api_key: str | None = None, **kwargs):
        super().__init__(**kwargs)
        key = pinecone_api_key or os.environ.get("PINECONE_API_KEY")
        if not key:
            raise ValueError(
                "Pinecone API key required. "
                "Set $env:PINECONE_API_KEY='your_key' or pass --pinecone-api-key"
            )
        self.rag          = BalatroRAG(pinecone_api_key=key)
        self._qual_logger = QualitativeLogger("rag_decisions.jsonl")

    def _build_rag_prompt(self, state: dict) -> str:
        """Build base prompt then inject relevant rules above the action section."""
        state_name = state.get("state","")
        builders   = {
            "SELECTING_HAND":       build_hand_prompt,
            "SHOP":                 build_shop_prompt,
            "BLIND_SELECT":         build_blind_prompt,
            "SMODS_BOOSTER_OPENED": build_pack_prompt,
        }
        builder        = builders.get(state_name, build_hand_prompt)
        base_prompt    = builder(state)
        relevant_rules = self.rag.retrieve_for_state(state)

        if relevant_rules:
            rag_section = (
                f"RELEVANT RULES (from official Balatro rulebook):\n"
                f"{relevant_rules}\n\n"
            )
            # Insert rules after the first two lines (header + blank line)
            lines = base_prompt.split("\n", 2)
            if len(lines) >= 2:
                return (lines[0] + "\n" + lines[1] + "\n\n" +
                        rag_section +
                        ("\n".join(lines[2:]) if len(lines) > 2 else ""))

        return base_prompt

    # ------------------------------------------------------------------
    # Override each action to use RAG-augmented prompt
    # ------------------------------------------------------------------

    def select_hand_action(self, state: dict) -> tuple[str, list[int]]:
        self._turn_number += 1
        hand   = get_hand_cards(state)
        prompt = self._build_rag_prompt(state)

        response_raw, in_tok, out_tok = self._call_llm(prompt)
        parsed, success = parse_llm_response(response_raw, "SELECTING_HAND")
        if not success: self._parse_failures += 1

        action = parsed.get("action","play")
        cards  = parsed.get("cards",[0,1,2,3,4])

        available_hands = detect_available_hands(hand)

        cards = [c for c in cards if isinstance(c,int) and 0<=c<len(hand)]
        if action == "play":
            cards = cards[:5]  # game only allows playing 1-5 cards
        if action == "discard":
            cards = cards[:5]  # game only allows discarding 1-5 cards
        if not cards:
            cards = list(range(min(5,len(hand))))
            logger.warning(f"RAG card indices out of range, using fallback {cards}")

        if action == "discard" and get_discards_left(state) <= 0:
            logger.warning("RAG-LLM wanted to discard but none left, overriding to play")
            action = "play"
            cards  = list(range(min(5,len(hand))))

        round_info = state.get("round",{})
        self._qual_logger.log(
            bot_type      = self.BOT_TYPE,
            seed          = self._current_seed,
            ante          = get_ante(state),
            round_num     = state.get("round_num",0),
            state_name    = "SELECTING_HAND",
            prompt        = prompt,
            response_raw  = response_raw,
            parsed_action = parsed,
            parse_success = success,
            input_tokens  = in_tok,
            output_tokens = out_tok,
            extra={
                "turn":               self._turn_number,
                "available_hands":    available_hands,
                "cards_chosen":       cards,
                "action_chosen":      action,
                "chips_before":       round_info.get("chips",0),
                "chips_needed":       self._get_chips_needed(state),
                "hands_remaining":    get_hands_left(state),
                "discards_remaining": get_discards_left(state),
            }
        )

        if parsed.get("reasoning"):
            logger.info(f"RAG-LLM: {parsed['reasoning']}")

        self._last_hand_type = action
        return action, cards

    def select_shop_action(self, state: dict) -> list[dict]:
        prompt = self._build_rag_prompt(state)
        response_raw, in_tok, out_tok = self._call_llm(prompt)
        parsed, success = parse_llm_response(response_raw, "SHOP")
        if not success: self._parse_failures += 1

        actions = parsed.get("actions",[{"action":"end_shop"}])

        self._qual_logger.log(
            bot_type      = self.BOT_TYPE,
            seed          = self._current_seed,
            ante          = get_ante(state),
            round_num     = state.get("round_num",0),
            state_name    = "SHOP",
            prompt        = prompt,
            response_raw  = response_raw,
            parsed_action = parsed,
            parse_success = success,
            input_tokens  = in_tok,
            output_tokens = out_tok,
            extra={"money": get_money(state)}
        )

        if parsed.get("reasoning"):
            logger.info(f"RAG-LLM: {parsed['reasoning']}")

        valid = []
        shop  = state.get("shop",{}).get("cards",[])
        money = get_money(state)

        for ad in actions:
            a = ad.get("action")
            if a == "buy_card":
                idx      = ad.get("index",0)
                if idx < len(shop):
                    cost_raw = shop[idx].get("cost",{})
                    cost     = cost_raw.get("buy",999) if isinstance(cost_raw,dict) else 999
                    if cost <= money and cost > 0:
                        valid.append({"action":"buy_card","index":idx})
                        money -= cost
                    else:
                        logger.warning(f"RAG cannot afford shop card {idx} cost={cost} money={money}")
            elif a == "buy_voucher":
                valid.append(ad)
            elif a == "buy_pack":
                valid.append(ad)
            elif a == "reroll":
                rc = state.get("round",{}).get("reroll_cost",5)
                if money >= rc:
                    valid.append({"action":"reroll"})
                    money -= rc
            elif a == "end_shop":
                valid.append({"action":"end_shop"})
                break

        if not valid or valid[-1].get("action") != "end_shop":
            valid.append({"action":"end_shop"})
        return valid

    def select_blind_action(self, state: dict) -> str:
        prompt = self._build_rag_prompt(state)
        response_raw, in_tok, out_tok = self._call_llm(prompt)
        parsed, success = parse_llm_response(response_raw, "BLIND_SELECT")
        if not success: self._parse_failures += 1

        action = parsed.get("action","select")

        self._qual_logger.log(
            bot_type      = self.BOT_TYPE,
            seed          = self._current_seed,
            ante          = get_ante(state),
            round_num     = state.get("round_num",0),
            state_name    = "BLIND_SELECT",
            prompt        = prompt,
            response_raw  = response_raw,
            parsed_action = parsed,
            parse_success = success,
            input_tokens  = in_tok,
            output_tokens = out_tok,
        )

        if parsed.get("reasoning"):
            logger.info(f"RAG-LLM: {parsed['reasoning']}")

        if get_blind_type(state) == "boss" and action == "skip":
            logger.warning("RAG-LLM tried to skip boss blind, overriding to select")
            action = "select"

        return action

    def select_pack_action(self, state: dict) -> dict:
        prompt = self._build_rag_prompt(state)
        response_raw, in_tok, out_tok = self._call_llm(prompt)

        try:
            json_match = re.search(r'\{.*\}', response_raw, re.DOTALL)
            parsed     = json.loads(json_match.group()) if json_match else {}
            success    = True
        except (json.JSONDecodeError, AttributeError):
            parsed  = {"action":"skip","cards":[],"reasoning":"parse_failure_fallback"}
            success = False
            self._parse_failures += 1

        self._qual_logger.log(
            bot_type      = self.BOT_TYPE,
            seed          = self._current_seed,
            ante          = get_ante(state),
            round_num     = state.get("round_num",0),
            state_name    = "SMODS_BOOSTER_OPENED",
            prompt        = prompt,
            response_raw  = response_raw,
            parsed_action = parsed,
            parse_success = success,
            input_tokens  = in_tok,
            output_tokens = out_tok,
        )

        if parsed.get("reasoning"):
            logger.info(f"RAG-LLM pack: {parsed['reasoning']}")

        return parsed


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

BENCHMARK_SEEDS = [f"SEED{str(i).zfill(3)}" for i in range(1,101)]
RUNS_PER_SEED   = 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG pipeline for Balatro")
    parser.add_argument("--build-index",       action="store_true")
    parser.add_argument("--api-key",           help="Anthropic API key")
    parser.add_argument("--pinecone-api-key",  help="Pinecone API key")
    parser.add_argument("--model",             default=MODEL)
    parser.add_argument("--seeds",             nargs="+", default=BENCHMARK_SEEDS)
    parser.add_argument("--runs-per-seed",     type=int,  default=RUNS_PER_SEED)
    parser.add_argument("--results",           default="results.csv")
    parser.add_argument("--port",              type=int,  default=12346)
    parser.add_argument("--deck",              default="RED")
    parser.add_argument("--stake",             default="WHITE")
    args = parser.parse_args()

    pinecone_key = args.pinecone_api_key or os.environ.get("PINECONE_API_KEY")
    if not pinecone_key:
        print("ERROR: Pinecone API key required."); exit(1)

    if args.build_index:
        build_index(pinecone_key)
        exit(0)

    bot = RAGLLMBot(
        api_key          = args.api_key,
        pinecone_api_key = pinecone_key,
        model            = args.model,
        port             = args.port,
        results_path     = args.results,
        deck             = args.deck,
        stake            = args.stake,
    )

    if not bot.client.health():
        print("ERROR: Cannot connect to Balatro."); exit(1)

    print(f"Running RAGLLMBot on {len(args.seeds)} seeds x {args.runs_per_seed} runs")
    results   = bot.run_experiment(seeds=args.seeds, runs_per_seed=args.runs_per_seed)
    completed = [r for r in results if r["outcome"] in ("won","lost")]
    if completed:
        print(f"\nSummary:")
        print(f"  Completed:      {len(completed)}/{len(results)}")
        print(f"  Wins:           {sum(1 for r in completed if r['outcome']=='won')}")
        print(f"  Avg ante:       {sum(r['final_ante'] for r in completed)/len(completed):.2f}")
        print(f"  Avg round:      {sum(r['final_round'] for r in completed)/len(completed):.2f}")
        print(f"  LLM calls:      {bot._total_llm_calls}")
        print(f"  Input tokens:   {bot._total_input_tokens}")
        print(f"  Output tokens:  {bot._total_output_tokens}")
        from llm_bot import INPUT_COST_PER_1M, OUTPUT_COST_PER_1M
        cost = ((bot._total_input_tokens/1_000_000)*INPUT_COST_PER_1M +
                (bot._total_output_tokens/1_000_000)*OUTPUT_COST_PER_1M)
        print(f"  Estimated cost: ${cost:.4f}")
        print(f"  Parse failures: {bot._parse_failures}")
        print(f"  Decisions logged to: rag_decisions.jsonl")