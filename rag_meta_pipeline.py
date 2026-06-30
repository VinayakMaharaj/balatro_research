"""
rag_meta_pipeline.py
RAG agent augmented with MetaBot v2 strategy corpus.

Separate from rag_pipeline.py (rules only). This pipeline retrieves
strategy knowledge that exactly mirrors MetaBot v2's decision logic,
encoded as retrievable text chunks in Pinecone.

Research question: given identical strategic knowledge, does an LLM
executing via reasoning match a deterministic agent executing via code?

Index: balatro-strategy (separate from balatro-rules)
Corpus: data/strategy.json (15 chunks mirroring MetaBot v2)

Build index (one-time):
    python rag_meta_pipeline.py --build-index

Run bot:
    python rag_meta_pipeline.py --seeds SEED001 ... SEED050 --results results_final.csv
"""

import os
import re
import json
import logging
import argparse
import unicodedata

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from llm_bot import (
    LLMBot, parse_llm_response,
    MODEL, MAX_TOKENS, QualitativeLogger,
    build_hand_prompt, build_shop_prompt, build_blind_prompt, build_pack_prompt,
    detect_available_hands, validate_hand_action, validate_shop_actions,
    validate_pack_action, count_self_corrections,
)
from base_bot import (
    get_hand_cards, get_discards_left, get_hands_left,
    get_shop_cards, get_money, get_blind_type, get_ante, get_joker_count,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PINECONE_INDEX = "balatro-strategy"
EMBED_MODEL    = "all-MiniLM-L6-v2"
TOP_K          = 5
STRATEGY_PATH  = "data/strategy.json"


# ---------------------------------------------------------------------------
# Corpus loader
# ---------------------------------------------------------------------------

def build_corpus() -> list[dict]:
    """
    Load strategy corpus from data/strategy.json.
    These chunks exactly mirror MetaBot v2's decision logic as retrievable text.
    No rules content — purely strategy knowledge.
    """
    chunks = []
    try:
        with open(STRATEGY_PATH, "r", encoding="utf-8") as f:
            strategy = json.load(f)

        for entry in strategy:
            name = entry.get("name", "")
            desc = entry.get("description", "")
            if name and desc:
                chunk_id = f"strat_{name.lower().replace(' ', '_')[:40]}"
                chunk_id = unicodedata.normalize("NFKD", chunk_id).encode("ascii","ignore").decode("ascii")
                chunks.append({
                    "id":       chunk_id,
                    "category": "strategy",
                    "title":    name,
                    "text":     f"{name}: {desc}",
                })

        logger.info(f"Loaded {len(chunks)} strategy chunks from {STRATEGY_PATH}")

    except FileNotFoundError:
        logger.error(f"Strategy file not found: {STRATEGY_PATH}")
    except Exception as e:
        logger.error(f"Failed to load strategy corpus: {e}")

    return chunks


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def build_index(pinecone_api_key: str) -> None:
    logger.info("Loading embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL)

    logger.info("Building strategy corpus...")
    chunks = build_corpus()
    if not chunks:
        logger.error("No chunks to index. Check data/strategy.json exists.")
        return

    logger.info(f"Connecting to Pinecone index: {PINECONE_INDEX}")
    pc = Pinecone(api_key=pinecone_api_key)

    existing = [idx["name"] for idx in pc.list_indexes()]
    if PINECONE_INDEX not in existing:
        from pinecone import ServerlessSpec
        logger.info(f"Index '{PINECONE_INDEX}' not found, creating it...")
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=384,  # all-MiniLM-L6-v2 output dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        import time as _time
        logger.info("Waiting for index to be ready...")
        _time.sleep(10)

    index = pc.Index(PINECONE_INDEX)

    logger.info("Clearing existing vectors...")
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
    logger.info(f"Strategy index built. Total vectors: {stats.total_vector_count}")


# ---------------------------------------------------------------------------
# RAG retriever
# ---------------------------------------------------------------------------

class StrategyRAG:
    def __init__(self, pinecone_api_key: str):
        logger.info("Loading embedding model for strategy RAG...")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        pc            = Pinecone(api_key=pinecone_api_key)
        self.index    = pc.Index(PINECONE_INDEX)
        logger.info("Strategy RAG retriever ready")

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
        """Build query from game state to retrieve relevant strategy chunks."""
        state_name  = state.get("state", "")
        ante        = state.get("ante_num", 1)
        query_parts = []

        if state_name == "SELECTING_HAND":
            query_parts.append("hand selection priority discard strategy dominant hand")
            # Include boss blind effect for boss-specific strategy
            blinds = state.get("blinds", {})
            for bk in ["small","big","boss"]:
                b = blinds.get(bk, {})
                if b.get("status") in ("SELECT","CURRENT"):
                    if bk == "boss":
                        effect = b.get("effect","")
                        name   = b.get("name","")
                        if effect or name:
                            query_parts.append(f"boss blind {name} {effect} adaptation")
                    break

        elif state_name == "SHOP":
            if ante >= 3:
                query_parts.append("xmult joker priority blueprint brainstorm late game transition")
            else:
                query_parts.append("joker priority economy interest planet card dominant hand early game")
            query_parts.append("shop reroll slot management")

        elif state_name == "BLIND_SELECT":
            query_parts.append("blind selection skip strategy ante economy interest")

        elif state_name == "SMODS_BOOSTER_OPENED":
            query_parts.append("booster pack planet card strategy")

        query = " ".join(filter(None, query_parts))
        return self.retrieve(query)


# ---------------------------------------------------------------------------
# RAG Meta Bot
# ---------------------------------------------------------------------------

class RAGMetaBot(LLMBot):
    """
    LLM agent augmented with MetaBot v2 strategy corpus via Pinecone RAG.

    Retrieves strategy knowledge that exactly mirrors MetaBot v2's decision
    logic, encoded as text. Enables direct comparison:

    MetaBot v2         — same strategy, executed deterministically via code
    RAGMetaBot         — same strategy, executed via LLM reasoning from text

    If RAGMetaBot >= MetaBot v2: LLM can faithfully apply expert strategy
    If RAGMetaBot < MetaBot v2:  gap is execution reliability, not knowledge
    """

    BOT_TYPE   = "rag_meta_bot"
    WANDB_TAGS = ["rag_meta_bot","rag","strategy","zero_shot"]

    def __init__(self, pinecone_api_key: str | None = None, **kwargs):
        super().__init__(**kwargs)
        key = pinecone_api_key or os.environ.get("PINECONE_API_KEY")
        if not key:
            raise ValueError(
                "Pinecone API key required. "
                "Set $env:PINECONE_API_KEY='your_key' or pass --pinecone-api-key"
            )
        self.rag          = StrategyRAG(pinecone_api_key=key)
        self._qual_logger = QualitativeLogger("rag_meta_decisions.jsonl")

    def _build_rag_prompt(self, state: dict) -> str:
        """Build base prompt then inject relevant strategy above the action section."""
        state_name = state.get("state", "")
        builders   = {
            "SELECTING_HAND":       build_hand_prompt,
            "SHOP":                 build_shop_prompt,
            "BLIND_SELECT":         build_blind_prompt,
            "SMODS_BOOSTER_OPENED": build_pack_prompt,
        }
        builder           = builders.get(state_name, build_hand_prompt)
        base_prompt       = builder(state)
        relevant_strategy = self.rag.retrieve_for_state(state)

        if relevant_strategy:
            strategy_section = (
                f"RELEVANT STRATEGY (from expert Balatro strategy guide):\n"
                f"{relevant_strategy}\n\n"
            )
            lines = base_prompt.split("\n", 2)
            if len(lines) >= 2:
                return (lines[0] + "\n" + lines[1] + "\n\n" +
                        strategy_section +
                        ("\n".join(lines[2:]) if len(lines) > 2 else ""))

        return base_prompt

    # -----------------------------------------------------------------------
    # Action methods — same structure as rag_pipeline.py but logs to
    # rag_meta_decisions.jsonl and uses strategy retrieval
    # -----------------------------------------------------------------------

    def select_hand_action(self, state: dict) -> tuple[str, list[int]]:
        self._turn_number += 1
        hand          = get_hand_cards(state)
        discards_left = get_discards_left(state)
        prompt        = self._build_rag_prompt(state)
        round_info    = state.get("round", {})
        is_boss       = get_blind_type(state) == "boss"
        blind_eff     = ""
        for bk in ["small","big","boss"]:
            b = state.get("blinds",{}).get(bk,{})
            if b.get("status") in ("SELECT","CURRENT"):
                blind_eff = b.get("effect",""); break

        parsed, success, raw, in_tok, out_tok, retried = self._call_with_retry(prompt, "SELECTING_HAND")

        action          = parsed.get("action","play")
        cards           = parsed.get("cards",[0,1,2,3,4])
        available_hands = detect_available_hands(hand)
        action, cards   = validate_hand_action(action, cards, hand, discards_left)

        self._self_corrections_total += count_self_corrections(raw)
        if is_boss and blind_eff and blind_eff[:10].lower() in parsed.get("reasoning","").lower():
            self._boss_adaptations += 1

        if action == "play":
            self._hand_recognition_total += 1
            played      = [hand[i] for i in cards if 0<=i<len(hand)]
            played_type = detect_available_hands(played)[0] if played else "high_card"
            if played_type == available_hands[0]:
                self._hand_recognition_hits += 1

        self._qual_logger.log(
            bot_type      = self.BOT_TYPE,
            seed          = self._current_seed,
            ante          = get_ante(state),
            round_num     = state.get("round_num",0),
            state_name    = "SELECTING_HAND",
            prompt        = prompt,
            response_raw  = raw,
            parsed_action = parsed,
            parse_success = success,
            input_tokens  = in_tok,
            output_tokens = out_tok,
            retried       = retried,
            extra={
                "turn":               self._turn_number,
                "available_hands":    available_hands,
                "best_available":     available_hands[0],
                "cards_chosen":       cards,
                "action_chosen":      action,
                "chips_before":       round_info.get("chips",0),
                "chips_needed":       self._get_chips_needed(state),
                "hands_remaining":    get_hands_left(state),
                "discards_remaining": discards_left,
                "is_boss":            is_boss,
                "boss_effect":        blind_eff,
                "self_corrections":   count_self_corrections(raw),
            }
        )

        if parsed.get("reasoning"):
            logger.info(f"RAG-Meta: {parsed['reasoning']}")

        self._last_hand_type = action
        return action, cards

    def select_shop_action(self, state: dict) -> list[dict]:
        prompt                                           = self._build_rag_prompt(state)
        parsed, success, raw, in_tok, out_tok, retried  = self._call_with_retry(prompt, "SHOP")

        self._qual_logger.log(
            bot_type      = self.BOT_TYPE,
            seed          = self._current_seed,
            ante          = get_ante(state),
            round_num     = state.get("round_num",0),
            state_name    = "SHOP",
            prompt        = prompt,
            response_raw  = raw,
            parsed_action = parsed,
            parse_success = success,
            input_tokens  = in_tok,
            output_tokens = out_tok,
            retried       = retried,
            extra={
                "money":            get_money(state),
                "joker_count":      get_joker_count(state),
                "self_corrections": count_self_corrections(raw),
            }
        )

        if parsed.get("reasoning"):
            logger.info(f"RAG-Meta shop: {parsed['reasoning']}")

        return validate_shop_actions(parsed.get("actions",[{"action":"end_shop"}]), state)

    def select_blind_action(self, state: dict) -> str:
        prompt                                           = self._build_rag_prompt(state)
        parsed, success, raw, in_tok, out_tok, retried  = self._call_with_retry(prompt, "BLIND_SELECT")

        action     = parsed.get("action","select")
        ante       = get_ante(state)
        blind_type = get_blind_type(state)

        if action == "skip" and blind_type != "boss":
            self._blind_skips_total += 1
            if ante <= 1:
                self._blind_skips_ante1 += 1

        self._qual_logger.log(
            bot_type      = self.BOT_TYPE,
            seed          = self._current_seed,
            ante          = ante,
            round_num     = state.get("round_num",0),
            state_name    = "BLIND_SELECT",
            prompt        = prompt,
            response_raw  = raw,
            parsed_action = parsed,
            parse_success = success,
            input_tokens  = in_tok,
            output_tokens = out_tok,
            retried       = retried,
            extra={
                "blind_type":       blind_type,
                "joker_count":      get_joker_count(state),
                "skipped":          action == "skip",
                "skip_at_ante1":    ante <= 1 and action == "skip",
                "self_corrections": count_self_corrections(raw),
            }
        )

        if parsed.get("reasoning"):
            logger.info(f"RAG-Meta blind: {parsed['reasoning']}")

        if blind_type == "boss" and action == "skip":
            logger.warning("RAG-Meta tried to skip boss blind, overriding to select")
            action = "select"

        return action

    def select_pack_action(self, state: dict) -> dict:
        prompt                                           = self._build_rag_prompt(state)
        parsed, success, raw, in_tok, out_tok, retried  = self._call_with_retry(prompt, "SMODS_BOOSTER_OPENED")
        safe_parsed = validate_pack_action(parsed, state)

        self._qual_logger.log(
            bot_type      = self.BOT_TYPE,
            seed          = self._current_seed,
            ante          = get_ante(state),
            round_num     = state.get("round_num",0),
            state_name    = "SMODS_BOOSTER_OPENED",
            prompt        = prompt,
            response_raw  = raw,
            parsed_action = parsed,
            parse_success = success,
            input_tokens  = in_tok,
            output_tokens = out_tok,
            retried       = retried,
            extra={
                "pack_choices":     state.get("pack_cards",{}).get("choose",1),
                "pack_cards_count": len(state.get("pack_cards",{}).get("cards",[])),
                "action_taken":     safe_parsed.get("action","skip"),
                "self_corrections": count_self_corrections(raw),
            }
        )

        if parsed.get("reasoning"):
            logger.info(f"RAG-Meta pack: {parsed['reasoning']}")

        return safe_parsed


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

BENCHMARK_SEEDS = [f"SEED{str(i).zfill(3)}" for i in range(1, 101)]
RUNS_PER_SEED   = 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Meta pipeline for Balatro")
    parser.add_argument("--build-index",      action="store_true",
                        help="Build the balatro-strategy Pinecone index from data/strategy.json")
    parser.add_argument("--api-key",          help="Anthropic API key")
    parser.add_argument("--pinecone-api-key", help="Pinecone API key")
    parser.add_argument("--model",            default=MODEL)
    parser.add_argument("--seeds",            nargs="+", default=BENCHMARK_SEEDS)
    parser.add_argument("--runs-per-seed",    type=int,  default=RUNS_PER_SEED)
    parser.add_argument("--results",          default="results.csv")
    parser.add_argument("--port",             type=int,  default=12346)
    parser.add_argument("--deck",             default="RED")
    parser.add_argument("--stake",            default="WHITE")
    args = parser.parse_args()

    pinecone_key = args.pinecone_api_key or os.environ.get("PINECONE_API_KEY")
    if not pinecone_key:
        print("ERROR: Pinecone API key required."); exit(1)

    if args.build_index:
        build_index(pinecone_key)
        exit(0)

    bot = RAGMetaBot(
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

    print(f"Running RAGMetaBot on {len(args.seeds)} seeds x {args.runs_per_seed} runs")
    results   = bot.run_experiment(seeds=args.seeds, runs_per_seed=args.runs_per_seed)
    completed = [r for r in results if r["outcome"] in ("won","lost")]
    if completed:
        from llm_bot import INPUT_COST_PER_1M, OUTPUT_COST_PER_1M
        cost = ((bot._total_input_tokens/1_000_000)*INPUT_COST_PER_1M +
                (bot._total_output_tokens/1_000_000)*OUTPUT_COST_PER_1M)
        print(f"\nSummary:")
        print(f"  Completed:            {len(completed)}/{len(results)}")
        print(f"  Wins:                 {sum(1 for r in completed if r['outcome']=='won')}")
        print(f"  Avg ante:             {sum(r['final_ante'] for r in completed)/len(completed):.2f}")
        print(f"  Avg round:            {sum(r['final_round'] for r in completed)/len(completed):.2f}")
        print(f"  LLM calls:            {bot._total_llm_calls}")
        print(f"  Retries:              {bot._retries}")
        print(f"  Input tokens:         {bot._total_input_tokens}")
        print(f"  Output tokens:        {bot._total_output_tokens}")
        print(f"  Estimated cost:       ${cost:.4f}")
        print(f"  Parse failures:       {bot._parse_failures}")
        print(f"  Blind skips ante 1:   {bot._blind_skips_ante1}")
        print(f"  Blind skips total:    {bot._blind_skips_total}")
        print(f"  Self corrections:     {bot._self_corrections_total}")
        print(f"  Hand recog accuracy:  {bot._hand_recognition_hits}/{bot._hand_recognition_total}")
        print(f"  Boss adaptations:     {bot._boss_adaptations}")
        print(f"  Decisions logged to:  rag_meta_decisions.jsonl")