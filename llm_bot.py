"""
llm_bot.py
Zero-shot LLM agent for Balatro using Claude via Anthropic API.
"""

import os
import re
import json
import logging
import anthropic
from pathlib import Path
from datetime import datetime

from base_bot import (
    BaseBot,
    get_hand_cards,
    get_money,
    get_joker_count,
    get_joker_limit,
    get_shop_cards,
    get_shop_packs,
    get_ante,
    get_blind_type,
    get_hands_left,
    get_discards_left,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 512

# Cost per 1M tokens (Haiku input+output blended estimate)
COST_PER_1M_TOKENS = 0.80


# ---------------------------------------------------------------------------
# Qualitative decision logger
# ---------------------------------------------------------------------------

class QualitativeLogger:
    """
    Logs every LLM decision to a JSONL file for qualitative analysis.
    Each line is one decision with full context: state, prompt, response, parsed action.
    Used for the qualitative analysis section of the paper.
    """

    def __init__(self, log_path: str = "llm_decisions.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        bot_type: str,
        seed: str,
        ante: int,
        round_num: int,
        state_name: str,
        prompt: str,
        response: str,
        parsed_action: dict,
    ):
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "bot_type": bot_type,
            "seed": seed,
            "ante": ante,
            "round": round_num,
            "state": state_name,
            "prompt_length": len(prompt),
            "response": response[:500],  # truncate long responses
            "parsed_action": parsed_action,
            "reasoning": parsed_action.get("reasoning", ""),
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Game state formatter
# ---------------------------------------------------------------------------

def format_hand(cards: list[dict]) -> str:
    if not cards:
        return "  (empty)"
    lines = []
    for i, card in enumerate(cards):
        v = card.get("value", {})
        rank = v.get("rank", "?")
        suit = v.get("suit", "?")
        suit_name = {"H": "Hearts", "D": "Diamonds", "C": "Clubs", "S": "Spades"}.get(suit, suit)
        modifier = card.get("modifier", {})
        mods = []
        if isinstance(modifier, dict):
            if modifier.get("edition"):
                mods.append(modifier["edition"])
            if modifier.get("enhancement"):
                mods.append(modifier["enhancement"])
            if modifier.get("seal"):
                mods.append(modifier["seal"] + " seal")
        mod_str = f" [{', '.join(mods)}]" if mods else ""
        lines.append(f"  {i}: {rank} of {suit_name}{mod_str}")
    return "\n".join(lines)


def format_jokers(jokers_area: dict) -> str:
    cards = jokers_area.get("cards", [])
    count = jokers_area.get("count", 0)
    limit = jokers_area.get("limit", 5)
    if not cards:
        return f"  (none) [{count}/{limit} slots used]"
    lines = [f"  [{count}/{limit} slots used]"]
    for i, card in enumerate(cards):
        label = card.get("label", "Unknown")
        lines.append(f"  {i}: {label}")
    return "\n".join(lines)


def format_shop(shop_area: dict, packs_area: dict) -> str:
    lines = []
    cards = shop_area.get("cards", [])
    if cards:
        lines.append("  Cards/Jokers:")
        for i, card in enumerate(cards):
            label = card.get("label", "Unknown")
            card_set = card.get("set", "")
            cost = card.get("cost", {}).get("buy", "?")
            v = card.get("value", {})
            effect = v.get("effect", "")
            lines.append(f"    card {i}: {label} ({card_set}) - ${cost} - {effect}")
    else:
        lines.append("  Cards/Jokers: (none)")

    packs = packs_area.get("cards", [])
    if packs:
        lines.append("  Packs:")
        for i, pack in enumerate(packs):
            label = pack.get("label", "Unknown")
            cost = pack.get("cost", {}).get("buy", "?")
            lines.append(f"    pack {i}: {label} - ${cost}")

    return "\n".join(lines)


def format_blinds(blinds: dict) -> str:
    lines = []
    for blind_key in ["small", "big", "boss"]:
        b = blinds.get(blind_key, {})
        name = b.get("name", blind_key)
        score = b.get("score", 0)
        status = b.get("status", "UPCOMING")
        effect = b.get("effect", "")
        tag = b.get("tag_name", "")
        tag_str = f" | Skip reward: {tag}" if tag else ""
        effect_str = f" | Effect: {effect}" if effect else ""
        lines.append(f"  {blind_key.upper()}: {name} (need {score} chips) [{status}]{effect_str}{tag_str}")
    return "\n".join(lines)


def format_state_for_llm(state: dict) -> str:
    state_name = state.get("state", "UNKNOWN")
    ante = state.get("ante_num", 0)
    round_num = state.get("round_num", 0)
    money = state.get("money", 0)

    round_info = state.get("round", {})
    hands_left = round_info.get("hands_left", 0)
    discards_left = round_info.get("discards_left", 0)
    chips = round_info.get("chips", 0)

    blinds = state.get("blinds", {})
    jokers = state.get("jokers", {})
    hand = state.get("hand", {})
    shop = state.get("shop", {})
    packs = state.get("packs", {})

    prompt = f"You are playing Balatro. Ante: {ante} | Round: {round_num} | Money: ${money}\n\n"

    if state_name == "SELECTING_HAND":
        current_blind = None
        for bk in ["small", "big", "boss"]:
            b = blinds.get(bk, {})
            if b.get("status") in ("SELECT", "CURRENT"):
                current_blind = b
                break

        chips_needed = current_blind.get("score", 0) if current_blind else 0
        blind_name = current_blind.get("name", "?") if current_blind else "?"
        blind_effect = current_blind.get("effect", "") if current_blind else ""

        prompt += f"""BLIND: {blind_name} | Need: {chips_needed} chips | Scored: {chips}
{f"Blind effect: {blind_effect}" if blind_effect else ""}
Hands left: {hands_left} | Discards left: {discards_left}

HAND:
{format_hand(hand.get("cards", []))}

JOKERS:
{format_jokers(jokers)}

Respond ONLY in this JSON format:
{{"action": "play" or "discard", "cards": [indices], "reasoning": "1 sentence max"}}

Rules: play 1-5 cards to score, discard 1-5 to draw new. Beat {chips_needed} total chips.
Best hands: Flush > Straight > Four of a Kind > Full House > Three of a Kind > Two Pair > Pair
Discard strategy: if you have no pair or better, discard your 3 weakest cards to fish for a stronger hand. Use discards aggressively early — wasted discards are wasted value."""

    elif state_name == "SHOP":
        reroll_cost = round_info.get("reroll_cost", 5)
        prompt += f"""SHOP - Money: ${money} | Reroll: ${reroll_cost}

{format_shop(shop, packs)}

JOKERS: {get_joker_count(state)}/{get_joker_limit(state)} slots
{format_jokers(jokers)}

Respond ONLY in this JSON format:
{{"actions": [{{"action": "buy_card", "index": 0}}, {{"action": "end_shop"}}]}}

Valid actions: buy_card, buy_pack, reroll, end_shop. Always end with end_shop.
Economy rules:
- INTEREST: you earn $1 per $5 held at end of shop (max $5 bonus). Holding $20+ is worth $4/round.
- SPEND FLOOR: only buy if your money after purchase stays >= $6. Never spend down to $0-$5.
- SKIP REWARD: skipping small or big blind gives a tag (free card/joker). Skip if your hand is strong enough to beat the blind easily.
- PRIORITY: Jokers > consumables > packs. Only reroll if you have $10+ after reroll cost."""

    elif state_name == "BLIND_SELECT":
        prompt += f"""BLIND SELECTION:
{format_blinds(blinds)}

JOKERS:
{format_jokers(jokers)}

Respond ONLY in this JSON format:
{{"action": "select" or "skip", "reasoning": "brief"}}

Rules: skipping small/big gives a tag (free joker, free card, or other bonus) — skip if you have a strong joker setup or a comfortable chip lead. Boss CANNOT be skipped. When in doubt on small blind, skip it."""

    return prompt


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def parse_llm_response(response_text: str, state_name: str) -> dict:
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            return parsed
    except json.JSONDecodeError:
        pass

    logger.warning(f"Failed to parse LLM response, using fallback. Response: {response_text[:200]}")

    if state_name == "SELECTING_HAND":
        return {"action": "play", "cards": [0, 1, 2, 3, 4], "reasoning": "fallback"}
    elif state_name == "SHOP":
        return {"actions": [{"action": "end_shop", "reasoning": "fallback"}]}
    elif state_name == "BLIND_SELECT":
        return {"action": "select", "reasoning": "fallback"}

    return {}


# ---------------------------------------------------------------------------
# LLM Bot
# ---------------------------------------------------------------------------

class LLMBot(BaseBot):
    BOT_TYPE = "llm_bot"
    WANDB_TAGS = ["llm_bot", "zero_shot"]

    def __init__(self, api_key: str | None = None, model: str = MODEL, **kwargs):
        super().__init__(**kwargs)
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("Anthropic API key required.")
        from balatro_client import BalatroClient
        self.client = BalatroClient(
            host=kwargs.get("host", "127.0.0.1"),
            port=kwargs.get("port", 12346),
        )
        self.llm = anthropic.Anthropic(api_key=key)
        self.model = model
        self._total_tokens_used = 0
        self._total_llm_calls = 0
        self._qual_logger = QualitativeLogger("llm_decisions.jsonl")
        self._current_seed = "unknown"

    def _call_llm(self, prompt: str) -> str:
        self._total_llm_calls += 1
        message = self.llm.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = message.content[0].text
        tokens_used = message.usage.input_tokens + message.usage.output_tokens
        self._total_tokens_used += tokens_used
        cost = (self._total_tokens_used / 1_000_000) * COST_PER_1M_TOKENS
        logger.info(
            f"LLM call #{self._total_llm_calls} | "
            f"tokens={tokens_used} | "
            f"total={self._total_tokens_used} | "
            f"cost=${cost:.4f}"
        )
        return response_text

    def run_game(self, seed: str) -> dict:
        self._current_seed = seed
        return super().run_game(seed)

    def select_hand_action(self, state: dict) -> tuple[str, list[int]]:
        prompt = format_state_for_llm(state)
        response = self._call_llm(prompt)
        parsed = parse_llm_response(response, "SELECTING_HAND")

        action = parsed.get("action", "play")
        cards = parsed.get("cards", [0, 1, 2, 3, 4])
        reasoning = parsed.get("reasoning", "")

        if reasoning:
            logger.info(f"LLM: {reasoning}")

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
            logger.warning("LLM wanted to discard but none left, playing instead")
            action = "play"
            cards = list(range(min(5, len(hand))))

        return action, cards

    def select_shop_action(self, state: dict) -> list[dict]:
        prompt = format_state_for_llm(state)
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
        prompt = format_state_for_llm(state)
        response = self._call_llm(prompt)
        parsed = parse_llm_response(response, "BLIND_SELECT")

        action = parsed.get("action", "select")
        reasoning = parsed.get("reasoning", "")

        if reasoning:
            logger.info(f"LLM: {reasoning}")

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
            logger.warning("LLM tried to skip boss blind, overriding to select")
            action = "select"

        return action


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

BENCHMARK_SEEDS = ["AAAAAAA", "BBBBBBB", "CCCCCCC", "DDDDDDD", "EEEEEEE"]
RUNS_PER_SEED = 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LLM Balatro bot")
    parser.add_argument("--api-key", help="Anthropic API key")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--seeds", nargs="+", default=BENCHMARK_SEEDS)
    parser.add_argument("--runs-per-seed", type=int, default=RUNS_PER_SEED)
    parser.add_argument("--results", default="results.csv")
    parser.add_argument("--port", type=int, default=12346)
    parser.add_argument("--deck", default="RED")
    parser.add_argument("--stake", default="WHITE")
    args = parser.parse_args()

    bot = LLMBot(
        api_key=args.api_key,
        model=args.model,
        port=args.port,
        results_path=args.results,
        deck=args.deck,
        stake=args.stake,
    )

    if not bot.client.health():
        print("ERROR: Cannot connect to Balatro.")
        exit(1)

    print(f"Running LLMBot ({args.model}) on {len(args.seeds)} seeds x {args.runs_per_seed} runs")

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
        print(f"  Estimated cost: ${(bot._total_tokens_used / 1_000_000) * COST_PER_1M_TOKENS:.4f}")
        print(f"  Decisions logged to: llm_decisions.jsonl")
