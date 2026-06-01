"""
llm_bot.py
Zero-shot LLM agent for Balatro using Claude via Anthropic API.

Inherits from BaseBot. At each decision point, formats the current
game state as a structured prompt and asks Claude what to do.
Parses the response into a concrete action.

This is the second tier in the three-tier comparison:
    FlushBot (floor) -> LLMBot (zero-shot) -> RLBot (ceiling)

Run with:
    python llm_bot.py --api-key YOUR_KEY --runs-per-seed 1

Or set environment variable:
    set ANTHROPIC_API_KEY=YOUR_KEY
    python llm_bot.py --runs-per-seed 1
"""

import os
import re
import json
import logging
import anthropic

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

# Model to use - claude-sonnet-4-6 matches BalatroBench for direct comparison
MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 256


# ---------------------------------------------------------------------------
# Game state formatter
# ---------------------------------------------------------------------------

def format_hand(cards: list[dict]) -> str:
    """Format hand cards as readable string with indices."""
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
    """Format jokers as readable string."""
    cards = jokers_area.get("cards", [])
    count = jokers_area.get("count", 0)
    limit = jokers_area.get("limit", 5)
    if not cards:
        return f"  (none) [{count}/{limit} slots used]"
    lines = [f"  [{count}/{limit} slots used]"]
    for i, card in enumerate(cards):
        label = card.get("label", "Unknown")
        v = card.get("value", {})
        effect = v.get("effect", "")
        lines.append(f"  {i}: {label} - {effect}")
    return "\n".join(lines)


def format_shop(shop_area: dict, packs_area: dict) -> str:
    """Format shop cards as readable string with indices and costs."""
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
    """Format blind information."""
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
    """
    Format the full game state as a structured prompt for the LLM.
    Designed to be clear, information-dense, and actionable.
    """
    state_name = state.get("state", "UNKNOWN")
    ante = state.get("ante_num", 0)
    round_num = state.get("round_num", 0)
    money = state.get("money", 0)
    deck = state.get("deck", "RED")
    stake = state.get("stake", "WHITE")
    seed = state.get("seed", "?")

    round_info = state.get("round", {})
    hands_left = round_info.get("hands_left", 0)
    discards_left = round_info.get("discards_left", 0)
    chips = round_info.get("chips", 0)

    blinds = state.get("blinds", {})
    jokers = state.get("jokers", {})
    hand = state.get("hand", {})
    shop = state.get("shop", {})
    packs = state.get("packs", {})

    prompt = f"""You are playing Balatro, a roguelike deckbuilder card game.

CURRENT GAME STATE:
- State: {state_name}
- Ante: {ante} | Round: {round_num}
- Money: ${money}
- Deck: {deck} | Stake: {stake} | Seed: {seed}

"""

    if state_name == "SELECTING_HAND":
        # Get blind score requirement
        current_blind = None
        for bk in ["small", "big", "boss"]:
            b = blinds.get(bk, {})
            if b.get("status") in ("SELECT", "CURRENT"):
                current_blind = b
                break

        chips_needed = current_blind.get("score", 0) if current_blind else 0
        blind_name = current_blind.get("name", "?") if current_blind else "?"
        blind_effect = current_blind.get("effect", "") if current_blind else ""

        prompt += f"""BLIND: {blind_name} | Need: {chips_needed} chips | Scored so far: {chips}
{f"Blind effect: {blind_effect}" if blind_effect else ""}

HANDS LEFT: {hands_left} | DISCARDS LEFT: {discards_left}

YOUR HAND (0-indexed):
{format_hand(hand.get("cards", []))}

YOUR JOKERS:
{format_jokers(jokers)}

DECIDE: What cards to play or discard?

Respond in this exact JSON format:
{{
  "action": "play" or "discard",
  "cards": [list of 0-based card indices],
  "reasoning": "brief explanation"
}}

Rules:
- Play 1-5 cards to score points (flush=5 same suit, straight=5 consecutive, pairs, etc)
- Discard 1-5 cards to draw new ones (costs 1 discard)
- You must beat {chips_needed} chips total to win this blind
- Best hands: Flush > Straight > Four of a Kind > Full House > Three of a Kind > Two Pair > Pair > High Card
"""

    elif state_name == "SHOP":
        reroll_cost = round_info.get("reroll_cost", 5)
        used_vouchers = state.get("used_vouchers", {})

        prompt += f"""SHOP - Money: ${money} | Reroll cost: ${reroll_cost}

AVAILABLE ITEMS:
{format_shop(shop, packs)}

YOUR JOKERS:
{format_jokers(jokers)}



DECIDE: What to buy, or leave the shop?

Respond in this exact JSON format:
{{
  "actions": [
    {{"action": "buy_card", "index": 0, "reasoning": "..."}},
    {{"action": "buy_pack", "index": 0, "reasoning": "..."}},
    {{"action": "reroll", "reasoning": "..."}},
    {{"action": "end_shop", "reasoning": "..."}}
  ]
}}

Rules:
- You can take multiple actions (buy multiple items, reroll, etc)
- Always end with end_shop
- Jokers multiply your score - buy them early
- You have {get_joker_count(state)}/{get_joker_limit(state)} joker slots used
- Keep some money for interest (earn $1 per $5 held at end of round)
"""

    elif state_name == "BLIND_SELECT":
        prompt += f"""BLIND SELECTION - Ante {ante}

BLINDS:
{format_blinds(blinds)}

YOUR JOKERS:
{format_jokers(jokers)}

Money: ${money}

DECIDE: Select or skip the current blind?

Respond in this exact JSON format:
{{
  "action": "select" or "skip",
  "reasoning": "brief explanation"
}}

Rules:
- You must defeat all 3 blinds in an ante to advance
- Skipping small/big blinds gives you a tag reward (shown above)
- Boss blinds CANNOT be skipped
- Skipping is free and you still advance to the next blind
"""

    return prompt


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def parse_llm_response(response_text: str, state_name: str) -> dict:
    """
    Parse Claude's JSON response into an action dict.
    Returns a safe fallback action on parse failure.
    """
    # Extract JSON from response
    try:
        # Try to find JSON block
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            return parsed
    except json.JSONDecodeError:
        pass

    # Fallback actions by state
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
    """
    Zero-shot LLM agent using Claude.

    Formats game state as a structured prompt, asks Claude for an action,
    parses the JSON response, and executes it.

    This is the core LLM tier for the paper. No RAG, no fine-tuning,
    pure zero-shot reasoning from the model's pretrained knowledge.
    """

    BOT_TYPE = "llm_bot"

    def __init__(self, api_key: str | None = None, model: str = MODEL, **kwargs):
        super().__init__(**kwargs)
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("Anthropic API key required. Pass --api-key or set ANTHROPIC_API_KEY env var.")
        self.client = anthropic.Anthropic(api_key=key)
        self.balatro_client = self.client  # will be overwritten by BaseBot
        # Re-init the balatro client separately
        from balatro_client import BalatroClient
        self.client = BalatroClient(
            host=kwargs.get("host", "127.0.0.1"),
            port=kwargs.get("port", 12346),
        )
        self.llm = anthropic.Anthropic(api_key=key)
        self.model = model
        self._total_tokens_used = 0
        self._total_llm_calls = 0

    def _call_llm(self, prompt: str) -> str:
        """Call Claude and return the response text."""
        self._total_llm_calls += 1
        message = self.llm.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = message.content[0].text
        tokens_used = message.usage.input_tokens + message.usage.output_tokens
        self._total_tokens_used += tokens_used
        logger.info(f"LLM call #{self._total_llm_calls} | tokens={tokens_used} | total_tokens={self._total_tokens_used}")
        return response_text

    def select_hand_action(self, state: dict) -> tuple[str, list[int]]:
        """Ask Claude what to play or discard."""
        prompt = format_state_for_llm(state)
        response = self._call_llm(prompt)
        logger.debug(f"LLM response (hand): {response[:300]}")

        parsed = parse_llm_response(response, "SELECTING_HAND")
        action = parsed.get("action", "play")
        cards = parsed.get("cards", [0, 1, 2, 3, 4])
        reasoning = parsed.get("reasoning", "")

        if reasoning:
            logger.info(f"LLM reasoning: {reasoning}")

        # Validate cards are in range
        hand = get_hand_cards(state)
        cards = [c for c in cards if 0 <= c < len(hand)]
        if not cards:
            cards = list(range(min(5, len(hand))))

        # Validate discards available
        if action == "discard" and get_discards_left(state) <= 0:
            logger.warning("LLM wanted to discard but no discards left, playing instead")
            action = "play"
            cards = list(range(min(5, len(hand))))

        return action, cards

    def select_shop_action(self, state: dict) -> list[dict]:
        """Ask Claude what to buy in the shop."""
        prompt = format_state_for_llm(state)
        response = self._call_llm(prompt)
        logger.debug(f"LLM response (shop): {response[:300]}")

        parsed = parse_llm_response(response, "SHOP")
        actions = parsed.get("actions", [{"action": "end_shop", "reasoning": "fallback"}])

        # Validate and clean actions
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
                        logger.warning(f"LLM wanted to buy card {idx} but cannot afford it (cost={cost}, money={money})")

            elif action == "buy_pack":
                idx = action_dict.get("index", 0)
                if idx < len(packs):
                    cost = packs[idx].get("cost", {}).get("buy", 999)
                    if cost <= money:
                        valid_actions.append({"action": "buy_pack", "index": idx})
                        money -= cost
                    else:
                        logger.warning(f"LLM wanted to buy pack {idx} but cannot afford it")

            elif action == "reroll":
                reroll_cost = state.get("round", {}).get("reroll_cost", 5)
                if money >= reroll_cost:
                    valid_actions.append({"action": "reroll"})
                    money -= reroll_cost
                else:
                    logger.warning(f"LLM wanted to reroll but cannot afford it")

            elif action == "end_shop":
                valid_actions.append({"action": "end_shop"})
                break

        # Always end with end_shop
        if not valid_actions or valid_actions[-1].get("action") != "end_shop":
            valid_actions.append({"action": "end_shop"})

        return valid_actions

    def select_blind_action(self, state: dict) -> str:
        """Ask Claude whether to select or skip the blind."""
        prompt = format_state_for_llm(state)
        response = self._call_llm(prompt)
        logger.debug(f"LLM response (blind): {response[:300]}")

        parsed = parse_llm_response(response, "BLIND_SELECT")
        action = parsed.get("action", "select")
        reasoning = parsed.get("reasoning", "")

        if reasoning:
            logger.info(f"LLM reasoning: {reasoning}")

        # Cannot skip boss blind
        blind_type = get_blind_type(state)
        if blind_type == "boss" and action == "skip":
            logger.warning("LLM tried to skip boss blind, overriding to select")
            action = "select"

        return action


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

BENCHMARK_SEEDS = ["AAAAAAA", "BBBBBBB", "CCCCCCC", "DDDDDDD", "EEEEEEE"]
RUNS_PER_SEED = 1  # Start with 1 for testing, scale to 20 for full experiment


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LLM Balatro bot")
    parser.add_argument("--api-key", help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--model", default=MODEL, help=f"Model to use (default: {MODEL})")
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
        print("ERROR: Cannot connect to Balatro. Make sure the game is running with the BalatroBot mod loaded.")
        exit(1)

    print(f"Running LLMBot ({args.model}) on {len(args.seeds)} seeds x {args.runs_per_seed} runs")
    print(f"Total games: {len(args.seeds) * args.runs_per_seed}")
    print(f"Results: {args.results}")

    results = bot.run_experiment(seeds=args.seeds, runs_per_seed=args.runs_per_seed)

    completed = [r for r in results if r["outcome"] in ("won", "lost")]
    if completed:
        avg_round = sum(r["final_round"] for r in completed) / len(completed)
        avg_ante = sum(r["final_ante"] for r in completed) / len(completed)
        wins = sum(1 for r in results if r["outcome"] == "won")
        print(f"\nResults summary:")
        print(f"  Games completed: {len(completed)}/{len(results)}")
        print(f"  Wins: {wins}")
        print(f"  Avg final round: {avg_round:.2f}")
        print(f"  Avg final ante:  {avg_ante:.2f}")
        print(f"  Total LLM calls: {bot._total_llm_calls}")
        print(f"  Total tokens used: {bot._total_tokens_used}")