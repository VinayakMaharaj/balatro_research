"""
llm_bot.py
Zero-shot LLM agent for Balatro using Claude via Anthropic API.

Design principles:
- ZERO strategy bias in prompts — only mechanical game state description
- Claude figures out hand rankings, discard strategy, economy on its own
- Full reasoning logged per decision for qualitative paper analysis
- Input/output tokens tracked separately for accurate cost reporting
- Pack actions enabled — Claude engages with full game
- Parse failures logged explicitly, never silently swallowed
"""

import os
import re
import json
import time
import logging
from collections import Counter
from pathlib import Path
from datetime import datetime

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
    get_jokers,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MODEL            = "claude-sonnet-4-6"
MAX_TOKENS       = 1024
INPUT_COST_PER_1M  = 3.00   # Sonnet 4.6 input
OUTPUT_COST_PER_1M = 15.00  # Sonnet 4.6 output

RANK_VALUES = {"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,
               "T":10,"J":11,"Q":12,"K":13,"A":14}
SUIT_NAMES  = {"H":"Hearts","D":"Diamonds","C":"Clubs","S":"Spades"}

# ---------------------------------------------------------------------------
# Hand availability detector — for qualitative logging
# ---------------------------------------------------------------------------

def detect_available_hands(cards: list[dict]) -> list[str]:
    """
    Detect which hand types are actually available in the current hand.
    Used purely for logging — never fed to Claude.
    """
    if not cards: return []
    available = []

    def parse(c):
        v = c.get("value",{})
        return v.get("rank","2"), v.get("suit","S")

    ranks = [parse(c)[0] for c in cards]
    suits = [parse(c)[1] for c in cards]
    rc    = Counter(ranks)
    sc    = Counter(suits)
    vals  = sorted(RANK_VALUES.get(r,0) for r in ranks)
    counts = sorted(rc.values(), reverse=True)

    is_flush    = max(sc.values()) >= 5 if sc else False
    uv          = sorted(set(vals))
    is_straight = len(uv) >= 5 and any(uv[i+4]-uv[i]==4 for i in range(len(uv)-4))

    # Check straight flush
    for suit, count in sc.items():
        if count >= 5:
            sv = sorted(RANK_VALUES.get(parse(c)[0],0) for c in cards if parse(c)[1]==suit)
            uv2 = sorted(set(sv))
            if any(uv2[i+4]-uv2[i]==4 for i in range(len(uv2)-4)):
                available.append("straight_flush")
                break

    if counts[0] == 4:                                    available.append("four_of_a_kind")
    if counts[0]==3 and len(counts)>1 and counts[1]==2:   available.append("full_house")
    if is_flush:                                          available.append("flush")
    if is_straight:                                       available.append("straight")
    if counts[0] == 3:                                    available.append("three_of_a_kind")
    if counts[0]==2 and len(counts)>1 and counts[1]==2:   available.append("two_pair")
    if counts[0] == 2:                                    available.append("pair")
    available.append("high_card")

    return available


# ---------------------------------------------------------------------------
# Qualitative decision logger
# ---------------------------------------------------------------------------

class QualitativeLogger:
    """
    Logs every LLM decision to JSONL for qualitative paper analysis.
    Captures: full prompt, full response, parse success/failure,
    available hands vs chosen, token usage, reasoning text.
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
        response_raw: str,
        parsed_action: dict,
        parse_success: bool,
        input_tokens: int,
        output_tokens: int,
        extra: dict = None,
    ):
        record = {
            "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "bot_type":      bot_type,
            "seed":          seed,
            "ante":          ante,
            "round":         round_num,
            "state":         state_name,
            "prompt_chars":  len(prompt),
            "response_raw":  response_raw,           # full, not truncated
            "parsed_action": parsed_action,
            "reasoning":     parsed_action.get("reasoning",""),
            "parse_success": parse_success,
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
            **(extra or {}),
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Game state formatters — MECHANICAL ONLY, zero strategy
# ---------------------------------------------------------------------------

def format_hand(cards: list[dict]) -> str:
    """Show cards with all properties. No hints about what's good."""
    if not cards: return "  (empty)"
    lines = []
    for i, card in enumerate(cards):
        v     = card.get("value",{})
        rank  = v.get("rank","?")
        suit  = SUIT_NAMES.get(v.get("suit","?"), v.get("suit","?"))
        parts = [f"{i}: {rank} of {suit}"]

        # Enhancement — describe mechanically what it does
        enh = card.get("enhancement","") or card.get("label","")
        enh_desc = {
            "glass":  "Glass (x2 mult when scored, 1/4 chance to break)",
            "steel":  "Steel (x1.5 mult while held in hand)",
            "gold":   "Gold (+$3 when hand is played)",
            "mult":   "Mult (+4 mult when scored)",
            "bonus":  "Bonus (+30 chips when scored)",
            "wild":   "Wild (counts as any suit)",
            "lucky":  "Lucky (1/5 chance +20 mult, 1/15 chance +$20)",
        }
        if enh and enh.lower() in enh_desc:
            parts.append(enh_desc[enh.lower()])

        # Seal
        seal = card.get("seal","")
        seal_desc = {
            "red":    "Red seal (retrigger card scoring once)",
            "blue":   "Blue seal (create planet card for final hand if held)",
            "gold":   "Gold seal (+$3 when hand containing this card is played)",
            "purple": "Purple seal (create tarot card when discarded)",
        }
        if seal and seal.lower() in seal_desc:
            parts.append(seal_desc[seal.lower()])

        # Debuff
        if card.get("debuff", False):
            parts.append("DEBUFFED (this card does not score)")

        lines.append("  " + " | ".join(parts))
    return "\n".join(lines)


def format_jokers(state: dict) -> str:
    joker_data = state.get("jokers",{})
    if isinstance(joker_data, dict):
        cards = joker_data.get("cards",[])
        count = joker_data.get("count",0)
        limit = joker_data.get("limit",5)
    else:
        cards = joker_data if isinstance(joker_data,list) else []
        count = len(cards)
        limit = 5

    if not cards:
        return f"  (none) [{count}/{limit} slots]"

    lines = [f"  [{count}/{limit} slots used]"]
    for i, card in enumerate(cards):
        label = card.get("label","Unknown")
        desc  = card.get("description","")
        lines.append(f"  {i}: {label}" + (f" — {desc}" if desc else ""))
    return "\n".join(lines)


def format_shop(state: dict) -> str:
    shop      = state.get("shop",{})
    cards     = shop.get("cards",[])
    money     = get_money(state)
    rc        = state.get("round",{}).get("reroll_cost",5)
    lines     = [f"  Money: ${money} | Reroll cost: ${rc}"]

    if cards:
        lines.append("  Available:")
        for i, card in enumerate(cards):
            label    = card.get("label","Unknown")
            card_set = card.get("set","") or card.get("ability",{}).get("set","")
            cost_raw = card.get("cost",{})
            cost     = cost_raw.get("buy","?") if isinstance(cost_raw,dict) else cost_raw
            desc     = card.get("description","") or card.get("value",{}).get("effect","")
            can      = "✓" if isinstance(cost,int) and cost<=money else "✗ (can't afford)"
            lines.append(f"  card {i}: {label} ({card_set}) ${cost} {can}" +
                        (f" — {desc}" if desc else ""))
    else:
        lines.append("  (no cards available)")

    # Vouchers
    vouchers = state.get("shop_vouchers",{})
    if isinstance(vouchers,dict): vouchers = vouchers.get("cards",[])
    if vouchers:
        lines.append("  Vouchers:")
        for i, v in enumerate(vouchers):
            cost_raw = v.get("cost",{})
            cost     = cost_raw.get("buy","?") if isinstance(cost_raw,dict) else cost_raw
            lines.append(f"  voucher {i}: {v.get('label','?')} ${cost} — {v.get('description','')}")

    # Booster packs
    boosters = state.get("shop_booster",{})
    if isinstance(boosters,dict): boosters = boosters.get("cards",[])
    if boosters:
        lines.append("  Booster packs:")
        for i, p in enumerate(boosters):
            cost_raw = p.get("cost",{})
            cost     = cost_raw.get("buy","?") if isinstance(cost_raw,dict) else cost_raw
            lines.append(f"  pack {i}: {p.get('label','?')} ${cost} — {p.get('description','')}")

    return "\n".join(lines)


def format_blinds(state: dict) -> str:
    blinds = state.get("blinds",{})
    lines  = []
    for bk in ["small","big","boss"]:
        b      = blinds.get(bk,{})
        name   = b.get("name",bk)
        score  = b.get("score",0)
        status = b.get("status","UPCOMING")
        effect = b.get("effect","")
        tag    = b.get("tag_name","")
        line   = f"  {bk.upper()}: {name} | Chips required: {score} | [{status}]"
        if effect: line += f" | Effect: {effect}"
        if tag:    line += f" | Skip reward: {tag}"
        lines.append(line)
    return "\n".join(lines)


def format_pack_cards(state: dict) -> str:
    pack_cards = state.get("pack_cards",{}).get("cards",[])
    choices    = state.get("pack_cards",{}).get("choose",1)
    if not pack_cards: return "  (no cards available)"
    lines = [f"  Must choose {choices} card(s) (or skip all):"]
    for i, card in enumerate(pack_cards):
        label    = card.get("label","Unknown")
        card_set = card.get("set","")
        desc     = card.get("description","") or card.get("value",{}).get("effect","")
        lines.append(f"  {i}: {label} ({card_set})" + (f" — {desc}" if desc else ""))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt builders — ZERO STRATEGY, purely mechanical
# ---------------------------------------------------------------------------

def build_hand_prompt(state: dict) -> str:
    ante     = state.get("ante_num",0)
    round_n  = state.get("round_num",0)
    money    = get_money(state)
    round_i  = state.get("round",{})
    hands_l  = round_i.get("hands_left",0)
    disc_l   = round_i.get("discards_left",0)
    chips    = round_i.get("chips",0)

    blinds      = state.get("blinds",{})
    blind_name  = "?"
    chips_need  = 0
    blind_eff   = ""
    for bk in ["small","big","boss"]:
        b = blinds.get(bk,{})
        if b.get("status") in ("SELECT","CURRENT"):
            blind_name = b.get("name",bk)
            chips_need = b.get("score",0)
            blind_eff  = b.get("effect","")
            break

    return f"""You are playing Balatro, a poker-based roguelike card game.

GAME STATE:
  Ante: {ante} | Round: {round_n} | Money: ${money}
  Blind: {blind_name} | Chips needed: {chips_need} | Chips scored so far: {chips}
  Hands remaining: {hands_l} | Discards remaining: {disc_l}
{f"  Blind effect: {blind_eff}" if blind_eff else ""}

YOUR HAND ({len(get_hand_cards(state))} cards):
{format_hand(get_hand_cards(state))}

YOUR JOKERS:
{format_jokers(state)}

ACTIONS:
  play: select 1-5 card indices to play as a poker hand. The hand type is determined automatically from the cards played. Scored chips go toward the blind requirement.
  discard: select 1-5 card indices to discard and draw new cards. Does not score chips. Cannot discard if discards_remaining is 0.

Respond ONLY in this exact JSON format:
{{"action": "play" or "discard", "cards": [list of card indices], "reasoning": "your reasoning in 2 sentences max"}}"""


def build_shop_prompt(state: dict) -> str:
    ante  = state.get("ante_num",0)
    money = get_money(state)

    return f"""You are playing Balatro, a poker-based roguelike card game.

GAME STATE:
  Ante: {ante} | Money: ${money}

SHOP:
{format_shop(state)}

YOUR JOKERS:
{format_jokers(state)}

SHOP MECHANICS:
  buy_card: purchase a card from the shop (joker, planet, tarot, spectral, or playing card)
  buy_voucher: purchase a voucher (permanent upgrade)
  buy_pack: purchase a booster pack (opens and lets you choose cards)
  reroll: spend money to refresh shop cards
  end_shop: leave the shop and proceed to next blind
  You may perform multiple actions before end_shop. Always end with end_shop.
  You cannot buy cards you cannot afford.
  Interest: at end of each round you earn $1 per $5 you hold (maximum $5 interest per round).

Respond ONLY in this exact JSON format:
{{"actions": [{{"action": "buy_card", "index": 0}}, {{"action": "end_shop"}}], "reasoning": "your reasoning in 2 sentences max"}}"""


def build_blind_prompt(state: dict) -> str:
    ante = state.get("ante_num",0)

    return f"""You are playing Balatro, a poker-based roguelike card game.

GAME STATE:
  Ante: {ante}

BLIND SELECTION:
{format_blinds(state)}

YOUR JOKERS:
{format_jokers(state)}

BLIND MECHANICS:
  select: play this blind to earn money and progress
  skip: skip small or big blind to receive the tag reward instead (no money earned from blind)
  The boss blind cannot be skipped — you must always select it.
  You are currently choosing whether to select or skip the blind shown as SELECT status.

Respond ONLY in this exact JSON format:
{{"action": "select" or "skip", "reasoning": "your reasoning in 1 sentence max"}}"""


def build_pack_prompt(state: dict) -> str:
    ante = state.get("ante_num",0)

    return f"""You are playing Balatro, a poker-based roguelike card game.

GAME STATE:
  Ante: {ante}

BOOSTER PACK:
{format_pack_cards(state)}

YOUR JOKERS:
{format_jokers(state)}

PACK MECHANICS:
  pick: choose exactly the required number of cards (specified above)
  skip: take nothing from this pack
  Cards chosen go directly into your collection (jokers to joker slots, consumables to consumable slots, playing cards to deck).

Respond ONLY in this exact JSON format:
{{"action": "pick" or "skip", "cards": [list of chosen indices, empty if skipping], "reasoning": "your reasoning in 1 sentence max"}}"""


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def parse_llm_response(response_text: str, state_name: str) -> tuple[dict, bool]:
    """
    Returns (parsed_dict, parse_success).
    Never silently swallows failures — always returns explicit success flag.
    """
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            return parsed, True
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error: {e} | Response: {response_text[:300]}")

    logger.warning(f"Parse FAILED for state={state_name} | Response: {response_text[:300]}")

    # Fallbacks — minimal safe defaults
    if state_name == "SELECTING_HAND":
        return {"action":"play","cards":[0,1,2,3,4],"reasoning":"parse_failure_fallback"}, False
    elif state_name == "SHOP":
        return {"actions":[{"action":"end_shop"}],"reasoning":"parse_failure_fallback"}, False
    elif state_name == "BLIND_SELECT":
        return {"action":"select","reasoning":"parse_failure_fallback"}, False
    elif state_name == "SMODS_BOOSTER_OPENED":
        return {"action":"skip","cards":[],"reasoning":"parse_failure_fallback"}, False
    return {}, False


# ---------------------------------------------------------------------------
# LLM Bot
# ---------------------------------------------------------------------------

class LLMBot(BaseBot):
    BOT_TYPE   = "llm_bot"
    WANDB_TAGS = ["llm_bot","zero_shot"]

    def __init__(self, api_key: str | None = None, model: str = MODEL, **kwargs):
        super().__init__(**kwargs)
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("Anthropic API key required.")
        self.llm   = anthropic.Anthropic(api_key=key)
        self.model = model

        self._total_input_tokens  = 0
        self._total_output_tokens = 0
        self._total_tokens_used   = 0
        self._total_llm_calls     = 0
        self._parse_failures      = 0
        self._cost_per_1m_tokens  = (INPUT_COST_PER_1M + OUTPUT_COST_PER_1M) / 2  # blended for base_bot compat

        self._qual_logger  = QualitativeLogger("llm_decisions.jsonl")
        self._current_seed = "unknown"
        self._turn_number  = 0

    def _call_llm(self, prompt: str) -> tuple[str, int, int]:
        """Returns (response_text, input_tokens, output_tokens)."""
        self._total_llm_calls += 1
        message = self.llm.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            messages=[{"role":"user","content":prompt}],
        )
        response_text  = message.content[0].text
        input_tokens   = message.usage.input_tokens
        output_tokens  = message.usage.output_tokens

        self._total_input_tokens  += input_tokens
        self._total_output_tokens += output_tokens
        self._total_tokens_used   += input_tokens + output_tokens

        cost = ((self._total_input_tokens  / 1_000_000) * INPUT_COST_PER_1M +
                (self._total_output_tokens / 1_000_000) * OUTPUT_COST_PER_1M)
        logger.info(
            f"LLM call #{self._total_llm_calls} | "
            f"in={input_tokens} out={output_tokens} | "
            f"total_tokens={self._total_tokens_used} | "
            f"cost=${cost:.4f}"
        )
        return response_text, input_tokens, output_tokens

    def _on_game_start(self, seed: str):
        self._current_seed    = seed
        self._turn_number     = 0
        self._total_input_tokens  = 0
        self._total_output_tokens = 0
        self._total_tokens_used   = 0
        self._total_llm_calls     = 0
        self._parse_failures      = 0

    def select_hand_action(self, state: dict) -> tuple[str, list[int]]:
        self._turn_number += 1
        hand   = get_hand_cards(state)
        prompt = build_hand_prompt(state)

        response_raw, in_tok, out_tok = self._call_llm(prompt)
        parsed, success = parse_llm_response(response_raw, "SELECTING_HAND")
        if not success: self._parse_failures += 1

        action  = parsed.get("action","play")
        cards   = parsed.get("cards",[0,1,2,3,4])

        # Detect what was actually available — for logging only, never fed to Claude
        available_hands = detect_available_hands(hand)

        # Clamp card indices to valid range
        cards = [c for c in cards if isinstance(c,int) and 0<=c<len(hand)]
        if not cards:
            cards = list(range(min(5,len(hand))))
            logger.warning(f"Card indices out of range, using fallback {cards}")

        # Enforce discard availability
        if action == "discard" and get_discards_left(state) <= 0:
            logger.warning("LLM wanted to discard but none left, overriding to play")
            action = "play"
            cards  = list(range(min(5,len(hand))))

        # Log full decision
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
                "turn":              self._turn_number,
                "available_hands":   available_hands,
                "cards_chosen":      cards,
                "action_chosen":     action,
                "chips_before":      round_info.get("chips",0),
                "chips_needed":      self._get_chips_needed(state),
                "hands_remaining":   get_hands_left(state),
                "discards_remaining":get_discards_left(state),
            }
        )

        if parsed.get("reasoning"):
            logger.info(f"LLM ({self.BOT_TYPE}): {parsed['reasoning']}")

        self._last_hand_type = action
        return action, cards

    def select_shop_action(self, state: dict) -> list[dict]:
        prompt = build_shop_prompt(state)
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
            logger.info(f"LLM ({self.BOT_TYPE}): {parsed['reasoning']}")

        # Validate and filter actions
        valid   = []
        shop    = state.get("shop",{}).get("cards",[])
        money   = get_money(state)

        for ad in actions:
            a = ad.get("action")
            if a == "buy_card":
                idx  = ad.get("index",0)
                if idx < len(shop):
                    cost_raw = shop[idx].get("cost",{})
                    cost     = cost_raw.get("buy",999) if isinstance(cost_raw,dict) else 999
                    if cost <= money and cost > 0:
                        valid.append({"action":"buy_card","index":idx})
                        money -= cost
                    else:
                        logger.warning(f"Cannot afford shop card {idx} cost={cost} money={money}")
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
        prompt = build_blind_prompt(state)
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
            logger.info(f"LLM ({self.BOT_TYPE}): {parsed['reasoning']}")

        # Hard override — boss blind cannot be skipped
        if get_blind_type(state) == "boss" and action == "skip":
            logger.warning("LLM tried to skip boss blind, overriding to select")
            action = "select"

        return action

    def select_pack_action(self, state: dict) -> dict:
        """
        LLMBot engages with packs — not auto-skipped like RLBot.
        Claude decides what to pick based on pack contents and joker state.
        """
        prompt = build_pack_prompt(state)
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
            logger.info(f"LLM pack ({self.BOT_TYPE}): {parsed['reasoning']}")

        return parsed

    def _get_chips_needed(self, state: dict) -> int:
        blinds = state.get("blinds",{})
        for bk in ["small","big","boss"]:
            b = blinds.get(bk,{})
            if b.get("status") in ("SELECT","CURRENT"):
                return b.get("score",0)
        return 0

    def run_game(self, seed: str) -> dict:
        self._current_seed = seed
        return super().run_game(seed)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

BENCHMARK_SEEDS = [f"SEED{str(i).zfill(3)}" for i in range(1,101)]
RUNS_PER_SEED   = 1

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key",       help="Anthropic API key")
    parser.add_argument("--model",         default=MODEL)
    parser.add_argument("--seeds",         nargs="+", default=BENCHMARK_SEEDS)
    parser.add_argument("--runs-per-seed", type=int,  default=RUNS_PER_SEED)
    parser.add_argument("--results",       default="results.csv")
    parser.add_argument("--port",          type=int,  default=12346)
    parser.add_argument("--deck",          default="RED")
    parser.add_argument("--stake",         default="WHITE")
    args = parser.parse_args()

    bot = LLMBot(
        api_key      = args.api_key,
        model        = args.model,
        port         = args.port,
        results_path = args.results,
        deck         = args.deck,
        stake        = args.stake,
    )

    if not bot.client.health():
        print("ERROR: Cannot connect to Balatro."); exit(1)

    print(f"Running LLMBot zero-shot ({args.model}) on {len(args.seeds)} seeds x {args.runs_per_seed} runs")
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
        cost = ((bot._total_input_tokens/1_000_000)*INPUT_COST_PER_1M +
                (bot._total_output_tokens/1_000_000)*OUTPUT_COST_PER_1M)
        print(f"  Estimated cost: ${cost:.4f}")
        print(f"  Parse failures: {bot._parse_failures}")
        print(f"  Decisions logged to: llm_decisions.jsonl")