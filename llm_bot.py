"""
llm_bot.py
Zero-shot LLM agent for Balatro using Claude via Anthropic API.
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
    BaseBot, get_hand_cards, get_money, get_joker_count,
    get_joker_limit, get_shop_cards, get_shop_packs, get_ante,
    get_blind_type, get_hands_left, get_discards_left, get_jokers,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL              = "claude-sonnet-4-6"
MAX_TOKENS         = 1024
INPUT_COST_PER_1M  = 3.00
OUTPUT_COST_PER_1M = 15.00

RANK_VALUES = {"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,"T":10,"J":11,"Q":12,"K":13,"A":14}
SUIT_NAMES  = {"H":"Hearts","D":"Diamonds","C":"Clubs","S":"Spades"}

RETRY_PROMPT_SUFFIX = "\n\nYour previous response could not be parsed as valid JSON. Respond ONLY with a valid JSON object, no other text."

def detect_available_hands(cards):
    if not cards: return ["high_card"]
    def parse(c):
        v = c.get("value",{})
        return v.get("rank","2"), v.get("suit","S")
    ranks  = [parse(c)[0] for c in cards]
    suits  = [parse(c)[1] for c in cards]
    rc     = Counter(ranks)
    sc     = Counter(suits)
    vals   = sorted(RANK_VALUES.get(r,0) for r in ranks)
    counts = sorted(rc.values(), reverse=True)
    is_flush    = max(sc.values()) >= 5 if sc else False
    uv          = sorted(set(vals))
    is_straight = len(uv) >= 5 and any(uv[i+4]-uv[i]==4 for i in range(len(uv)-4))
    available = []
    for suit, count in sc.items():
        if count >= 5:
            sv  = sorted(RANK_VALUES.get(parse(c)[0],0) for c in cards if parse(c)[1]==suit)
            uv2 = sorted(set(sv))
            if any(uv2[i+4]-uv2[i]==4 for i in range(len(uv2)-4)):
                available.append("straight_flush"); break
    if counts[0] == 4:                                    available.append("four_of_a_kind")
    if counts[0]==3 and len(counts)>1 and counts[1]==2:   available.append("full_house")
    if is_flush:                                          available.append("flush")
    if is_straight:                                       available.append("straight")
    if counts[0] == 3:                                    available.append("three_of_a_kind")
    if counts[0]==2 and len(counts)>1 and counts[1]==2:   available.append("two_pair")
    if counts[0] == 2:                                    available.append("pair")
    available.append("high_card")
    return available

def count_self_corrections(text):
    count = 0
    for phrase in ["wait,","wait -","actually,","let me reconsider","let me re-read","i made an error","correction:"]:
        count += text.lower().count(phrase)
    return count

class QualitativeLogger:
    def __init__(self, log_path="llm_decisions.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
    def log(self, bot_type, seed, ante, round_num, state_name, prompt, response_raw,
            parsed_action, parse_success, input_tokens, output_tokens, retried=False, extra=None):
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "bot_type": bot_type, "seed": seed, "ante": ante, "round": round_num,
            "state": state_name, "prompt_chars": len(prompt), "response_raw": response_raw,
            "parsed_action": parsed_action, "reasoning": parsed_action.get("reasoning",""),
            "parse_success": parse_success, "retried": retried,
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "self_corrections": count_self_corrections(response_raw),
            "reasoning_length": len(parsed_action.get("reasoning","")),
            **(extra or {}),
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

def format_hand(cards):
    if not cards: return "  (empty)"
    lines = []
    for i, card in enumerate(cards):
        v    = card.get("value",{})
        rank = v.get("rank","?")
        suit = SUIT_NAMES.get(v.get("suit","?"), v.get("suit","?"))
        parts = [f"{i}: {rank} of {suit}"]
        enh = (card.get("enhancement","") or "none").lower()
        enh_desc = {
            "glass":"Glass (x2 mult when scored, 1/4 chance to break)",
            "steel":"Steel (x1.5 mult while held in hand)",
            "gold":"Gold (+$3 when held at end of round)",
            "mult":"Mult (+4 mult when scored)",
            "bonus":"Bonus (+30 chips when scored)",
            "wild":"Wild (counts as any suit)",
            "lucky":"Lucky (1/5 chance +20 mult, 1/15 chance +$20 when scored)",
        }
        if enh in enh_desc: parts.append(enh_desc[enh])
        seal = (card.get("seal","") or "none").lower()
        seal_desc = {
            "red":"Red seal (retrigger card scoring once)",
            "blue":"Blue seal (create planet card for final hand if held)",
            "gold":"Gold seal (+$3 when held at end of round)",
            "purple":"Purple seal (create tarot card when discarded)",
        }
        if seal in seal_desc: parts.append(seal_desc[seal])
        if card.get("debuff",False): parts.append("DEBUFFED (scores 0 chips)")
        lines.append("  " + " | ".join(parts))
    return "\n".join(lines)

def format_jokers(state):
    jd = state.get("jokers",{})
    if isinstance(jd,dict):
        cards = jd.get("cards",[]); count = jd.get("count",0); limit = jd.get("limit",5)
    else:
        cards = jd if isinstance(jd,list) else []; count = len(cards); limit = 5
    if not cards: return f"  (none) [{count}/{limit} slots]"
    lines = [f"  [{count}/{limit} slots used]"]
    for i, card in enumerate(cards):
        label = card.get("label","Unknown"); desc = card.get("description","")
        lines.append(f"  {i}: {label}" + (f" — {desc}" if desc else ""))
    return "\n".join(lines)

def format_shop(state):
    shop = state.get("shop",{}); cards = shop.get("cards",[]); money = get_money(state)
    rc   = state.get("round",{}).get("reroll_cost",5)
    lines = [f"  Money: ${money} | Reroll cost: ${rc}"]
    if cards:
        lines.append("  Available:")
        for i, card in enumerate(cards):
            label    = card.get("label","Unknown")
            card_set = card.get("set","") or card.get("ability",{}).get("set","")
            cost_raw = card.get("cost",{})
            cost     = cost_raw.get("buy","?") if isinstance(cost_raw,dict) else cost_raw
            desc     = card.get("description","") or card.get("value",{}).get("effect","")
            can      = "✓" if isinstance(cost,int) and cost<=money else "✗ (can't afford)"
            lines.append(f"  card {i}: {label} ({card_set}) ${cost} {can}" + (f" — {desc}" if desc else ""))
    else:
        lines.append("  (no cards available)")
    return "\n".join(lines)

def format_blinds(state):
    blinds = state.get("blinds",{}); lines = []
    for bk in ["small","big","boss"]:
        b = blinds.get(bk,{}); name = b.get("name",bk); score = b.get("score",0)
        status = b.get("status","UPCOMING"); effect = b.get("effect",""); tag = b.get("tag_name","")
        line = f"  {bk.upper()}: {name} | Chips required: {score} | [{status}]"
        if effect: line += f" | Effect: {effect}"
        if tag:    line += f" | Skip reward: {tag}"
        lines.append(line)
    return "\n".join(lines)

def format_pack_cards(state):
    pack_cards = state.get("pack_cards",{}).get("cards",[]); choices = state.get("pack_cards",{}).get("choose",1)
    if not pack_cards: return "  (no cards available)"
    lines = [f"  Must choose {choices} card(s) (or skip all):"]
    for i, card in enumerate(pack_cards):
        label = card.get("label","Unknown"); card_set = card.get("set","")
        desc  = card.get("description","") or card.get("value",{}).get("effect","")
        lines.append(f"  {i}: {label} ({card_set})" + (f" — {desc}" if desc else ""))
    return "\n".join(lines)

def build_hand_prompt(state):
    ante = state.get("ante_num",0); round_n = state.get("round_num",0); money = get_money(state)
    round_i = state.get("round",{}); hands_l = round_i.get("hands_left",0); disc_l = round_i.get("discards_left",0)
    chips = round_i.get("chips",0); blind_name = "?"; chips_need = 0; blind_eff = ""
    for bk in ["small","big","boss"]:
        b = state.get("blinds",{}).get(bk,{})
        if b.get("status") in ("SELECT","CURRENT"):
            blind_name = b.get("name",bk); chips_need = b.get("score",0); blind_eff = b.get("effect",""); break
    hand = get_hand_cards(state)
    return f"""You are playing Balatro, a poker-based roguelike card game.

GAME STATE:
  Ante: {ante} | Round: {round_n} | Money: ${money}
  Blind: {blind_name} | Chips needed: {chips_need} | Chips scored so far: {chips}
  Hands remaining: {hands_l} | Discards remaining: {disc_l}
{f"  Blind effect: {blind_eff}" if blind_eff else ""}

YOUR HAND ({len(hand)} cards):
{format_hand(hand)}

YOUR JOKERS:
{format_jokers(state)}

MECHANICS:
  play: select 1-5 card indices to play as a poker hand. The game detects the best hand type automatically and scores chips. You must reach {chips_need} total chips to beat this blind.
  discard: select 1-5 card indices to discard and draw new cards. Does not score chips. Cannot discard if discards_remaining is 0.
  If hands_remaining reaches 0 before beating the blind, you lose the run.

Respond ONLY in this exact JSON format:
{{"action": "play" or "discard", "cards": [list of card indices 0-{len(hand)-1}], "reasoning": "your reasoning in 2 sentences max"}}"""

def build_shop_prompt(state):
    ante = state.get("ante_num",0); money = get_money(state)
    return f"""You are playing Balatro, a poker-based roguelike card game.

GAME STATE:
  Ante: {ante} | Money: ${money}

SHOP:
{format_shop(state)}

YOUR JOKERS:
{format_jokers(state)}

SHOP MECHANICS:
  buy_card: purchase a card. Jokers provide passive bonuses each hand. Planet cards permanently level up a poker hand type for the rest of the run. Tarot cards are one-use consumables.
  reroll: spend money to refresh shop cards (starts at $5, increases by $1 each reroll)
  end_shop: leave the shop and proceed to the next blind
  You may perform multiple actions. Always end with end_shop. You cannot buy cards you cannot afford.
  Interest: at end of each round you earn $1 per $5 you hold, up to $5 maximum per round.

Respond ONLY in this exact JSON format:
{{"actions": [{{"action": "buy_card", "index": 0}}, {{"action": "end_shop"}}], "reasoning": "your reasoning in 2 sentences max"}}"""

def build_blind_prompt(state):
    ante = state.get("ante_num",0); joker_count = get_joker_count(state)
    return f"""You are playing Balatro, a poker-based roguelike card game.

GAME STATE:
  Ante: {ante} | Jokers owned: {joker_count}

BLIND SELECTION:
{format_blinds(state)}

YOUR JOKERS:
{format_jokers(state)}

BLIND MECHANICS:
  select: play this blind. You earn money for beating it and visit a shop afterward where you can buy jokers and other cards.
  skip: skip this blind and receive the tag reward shown instead. You do NOT earn money from the blind and do NOT visit a shop for that blind — you lose that shop visit entirely.
  The boss blind CANNOT be skipped — you must always select it.
  Note: Skipping a blind means losing the money reward AND the shop visit for that blind. Each shop visit is an opportunity to buy jokers which increase your scoring power.

Respond ONLY in this exact JSON format:
{{"action": "select" or "skip", "reasoning": "your reasoning in 1 sentence max"}}"""

def build_pack_prompt(state):
    ante = state.get("ante_num",0); choices = state.get("pack_cards",{}).get("choose",1)
    return f"""You are playing Balatro, a poker-based roguelike card game.

GAME STATE:
  Ante: {ante}

BOOSTER PACK:
{format_pack_cards(state)}

YOUR JOKERS:
{format_jokers(state)}

PACK MECHANICS:
  pick: choose exactly {choices} card(s) by index. Cards go into your collection immediately.
  skip: take nothing from this pack.

Respond ONLY in this exact JSON format:
{{"action": "pick" or "skip", "cards": [list of exactly {choices} indices if picking, empty list if skipping], "reasoning": "your reasoning in 1 sentence max"}}"""

def parse_llm_response(response_text, state_name):
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            return parsed, True
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error: {e}")
    logger.warning(f"Parse FAILED for state={state_name}")
    defaults = {
        "SELECTING_HAND":      {"action":"play","cards":[0,1,2,3,4],"reasoning":"parse_failure_fallback"},
        "SHOP":                {"actions":[{"action":"end_shop"}],"reasoning":"parse_failure_fallback"},
        "BLIND_SELECT":        {"action":"select","reasoning":"parse_failure_fallback"},
        "SMODS_BOOSTER_OPENED":{"action":"skip","cards":[],"reasoning":"parse_failure_fallback"},
    }
    return defaults.get(state_name, {}), False

def validate_hand_action(action, cards, hand, discards_left):
    cards = [c for c in cards if isinstance(c,int) and 0<=c<len(hand)]
    if action == "play":
        cards = cards[:5]  # game only allows playing 1-5 cards
    if action == "discard":
        cards = cards[:5]  # game only allows discarding 1-5 cards
    if not cards: cards = list(range(min(5,len(hand))))
    if action == "discard" and discards_left <= 0: action = "play"
    if not cards: action = "play"; cards = list(range(min(5,len(hand))))
    return action, cards

def validate_shop_actions(actions, state):
    valid = []; shop = state.get("shop",{}).get("cards",[]); money = get_money(state)
    for ad in actions:
        a = ad.get("action")
        if a == "buy_card":
            idx = ad.get("index",0)
            if not isinstance(idx,int): continue
            if idx < len(shop):
                cost_raw = shop[idx].get("cost",{})
                cost = cost_raw.get("buy",999) if isinstance(cost_raw,dict) else 999
                if isinstance(cost,int) and cost <= money and cost > 0:
                    valid.append({"action":"buy_card","index":idx}); money -= cost
        elif a in ("buy_voucher","buy_pack"): valid.append(ad)
        elif a == "reroll":
            rc = state.get("round",{}).get("reroll_cost",5)
            if money >= rc: valid.append({"action":"reroll"}); money -= rc
        elif a == "end_shop": valid.append({"action":"end_shop"}); break
    if not valid or valid[-1].get("action") != "end_shop": valid.append({"action":"end_shop"})
    return valid

def validate_pack_action(parsed, state):
    choices = state.get("pack_cards",{}).get("choose",1)
    pack_cards = state.get("pack_cards",{}).get("cards",[])
    action = parsed.get("action","skip"); cards = parsed.get("cards",[])
    if action == "skip": return {"action":"skip","cards":[]}
    if action == "pick":
        cards = [c for c in cards if isinstance(c,int) and 0<=c<len(pack_cards)]
        if len(cards) != choices:
            logger.warning(f"Pack: wanted {choices} cards got {len(cards)}, skipping")
            return {"action":"skip","cards":[]}
        return {"action":"pick","cards":cards}
    return {"action":"skip","cards":[]}

class LLMBot(BaseBot):
    BOT_TYPE   = "llm_bot"
    WANDB_TAGS = ["llm_bot","zero_shot"]

    def __init__(self, api_key=None, model=MODEL, **kwargs):
        super().__init__(**kwargs)
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key: raise ValueError("Anthropic API key required.")
        self.llm = anthropic.Anthropic(api_key=key); self.model = model
        self._total_input_tokens = 0; self._total_output_tokens = 0
        self._total_tokens_used  = 0; self._total_llm_calls = 0
        self._parse_failures = 0; self._retries = 0
        self._cost_per_1m_tokens = (INPUT_COST_PER_1M + OUTPUT_COST_PER_1M) / 2
        self._blind_skips_ante1 = 0; self._blind_skips_total = 0
        self._self_corrections_total = 0; self._hand_recognition_hits = 0
        self._hand_recognition_total = 0; self._boss_adaptations = 0
        self._qual_logger = QualitativeLogger("llm_decisions.jsonl")
        self._current_seed = "unknown"; self._turn_number = 0

    def _call_llm(self, prompt):
        self._total_llm_calls += 1
        message = self.llm.messages.create(model=self.model, max_tokens=MAX_TOKENS,
                                           messages=[{"role":"user","content":prompt}])
        response_text = message.content[0].text
        in_tok = message.usage.input_tokens; out_tok = message.usage.output_tokens
        self._total_input_tokens += in_tok; self._total_output_tokens += out_tok
        self._total_tokens_used  += in_tok + out_tok
        cost = ((self._total_input_tokens/1_000_000)*INPUT_COST_PER_1M +
                (self._total_output_tokens/1_000_000)*OUTPUT_COST_PER_1M)
        logger.info(f"LLM call #{self._total_llm_calls} | in={in_tok} out={out_tok} | cost=${cost:.4f}")
        return response_text, in_tok, out_tok

    def _call_with_retry(self, prompt, state_name):
        raw, in_tok, out_tok = self._call_llm(prompt)
        parsed, success = parse_llm_response(raw, state_name)
        if not success:
            self._parse_failures += 1
            raw2, in2, out2 = self._call_llm(prompt + RETRY_PROMPT_SUFFIX)
            parsed2, success2 = parse_llm_response(raw2, state_name)
            in_tok += in2; out_tok += out2
            if success2:
                self._retries += 1
                return parsed2, True, raw2, in_tok, out_tok, True
            return parsed, False, raw, in_tok, out_tok, True
        return parsed, success, raw, in_tok, out_tok, False

    def _on_game_start(self, seed):
        self._current_seed = seed; self._turn_number = 0
        self._total_input_tokens = 0; self._total_output_tokens = 0
        self._total_tokens_used  = 0; self._total_llm_calls = 0
        self._parse_failures = 0; self._retries = 0
        self._blind_skips_ante1 = 0; self._blind_skips_total = 0
        self._self_corrections_total = 0; self._hand_recognition_hits = 0
        self._hand_recognition_total = 0; self._boss_adaptations = 0

    def _get_chips_needed(self, state):
        for bk in ["small","big","boss"]:
            b = state.get("blinds",{}).get(bk,{})
            if b.get("status") in ("SELECT","CURRENT"): return b.get("score",0)
        return 0

    def select_hand_action(self, state):
        self._turn_number += 1
        hand = get_hand_cards(state); discards_left = get_discards_left(state)
        prompt = build_hand_prompt(state); round_i = state.get("round",{})
        is_boss = get_blind_type(state) == "boss"; blind_eff = ""
        for bk in ["small","big","boss"]:
            b = state.get("blinds",{}).get(bk,{})
            if b.get("status") in ("SELECT","CURRENT"): blind_eff = b.get("effect",""); break

        parsed, success, raw, in_tok, out_tok, retried = self._call_with_retry(prompt, "SELECTING_HAND")
        action = parsed.get("action","play"); cards = parsed.get("cards",[0,1,2,3,4])
        self._self_corrections_total += count_self_corrections(raw)
        if is_boss and blind_eff and blind_eff[:10].lower() in parsed.get("reasoning","").lower():
            self._boss_adaptations += 1

        available_hands = detect_available_hands(hand)
        action, cards   = validate_hand_action(action, cards, hand, discards_left)

        if action == "play":
            self._hand_recognition_total += 1
            played = [hand[i] for i in cards if 0<=i<len(hand)]
            played_type = detect_available_hands(played)[0] if played else "high_card"
            if played_type == available_hands[0]: self._hand_recognition_hits += 1

        self._qual_logger.log(
            bot_type=self.BOT_TYPE, seed=self._current_seed, ante=get_ante(state),
            round_num=state.get("round_num",0), state_name="SELECTING_HAND",
            prompt=prompt, response_raw=raw, parsed_action=parsed,
            parse_success=success, input_tokens=in_tok, output_tokens=out_tok, retried=retried,
            extra={"turn":self._turn_number,"available_hands":available_hands,
                   "best_available":available_hands[0],"cards_chosen":cards,"action_chosen":action,
                   "chips_before":round_i.get("chips",0),"chips_needed":self._get_chips_needed(state),
                   "hands_remaining":get_hands_left(state),"discards_remaining":discards_left,
                   "is_boss":is_boss,"boss_effect":blind_eff,"self_corrections":count_self_corrections(raw)})
        if parsed.get("reasoning"): logger.info(f"LLM ({self.BOT_TYPE}): {parsed['reasoning']}")
        self._last_hand_type = action
        return action, cards

    def select_shop_action(self, state):
        prompt = build_shop_prompt(state)
        parsed, success, raw, in_tok, out_tok, retried = self._call_with_retry(prompt, "SHOP")
        self._qual_logger.log(
            bot_type=self.BOT_TYPE, seed=self._current_seed, ante=get_ante(state),
            round_num=state.get("round_num",0), state_name="SHOP",
            prompt=prompt, response_raw=raw, parsed_action=parsed,
            parse_success=success, input_tokens=in_tok, output_tokens=out_tok, retried=retried,
            extra={"money":get_money(state),"joker_count":get_joker_count(state),
                   "self_corrections":count_self_corrections(raw)})
        if parsed.get("reasoning"): logger.info(f"LLM ({self.BOT_TYPE}): {parsed['reasoning']}")
        return validate_shop_actions(parsed.get("actions",[{"action":"end_shop"}]), state)

    def select_blind_action(self, state):
        prompt = build_blind_prompt(state)
        parsed, success, raw, in_tok, out_tok, retried = self._call_with_retry(prompt, "BLIND_SELECT")
        action = parsed.get("action","select"); ante = get_ante(state); blind_type = get_blind_type(state)
        if action == "skip" and blind_type != "boss":
            self._blind_skips_total += 1
            if ante <= 1: self._blind_skips_ante1 += 1
        self._qual_logger.log(
            bot_type=self.BOT_TYPE, seed=self._current_seed, ante=ante,
            round_num=state.get("round_num",0), state_name="BLIND_SELECT",
            prompt=prompt, response_raw=raw, parsed_action=parsed,
            parse_success=success, input_tokens=in_tok, output_tokens=out_tok, retried=retried,
            extra={"blind_type":blind_type,"joker_count":get_joker_count(state),
                   "skipped":action=="skip","skip_at_ante1":ante<=1 and action=="skip",
                   "self_corrections":count_self_corrections(raw)})
        if parsed.get("reasoning"): logger.info(f"LLM ({self.BOT_TYPE}): {parsed['reasoning']}")
        if blind_type == "boss" and action == "skip":
            logger.warning("LLM tried to skip boss blind, overriding to select"); action = "select"
        return action

    def select_pack_action(self, state):
        prompt = build_pack_prompt(state)
        parsed, success, raw, in_tok, out_tok, retried = self._call_with_retry(prompt, "SMODS_BOOSTER_OPENED")
        safe_parsed = validate_pack_action(parsed, state)
        self._qual_logger.log(
            bot_type=self.BOT_TYPE, seed=self._current_seed, ante=get_ante(state),
            round_num=state.get("round_num",0), state_name="SMODS_BOOSTER_OPENED",
            prompt=prompt, response_raw=raw, parsed_action=parsed,
            parse_success=success, input_tokens=in_tok, output_tokens=out_tok, retried=retried,
            extra={"pack_choices":state.get("pack_cards",{}).get("choose",1),
                   "pack_cards_count":len(state.get("pack_cards",{}).get("cards",[])),
                   "action_taken":safe_parsed.get("action","skip"),
                   "self_corrections":count_self_corrections(raw)})
        if parsed.get("reasoning"): logger.info(f"LLM pack ({self.BOT_TYPE}): {parsed['reasoning']}")
        return safe_parsed

    def run_game(self, seed):
        self._current_seed = seed
        return super().run_game(seed)

BENCHMARK_SEEDS = [f"SEED{str(i).zfill(3)}" for i in range(1,101)]
RUNS_PER_SEED   = 1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key"); parser.add_argument("--model", default=MODEL)
    parser.add_argument("--seeds", nargs="+", default=BENCHMARK_SEEDS)
    parser.add_argument("--runs-per-seed", type=int, default=RUNS_PER_SEED)
    parser.add_argument("--results", default="results.csv"); parser.add_argument("--port", type=int, default=12346)
    parser.add_argument("--deck", default="RED"); parser.add_argument("--stake", default="WHITE")
    args = parser.parse_args()
    bot = LLMBot(api_key=args.api_key, model=args.model, port=args.port,
                 results_path=args.results, deck=args.deck, stake=args.stake)
    if not bot.client.health(): print("ERROR: Cannot connect to Balatro."); exit(1)
    print(f"Running LLMBot zero-shot ({args.model}) on {len(args.seeds)} seeds x {args.runs_per_seed} runs")
    results   = bot.run_experiment(seeds=args.seeds, runs_per_seed=args.runs_per_seed)
    completed = [r for r in results if r["outcome"] in ("won","lost")]
    if completed:
        print(f"\nSummary:")
        print(f"  Completed:            {len(completed)}/{len(results)}")
        print(f"  Wins:                 {sum(1 for r in completed if r['outcome']=='won')}")
        print(f"  Avg ante:             {sum(r['final_ante'] for r in completed)/len(completed):.2f}")
        print(f"  Avg round:            {sum(r['final_round'] for r in completed)/len(completed):.2f}")
        print(f"  LLM calls:            {bot._total_llm_calls}")
        print(f"  Retries:              {bot._retries}")
        print(f"  Input tokens:         {bot._total_input_tokens}")
        print(f"  Output tokens:        {bot._total_output_tokens}")
        cost = ((bot._total_input_tokens/1_000_000)*INPUT_COST_PER_1M + (bot._total_output_tokens/1_000_000)*OUTPUT_COST_PER_1M)
        print(f"  Estimated cost:       ${cost:.4f}")
        print(f"  Parse failures:       {bot._parse_failures}")
        print(f"  Blind skips ante 1:   {bot._blind_skips_ante1}")
        print(f"  Blind skips total:    {bot._blind_skips_total}")
        print(f"  Self corrections:     {bot._self_corrections_total}")
        print(f"  Hand recog accuracy:  {bot._hand_recognition_hits}/{bot._hand_recognition_total}")
        print(f"  Boss adaptations:     {bot._boss_adaptations}")
        print(f"  Decisions logged to:  llm_decisions.jsonl")