"""
heuristic_bots.py
Rule-based heuristic agents. Both inherit from BaseBot.

FlushBot  - Always plays the best flush available, then best pair hand.
            Never buys anything. Absolute floor baseline.

MetaBot   - Meta-informed heuristics based on community strategy guides and wiki.
            Key principles:
            - Pick a dominant hand type early and stick to it
            - Buy planet cards to level dominant hand (compounds across run)
            - Balance chip joker + scaling mult joker + xmult joker
            - Hold $25 for max interest ($5/round) — don't overspend
            - Only skip blinds at ante 3+ when tag reward is clearly worth it
            - Hand priority: straight_flush > four_of_a_kind > full_house >
                             flush > straight > three_of_a_kind > two_pair >
                             pair > high_card
            - Discard to fish for dominant hand type (flush or straight),
              but save 1 discard for boss blind
"""

import logging
from collections import Counter
from base_bot import (
    BaseBot,
    get_hand_cards,
    get_money,
    get_joker_count,
    get_joker_limit,
    get_shop_cards,
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

# ---------------------------------------------------------------------------
# Card parsing helpers
# ---------------------------------------------------------------------------

def parse_card(card: dict) -> tuple[str, str]:
    value = card.get("value", {})
    return value.get("rank", "?"), value.get("suit", "?")

def get_suits(cards): return [parse_card(c)[1] for c in cards]
def get_ranks(cards): return [parse_card(c)[0] for c in cards]

RANK_ORDER  = ["2","3","4","5","6","7","8","9","T","J","Q","K","A"]
RANK_VALUES = {r: i for i, r in enumerate(RANK_ORDER)}
FACE_CARDS  = {"J","Q","K"}

PLANET_HAND_MAP = {
    "Mercury":"high_card","Venus":"three_of_a_kind","Earth":"full_house",
    "Mars":"four_of_a_kind","Jupiter":"flush","Saturn":"straight",
    "Uranus":"two_pair","Neptune":"straight_flush","Pluto":"pair",
}

def rank_value(rank: str) -> int:
    return RANK_VALUES.get(rank, 0)


# ---------------------------------------------------------------------------
# Hand detection helpers — full priority hierarchy
# ---------------------------------------------------------------------------

def find_straight_flush(cards: list[dict]) -> list[int] | None:
    suits = get_suits(cards)
    sc    = Counter(suits)
    for suit, count in sc.items():
        if count < 5: continue
        suited = [(i, rank_value(parse_card(c)[0]))
                  for i, c in enumerate(cards) if parse_card(c)[1] == suit]
        suited.sort(key=lambda x: x[1])
        seen = {}
        for idx, rv in suited:
            if rv not in seen: seen[rv] = idx
        unique = sorted(seen.keys())
        for start in range(len(unique) - 4):
            w = unique[start:start+5]
            if w[-1]-w[0]==4 and len(set(w))==5:
                return [seen[r] for r in w]
    return None


def find_four_of_a_kind(cards: list[dict]) -> list[int] | None:
    ranks = get_ranks(cards)
    rc    = Counter(ranks)
    for rank, count in rc.most_common():
        if count >= 4:
            idxs = [i for i,c in enumerate(cards) if parse_card(c)[0]==rank][:4]
            rest = sorted([i for i in range(len(cards)) if i not in idxs],
                          key=lambda i: rank_value(parse_card(cards[i])[0]), reverse=True)
            return (idxs + rest[:1])[:5]
    return None


def find_full_house(cards: list[dict]) -> list[int] | None:
    ranks  = get_ranks(cards)
    rc     = Counter(ranks)
    threes = [r for r,c in rc.items() if c>=3]
    if not threes: return None
    three_rank = max(threes, key=lambda r: rank_value(r))
    pair_opts  = [r for r,c in rc.items() if c>=2 and r!=three_rank]
    if not pair_opts: return None
    pair_rank = max(pair_opts, key=lambda r: rank_value(r))
    return ([i for i,c in enumerate(cards) if parse_card(c)[0]==three_rank][:3] +
            [i for i,c in enumerate(cards) if parse_card(c)[0]==pair_rank][:2])


def find_flush(cards: list[dict]) -> list[int] | None:
    suits = get_suits(cards)
    sc    = Counter(suits)
    for suit, count in sc.most_common():
        if count >= 5:
            idxs = [i for i,c in enumerate(cards) if parse_card(c)[1]==suit]
            idxs.sort(key=lambda i: rank_value(parse_card(cards[i])[0]), reverse=True)
            return idxs[:5]
    return None


def find_straight(cards: list[dict]) -> list[int] | None:
    indexed = [(i, rank_value(parse_card(c)[0])) for i,c in enumerate(cards)]
    indexed.sort(key=lambda x: x[1])
    seen = {}
    for idx, rv in indexed:
        if rv not in seen: seen[rv] = idx
    unique = sorted(seen.keys())
    for start in range(len(unique)-4):
        w = unique[start:start+5]
        if w[-1]-w[0]==4 and len(set(w))==5:
            return [seen[r] for r in w]
    # Wheel straight: A-2-3-4-5
    ace_idxs = [i for i,c in enumerate(cards) if parse_card(c)[0]=="A"]
    low = {r for r in unique if r <= rank_value("5")}
    if ace_idxs and len(low) >= 4:
        low4 = sorted(low)[:4]
        if low4 == [0,1,2,3]:
            return [ace_idxs[0]] + [seen[r] for r in low4]
    return None


def find_three_of_a_kind(cards: list[dict]) -> list[int] | None:
    ranks = get_ranks(cards)
    rc    = Counter(ranks)
    for rank, count in rc.most_common():
        if count >= 3:
            idxs = [i for i,c in enumerate(cards) if parse_card(c)[0]==rank][:3]
            rest = sorted([i for i in range(len(cards)) if i not in idxs],
                          key=lambda i: rank_value(parse_card(cards[i])[0]), reverse=True)
            return (idxs + rest[:2])[:5]
    return None


def find_two_pair(cards: list[dict]) -> list[int] | None:
    ranks  = get_ranks(cards)
    rc     = Counter(ranks)
    pairs  = sorted([r for r,c in rc.items() if c>=2],
                    key=lambda r: rank_value(r), reverse=True)
    if len(pairs) < 2: return None
    idxs = ([i for i,c in enumerate(cards) if parse_card(c)[0]==pairs[0]][:2] +
            [i for i,c in enumerate(cards) if parse_card(c)[0]==pairs[1]][:2])
    rest = sorted([i for i in range(len(cards)) if i not in idxs],
                  key=lambda i: rank_value(parse_card(cards[i])[0]), reverse=True)
    return (idxs + rest[:1])[:5]


def find_pair(cards: list[dict]) -> list[int] | None:
    ranks = get_ranks(cards)
    rc    = Counter(ranks)
    for rank, count in rc.most_common():
        if count >= 2:
            idxs = [i for i,c in enumerate(cards) if parse_card(c)[0]==rank][:2]
            rest = sorted([i for i in range(len(cards)) if i not in idxs],
                          key=lambda i: rank_value(parse_card(cards[i])[0]), reverse=True)
            return (idxs + rest[:3])[:5]
    return None


def find_best_hand(cards: list[dict]) -> tuple[str, list[int]]:
    """Find best available hand in full priority order."""
    sf = find_straight_flush(cards)
    if sf:   return "straight_flush", sf
    foak = find_four_of_a_kind(cards)
    if foak: return "four_of_a_kind", foak
    fh = find_full_house(cards)
    if fh:   return "full_house", fh
    fl = find_flush(cards)
    if fl:   return "flush", fl
    st = find_straight(cards)
    if st:   return "straight", st
    toak = find_three_of_a_kind(cards)
    if toak: return "three_of_a_kind", toak
    tp = find_two_pair(cards)
    if tp:   return "two_pair", tp
    pr = find_pair(cards)
    if pr:   return "pair", pr
    # High card — play highest ranked card
    best = sorted(range(len(cards)),
                  key=lambda i: rank_value(parse_card(cards[i])[0]), reverse=True)
    return "high_card", best[:5]


# ---------------------------------------------------------------------------
# FlushBot — absolute floor baseline
# Always plays best flush available, falls back to best pair hand.
# Never buys anything, never skips.
# ---------------------------------------------------------------------------

class FlushBot(BaseBot):
    BOT_TYPE   = "flush_bot"
    WANDB_TAGS = ["flush_bot","heuristic","baseline"]

    def select_hand_action(self, state: dict) -> tuple[str, list[int]]:
        cards = get_hand_cards(state)
        if not cards:
            self._last_hand_type = "high_card"
            return "play", [0]

        fl = find_flush(cards)
        if fl:
            self._last_hand_type = "flush"
            return "play", fl

        hand_name, indices = find_best_hand(cards)
        self._last_hand_type = hand_name
        return "play", indices

    def select_shop_action(self, state: dict) -> list[dict]:
        return [{"action": "end_shop"}]

    def select_blind_action(self, state: dict) -> str:
        return "select"


# ---------------------------------------------------------------------------
# MetaBot — meta-informed heuristic agent
# Based on community strategy guides and Balatro wiki.
# ---------------------------------------------------------------------------

# Joker tier list — based on wiki and community guides
# Three-part combo goal: chip joker + scaling mult joker + xmult joker
# Priority reflects early/mid game value on White Stake Red Deck

JOKER_TIERS = {
    # S+: game-winning, always buy if affordable
    "S+": [
        "Blueprint","Brainstorm","Cavendish","Hologram",
        "Baron","Triboulet","Chicot","Perkeo","Canio","Yorick",
        "Obelisk","Vampire","Glass Joker","Campfire",
    ],
    # S: strong, buy unless joker slots full with S+
    "S": [
        "Ride the Bus","Green Joker","Supernova","Constellation",
        "Fibonacci","Abstract Joker","Swashbuckler","Acrobat",
        "Madness","The Duo","The Trio","The Family","The Order","The Tribe",
        "Dusk","Sock and Buskin","Card Sharp",
    ],
    # A: good if it fits current hand type
    "A": [
        "Scary Face","Smiley Face","Scholar","Even Steven","Odd Todd",
        "Bull","Blue Joker","Bootstraps","Hack","Raised Fist",
        "Jolly Joker","Zany Joker","Mad Joker","Crazy Joker","Droll Joker",
        "Sly Joker","Wily Joker","Clever Joker","Devious Joker","Crafty Joker",
        "Blackboard","Flower Pot","Bloodstone","Arrowhead","Onyx Agate",
        "Runner","Square Joker","Spare Trousers","Fibonacci",
    ],
    # B: filler — only buy if joker slot open and nothing better
    "B": [
        "Joker","Gros Michel","Misprint","Riff-Raff","Ice Cream",
        "Popcorn","Ramen","Egg","Rocket","Satellite",
    ],
}

JOKER_PRIORITY = {}
for tier, jokers in JOKER_TIERS.items():
    score = {"S+":100,"S":70,"A":40,"B":15}[tier]
    for j in jokers:
        JOKER_PRIORITY[j.lower()] = score

# Hand-type joker affinity — prefer jokers that match dominant hand
JOKER_HAND_AFFINITY = {
    "jolly joker":"pair","sly joker":"pair",
    "zany joker":"three_of_a_kind","wily joker":"three_of_a_kind",
    "mad joker":"two_pair","clever joker":"two_pair",
    "crazy joker":"straight","devious joker":"straight",
    "droll joker":"flush","crafty joker":"flush",
    "the duo":"pair","the trio":"three_of_a_kind",
    "the family":"four_of_a_kind","the order":"straight","the tribe":"flush",
    "spare trousers":"two_pair","runner":"straight",
}

def joker_score(label: str, dominant_hand: str) -> int:
    base = JOKER_PRIORITY.get(label.lower(), 5)
    # Boost score if joker matches our dominant hand
    affinity = JOKER_HAND_AFFINITY.get(label.lower())
    if affinity and affinity == dominant_hand:
        base = min(base + 20, 120)
    return base


class MetaBot(BaseBot):
    BOT_TYPE   = "meta_bot"
    WANDB_TAGS = ["meta_bot","heuristic"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Track dominant hand across the game
        self._hand_counts   = Counter()
        self._dominant_hand = "flush"  # default target — most beginner-friendly
        self._hand_levels   = {ht:1 for ht in [
            "straight_flush","four_of_a_kind","full_house","flush",
            "straight","three_of_a_kind","two_pair","pair","high_card"
        ]}

    def _on_game_start(self, seed: str) -> None:
        self._hand_counts   = Counter()
        self._dominant_hand = "flush"
        self._hand_levels   = {ht:1 for ht in self._hand_levels}

    def _update_dominant(self, hand_type: str):
        self._hand_counts[hand_type] += 1
        self._dominant_hand = self._hand_counts.most_common(1)[0][0]

    # -----------------------------------------------------------------------
    # Hand selection — full priority hierarchy with discard conservation
    # -----------------------------------------------------------------------

    def select_hand_action(self, state: dict) -> tuple[str, list[int]]:
        cards         = get_hand_cards(state)
        discards_left = get_discards_left(state)
        hands_left    = get_hands_left(state)
        blind_type    = get_blind_type(state)
        is_boss       = blind_type == "boss"

        if not cards:
            self._last_hand_type = "high_card"
            return "play", [0]

        # Find best available hand
        hand_name, indices = find_best_hand(cards)

        # Discard strategy:
        # - Save last discard for boss blind (never use final discard on small/big)
        # - Don't discard if we have a flush, straight, or better
        # - Fish for dominant hand type if hand is weak (pair or worse)
        hand_strength = ["straight_flush","four_of_a_kind","full_house",
                         "flush","straight","three_of_a_kind","two_pair",
                         "pair","high_card"].index(hand_name)
        is_weak = hand_strength >= 6  # two_pair or worse

        save_last   = not is_boss and discards_left == 1
        can_discard = discards_left > 0 and not save_last and hands_left > 1

        if can_discard and is_weak:
            discard_idxs = self._find_discard(cards)
            if discard_idxs:
                return "discard", discard_idxs

        self._update_dominant(hand_name)
        self._last_hand_type = hand_name
        return "play", indices

    def _find_discard(self, cards: list[dict]) -> list[int] | None:
        """
        Discard cards that don't contribute to flush or straight.
        Flush-first if close (4+ of same suit), straight-second (4 consecutive ranks).
        """
        suits   = get_suits(cards)
        sc      = Counter(suits)
        best_suit, best_count = sc.most_common(1)[0]

        # Flush fishing: discard non-flush-suit cards if 4+ of same suit
        if best_count >= 4:
            discard = [i for i,c in enumerate(cards) if parse_card(c)[1] != best_suit]
            if discard:
                return discard[:5]

        # Straight fishing: keep 4 consecutive ranks, discard rest
        ranks  = get_ranks(cards)
        vals   = sorted(set(rank_value(r) for r in ranks))
        for start in range(len(vals)-3):
            w = vals[start:start+4]
            if w[-1]-w[0] <= 4:  # 4 cards within a 5-rank window
                keep_ranks = {RANK_ORDER[v] for v in w}
                discard    = [i for i,c in enumerate(cards)
                              if parse_card(c)[0] not in keep_ranks]
                if discard:
                    return discard[:5]

        return None

    # -----------------------------------------------------------------------
    # Shop — balanced economy: interest + jokers + planet cards
    # -----------------------------------------------------------------------

    def select_shop_action(self, state: dict) -> list[dict]:
        actions     = []
        money       = get_money(state)
        joker_count = get_joker_count(state)
        joker_limit = get_joker_limit(state)
        shop_cards  = get_shop_cards(state)
        ante        = get_ante(state)

        # Economy rule: keep $25 for max interest ($5/round)
        # Early game (ante 1-2): keep $20 floor to build interest base
        # Mid game (ante 3+): keep $25 floor
        interest_floor = 20 if ante <= 2 else 25
        spendable      = max(0, money - interest_floor)

        # Emergency: if no jokers at all, buy anything affordable
        emergency_buy = joker_count == 0 and ante <= 2

        bought = False

        # --- Priority 1: Planet card for dominant hand ---
        # Always buy planet matching dominant hand — compounds across entire run
        for i, card in enumerate(shop_cards):
            card_set = card.get("set","") or card.get("ability",{}).get("set","")
            label    = card.get("label","")
            cost_raw = card.get("cost",{})
            cost     = cost_raw.get("buy",999) if isinstance(cost_raw,dict) else 999

            if card_set not in ("PLANET","Planet"): continue
            target_hand = PLANET_HAND_MAP.get(label,"")
            if target_hand != self._dominant_hand: continue
            if cost > (money if emergency_buy else max(spendable, money-5)):
                continue

            logger.info(f"MetaBot buying planet {label} for {self._dominant_hand} hand")
            actions.append({"action":"buy_card","index":i})
            money -= cost
            spendable = max(0, money - interest_floor)
            bought = True
            break

        # --- Priority 2: Best joker if slots open ---
        if joker_count < joker_limit and (spendable > 0 or emergency_buy):
            best_idx   = None
            best_score = -1
            min_score  = 5 if emergency_buy else (15 if joker_count == 0 else 40)

            for i, card in enumerate(shop_cards):
                label    = card.get("label","")
                card_set = card.get("set","") or card.get("ability",{}).get("set","")
                cost_raw = card.get("cost",{})
                cost     = cost_raw.get("buy",999) if isinstance(cost_raw,dict) else 999

                if card_set not in ("JOKER","Joker"): continue
                if "pack" in card_set.lower() or "booster" in card_set.lower(): continue
                budget = money if emergency_buy else spendable
                if cost > budget or cost <= 0: continue

                score = joker_score(label, self._dominant_hand)
                if score >= min_score and score > best_score:
                    best_score = score
                    best_idx   = i

            if best_idx is not None:
                cost_raw = shop_cards[best_idx].get("cost",{})
                cost     = cost_raw.get("buy",0) if isinstance(cost_raw,dict) else 0
                logger.info(f"MetaBot buying joker {shop_cards[best_idx].get('label','')} score={best_score}")
                actions.append({"action":"buy_card","index":best_idx})
                money    -= cost
                spendable = max(0, money - interest_floor)

        # --- Priority 3: Any planet card if nothing else bought ---
        if not bought:
            for i, card in enumerate(shop_cards):
                card_set = card.get("set","") or card.get("ability",{}).get("set","")
                cost_raw = card.get("cost",{})
                cost     = cost_raw.get("buy",999) if isinstance(cost_raw,dict) else 999
                if card_set not in ("PLANET","Planet"): continue
                if cost > spendable: continue
                logger.info(f"MetaBot buying off-hand planet {card.get('label','')}")
                actions.append({"action":"buy_card","index":i})
                money    -= cost
                spendable = max(0, money - interest_floor)
                break

        actions.append({"action":"end_shop"})
        return actions

    # -----------------------------------------------------------------------
    # Blind selection — conservative skip strategy
    # Wiki: skipping is generally ill-advised — loses shop visits, interest, scaling
    # Only skip at ante 3+ when tag is clearly valuable
    # -----------------------------------------------------------------------

    def select_blind_action(self, state: dict) -> str:
        blind_type = get_blind_type(state)
        ante       = get_ante(state)

        # Boss blind can never be skipped
        if blind_type == "boss":
            return "select"

        # Ante 1-2: always select — need money, interest, shop visits
        if ante <= 2:
            return "select"

        # Ante 3+: only skip small blind (not big blind — too much money lost)
        # Big blind gives more money and shop visit — usually worth playing
        if ante >= 3 and blind_type == "small":
            return "skip"

        return "select"


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

BENCHMARK_SEEDS = [f"SEED{str(i).zfill(3)}" for i in range(1, 101)]
RUNS_PER_SEED   = 1

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run heuristic Balatro bots")
    parser.add_argument("--bot",           choices=["flush","meta"], default="flush")
    parser.add_argument("--seeds",         nargs="+", default=BENCHMARK_SEEDS)
    parser.add_argument("--runs-per-seed", type=int,  default=RUNS_PER_SEED)
    parser.add_argument("--results",       default="results.csv")
    parser.add_argument("--port",          type=int,  default=12346)
    parser.add_argument("--deck",          default="RED")
    parser.add_argument("--stake",         default="WHITE")
    args = parser.parse_args()

    BotClass = FlushBot if args.bot == "flush" else MetaBot
    bot = BotClass(
        port         = args.port,
        results_path = args.results,
        deck         = args.deck,
        stake        = args.stake,
    )

    if not bot.client.health():
        print("ERROR: Cannot connect to Balatro."); exit(1)

    print(f"Running {BotClass.BOT_TYPE} on {len(args.seeds)} seeds x {args.runs_per_seed} runs")
    results   = bot.run_experiment(seeds=args.seeds, runs_per_seed=args.runs_per_seed)
    completed = [r for r in results if r["outcome"] in ("won","lost")]
    if completed:
        print(f"\nResults summary:")
        print(f"  Completed: {len(completed)}/{len(results)}")
        print(f"  Wins:      {sum(1 for r in results if r['outcome']=='won')}")
        print(f"  Avg ante:  {sum(r['final_ante'] for r in completed)/len(completed):.2f}")
        print(f"  Avg round: {sum(r['final_round'] for r in completed)/len(completed):.2f}")
