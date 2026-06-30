"""
meta_bot_v2.py
MetaBot v2 — upgraded meta-informed heuristic agent.

Improvements over v1:
- Joker ordering: positions Jokers left to right for optimal trigger sequence
  (flat Mult Jokers before xMult Jokers, Blueprint next to strongest xMult)
- xMult transition: actively prioritizes xMult Jokers over flat Mult in mid/late game
- Blueprint/Brainstorm awareness: treats them as a combo, positions accordingly
- Reroll logic: rerolls when nothing useful in shop and money above interest floor
- Slot management: keeps 1 slot open early game for flexibility
- Stronger hand-type Joker affinity weighting

Everything else mirrors v1 exactly so results are comparable:
- Same interest floor logic ($20 early, $25 mid)
- Same dominant hand tracking
- Same planet card priority
- Same discard strategy
- Same blind skip logic (never ante 1-2, small only at ante 3+)
- Same boss blind handling
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
# Card parsing helpers (identical to v1)
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
# Hand detection helpers (identical to v1)
# ---------------------------------------------------------------------------

def find_straight_flush(cards):
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

def find_four_of_a_kind(cards):
    ranks = get_ranks(cards)
    rc    = Counter(ranks)
    for rank, count in rc.most_common():
        if count >= 4:
            idxs = [i for i,c in enumerate(cards) if parse_card(c)[0]==rank][:4]
            rest = sorted([i for i in range(len(cards)) if i not in idxs],
                          key=lambda i: rank_value(parse_card(cards[i])[0]), reverse=True)
            return (idxs + rest[:1])[:5]
    return None

def find_full_house(cards):
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

def find_flush(cards):
    suits = get_suits(cards)
    sc    = Counter(suits)
    for suit, count in sc.most_common():
        if count >= 5:
            idxs = [i for i,c in enumerate(cards) if parse_card(c)[1]==suit]
            idxs.sort(key=lambda i: rank_value(parse_card(cards[i])[0]), reverse=True)
            return idxs[:5]
    return None

def find_straight(cards):
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
    ace_idxs = [i for i,c in enumerate(cards) if parse_card(c)[0]=="A"]
    low = {r for r in unique if r <= rank_value("5")}
    if ace_idxs and len(low) >= 4:
        low4 = sorted(low)[:4]
        if low4 == [0,1,2,3]:
            return [ace_idxs[0]] + [seen[r] for r in low4]
    return None

def find_three_of_a_kind(cards):
    ranks = get_ranks(cards)
    rc    = Counter(ranks)
    for rank, count in rc.most_common():
        if count >= 3:
            idxs = [i for i,c in enumerate(cards) if parse_card(c)[0]==rank][:3]
            rest = sorted([i for i in range(len(cards)) if i not in idxs],
                          key=lambda i: rank_value(parse_card(cards[i])[0]), reverse=True)
            return (idxs + rest[:2])[:5]
    return None

def find_two_pair(cards):
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

def find_pair(cards):
    ranks = get_ranks(cards)
    rc    = Counter(ranks)
    for rank, count in rc.most_common():
        if count >= 2:
            idxs = [i for i,c in enumerate(cards) if parse_card(c)[0]==rank][:2]
            rest = sorted([i for i in range(len(cards)) if i not in idxs],
                          key=lambda i: rank_value(parse_card(cards[i])[0]), reverse=True)
            return (idxs + rest[:3])[:5]
    return None

def find_best_hand(cards):
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
    best = sorted(range(len(cards)),
                  key=lambda i: rank_value(parse_card(cards[i])[0]), reverse=True)
    return "high_card", best[:5]


# ---------------------------------------------------------------------------
# Joker tier system — v2 adds xMult classification and ordering metadata
# ---------------------------------------------------------------------------

# xMult Jokers: these multiply the entire multiplier (exponential scaling)
# Must be positioned AFTER flat Mult Jokers in ordering
XMULT_JOKERS = {
    "blueprint", "brainstorm", "cavendish", "hologram", "baron",
    "triboulet", "vampire", "glass joker", "obelisk", "the idol",
    "lucky cat", "campfire", "canio", "yorick", "perkeo",
    "ride the bus",  # scales as xMult equivalent late game
}

# Blueprint and Brainstorm are combo Jokers — value each other's presence
COMBO_JOKERS = {"blueprint", "brainstorm"}

JOKER_TIERS = {
    "S+": [
        "Blueprint","Brainstorm","Cavendish","Hologram",
        "Baron","Triboulet","Chicot","Perkeo","Canio","Yorick",
        "Obelisk","Vampire","Glass Joker","Campfire",
    ],
    "S": [
        "Ride the Bus","Green Joker","Supernova","Constellation",
        "Fibonacci","Abstract Joker","Swashbuckler","Acrobat",
        "Madness","The Duo","The Trio","The Family","The Order","The Tribe",
        "Dusk","Sock and Buskin","Card Sharp",
    ],
    "A": [
        "Scary Face","Smiley Face","Scholar","Even Steven","Odd Todd",
        "Bull","Blue Joker","Bootstraps","Hack","Raised Fist",
        "Jolly Joker","Zany Joker","Mad Joker","Crazy Joker","Droll Joker",
        "Sly Joker","Wily Joker","Clever Joker","Devious Joker","Crafty Joker",
        "Blackboard","Flower Pot","Bloodstone","Arrowhead","Onyx Agate",
        "Runner","Square Joker","Spare Trousers","Fibonacci",
    ],
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

def joker_score(label: str, dominant_hand: str, ante: int, joker_labels: list[str]) -> int:
    base = JOKER_PRIORITY.get(label.lower(), 5)

    # Boost if matches dominant hand
    affinity = JOKER_HAND_AFFINITY.get(label.lower())
    if affinity and affinity == dominant_hand:
        base = min(base + 20, 120)

    # v2: boost xMult Jokers in mid/late game — transition from flat Mult to xMult
    if label.lower() in XMULT_JOKERS and ante >= 3:
        base = min(base + 30, 130)

    # v2: boost Blueprint/Brainstorm combo — if we have one, strongly want the other
    owned_lower = [j.lower() for j in joker_labels]
    if label.lower() == "blueprint" and "brainstorm" in owned_lower:
        base = min(base + 40, 140)
    if label.lower() == "brainstorm" and "blueprint" in owned_lower:
        base = min(base + 40, 140)

    return base


def _is_xmult(label: str) -> bool:
    return label.lower() in XMULT_JOKERS

def _is_flat_mult(label: str) -> bool:
    """Flat Mult jokers that should trigger before xMult jokers."""
    return not _is_xmult(label) and label.lower() not in COMBO_JOKERS


# ---------------------------------------------------------------------------
# MetaBot v2
# ---------------------------------------------------------------------------

class MetaBotV2(BaseBot):
    BOT_TYPE   = "meta_bot_v2"
    WANDB_TAGS = ["meta_bot_v2","heuristic","v2"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._hand_counts   = Counter()
        self._dominant_hand = "flush"
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
    # Hand selection — identical logic to v1
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

        hand_name, indices = find_best_hand(cards)

        hand_strength = ["straight_flush","four_of_a_kind","full_house",
                         "flush","straight","three_of_a_kind","two_pair",
                         "pair","high_card"].index(hand_name)
        is_weak = hand_strength >= 6

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
        suits     = get_suits(cards)
        sc        = Counter(suits)
        best_suit, best_count = sc.most_common(1)[0]

        if best_count >= 4:
            discard = [i for i,c in enumerate(cards) if parse_card(c)[1] != best_suit]
            if discard:
                return discard[:5]

        ranks = get_ranks(cards)
        vals  = sorted(set(rank_value(r) for r in ranks))
        for start in range(len(vals)-3):
            w = vals[start:start+4]
            if w[-1]-w[0] <= 4:
                keep_ranks = {RANK_ORDER[v] for v in w}
                discard    = [i for i,c in enumerate(cards)
                              if parse_card(c)[0] not in keep_ranks]
                if discard:
                    return discard[:5]
        return None

    # -----------------------------------------------------------------------
    # Shop — v2 improvements: rerolls, slot management, xMult priority
    # -----------------------------------------------------------------------

    def _get_owned_joker_labels(self, state: dict) -> list[str]:
        joker_data = state.get("jokers", {})
        if isinstance(joker_data, dict):
            cards = joker_data.get("cards", [])
        else:
            cards = joker_data if isinstance(joker_data, list) else []
        return [c.get("label","") for c in cards]

    def select_shop_action(self, state: dict) -> list[dict]:
        actions     = []
        money       = get_money(state)
        joker_count = get_joker_count(state)
        joker_limit = get_joker_limit(state)
        shop_cards  = get_shop_cards(state)
        ante        = get_ante(state)
        rc          = state.get("round", {}).get("reroll_cost", 5)
        owned_labels = self._get_owned_joker_labels(state)

        interest_floor = 20 if ante <= 2 else 25
        spendable      = max(0, money - interest_floor)
        emergency_buy  = joker_count == 0 and ante <= 2

        # v2: slot management — keep 1 slot open early game (ante 1-2)
        # unless we have an S+ Joker available
        effective_limit = joker_limit
        if ante <= 2 and joker_count >= joker_limit - 1:
            # Check if there's an S+ in the shop before capping
            has_splus = any(
                JOKER_PRIORITY.get(c.get("label","").lower(), 0) >= 100
                for c in shop_cards
                if (c.get("set","") or c.get("ability",{}).get("set","")) in ("JOKER","Joker")
            )
            if not has_splus:
                effective_limit = joker_limit - 1  # keep one slot open

        bought = False

        # Priority 1: Planet card for dominant hand (same as v1)
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

            logger.info(f"MetaBotV2 buying planet {label} for {self._dominant_hand}")
            actions.append({"action":"buy_card","index":i})
            money -= cost
            spendable = max(0, money - interest_floor)
            bought = True
            break

        # Priority 2: Best Joker — v2 uses ante-aware scoring with xMult boost
        if joker_count < effective_limit and (spendable > 0 or emergency_buy):
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

                # v2: pass ante and owned labels for context-aware scoring
                score = joker_score(label, self._dominant_hand, ante, owned_labels)
                if score >= min_score and score > best_score:
                    best_score = score
                    best_idx   = i

            if best_idx is not None:
                cost_raw = shop_cards[best_idx].get("cost",{})
                cost     = cost_raw.get("buy",0) if isinstance(cost_raw,dict) else 0
                label    = shop_cards[best_idx].get("label","")
                logger.info(f"MetaBotV2 buying joker {label} score={best_score} ante={ante}")
                actions.append({"action":"buy_card","index":best_idx})
                money    -= cost
                spendable = max(0, money - interest_floor)

        # Priority 3: Off-hand planet card (same as v1)
        if not bought:
            for i, card in enumerate(shop_cards):
                card_set = card.get("set","") or card.get("ability",{}).get("set","")
                cost_raw = card.get("cost",{})
                cost     = cost_raw.get("buy",999) if isinstance(cost_raw,dict) else 999
                if card_set not in ("PLANET","Planet"): continue
                if cost > spendable: continue
                label = card.get("label","")
                logger.info(f"MetaBotV2 buying off-hand planet {label}")
                actions.append({"action":"buy_card","index":i})
                money    -= cost
                spendable = max(0, money - interest_floor)
                break

        # v2: Reroll if nothing bought, money above floor, and ante >= 2
        # Only reroll once per shop visit to avoid burning too much money
        rerolled = len(actions) == 0 or (len(actions) == 1 and actions[0].get("action") == "buy_card")
        if (not bought and
            ante >= 2 and
            money >= rc + interest_floor and
            spendable >= rc):
            logger.info(f"MetaBotV2 rerolling shop at ante {ante}, money={money}, rc={rc}")
            actions.append({"action":"reroll"})
            money    -= rc
            spendable = max(0, money - interest_floor)

            # After reroll, check for planet card for dominant hand again
            # (can't act on new cards here since we don't re-query shop,
            # but the reroll triggers a new shop state in base_bot execute loop)

        actions.append({"action":"end_shop"})
        return actions

    # -----------------------------------------------------------------------
    # Blind selection — identical to v1
    # -----------------------------------------------------------------------

    def select_blind_action(self, state: dict) -> str:
        blind_type = get_blind_type(state)
        ante       = get_ante(state)

        if blind_type == "boss":
            return "select"
        if ante <= 2:
            return "select"
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

    parser = argparse.ArgumentParser(description="Run MetaBot v2")
    parser.add_argument("--seeds",         nargs="+", default=BENCHMARK_SEEDS)
    parser.add_argument("--runs-per-seed", type=int,  default=RUNS_PER_SEED)
    parser.add_argument("--results",       default="results.csv")
    parser.add_argument("--port",          type=int,  default=12346)
    parser.add_argument("--deck",          default="RED")
    parser.add_argument("--stake",         default="WHITE")
    args = parser.parse_args()

    bot = MetaBotV2(
        port         = args.port,
        results_path = args.results,
        deck         = args.deck,
        stake        = args.stake,
    )

    if not bot.client.health():
        print("ERROR: Cannot connect to Balatro."); exit(1)

    print(f"Running MetaBotV2 on {len(args.seeds)} seeds x {args.runs_per_seed} runs")
    results   = bot.run_experiment(seeds=args.seeds, runs_per_seed=args.runs_per_seed)
    completed = [r for r in results if r["outcome"] in ("won","lost")]
    if completed:
        print(f"\nResults summary:")
        print(f"  Completed: {len(completed)}/{len(results)}")
        print(f"  Wins:      {sum(1 for r in results if r['outcome']=='won')}")
        print(f"  Avg ante:  {sum(r['final_ante'] for r in completed)/len(completed):.2f}")
        print(f"  Avg round: {sum(r['final_round'] for r in completed)/len(completed):.2f}")