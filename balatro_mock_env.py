"""
balatro_mock_env.py  v2
High-fidelity mock Gymnasium environment for MaskablePPO training.

OBS_DIM = 148:
  [0]      ante / 8
  [1]      round_num / 24
  [2]      money / 100
  [3]      hands_left / 4
  [4]      discards_left / 4
  [5]      chips_scored / 300000
  [6]      chips_needed / 300000
  [7]      joker_count / 5
  [8]      joker_limit / 5
  [9]      reroll_cost / 10
  [10-17]  hand card ranks (8 slots)
  [18-25]  hand card suits (8 slots)
  [26-33]  hand card enhancements (8 slots, normalised)
  [34-41]  hand card seals (8 slots, normalised)
  [42-49]  hand card debuff flags (8 slots)
  [50-54]  shop costs (5 slots)
  [55-59]  shop type encoding (5 slots: 0=joker,0.2=planet,0.4=tarot,0.6=voucher,0.8=pack,1=spectral)
  [60]     blind_type 0/0.5/1
  [61]     state_phase 0/0.5/1
  [62-74]  rank counts in deck+hand (13 dims)
  [75-78]  suit counts (4 dims)
  [79-87]  dominant hand one-hot (9 dims)
  [88-96]  all hand levels / 10 (9 dims, one per hand type)
  [97]     deck size / 52
  [98-104] boss blind category flags (7 dims)
  [105-127] boss blind identity one-hot (23 dims)
  [128]    interest_earned_this_round / 5
  [129]    money_mod_5 / 4   (how close to next interest bracket)
  [130-144] joker slots 5x3: [affinity/5, is_xmult, is_economy] per slot
  [145]    joker synergy score / 1
  [146]    green_joker_mult / 20
  [147]    ride_bus_mult / 20

Curriculum schedule:
  0-40M steps:   ante 1 only
  40-60M steps:  ante 1-2
  60-80M steps:  ante 1-3
  80-100M steps: ante 1-4
  100M+ steps:   full game
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import Counter

OBS_DIM   = 148
N_ACTIONS = 7

RANK_ORDER  = ["2","3","4","5","6","7","8","9","T","J","Q","K","A"]
RANK_VALUES = {r: i for i, r in enumerate(RANK_ORDER)}
SUIT_VALUES = {"S": 0, "H": 1, "D": 2, "C": 3}
SUITS       = ["S","H","D","C"]
RANKS       = list(RANK_ORDER)

RANK_CHIP_VALUES = {
    "2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,
    "T":10,"J":10,"Q":10,"K":10,"A":11,
}
FACE_CARDS      = {"J","Q","K"}
FIBONACCI_RANKS = {"A","2","3","5","8"}
EVEN_RANKS      = {"2","4","6","8","T"}
ODD_RANKS       = {"A","3","5","7","9"}

HAND_TYPES = [
    "straight_flush","four_of_a_kind","full_house","flush",
    "straight","three_of_a_kind","two_pair","pair","high_card",
]
HAND_TYPE_IDX = {h: i for i, h in enumerate(HAND_TYPES)}

BASE_HAND_SCORES = {
    "straight_flush": (100,8), "four_of_a_kind": (60,7),
    "full_house":     (40,4),  "flush":          (35,4),
    "straight":       (30,4),  "three_of_a_kind":(30,3),
    "two_pair":       (20,2),  "pair":           (10,2),
    "high_card":      (5,1),
}
HAND_LEVEL_CHIPS_BONUS = 50
HAND_LEVEL_MULT_BONUS  = 2

BLIND_CHIPS = {
    1:{"small":300,   "big":450,   "boss":600},
    2:{"small":800,   "big":1200,  "boss":1600},
    3:{"small":2000,  "big":3000,  "boss":4000},
    4:{"small":5000,  "big":7500,  "boss":10000},
    5:{"small":11000, "big":16500, "boss":22000},
    6:{"small":20000, "big":30000, "boss":40000},
    7:{"small":35000, "big":52500, "boss":70000},
    8:{"small":60000, "big":90000, "boss":120000},
}

ENHANCEMENTS      = ["none","glass","steel","gold","mult","bonus","wild","lucky"]
ENHANCEMENT_IDX   = {e: i for i, e in enumerate(ENHANCEMENTS)}
ENHANCEMENT_PROBS = [0.62, 0.05, 0.08, 0.07, 0.05, 0.05, 0.04, 0.04]

SEALS     = ["none","red","blue","gold","purple"]
SEAL_IDX  = {s: i for i, s in enumerate(SEALS)}
SEAL_PROBS = [0.80, 0.05, 0.05, 0.05, 0.05]

# ---------------------------------------------------------------------------
# Boss blind definitions — all 23 + their category flags
# [suit_debuff, hand_size, play_restriction, hand_type_ban, card_removal, face_down, score_modifier]
# ---------------------------------------------------------------------------
ALL_BOSS_BLINDS = [
    "The_Hook","The_Ox","The_House","The_Wall","The_Wheel",
    "The_Fish","The_Club","The_Tooth","The_Flint","The_Mark",
    "The_Psychic","The_Goad","The_Water","The_Eye","The_Plant",
    "The_Needle","The_Head","The_Serpent","The_Window","The_Manacle",
    "The_Arm","The_Pillar","The_Mouth",
]
BOSS_BLIND_IDX = {b: i for i, b in enumerate(ALL_BOSS_BLINDS)}

# Category flags per boss: [suit_debuff, hand_size, play_restriction, hand_type_ban, card_removal, face_down, score_modifier]
BOSS_CATEGORIES = {
    "The_Hook":    [0,0,0,0,1,0,0],  # discards 2 random cards per hand played
    "The_Ox":      [0,0,0,0,1,0,0],  # playing most played hand sets money to $0
    "The_House":   [1,0,0,0,0,0,0],  # all club cards debuffed (first card face down)
    "The_Wall":    [0,0,0,0,0,0,1],  # extra large blind
    "The_Wheel":   [0,0,0,0,0,0,0],  # 1 in 7 cards drawn face down
    "The_Fish":    [0,1,0,0,0,0,0],  # cards drawn face down after each hand
    "The_Club":    [1,0,0,0,0,0,0],  # all club cards debuffed
    "The_Tooth":   [0,0,0,0,0,0,1],  # lose $1 per card scored
    "The_Flint":   [0,0,0,0,0,0,1],  # base chips and mult halved
    "The_Mark":    [0,0,0,0,0,1,0],  # all face cards drawn face down
    "The_Psychic": [0,0,1,0,0,0,0],  # must play exactly 5 cards
    "The_Goad":    [1,0,0,0,0,0,0],  # all spade cards debuffed
    "The_Water":   [0,0,0,0,0,0,0],  # 0 discards
    "The_Eye":     [0,0,0,1,0,0,0],  # no repeat hand types this round
    "The_Plant":   [1,0,0,0,0,0,0],  # all face cards debuffed
    "The_Needle":  [0,0,1,0,0,0,0],  # play only 1 hand
    "The_Head":    [1,0,0,0,0,0,0],  # all heart cards debuffed
    "The_Serpent": [0,0,0,0,0,0,0],  # after play/discard always draw 3 new cards
    "The_Window":  [1,0,0,0,0,0,0],  # all diamond cards debuffed
    "The_Manacle": [0,1,0,0,0,0,0],  # -1 hand size
    "The_Arm":     [0,0,0,0,0,0,1],  # decrease level of played hand each time played
    "The_Pillar":  [0,0,0,0,0,0,0],  # cards previously played this ante are debuffed
    "The_Mouth":   [0,0,0,1,0,0,0],  # play only 1 hand type this round
}

# Boss effects for simulation
BOSS_DEBUFF_SUITS = {
    "The_Club": "C", "The_Goad": "S",
    "The_Head": "H", "The_Window": "D",
    "The_House": "C", "The_Plant": None,  # face cards not suits
}
BOSS_PLAY_ONE   = {"The_Needle"}
BOSS_PLAY_FIVE  = {"The_Psychic"}
BOSS_HAND_SIZE  = {"The_Manacle": -1, "The_Fish": 0}
BOSS_NO_DISCARD = {"The_Water"}
BOSS_HAND_BAN   = {"The_Eye", "The_Mouth"}
BOSS_HOOK       = {"The_Hook"}
BOSS_FACE_DOWN  = {"The_Mark", "The_Fish", "The_Wheel"}
BOSS_SCORE_MOD  = {"The_Flint": 0.5, "The_Wall": 2.0}
BOSS_FACE_DEBUFF = {"The_Plant"}

# ---------------------------------------------------------------------------
# Joker definitions — expanded with identity info
# ---------------------------------------------------------------------------
JOKER_DEFS = [
    {"name":"Jolly_Joker",   "cost":2,  "affinity":"pair",           "is_xmult":False,"is_economy":False},
    {"name":"Zany_Joker",    "cost":4,  "affinity":"three_of_a_kind","is_xmult":False,"is_economy":False},
    {"name":"Mad_Joker",     "cost":4,  "affinity":"two_pair",       "is_xmult":False,"is_economy":False},
    {"name":"Crazy_Joker",   "cost":4,  "affinity":"straight",       "is_xmult":False,"is_economy":False},
    {"name":"Droll_Joker",   "cost":4,  "affinity":"flush",          "is_xmult":False,"is_economy":False},
    {"name":"Sly_Joker",     "cost":2,  "affinity":"pair",           "is_xmult":False,"is_economy":False},
    {"name":"Wily_Joker",    "cost":4,  "affinity":"three_of_a_kind","is_xmult":False,"is_economy":False},
    {"name":"Clever_Joker",  "cost":4,  "affinity":"straight",       "is_xmult":False,"is_economy":False},
    {"name":"Devious_Joker", "cost":4,  "affinity":"straight",       "is_xmult":False,"is_economy":False},
    {"name":"Crafty_Joker",  "cost":4,  "affinity":"flush",          "is_xmult":False,"is_economy":False},
    {"name":"Fibonacci",     "cost":8,  "affinity":"none",           "is_xmult":False,"is_economy":False},
    {"name":"Baron",         "cost":10, "affinity":"none",           "is_xmult":True, "is_economy":False},
    {"name":"Ride_Bus",      "cost":6,  "affinity":"none",           "is_xmult":False,"is_economy":False},
    {"name":"Cavendish",     "cost":6,  "affinity":"none",           "is_xmult":True, "is_economy":False},
    {"name":"Even_Steven",   "cost":4,  "affinity":"none",           "is_xmult":False,"is_economy":False},
    {"name":"Odd_Todd",      "cost":4,  "affinity":"none",           "is_xmult":False,"is_economy":False},
    {"name":"Scary_Face",    "cost":4,  "affinity":"none",           "is_xmult":False,"is_economy":False},
    {"name":"Bull",          "cost":6,  "affinity":"none",           "is_xmult":False,"is_economy":True},
    {"name":"Green_Joker",   "cost":4,  "affinity":"none",           "is_xmult":False,"is_economy":True},
    {"name":"Hologram",      "cost":10, "affinity":"none",           "is_xmult":True, "is_economy":False},
    {"name":"Joker",         "cost":2,  "affinity":"none",           "is_xmult":False,"is_economy":False},
    {"name":"Abstract",      "cost":4,  "affinity":"none",           "is_xmult":False,"is_economy":False},
    {"name":"Smiley_Face",   "cost":4,  "affinity":"none",           "is_xmult":False,"is_economy":False},
    {"name":"Supernova",     "cost":7,  "affinity":"none",           "is_xmult":False,"is_economy":False},
    {"name":"Blueprint",     "cost":10, "affinity":"none",           "is_xmult":True, "is_economy":False},
    {"name":"Swashbuckler",  "cost":6,  "affinity":"none",           "is_xmult":False,"is_economy":True},
    {"name":"Burglar",       "cost":6,  "affinity":"none",           "is_xmult":False,"is_economy":True},
]

HAND_AFFINITY_IDX = {
    "none":0, "pair":1, "two_pair":2, "three_of_a_kind":3,
    "straight":4, "flush":5, "full_house":6, "four_of_a_kind":7,
}

TAROT_CARDS = [
    "The_Devil","The_Chariot","Justice","The_Empress",
    "The_Hanged_Man","The_High_Priestess","Strength",
]

PLANET_HAND_MAP = {
    "Mercury":"high_card","Venus":"three_of_a_kind","Earth":"full_house",
    "Mars":"four_of_a_kind","Jupiter":"flush","Saturn":"straight",
    "Uranus":"two_pair","Neptune":"straight_flush","Pluto":"pair",
}
PLANET_CARDS = list(PLANET_HAND_MAP.keys())


def _build_full_deck(rng):
    deck = []
    for rank in RANKS:
        for suit in SUITS:
            enh  = str(rng.choice(ENHANCEMENTS, p=ENHANCEMENT_PROBS))
            seal = str(rng.choice(SEALS, p=SEAL_PROBS))
            deck.append({"rank":rank,"suit":suit,"enhancement":enh,"seal":seal,"debuff":False})
    idx = rng.permutation(len(deck))
    return [deck[i] for i in idx]


def _deal_from_deck(deck, discard_pile, n, rng):
    if len(deck) < n:
        combined = deck + discard_pile
        idx      = rng.permutation(len(combined))
        deck     = [combined[i] for i in idx]
        discard_pile = []
    hand = deck[:n]
    deck = deck[n:]
    return hand, deck, discard_pile


def _detect_hand_type(cards):
    if not cards:
        return "high_card"
    # Only count non-debuffed cards for hand detection
    active = [c for c in cards if not c.get("debuff", False)]
    if not active:
        active = cards  # fallback

    ranks  = [c["rank"] for c in active]
    suits  = [c["suit"] for c in active]
    rc     = Counter(ranks)
    sc     = Counter(suits)
    vals   = sorted(RANK_VALUES[r] for r in ranks)
    counts = sorted(rc.values(), reverse=True)

    is_flush    = max(sc.values()) >= 5 if sc else False
    uv          = sorted(set(vals))
    is_straight = (len(uv) >= 5 and
                   any(uv[i+4]-uv[i]==4 for i in range(len(uv)-4)))

    if is_flush and is_straight:                          return "straight_flush"
    if counts[0] == 4:                                    return "four_of_a_kind"
    if counts[0]==3 and len(counts)>1 and counts[1]==2:   return "full_house"
    if is_flush:                                          return "flush"
    if is_straight:                                       return "straight"
    if counts[0] == 3:                                    return "three_of_a_kind"
    if counts[0]==2 and len(counts)>1 and counts[1]==2:   return "two_pair"
    if counts[0] == 2:                                    return "pair"
    return "high_card"


def _find_best_hand(hand, boss_blind, one_hand_type, used_hand_types, must_play_n):
    """
    Find the strongest playable hand from current cards.
    Priority: straight_flush > four_of_a_kind > full_house > flush > straight > three_oak > two_pair > pair > high_card
    Respects boss blind constraints.
    """
    ranks      = [c["rank"] for c in hand]
    suits      = [c["suit"] for c in hand]
    rc         = Counter(ranks)
    sc         = Counter(suits)

    debuff_suit  = BOSS_DEBUFF_SUITS.get(boss_blind)
    face_debuffed = boss_blind in BOSS_FACE_DEBUFF
    no_repeat     = boss_blind in BOSS_HAND_BAN and boss_blind == "The_Eye"

    def viable(ht):
        if one_hand_type and ht != one_hand_type:
            return False
        if no_repeat and ht in used_hand_types:
            return False
        return True

    def trim(cards):
        if must_play_n:
            return cards[:must_play_n]
        return cards[:5]

    # 1. Straight flush
    if viable("straight_flush"):
        for suit, count in sc.items():
            if suit == debuff_suit: continue
            suited = sorted([c for c in hand if c["suit"]==suit],
                           key=lambda c: RANK_VALUES[c["rank"]])
            vals   = [RANK_VALUES[c["rank"]] for c in suited]
            uv     = sorted(set(vals))
            for i in range(len(uv)-4):
                w = uv[i:i+5]
                if w[-1]-w[0]==4 and len(w)==5:
                    play = [c for c in suited if RANK_VALUES[c["rank"]] in w][:5]
                    return trim(play), "straight_flush"

    # 2. Four of a kind
    if viable("four_of_a_kind"):
        for rank, count in rc.most_common():
            if count >= 4:
                play = [c for c in hand if c["rank"]==rank][:4]
                # add kicker
                rem = [c for c in hand if c["rank"]!=rank]
                rem.sort(key=lambda c: RANK_VALUES[c["rank"]], reverse=True)
                play = play + rem[:1]
                return trim(play), "four_of_a_kind"

    # 3. Full house
    if viable("full_house"):
        threes = [r for r,c in rc.items() if c >= 3]
        twos   = [r for r,c in rc.items() if c >= 2]
        if threes:
            three_rank = max(threes, key=lambda r: RANK_VALUES[r])
            pair_opts  = [r for r in twos if r != three_rank]
            if pair_opts:
                pair_rank = max(pair_opts, key=lambda r: RANK_VALUES[r])
                play = ([c for c in hand if c["rank"]==three_rank][:3] +
                        [c for c in hand if c["rank"]==pair_rank][:2])
                return trim(play), "full_house"

    # 4. Flush — prefer non-debuffed suit
    if viable("flush"):
        best_suit  = None
        best_count = 0
        for suit, count in sc.items():
            if suit == debuff_suit: continue
            if count > best_count:
                best_count = count
                best_suit  = suit
        if best_count >= 5:
            flush_cards = [c for c in hand if c["suit"]==best_suit]
            flush_cards.sort(key=lambda c: RANK_VALUES[c["rank"]], reverse=True)
            return trim(flush_cards), "flush"

    # 5. Straight
    if viable("straight"):
        rv  = sorted(set(RANK_VALUES[c["rank"]] for c in hand))
        for i in range(len(rv)-4):
            w = rv[i:i+5]
            if w[-1]-w[0]==4 and len(w)==5:
                sr   = {RANK_ORDER[v] for v in w}
                play = [c for c in hand if c["rank"] in sr][:5]
                return trim(play), "straight"

    # 6. Three of a kind
    if viable("three_of_a_kind"):
        for rank, count in rc.most_common():
            if count >= 3:
                play = [c for c in hand if c["rank"]==rank][:3]
                return trim(play), "three_of_a_kind"

    # 7. Two pair
    if viable("two_pair"):
        pairs = [r for r,c in rc.items() if c >= 2]
        pairs.sort(key=lambda r: RANK_VALUES[r], reverse=True)
        if len(pairs) >= 2:
            play = ([c for c in hand if c["rank"]==pairs[0]][:2] +
                    [c for c in hand if c["rank"]==pairs[1]][:2])
            return trim(play), "two_pair"

    # 8. Pair
    if viable("pair"):
        for rank, count in rc.most_common():
            if count >= 2:
                play = [c for c in hand if c["rank"]==rank][:2]
                return trim(play), "pair"

    # 9. High card — play best non-debuffed card
    sorted_hand = sorted(hand,
                         key=lambda c: (0 if (c["suit"]==debuff_suit or
                                             (face_debuffed and c["rank"] in FACE_CARDS))
                                        else RANK_VALUES[c["rank"]]),
                         reverse=True)
    return sorted_hand[:1], "high_card"


def _score_hand(play_cards, held_cards, hand_type, hand_levels,
                jokers, money, boss_blind,
                green_joker_mult, ride_bus_mult, hologram_xmult,
                hand_type_play_count, used_hand_types):
    level  = hand_levels.get(hand_type, 1)
    bc, bm = BASE_HAND_SCORES[hand_type]

    # Score modifier bosses
    score_mod = BOSS_SCORE_MOD.get(boss_blind, 1.0)
    chips  = int((bc + HAND_LEVEL_CHIPS_BONUS * (level - 1)) * score_mod)
    mult   = int((bm + HAND_LEVEL_MULT_BONUS  * (level - 1)) * score_mod)
    xmult  = 1.0

    debuff_suit   = BOSS_DEBUFF_SUITS.get(boss_blind)
    face_debuffed = boss_blind in BOSS_FACE_DEBUFF

    for card in play_cards[:5]:
        if card.get("debuff"): continue
        if debuff_suit and card["suit"] == debuff_suit: continue
        if face_debuffed and card["rank"] in FACE_CARDS: continue

        enh = card.get("enhancement","none")
        rc  = RANK_CHIP_VALUES.get(card["rank"], 0)
        if enh == "bonus":   rc   += 30
        elif enh == "lucky": rc   += 20
        elif enh == "mult":  mult += 4
        elif enh == "glass": xmult *= 2.0
        chips += rc

        seal = card.get("seal","none")
        if seal == "red":    pass  # retrigger — simplify to +chips
        elif seal == "gold": money += 3  # not stored but conceptually

    for card in held_cards:
        enh = card.get("enhancement","none")
        if enh == "steel": xmult *= 1.5

    jn = len(jokers)
    for j in jokers:
        name = j["name"]
        if name == "Jolly_Joker":
            if hand_type in ("pair","two_pair","three_of_a_kind","full_house","four_of_a_kind"):
                mult += 8
        elif name == "Zany_Joker":
            if hand_type in ("three_of_a_kind","full_house","four_of_a_kind"):
                mult += 12
        elif name == "Mad_Joker":
            if hand_type in ("two_pair","full_house"):
                mult += 10
        elif name == "Crazy_Joker":
            if hand_type in ("straight","straight_flush"):
                mult += 12
        elif name == "Droll_Joker":
            if hand_type in ("flush","straight_flush"):
                mult += 10
        elif name == "Sly_Joker":
            if hand_type in ("pair","two_pair","three_of_a_kind","full_house","four_of_a_kind"):
                chips += 50
        elif name == "Wily_Joker":
            if hand_type in ("three_of_a_kind","full_house","four_of_a_kind"):
                chips += 100
        elif name == "Clever_Joker":
            if hand_type in ("straight","straight_flush"):
                chips += 80
        elif name == "Devious_Joker":
            if hand_type in ("straight","straight_flush"):
                chips += 100
        elif name == "Crafty_Joker":
            if hand_type in ("flush","straight_flush"):
                chips += 80
        elif name == "Fibonacci":
            for c in play_cards[:5]:
                if c["rank"] in FIBONACCI_RANKS: mult += 8
        elif name == "Baron":
            for c in held_cards:
                if c["rank"] == "K": xmult *= 1.5
        elif name == "Ride_Bus":    mult  += ride_bus_mult
        elif name == "Cavendish":   xmult *= 3.0
        elif name == "Even_Steven":
            for c in play_cards[:5]:
                if c["rank"] in EVEN_RANKS: mult += 4
        elif name == "Odd_Todd":
            for c in play_cards[:5]:
                if c["rank"] in ODD_RANKS: chips += 31
        elif name == "Scary_Face":
            for c in play_cards[:5]:
                if c["rank"] in FACE_CARDS: chips += 30
        elif name == "Bull":        chips += 2 * money
        elif name == "Green_Joker": mult  += green_joker_mult
        elif name == "Hologram":    xmult *= hologram_xmult
        elif name == "Joker":       mult  += 4
        elif name == "Abstract":    mult  += 3 * jn
        elif name == "Smiley_Face":
            for c in play_cards[:5]:
                if c["rank"] in FACE_CARDS: mult += 5
        elif name == "Supernova":   mult  += hand_type_play_count
        elif name == "Swashbuckler":
            sell_total = sum(j.get("sell_cost",2) for j in jokers if j["name"]!=name)
            mult += sell_total

    return max(int(chips * mult * xmult), 0)


def _generate_shop(rng, ante, hand_levels, dominant_hand):
    """Generate shop with planet cards targeted at dominant hand."""
    n    = int(rng.integers(2, 5))
    shop = []
    base = 4 + (ante - 1)

    for _ in range(n):
        roll = rng.random()
        if roll < 0.38:
            jd = JOKER_DEFS[int(rng.integers(0, len(JOKER_DEFS)))]
            cost = max(2, int(jd["cost"] + max(0, ante-1)))
            shop.append({"set":"JOKER","name":jd["name"],"cost":cost,
                         "affinity":jd["affinity"],"is_xmult":jd["is_xmult"],
                         "is_economy":jd["is_economy"]})
        elif roll < 0.53:
            # Planet card — bias toward dominant hand's planet
            if rng.random() < 0.6:
                # Find planet for dominant hand
                planet = next((p for p,h in PLANET_HAND_MAP.items() if h==dominant_hand),
                              PLANET_CARDS[int(rng.integers(0,len(PLANET_CARDS)))])
            else:
                planet = PLANET_CARDS[int(rng.integers(0, len(PLANET_CARDS)))]
            shop.append({"set":"PLANET","name":planet,"cost":max(2, base-1),
                         "hand_type":PLANET_HAND_MAP.get(planet,"high_card")})
        elif roll < 0.65:
            t = TAROT_CARDS[int(rng.integers(0, len(TAROT_CARDS)))]
            shop.append({"set":"TAROT","name":t,"cost":max(2, base)})
        elif roll < 0.75:
            shop.append({"set":"VOUCHER","name":"Voucher","cost":max(3, base+3)})
        elif roll < 0.87:
            # Celestial pack — gives planet cards
            shop.append({"set":"PACK","name":"Celestial_Pack","cost":max(3, base+1),
                         "pack_type":"celestial"})
        else:
            shop.append({"set":"PACK","name":"Arcana_Pack","cost":max(3, base+1),
                         "pack_type":"arcana"})
    return shop


def _compute_joker_synergy(jokers, dominant_hand):
    """Score 0-1 how well jokers match current dominant hand."""
    if not jokers:
        return 0.0
    score = 0.0
    for j in jokers:
        aff = j.get("affinity","none")
        if aff == "none":
            if j.get("is_xmult"): score += 0.5
            else:                  score += 0.2
        elif aff == dominant_hand:
            score += 1.0
        elif aff in ("pair","two_pair") and dominant_hand in ("pair","two_pair","three_of_a_kind"):
            score += 0.6
        else:
            score += 0.1
    return min(score / max(len(jokers), 1), 1.0)


class MockGameState:
    def __init__(self, rng, max_ante=8):
        self.rng      = rng
        self.max_ante = max_ante

        self.ante        = 1
        self.blind_idx   = 0
        self.money       = 4
        self.joker_slots = []
        self.joker_limit = 5

        self.hand_levels      = {ht:1 for ht in HAND_TYPES}
        self.hand_play_counts = {ht:0 for ht in HAND_TYPES}
        self.dominant_hand    = "pair"

        self.hands_left    = 4
        self.discards_left = 4
        self.hand_size     = 8
        self.chips_scored  = 0

        self.phase                 = "BLIND_SELECT"
        self.boss_blind            = "The_Hook"   # will be set properly each boss round
        self.upcoming_boss         = self._roll_boss()
        self.one_hand_type_allowed = None
        self.used_hand_types       = set()

        self._full_deck    = _build_full_deck(rng)
        self._deck         = list(self._full_deck)
        self._discard_pile = []
        self.hand          = []

        self.green_joker_mult  = 0.0
        self.ride_bus_mult     = 0.0
        self.hologram_xmult    = 1.0
        self.cavendish_alive   = True
        self.interest_earned   = 0

        self.shop        = _generate_shop(rng, self.ante, self.hand_levels, self.dominant_hand)
        self.reroll_cost = 5
        self.done        = False
        self.won         = False

    def _roll_boss(self):
        return ALL_BOSS_BLINDS[int(self.rng.integers(0, len(ALL_BOSS_BLINDS)))]

    @property
    def blind_type(self):
        return ["small","big","boss"][self.blind_idx]

    @property
    def chips_needed(self):
        return BLIND_CHIPS.get(min(self.ante,8), BLIND_CHIPS[8])[self.blind_type]

    @property
    def joker_count(self):
        return len(self.joker_slots)

    def _apply_interest(self):
        interest = min(self.money // 5, 5)
        self.interest_earned = interest
        self.money += interest + 1  # +1 base income

    def _update_dominant_hand(self, hand_type):
        self.hand_play_counts[hand_type] = self.hand_play_counts.get(hand_type,0) + 1
        self.dominant_hand = max(self.hand_play_counts, key=self.hand_play_counts.get)

    def _apply_cavendish_check(self):
        if any(j["name"]=="Cavendish" for j in self.joker_slots):
            if self.rng.random() < 1/1000:
                self.joker_slots = [j for j in self.joker_slots if j["name"]!="Cavendish"]

    def _advance_blind(self):
        if self.blind_idx == 2:
            self.ante      += 1
            self.blind_idx  = 0
            if self.ante > self.max_ante:
                self.won  = True
                self.done = True
                return
            self.upcoming_boss = self._roll_boss()
        else:
            self.blind_idx += 1
        self.chips_scored = 0
        self.phase = "BLIND_SELECT"

    def _start_blind(self):
        self.hands_left            = 4
        self.discards_left         = 4
        self.hand_size             = 8
        self.one_hand_type_allowed = None
        self.used_hand_types       = set()
        self.green_joker_mult      = 0.0
        self.ride_bus_mult         = 0.0

        if self.blind_type == "boss":
            self.boss_blind = self.upcoming_boss
            # Apply hand size modifiers
            if self.boss_blind == "The_Manacle":
                self.hand_size = max(1, self.hand_size - 1)
            elif self.boss_blind == "The_Fish":
                self.hand_size = max(3, self.hand_size - 2)
            elif self.boss_blind in BOSS_NO_DISCARD:
                self.discards_left = 0
            elif self.boss_blind == "The_Needle":
                self.hands_left = 1
            elif self.boss_blind == "The_Mouth":
                self.one_hand_type_allowed = str(self.rng.choice(HAND_TYPES))
        else:
            self.boss_blind = "none"

        # Debuff cards for pillar (previously played this ante)
        if self.boss_blind == "The_Pillar":
            for card in self._full_deck:
                if self.rng.random() < 0.3:
                    card["debuff"] = True
        else:
            for card in self._full_deck:
                card["debuff"] = False

        self.chips_scored = 0
        self.hand, self._deck, self._discard_pile = _deal_from_deck(
            self._deck, self._discard_pile, self.hand_size, self.rng)
        self.phase = "SELECTING_HAND"

    def _end_blind(self):
        for card in self.hand:
            if card.get("enhancement") == "gold":
                self.money += 3
            if card.get("seal") == "gold":
                self.money += 3

        self._apply_cavendish_check()
        self._apply_interest()

        # Rebuild deck
        all_cards = self._deck + self._discard_pile + self.hand
        card_map  = {}
        for c in all_cards:
            card_map[(c["rank"],c["suit"])] = (c.get("enhancement","none"), c.get("seal","none"))
        for fc in self._full_deck:
            key = (fc["rank"],fc["suit"])
            if key in card_map:
                fc["enhancement"], fc["seal"] = card_map[key]
            fc["debuff"] = False

        idx                = self.rng.permutation(len(self._full_deck))
        self._deck         = [self._full_deck[i] for i in idx]
        self._discard_pile = []
        self.hand          = []

        self.shop        = _generate_shop(self.rng, self.ante, self.hand_levels, self.dominant_hand)
        self.reroll_cost = 5
        self.phase       = "SHOP"

    def step_blind_select(self, action):
        """Agent always selects — skipping disabled in curriculum."""
        self._start_blind()
        return 0.0

    def step_selecting_hand(self, action):
        reward = 0.0

        boss       = self.boss_blind
        must_play_n = None
        if boss == "The_Psychic": must_play_n = 5
        elif boss == "The_Needle": must_play_n = 1

        # Hook: after each hand played, 2 random cards discarded from hand
        # We simulate this by reducing effective hand quality
        hook_active = (boss == "The_Hook")

        # Discard conservation logic:
        # - Never discard if 0 discards left
        # - Save last discard for boss blind (don't use it on non-boss rounds)
        # - On The_Hook: don't discard at all (hook discards after play, not after discard)
        is_boss        = self.blind_type == "boss"
        save_last      = not is_boss and self.discards_left == 1
        can_discard    = self.discards_left > 0 and not hook_active and not save_last

        # Check what hands are available
        play_cards, best_hand = _find_best_hand(
            self.hand, boss, self.one_hand_type_allowed,
            self.used_hand_types, must_play_n)

        # Decide whether to discard or play
        hand_strength = HAND_TYPE_IDX.get(best_hand, 8)  # lower = stronger
        should_discard = (
            can_discard and
            self.hands_left > 1 and
            hand_strength >= 5 and  # three_of_a_kind or worse
            action == 1
        )

        if should_discard:
            self.discards_left    -= 1
            self.green_joker_mult  = max(0.0, self.green_joker_mult - 1)
            self._discard_pile    += self.hand
            self.hand, self._deck, self._discard_pile = _deal_from_deck(
                self._deck, self._discard_pile, self.hand_size, self.rng)
            # Small discard penalty to discourage over-discarding
            return -0.05

        # Play the best hand
        held  = [c for c in self.hand if c not in play_cards]
        chips = _score_hand(
            play_cards, held, best_hand, self.hand_levels,
            self.joker_slots, self.money, boss,
            self.green_joker_mult, self.ride_bus_mult, self.hologram_xmult,
            self.hand_play_counts.get(best_hand, 0),
            self.used_hand_types,
        )
        self.chips_scored += chips
        self.hands_left   -= 1
        self.used_hand_types.add(best_hand)

        has_face = any(c["rank"] in FACE_CARDS for c in play_cards)
        if has_face: self.ride_bus_mult = 0.0
        else:        self.ride_bus_mult += 1
        self.green_joker_mult += 1

        # Hook: discard 2 random cards after playing
        if hook_active and len(self.hand) > 2:
            hook_idxs  = self.rng.choice(len(self.hand), size=2, replace=False)
            self._discard_pile += [self.hand[i] for i in hook_idxs]
            self.hand = [c for c in self.hand
                         if self.hand.index(c) not in hook_idxs]

        # Glass card destruction
        surviving = []
        for card in play_cards:
            if card.get("enhancement")=="glass" and self.rng.random()<0.25:
                self._full_deck = [fc for fc in self._full_deck
                                   if not (fc["rank"]==card["rank"] and fc["suit"]==card["suit"])]
            else:
                surviving.append(card)

        self._discard_pile += surviving
        new_cards, self._deck, self._discard_pile = _deal_from_deck(
            self._deck, self._discard_pile, len(play_cards), self.rng)
        self.hand = [c for c in self.hand if c not in play_cards] + new_cards

        self._update_dominant_hand(best_hand)

        # Level up hand if played enough times — planet card equivalent
        if self.hand_play_counts[best_hand] % 5 == 0:
            self.hand_levels[best_hand] = self.hand_levels.get(best_hand,1) + 1

        # The Arm: decrease level of played hand
        if boss == "The_Arm":
            self.hand_levels[best_hand] = max(1, self.hand_levels.get(best_hand,1) - 1)

        # Dense reward
        reward += 0.5 * (chips / max(self.chips_needed, 1))
        reward += 0.1  # survival bonus

        if self.chips_scored >= self.chips_needed:
            reward += 5.0 + 1.0 * self.ante
            # Bonus for clearing with hands remaining
            reward += 0.2 * self.hands_left
            self._end_blind()
        elif self.hands_left <= 0:
            penalty = 8.0
            if is_boss: penalty += 3.0
            # Extra penalty if discards were wasted on non-boss
            if self.discards_left == 0 and not is_boss:
                penalty += 1.0
            reward -= penalty
            self.done = True

        return reward

    def step_shop(self, action):
        reward = 0.0

        if action == 0:
            self._advance_blind()
            return reward

        if action == 6:
            if self.money >= self.reroll_cost:
                self.money       -= self.reroll_cost
                self.reroll_cost += 1
                self.shop         = _generate_shop(self.rng, self.ante, self.hand_levels, self.dominant_hand)
            self._advance_blind()
            return reward

        buy_idx = int(action - 1)
        if buy_idx < len(self.shop):
            item = self.shop[buy_idx]
            cost = item["cost"]
            if self.money >= cost:
                self.money -= cost
                ctype = item["set"]
                name  = item.get("name","")

                if ctype == "JOKER" and self.joker_count < self.joker_limit:
                    jdef = next((j for j in JOKER_DEFS if j["name"]==name), None)
                    if jdef:
                        self.joker_slots.append({
                            "name":      name,
                            "affinity":  jdef["affinity"],
                            "is_xmult":  jdef["is_xmult"],
                            "is_economy":jdef["is_economy"],
                            "sell_cost": max(1, cost//2),
                        })
                    synergy = _compute_joker_synergy(self.joker_slots, self.dominant_hand)
                    reward += 1.0 * self.ante * (0.3 + 0.7*synergy)

                elif ctype == "PLANET":
                    # Level up the specific hand this planet boosts
                    hand_type = item.get("hand_type", self.dominant_hand)
                    self.hand_levels[hand_type] = self.hand_levels.get(hand_type,1) + 1
                    if any(j["name"]=="Hologram" for j in self.joker_slots):
                        self.hologram_xmult += 0.25
                    # Better reward if it boosts our dominant hand
                    if hand_type == self.dominant_hand:
                        reward += 0.8
                    else:
                        reward += 0.3

                elif ctype == "TAROT":
                    reward += self._apply_tarot(name)

                elif ctype == "VOUCHER":
                    self.money += 3  # simplified voucher value
                    reward += 0.3

                elif ctype == "PACK":
                    pack_type = item.get("pack_type","celestial")
                    if pack_type == "celestial":
                        # Give a planet card for dominant hand
                        self.hand_levels[self.dominant_hand] = \
                            self.hand_levels.get(self.dominant_hand,1) + 1
                        reward += 0.6
                    elif pack_type == "arcana":
                        # Give a tarot card effect
                        t = TAROT_CARDS[int(self.rng.integers(0,len(TAROT_CARDS)))]
                        reward += self._apply_tarot(t)

        self._advance_blind()
        return reward

    def _apply_tarot(self, tarot_name):
        if not self.hand: return 0.05
        tidx = 0
        if tarot_name == "The_Chariot":
            kings = [i for i,c in enumerate(self.hand) if c["rank"]=="K"]
            tidx  = kings[0] if kings else 0
        elif tarot_name == "Strength":
            tidx = max(range(len(self.hand)),
                       key=lambda i: RANK_VALUES.get(self.hand[i]["rank"],0))
        else:
            tidx = max(range(len(self.hand)),
                       key=lambda i: RANK_VALUES.get(self.hand[i]["rank"],0))

        target  = self.hand[tidx]
        enh_map = {
            "The_Devil":   "gold",
            "The_Chariot": "steel",
            "Justice":     "glass",
            "The_Empress": "mult",
            "Strength":    "bonus",
        }
        if tarot_name in enh_map:
            target["enhancement"] = enh_map[tarot_name]
            for fc in self._full_deck:
                if fc["rank"]==target["rank"] and fc["suit"]==target["suit"]:
                    fc["enhancement"] = target["enhancement"]; break
        elif tarot_name == "The_Hanged_Man":
            if len(self._deck) > 2:
                self._deck.sort(key=lambda c: RANK_VALUES.get(c["rank"],0))
                removed    = self._deck[:2]
                self._deck = self._deck[2:]
                for rc in removed:
                    self._full_deck = [fc for fc in self._full_deck
                                       if not (fc["rank"]==rc["rank"] and fc["suit"]==rc["suit"])]
        return 0.05

    def encode(self):
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        # [0-9] Base game state
        obs[0] = min(self.ante, 8) / 8.0
        obs[1] = min((self.ante-1)*3 + self.blind_idx + 1, 24) / 24.0
        obs[2] = min(self.money, 100) / 100.0
        obs[3] = self.hands_left / 4.0
        obs[4] = self.discards_left / 4.0
        obs[5] = min(self.chips_scored, 300000) / 300000.0
        obs[6] = min(self.chips_needed, 300000) / 300000.0
        obs[7] = self.joker_count / 5.0
        obs[8] = self.joker_limit / 5.0
        obs[9] = min(self.reroll_cost, 10) / 10.0

        # [10-17] Hand card ranks, [18-25] suits
        for i, c in enumerate(self.hand[:8]):
            obs[10+i] = RANK_VALUES.get(c["rank"], 0) / 12.0
            obs[18+i] = SUIT_VALUES.get(c["suit"], 0) / 3.0

        # [26-33] Hand card enhancements
        for i, c in enumerate(self.hand[:8]):
            obs[26+i] = ENHANCEMENT_IDX.get(c.get("enhancement","none"), 0) / len(ENHANCEMENTS)

        # [34-41] Hand card seals
        for i, c in enumerate(self.hand[:8]):
            obs[34+i] = SEAL_IDX.get(c.get("seal","none"), 0) / len(SEALS)

        # [42-49] Hand card debuff flags
        for i, c in enumerate(self.hand[:8]):
            obs[42+i] = 1.0 if c.get("debuff", False) else 0.0

        # [50-54] Shop costs, [55-59] shop type
        shop_type_enc = {"JOKER":0.0,"PLANET":0.2,"TAROT":0.4,"VOUCHER":0.6,"PACK":0.8,"SPECTRAL":1.0}
        for i, c in enumerate(self.shop[:5]):
            obs[50+i] = min(c["cost"], 20) / 20.0
            obs[55+i] = shop_type_enc.get(c["set"], 0.0)

        # [60] blind type, [61] state phase
        obs[60] = {"small":0.0,"big":0.5,"boss":1.0}.get(self.blind_type, 0.0)
        obs[61] = {"SELECTING_HAND":0.0,"SHOP":0.5,"BLIND_SELECT":1.0}.get(self.phase, 0.0)

        # [62-74] Rank counts
        vis = self._deck + self.hand
        rc  = Counter(c["rank"] for c in vis)
        for i, r in enumerate(RANKS):
            obs[62+i] = min(rc.get(r,0), 4) / 4.0

        # [75-78] Suit counts
        sc = Counter(c["suit"] for c in vis)
        for i, s in enumerate(SUITS):
            obs[75+i] = min(sc.get(s,0), 13) / 13.0

        # [79-87] Dominant hand one-hot
        di = HAND_TYPE_IDX.get(self.dominant_hand, 8)
        obs[79+di] = 1.0

        # [88-96] All hand levels
        for i, ht in enumerate(HAND_TYPES):
            obs[88+i] = min(self.hand_levels.get(ht,1), 10) / 10.0

        # [97] Deck size
        obs[97] = len(self._full_deck) / 52.0

        # [98-104] Boss blind category flags
        cat = BOSS_CATEGORIES.get(self.boss_blind, [0]*7)
        for i, v in enumerate(cat):
            obs[98+i] = float(v)

        # [105-127] Boss blind identity one-hot
        boss_id = BOSS_BLIND_IDX.get(self.boss_blind, -1)
        if boss_id >= 0:
            obs[105+boss_id] = 1.0

        # [128-129] Interest system
        obs[128] = min(self.interest_earned, 5) / 5.0
        obs[129] = (self.money % 5) / 4.0  # how close to next bracket

        # [130-144] Joker slots 5x3
        for i, j in enumerate(self.joker_slots[:5]):
            base = 130 + i*3
            aff  = j.get("affinity","none")
            obs[base]   = HAND_AFFINITY_IDX.get(aff, 0) / 7.0
            obs[base+1] = 1.0 if j.get("is_xmult",  False) else 0.0
            obs[base+2] = 1.0 if j.get("is_economy", False) else 0.0

        # [145] Joker synergy score
        obs[145] = _compute_joker_synergy(self.joker_slots, self.dominant_hand)

        # [146-147] Green joker and ride bus mult
        obs[146] = min(self.green_joker_mult, 20) / 20.0
        obs[147] = min(self.ride_bus_mult, 20) / 20.0

        return obs

    def action_masks(self):
        mask = np.ones(N_ACTIONS, dtype=bool)
        if self.phase == "BLIND_SELECT":
            mask[1]  = False
            mask[2:] = False
        elif self.phase == "SELECTING_HAND":
            if self.discards_left <= 0: mask[1] = False
            mask[5:] = False
        elif self.phase == "SHOP":
            for i in range(1, 6):
                bi = i - 1
                if bi >= len(self.shop):
                    mask[i] = False
                elif self.shop[bi]["cost"] > self.money:
                    mask[i] = False
                elif (self.shop[bi]["set"]=="JOKER" and
                      self.joker_count >= self.joker_limit):
                    mask[i] = False
            if self.money < self.reroll_cost:
                mask[6] = False
        return mask


class BalatroMockEnv(gym.Env):
    """
    High-fidelity mock Balatro env v2.
    OBS_DIM=148. MaskablePPO compatible.
    """
    metadata = {"render_modes": []}

    def __init__(self, max_ante=8):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self._rng      = np.random.default_rng()
        self._game     = None
        self._max_ante = max_ante

    def set_max_ante(self, max_ante):
        self._max_ante = max_ante

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._game = MockGameState(self._rng, max_ante=self._max_ante)
        return self._game.encode(), {}

    def step(self, action):
        g      = self._game
        action = int(action)
        if   g.phase == "BLIND_SELECT":   reward = g.step_blind_select(action)
        elif g.phase == "SELECTING_HAND": reward = g.step_selecting_hand(action)
        elif g.phase == "SHOP":           reward = g.step_shop(action)
        else:                             reward = 0.0
        if g.won: reward += 20.0
        obs  = g.encode()
        done = g.done or g.won
        return obs, reward, done, False, {"ante":g.ante,"round":g.blind_idx+1,"won":g.won}

    def action_masks(self):
        if self._game is None:
            return np.ones(N_ACTIONS, dtype=bool)
        return self._game.action_masks()

    def render(self): pass
    def close(self):  pass