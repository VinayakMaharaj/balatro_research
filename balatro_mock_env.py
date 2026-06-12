"""
balatro_mock_env.py
High-fidelity mock Gymnasium environment for MaskablePPO training.
Final training version — curriculum learning, all 15 fixes, optimised rewards.

OBS_DIM = 72:
  [0]     ante / 8
  [1]     round_num / 24
  [2]     money / 100
  [3]     hands_left / 4
  [4]     discards_left / 4
  [5]     chips_scored / 300000
  [6]     chips_needed / 300000
  [7]     joker_count / 5
  [8]     joker_limit / 5
  [9]     reroll_cost / 10
  [10-17] hand card ranks (8 slots)
  [18-25] hand card suits (8 slots)
  [26-30] shop costs (5 slots)
  [31-35] shop is_joker flags (5 slots)
  [36]    blind_type 0/0.5/1
  [37]    state_phase 0/0.5/1
  [38-50] rank counts in deck+hand (13 dims)
  [51-54] suit counts in deck+hand (4 dims)
  [55-63] dominant hand one-hot (9 dims)
  [64]    dominant hand level / 10
  [65]    deck size / 52
  [66]    has_suit_debuff 0/1
  [67]    debuffed_suit_idx / 3
  [68]    play_exactly_n / 5
  [69]    hand_size_penalty 0/1
  [70]    green_joker_mult / 20
  [71]    ride_bus_mult / 20

Curriculum learning:
  Stage 0 (0-25M steps):   ante 1 only — master basic hand play
  Stage 1 (25-50M steps):  ante 1-2
  Stage 2 (50-100M steps): ante 1-3
  Stage 3 (100M+ steps):   full game ante 1-8
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import Counter

OBS_DIM   = 72
N_ACTIONS = 7

# Curriculum stage — set externally by training loop
CURRICULUM_MAX_ANTE = 1  # starts at 1, increases during training

RANK_ORDER = ["2","3","4","5","6","7","8","9","T","J","Q","K","A"]
RANK_VALUES = {r: i for i, r in enumerate(RANK_ORDER)}
SUIT_VALUES = {"S": 0, "H": 1, "D": 2, "C": 3}
SUITS = ["S","H","D","C"]
RANKS = list(RANK_ORDER)

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
    "straight_flush": (100,8),"four_of_a_kind":(60,7),
    "full_house":     (40,4), "flush":         (35,4),
    "straight":       (30,4), "three_of_a_kind":(30,3),
    "two_pair":       (20,2), "pair":          (10,2),
    "high_card":      (5,1),
}
HAND_LEVEL_CHIPS_BONUS = 50
HAND_LEVEL_MULT_BONUS  = 2

BLIND_CHIPS = {
    1:{"small":300,  "big":450,  "boss":600},
    2:{"small":800,  "big":1200, "boss":1600},
    3:{"small":2000, "big":3000, "boss":4000},
    4:{"small":5000, "big":7500, "boss":10000},
    5:{"small":11000,"big":16500,"boss":22000},
    6:{"small":20000,"big":30000,"boss":40000},
    7:{"small":35000,"big":52500,"boss":70000},
    8:{"small":60000,"big":90000,"boss":120000},
}

ENHANCEMENTS      = ["none","glass","steel","gold","mult","bonus"]
ENHANCEMENT_PROBS = [0.70,0.05,0.08,0.07,0.05,0.05]

JOKER_DEFS = [
    {"name":"Fibonacci",   "cost":8},
    {"name":"Baron",       "cost":10},
    {"name":"Ride_Bus",    "cost":6},
    {"name":"Cavendish",   "cost":6},
    {"name":"Even_Steven", "cost":4},
    {"name":"Odd_Todd",    "cost":4},
    {"name":"Scary_Face",  "cost":4},
    {"name":"Bull",        "cost":6},
    {"name":"Green_Joker", "cost":4},
    {"name":"Hologram",    "cost":10},
    {"name":"Joker",       "cost":2},
    {"name":"Abstract",    "cost":4},
    {"name":"Smiley_Face", "cost":4},
    {"name":"Supernova",   "cost":7},
    {"name":"Blueprint",   "cost":10},
]

BOSS_DEBUFFS = [
    "debuff_hearts","debuff_spades","debuff_diamonds","debuff_clubs",
    "play_1","play_5","hand_size_minus1","unique_hands","one_hand_type","draw_1",
]
DEBUFF_SUITS = {
    "debuff_hearts":"H","debuff_spades":"S",
    "debuff_diamonds":"D","debuff_clubs":"C",
}
TAROT_CARDS = ["The_Devil","The_Chariot","Justice","The_Empress","The_Hanged_Man"]


def _build_full_deck(rng):
    deck = []
    for rank in RANKS:
        for suit in SUITS:
            enh = str(rng.choice(ENHANCEMENTS, p=ENHANCEMENT_PROBS))
            deck.append({"rank":rank,"suit":suit,"enhancement":enh})
    idx = rng.permutation(len(deck))
    return [deck[i] for i in idx]


def _deal_from_deck(deck, discard_pile, n, rng):
    if len(deck) < n:
        combined = deck + discard_pile
        idx = rng.permutation(len(combined))
        deck = [combined[i] for i in idx]
        discard_pile = []
    hand = deck[:n]
    deck = deck[n:]
    return hand, deck, discard_pile


def _detect_hand_type(cards):
    if not cards:
        return "high_card"
    ranks   = [c["rank"] for c in cards]
    suits   = [c["suit"] for c in cards]
    rc      = Counter(ranks)
    sc      = Counter(suits)
    vals    = sorted(RANK_VALUES[r] for r in ranks)
    counts  = sorted(rc.values(), reverse=True)
    is_flush = max(sc.values()) >= 5
    uv      = sorted(set(vals))
    is_straight = (len(uv) >= 5 and
                   any(uv[i+4]-uv[i]==4 for i in range(len(uv)-4)))
    if is_flush and is_straight:                           return "straight_flush"
    if counts[0] == 4:                                     return "four_of_a_kind"
    if counts[0]==3 and len(counts)>1 and counts[1]==2:    return "full_house"
    if is_flush:                                           return "flush"
    if is_straight:                                        return "straight"
    if counts[0] == 3:                                     return "three_of_a_kind"
    if counts[0]==2 and len(counts)>1 and counts[1]==2:    return "two_pair"
    if counts[0] == 2:                                     return "pair"
    return "high_card"


def _score_hand(play_cards, held_cards, hand_type, hand_levels,
                jokers, money, boss_debuff,
                green_joker_mult, ride_bus_mult, hologram_xmult,
                hand_type_play_count):
    level       = hand_levels.get(hand_type, 1)
    bc, bm      = BASE_HAND_SCORES[hand_type]
    chips       = bc + HAND_LEVEL_CHIPS_BONUS * (level - 1)
    mult        = bm + HAND_LEVEL_MULT_BONUS  * (level - 1)
    xmult       = 1.0
    debuff_suit = DEBUFF_SUITS.get(boss_debuff)

    for card in play_cards[:5]:
        if debuff_suit and card["suit"] == debuff_suit:
            continue
        enh = card.get("enhancement","none")
        rc  = RANK_CHIP_VALUES.get(card["rank"], 0)
        if enh == "bonus":   rc += 30
        elif enh == "glass": xmult *= 2.0
        chips += rc
        if enh == "mult":    mult += 4

    for card in held_cards:
        if card.get("enhancement") == "steel":
            xmult *= 1.5

    jn = len(jokers)
    for j in jokers:
        name = j["name"]
        if name == "Fibonacci":
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

    return max(int(chips * mult * xmult), 0)


def _select_play_cards(hand, action, boss_debuff, one_hand_type, must_play_n):
    ranks       = [c["rank"] for c in hand]
    suits       = [c["suit"] for c in hand]
    rc          = Counter(ranks)
    sc          = Counter(suits)
    debuff_suit = DEBUFF_SUITS.get(boss_debuff)

    # The Mouth constraint
    if one_hand_type == "flush"    and action != 2: action = 2
    elif one_hand_type == "straight" and action != 3: action = 3

    # Always try flush first (strongest hand, best chips) — prefer non-debuffed suit
    for suit, count in sc.most_common():
        if suit == debuff_suit: continue
        fc = [c for c in hand if c["suit"] == suit]
        if len(fc) >= 5:
            play = fc[:5]
            if must_play_n: play = play[:must_play_n]
            return play, _detect_hand_type(play)

    # Always try straight second
    rv = sorted(set(RANK_VALUES[r] for r in ranks))
    for i in range(len(rv)-4):
        w = rv[i:i+5]
        if w[-1]-w[0]==4 and len(w)==5:
            sr   = {RANK_ORDER[v] for v in w}
            play = [c for c in hand if c["rank"] in sr][:5]
            if must_play_n: play = play[:must_play_n]
            return play, "straight"

    src = sorted(rc.items(), key=lambda x:(x[1],RANK_VALUES[x[0]]), reverse=True)
    tr, tc = src[0]
    play   = [c for c in hand if c["rank"]==tr][:tc]
    if tc==3 and len(src)>1 and src[1][1]>=2:
        play += [c for c in hand if c["rank"]==src[1][0]][:2]
    elif tc==2 and len(src)>1 and src[1][1]==2:
        play += [c for c in hand if c["rank"]==src[1][0]][:2]
    rem  = [c for c in hand if c not in play]
    rem.sort(key=lambda x: RANK_VALUES[x["rank"]], reverse=True)
    play = (play+rem)[:5]
    if must_play_n is not None: play = play[:must_play_n]
    if not play: play = hand[:1]
    return play, _detect_hand_type(play)


def _generate_shop(rng, ante):
    n    = int(rng.integers(2,5))
    shop = []
    base = 4 + (ante-1)
    for _ in range(n):
        roll = rng.random()
        if roll < 0.40:
            jd = JOKER_DEFS[int(rng.integers(0,len(JOKER_DEFS)))]
            shop.append({"set":"JOKER",  "name":jd["name"], "cost":max(2,int(jd["cost"]+max(0,ante-1)))})
        elif roll < 0.55:
            shop.append({"set":"PLANET", "name":"Planet",   "cost":max(2,base-1)})
        elif roll < 0.68:
            t = TAROT_CARDS[int(rng.integers(0,len(TAROT_CARDS)))]
            shop.append({"set":"TAROT",  "name":t,          "cost":max(2,base)})
        elif roll < 0.80:
            shop.append({"set":"VOUCHER","name":"Voucher",   "cost":max(3,base+3)})
        else:
            shop.append({"set":"PACK",   "name":"Pack",     "cost":max(2,base+1)})
    return shop


def _smart_tarot_target(tarot_name, hand, jokers):
    if not hand: return 0
    jnames = {j["name"] for j in jokers}
    if tarot_name == "The_Chariot" and "Baron" in jnames:
        kings = [i for i,c in enumerate(hand) if c["rank"]=="K"]
        if kings: return kings[0]
    return max(range(len(hand)), key=lambda i: RANK_VALUES.get(hand[i]["rank"],0))


class MockGameState:
    def __init__(self, rng, max_ante=8):
        self.rng      = rng
        self.max_ante = max_ante  # curriculum: cap which ante agent can reach
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
        self.boss_debuff           = "none"
        self.one_hand_type_allowed = None
        self.used_hand_types       = set()

        self._full_deck    = _build_full_deck(rng)
        self._deck         = list(self._full_deck)
        self._discard_pile = []
        self.hand          = []

        self.green_joker_mult    = 0.0
        self.ride_bus_mult       = 0.0
        self.hologram_xmult      = 1.0
        self.cavendish_alive     = True
        self.interest_last_round = 0

        self.shop        = _generate_shop(rng, self.ante)
        self.reroll_cost = 5
        self.done        = False
        self.won         = False

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
        self.interest_last_round = interest
        self.money += interest + 1

    def _update_dominant_hand(self, hand_type):
        self.hand_play_counts[hand_type] = self.hand_play_counts.get(hand_type,0) + 1
        self.dominant_hand = max(self.hand_play_counts, key=self.hand_play_counts.get)

    def _apply_cavendish_check(self):
        if self.cavendish_alive and any(j["name"]=="Cavendish" for j in self.joker_slots):
            if self.rng.random() < 1/1000:
                self.cavendish_alive = False
                self.joker_slots = [j for j in self.joker_slots if j["name"]!="Cavendish"]

    def _advance_blind(self):
        if self.blind_idx == 2:
            self.ante     += 1
            self.blind_idx = 0
            # Curriculum: treat max_ante+1 as a win so agent gets full reward
            if self.ante > self.max_ante:
                self.won  = True
                self.done = True
                return
        else:
            self.blind_idx += 1
        self.chips_scored = 0
        self.phase = "BLIND_SELECT"

    def _start_blind(self):
        self.hands_left            = 4
        self.discards_left         = 4
        self.hand_size             = 8
        self.boss_debuff           = "none"
        self.one_hand_type_allowed = None
        self.used_hand_types       = set()
        self.green_joker_mult      = 0.0
        self.ride_bus_mult         = 0.0

        if self.blind_type == "boss":
            self.boss_debuff = str(self.rng.choice(BOSS_DEBUFFS))
            if self.boss_debuff == "hand_size_minus1":
                self.hand_size = max(1, self.hand_size-1)
            elif self.boss_debuff == "draw_1":
                self.hand_size = max(3, self.hand_size-2)
            elif self.boss_debuff == "one_hand_type":
                self.one_hand_type_allowed = str(self.rng.choice(HAND_TYPES))

        self.chips_scored = 0
        self.hand, self._deck, self._discard_pile = _deal_from_deck(
            self._deck, self._discard_pile, self.hand_size, self.rng)
        self.phase = "SELECTING_HAND"

    def _end_blind(self):
        for card in self.hand:
            if card.get("enhancement") == "gold":
                self.money += 3

        self._apply_cavendish_check()
        self._apply_interest()

        all_cards = self._deck + self._discard_pile + self.hand
        card_map  = {}
        for c in all_cards:
            card_map[(c["rank"],c["suit"])] = c["enhancement"]
        for fc in self._full_deck:
            key = (fc["rank"],fc["suit"])
            if key in card_map:
                fc["enhancement"] = card_map[key]

        idx                = self.rng.permutation(len(self._full_deck))
        self._deck         = [self._full_deck[i] for i in idx]
        self._discard_pile = []
        self.hand          = []

        self.shop        = _generate_shop(self.rng, self.ante)
        self.reroll_cost = 5
        self.phase       = "SHOP"

    def step_blind_select(self, action):
        # Skip always masked — agent must beat blinds
        self._start_blind()
        return 0.0

    def step_selecting_hand(self, action):
        reward      = 0.0
        must_play_n = None
        if self.boss_debuff == "play_1":  must_play_n = 1
        elif self.boss_debuff == "play_5": must_play_n = 5

        # Check if a strong hand is available
        ranks_now = [c["rank"] for c in self.hand]
        suits_now = [c["suit"] for c in self.hand]
        sc_now    = Counter(suits_now)
        rv_now    = sorted(set(RANK_VALUES[r] for r in ranks_now))
        debuff    = DEBUFF_SUITS.get(self.boss_debuff)
        has_flush = any(count >= 5 for suit, count in sc_now.items() if suit != debuff)
        has_straight = (
            len(rv_now) >= 5 and
            any(rv_now[i+4]-rv_now[i]==4 for i in range(len(rv_now)-4))
        )
        # Force discard if no strong hand and can afford to fish
        if (not has_flush and not has_straight and
                self.discards_left > 0 and
                self.hands_left > 1 and
                action != 1):
            action = 1

        if action == 1 and self.discards_left > 0:
            self.discards_left    -= 1
            self.green_joker_mult  = max(0.0, self.green_joker_mult - 1)
            self._discard_pile    += self.hand
            self.hand, self._deck, self._discard_pile = _deal_from_deck(
                self._deck, self._discard_pile, self.hand_size, self.rng)
            return 0.0

        play_cards, hand_type = _select_play_cards(
            self.hand, action, self.boss_debuff,
            self.one_hand_type_allowed, must_play_n)

        if self.boss_debuff == "unique_hands" and hand_type in self.used_hand_types:
            for ht in HAND_TYPES:
                if ht not in self.used_hand_types:
                    hand_type = ht; break
        self.used_hand_types.add(hand_type)

        has_face = any(c["rank"] in FACE_CARDS for c in play_cards)
        if has_face: self.ride_bus_mult = 0.0
        else:        self.ride_bus_mult += 1

        held  = [c for c in self.hand if c not in play_cards]
        chips = _score_hand(
            play_cards, held, hand_type, self.hand_levels,
            self.joker_slots, self.money, self.boss_debuff,
            self.green_joker_mult, self.ride_bus_mult, self.hologram_xmult,
            self.hand_play_counts.get(hand_type, 0),
        )
        self.chips_scored   += chips
        self.hands_left     -= 1
        self.green_joker_mult += 1

        surviving = []
        for card in play_cards:
            if card.get("enhancement") == "glass" and self.rng.random() < 0.25:
                self._full_deck = [fc for fc in self._full_deck
                                   if not (fc["rank"]==card["rank"] and fc["suit"]==card["suit"])]
            else:
                surviving.append(card)

        self._discard_pile += surviving
        new_cards, self._deck, self._discard_pile = _deal_from_deck(
            self._deck, self._discard_pile, len(play_cards), self.rng)
        self.hand = [c for c in self.hand if c not in play_cards] + new_cards

        self._update_dominant_hand(hand_type)
        if self.hand_play_counts[hand_type] % 5 == 0:
            self.hand_levels[hand_type] = self.hand_levels.get(hand_type,1) + 1

        # Dense progress reward — scaled so agent feels every hand
        reward += 0.5 * (chips / max(self.chips_needed, 1))
        # Small survival bonus — encourages using all hands
        reward += 0.1

        if self.chips_scored >= self.chips_needed:
            # Primary signal — must dominate all other rewards
            reward += 5.0 + 1.0 * self.ante
            self._end_blind()
        elif self.hands_left <= 0:
            # Flat harsh penalty — dying always bad regardless of ante
            reward -= 8.0
            self.done = True

        return reward

    def step_shop(self, action):
        reward = 0.0

        if action == 0:
            self._advance_blind(); return reward

        if action == 6:
            if self.money >= self.reroll_cost:
                self.money       -= self.reroll_cost
                self.reroll_cost += 1
                self.shop         = _generate_shop(self.rng, self.ante)
            self._advance_blind(); return reward

        buy_idx = int(action - 1)
        if buy_idx < len(self.shop):
            item  = self.shop[buy_idx]
            cost  = item["cost"]
            if self.money >= cost:
                self.money -= cost
                ctype = item["set"]
                name  = item.get("name","")

                if ctype == "JOKER" and self.joker_count < self.joker_limit:
                    self.joker_slots.append({"name":name})
                    reward += 1.0 * self.ante * self._joker_synergy(name)

                elif ctype == "PLANET":
                    ht = self.dominant_hand
                    self.hand_levels[ht] = self.hand_levels.get(ht,1) + 1
                    if any(j["name"]=="Hologram" for j in self.joker_slots):
                        self.hologram_xmult += 0.25
                    reward += 0.5

                elif ctype == "TAROT":
                    reward += self._apply_tarot(name)

                elif ctype == "VOUCHER":
                    self.money += 3

                elif ctype == "PACK":
                    if any(j["name"]=="Hologram" for j in self.joker_slots):
                        self.hologram_xmult += 0.25

        self._advance_blind(); return reward

    def _joker_synergy(self, name):
        dom   = self.dominant_hand
        great = {"Cavendish","Blueprint","Hologram","Fibonacci","Abstract","Supernova"}
        if name in great: return 0.8
        if dom=="flush"    and name in {"Crafty_Joker","Droll_Joker"}: return 1.0
        if dom=="straight" and name in {"Crazy_Joker","Devious_Joker"}: return 1.0
        if dom in ("pair","two_pair") and name in {"Jolly_Joker","Sly_Joker"}: return 1.0
        return 0.3

    def _apply_tarot(self, tarot_name):
        if tarot_name == "The_Hanged_Man":
            n = int(self.rng.integers(1,3))
            self._deck.sort(key=lambda c: RANK_VALUES.get(c["rank"],0))
            removed      = self._deck[:n]
            self._deck   = self._deck[n:]
            for rc in removed:
                self._full_deck = [fc for fc in self._full_deck
                                   if not (fc["rank"]==rc["rank"] and fc["suit"]==rc["suit"])]
            return 0.05
        if not self.hand: return 0.0
        tidx   = _smart_tarot_target(tarot_name, self.hand, self.joker_slots)
        target = self.hand[tidx]
        enh_map = {"The_Devil":"gold","The_Chariot":"steel",
                   "Justice":"glass","The_Empress":"mult"}
        if tarot_name in enh_map:
            target["enhancement"] = enh_map[tarot_name]
            for fc in self._full_deck:
                if fc["rank"]==target["rank"] and fc["suit"]==target["suit"]:
                    fc["enhancement"] = target["enhancement"]; break
        return 0.05

    def encode(self):
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[0]  = min(self.ante,8)/8.0
        obs[1]  = min((self.ante-1)*3+self.blind_idx+1,24)/24.0
        obs[2]  = min(self.money,100)/100.0
        obs[3]  = self.hands_left/4.0
        obs[4]  = self.discards_left/4.0
        obs[5]  = min(self.chips_scored,300000)/300000.0
        obs[6]  = min(self.chips_needed,300000)/300000.0
        obs[7]  = self.joker_count/5.0
        obs[8]  = self.joker_limit/5.0
        obs[9]  = min(self.reroll_cost,10)/10.0
        for i,c in enumerate(self.hand[:8]):
            obs[10+i] = RANK_VALUES.get(c["rank"],0)/12.0
            obs[18+i] = SUIT_VALUES.get(c["suit"],0)/3.0
        for i,c in enumerate(self.shop[:5]):
            obs[26+i] = min(c["cost"],20)/20.0
            obs[31+i] = 1.0 if c["set"]=="JOKER" else 0.0
        obs[36] = {"small":0.0,"big":0.5,"boss":1.0}.get(self.blind_type,0.0)
        obs[37] = {"SELECTING_HAND":0.0,"SHOP":0.5,"BLIND_SELECT":1.0}.get(self.phase,0.0)
        vis = self._deck + self.hand
        rc  = Counter(c["rank"] for c in vis)
        for i,r in enumerate(RANKS):  obs[38+i] = min(rc.get(r,0),4)/4.0
        sc  = Counter(c["suit"] for c in vis)
        for i,s in enumerate(SUITS):  obs[51+i] = min(sc.get(s,0),13)/13.0
        di  = HAND_TYPE_IDX.get(self.dominant_hand,8)
        obs[55+di] = 1.0
        obs[64] = min(self.hand_levels.get(self.dominant_hand,1),10)/10.0
        obs[65] = len(self._full_deck)/52.0
        ds  = DEBUFF_SUITS.get(self.boss_debuff)
        obs[66] = 1.0 if ds else 0.0
        obs[67] = SUIT_VALUES.get(ds,0)/3.0 if ds else 0.0
        if   self.boss_debuff=="play_1": obs[68] = 1.0/5.0
        elif self.boss_debuff=="play_5": obs[68] = 5.0/5.0
        obs[69] = 1.0 if self.boss_debuff in ("hand_size_minus1","draw_1") else 0.0
        obs[70] = min(self.green_joker_mult,20)/20.0
        obs[71] = min(self.ride_bus_mult,   20)/20.0
        return obs

    def action_masks(self):
        mask = np.ones(N_ACTIONS, dtype=bool)
        if self.phase == "BLIND_SELECT":
            mask[1]  = False   # skip always disabled
            mask[2:] = False
        elif self.phase == "SELECTING_HAND":
            if self.discards_left <= 0: mask[1] = False
            mask[5:] = False
        elif self.phase == "SHOP":
            for i in range(1,6):
                bi = i-1
                if bi >= len(self.shop):                    mask[i] = False
                elif self.shop[bi]["cost"] > self.money:    mask[i] = False
                elif (self.shop[bi]["set"]=="JOKER" and
                      self.joker_count >= self.joker_limit): mask[i] = False
            if self.money < self.reroll_cost: mask[6] = False
        return mask


class BalatroMockEnv(gym.Env):
    """
    High-fidelity mock Balatro env with curriculum learning.
    OBS_DIM=72. action_masks() for MaskablePPO.
    max_ante controls curriculum stage — set via env.set_max_ante(n).
    """
    metadata = {"render_modes": []}

    def __init__(self, max_ante=8):
        super().__init__()
        self.observation_space = spaces.Box(low=0.0,high=1.0,shape=(OBS_DIM,),dtype=np.float32)
        self.action_space      = spaces.Discrete(N_ACTIONS)
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
        if self._game is None: return np.ones(N_ACTIONS, dtype=bool)
        return self._game.action_masks()

    def render(self): pass
    def close(self):  pass