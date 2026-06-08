"""
balatro_mock_env.py
High-fidelity mock Gymnasium environment for PPO training.

Simulates Balatro game logic in pure Python — no HTTP calls.
Trains at ~1000+ it/s vs ~1 it/s on the real env.

Fidelity improvements over basic mock:
    - Hand level scaling (chips/mult increase as you play the same hand)
    - Joker value tiers (approximates S+/S/A tier effects)
    - Realistic shop (2-4 cards, vouchers, packs, correct cost scaling)
    - Boss blind debuffs (hand size reduction, forced discards)
    - Interest mechanic (exact: $1 per $5, max $5)
    - Blind skip tag values (weighted by tag type distribution)
    - Red deck bonus (+1 discard)
    - Hand size 8 cards
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import Counter

from balatro_env import OBS_DIM, N_ACTIONS, RANK_ORDER, RANK_VALUES, SUIT_VALUES

# ---------------------------------------------------------------------------
# Hand level scaling
# Each time you play a hand type, its level increases.
# Level N: base_chips += 50*(N-1), base_mult += 2*(N-1) (approximate)
# ---------------------------------------------------------------------------

BASE_HAND_SCORES = {
    "straight_flush":  (100, 8),
    "four_of_a_kind":  (60,  7),
    "full_house":      (40,  4),
    "flush":           (35,  4),
    "straight":        (30,  4),
    "three_of_a_kind": (30,  3),
    "two_pair":        (20,  2),
    "pair":            (10,  2),
    "high_card":       (5,   1),
}

HAND_LEVEL_CHIPS_BONUS = 50   # +chips per level above 1
HAND_LEVEL_MULT_BONUS  = 2    # +mult per level above 1

RANK_CHIP_VALUES = {
    "2": 2,  "3": 3,  "4": 4,  "5": 5,  "6": 6,
    "7": 7,  "8": 8,  "9": 9,  "T": 10,
    "J": 10, "Q": 10, "K": 10, "A": 11,
}

SUITS = ["S", "H", "D", "C"]
RANKS = list(RANK_ORDER)

# ---------------------------------------------------------------------------
# Blind chip targets (matches real game)
# ---------------------------------------------------------------------------

BLIND_CHIPS = {
    1: {"small": 300,   "big": 450,   "boss": 600},
    2: {"small": 800,   "big": 1200,  "boss": 1600},
    3: {"small": 2000,  "big": 3000,  "boss": 4000},
    4: {"small": 5000,  "big": 7500,  "boss": 10000},
    5: {"small": 11000, "big": 16500, "boss": 22000},
    6: {"small": 20000, "big": 30000, "boss": 40000},
    7: {"small": 35000, "big": 52500, "boss": 70000},
    8: {"small": 60000, "big": 90000, "boss": 120000},
}

# ---------------------------------------------------------------------------
# Joker tiers — approximate mult bonus per joker slot
# Weighted average of tier effects, not exact but directionally correct
# ---------------------------------------------------------------------------

JOKER_TIER_WEIGHTS = [0.05, 0.15, 0.35, 0.45]  # S+, S, A, common
JOKER_TIER_XMULT   = [3.0,  2.0,  1.5,  1.0]   # approximate Xmult per tier
JOKER_TIER_FLAT    = [0,    15,   8,    4]       # approximate flat mult per tier

# ---------------------------------------------------------------------------
# Boss blind debuff types (approximate distribution)
# ---------------------------------------------------------------------------

BOSS_DEBUFFS = ["hand_size_minus1", "forced_discard", "none", "none", "none"]

# ---------------------------------------------------------------------------
# Tag skip values (approximate $ equivalent of common tags)
# ---------------------------------------------------------------------------

TAG_VALUES = [2, 4, 6, 8, 3, 5]  # rough $ value distribution of tags

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deal_hand(rng: np.random.Generator, n: int = 8) -> list[tuple[str, str]]:
    deck = [(r, s) for r in RANKS for s in SUITS]
    indices = rng.choice(len(deck), size=min(n, len(deck)), replace=False)
    return [deck[i] for i in indices]


def _detect_hand_type(cards: list[tuple[str, str]]) -> str:
    ranks = [r for r, s in cards]
    suits = [s for r, s in cards]
    rank_counts = Counter(ranks)
    suit_counts = Counter(suits)
    rank_vals = sorted([RANK_VALUES[r] for r in ranks])
    counts = sorted(rank_counts.values(), reverse=True)

    is_flush = max(suit_counts.values()) >= 5
    unique_vals = sorted(set(rank_vals))
    is_straight = (
        len(unique_vals) >= 5 and
        any(
            unique_vals[i+4] - unique_vals[i] == 4
            for i in range(len(unique_vals) - 4)
        )
    )

    if is_flush and is_straight:
        return "straight_flush"
    if counts[0] == 4:
        return "four_of_a_kind"
    if counts[0] == 3 and len(counts) > 1 and counts[1] == 2:
        return "full_house"
    if is_flush:
        return "flush"
    if is_straight:
        return "straight"
    if counts[0] == 3:
        return "three_of_a_kind"
    if counts[0] == 2 and len(counts) > 1 and counts[1] == 2:
        return "two_pair"
    if counts[0] == 2:
        return "pair"
    return "high_card"


def _score_hand(
    cards: list[tuple[str, str]],
    hand_type: str,
    hand_levels: dict,
    joker_flat_mult: float,
    joker_xmult: float,
) -> int:
    level = hand_levels.get(hand_type, 1)
    base_chips, base_mult = BASE_HAND_SCORES[hand_type]

    # Level scaling
    chips = base_chips + HAND_LEVEL_CHIPS_BONUS * (level - 1)
    mult  = base_mult  + HAND_LEVEL_MULT_BONUS  * (level - 1)

    # Card chips (top 5 cards)
    card_chips = sum(RANK_CHIP_VALUES.get(r, 0) for r, s in cards[:5])
    chips += card_chips

    # Joker bonuses
    mult += joker_flat_mult
    score = int(chips * mult * joker_xmult)
    return score


def _select_play_cards(
    cards: list[tuple[str, str]],
    action: int
) -> tuple[list[tuple[str, str]], str]:
    """Select cards to play based on action, return (play_cards, hand_type)."""
    ranks = [r for r, s in cards]
    suits = [s for r, s in cards]
    rank_counts = Counter(ranks)
    suit_counts = Counter(suits)

    # Action 2: try flush
    if action == 2:
        best_suit = suit_counts.most_common(1)[0][0]
        flush_cards = [(r, s) for r, s in cards if s == best_suit]
        if len(flush_cards) >= 5:
            play = flush_cards[:5]
            return play, _detect_hand_type(play)

    # Action 3: try straight
    if action == 3:
        rank_vals = sorted(set(RANK_VALUES[r] for r in ranks))
        for i in range(len(rank_vals) - 4):
            w = rank_vals[i:i+5]
            if w[-1] - w[0] == 4 and len(w) == 5:
                straight_ranks = {RANK_ORDER[v] for v in w}
                play = [(r, s) for r, s in cards if r in straight_ranks][:5]
                return play, "straight"

    # Default: best pair-based hand
    sorted_rc = sorted(rank_counts.items(), key=lambda x: (x[1], RANK_VALUES[x[0]]), reverse=True)
    top_rank, top_count = sorted_rc[0]
    play = [(r, s) for r, s in cards if r == top_rank][:top_count]

    # Full house
    if top_count == 3 and len(sorted_rc) > 1 and sorted_rc[1][1] >= 2:
        r2 = sorted_rc[1][0]
        play += [(r, s) for r, s in cards if r == r2][:2]

    # Two pair
    elif top_count == 2 and len(sorted_rc) > 1 and sorted_rc[1][1] == 2:
        r2 = sorted_rc[1][0]
        play += [(r, s) for r, s in cards if r == r2][:2]

    # Fill to 5
    remaining = [(r, s) for r, s in cards if (r, s) not in play]
    remaining.sort(key=lambda x: RANK_VALUES[x[0]], reverse=True)
    play = (play + remaining)[:5]

    return play, _detect_hand_type(play)


def _generate_shop(rng: np.random.Generator, ante: int) -> list[dict]:
    """Generate a realistic shop with 2-4 items."""
    n_cards = rng.integers(2, 5)
    shop = []
    base_cost = 4 + (ante - 1)

    for _ in range(n_cards):
        roll = rng.random()
        if roll < 0.45:
            card_type = "JOKER"
            # Tier-weighted cost
            tier = rng.choice(4, p=JOKER_TIER_WEIGHTS)
            cost = base_cost + tier * 2 + int(rng.integers(0, 3))
            xmult = JOKER_TIER_XMULT[tier]
            flat  = JOKER_TIER_FLAT[tier]
        elif roll < 0.65:
            card_type = "PLANET"
            cost = base_cost - 1
            xmult = 1.0
            flat  = 0
        elif roll < 0.80:
            card_type = "TAROT"
            cost = base_cost
            xmult = 1.0
            flat  = 0
        elif roll < 0.90:
            card_type = "VOUCHER"
            cost = base_cost + 3
            xmult = 1.0
            flat  = 3  # vouchers give misc economy bonus
        else:
            card_type = "PACK"
            cost = base_cost + 1
            xmult = 1.0
            flat  = 0

        shop.append({
            "set":   card_type,
            "cost":  max(2, int(cost)),
            "xmult": xmult,
            "flat":  flat,
        })

    return shop


# ---------------------------------------------------------------------------
# Mock game state
# ---------------------------------------------------------------------------

class MockGameState:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.ante = 1
        self.blind_idx = 0          # 0=small, 1=big, 2=boss
        self.money = 4
        self.joker_count = 0
        self.joker_limit = 5
        self.joker_flat_mult = 0.0  # total flat mult from jokers
        self.joker_xmult = 1.0      # total Xmult from jokers
        self.hand_levels = {ht: 1 for ht in BASE_HAND_SCORES}
        self.hand_play_counts = {ht: 0 for ht in BASE_HAND_SCORES}
        self.hands_left = 4
        self.discards_left = 4      # Red deck: base 4
        self.hand_size = 8
        self.chips_scored = 0
        self.phase = "BLIND_SELECT"
        self.boss_debuff = "none"
        self.shop = _generate_shop(rng, self.ante)
        self.hand = _deal_hand(rng, self.hand_size)
        self.reroll_cost = 5
        self.done = False
        self.won = False

    @property
    def blind_type(self) -> str:
        return ["small", "big", "boss"][self.blind_idx]

    @property
    def chips_needed(self) -> int:
        ante_blinds = BLIND_CHIPS.get(min(self.ante, 8), BLIND_CHIPS[8])
        return ante_blinds[self.blind_type]

    def _apply_interest(self):
        interest = min(self.money // 5, 5)
        self.money += interest + 1  # +$1 base per round survived

    def _advance_blind(self):
        """Move to next blind or next ante."""
        if self.blind_idx == 2:
            self.ante += 1
            self.blind_idx = 0
            if self.ante > 8:
                self.won = True
                self.done = True
                return
        else:
            self.blind_idx += 1
        self.chips_scored = 0
        self.phase = "BLIND_SELECT"

    def _start_blind(self):
        self.hands_left = 4
        self.discards_left = 4  # Red deck

        # Apply boss debuff
        if self.blind_type == "boss":
            self.boss_debuff = self.rng.choice(BOSS_DEBUFFS)
            if self.boss_debuff == "hand_size_minus1":
                self.hand_size = max(1, 8 - 1)
            elif self.boss_debuff == "forced_discard":
                self.discards_left = max(0, self.discards_left - 1)
        else:
            self.boss_debuff = "none"
            self.hand_size = 8

        self.chips_scored = 0
        self.hand = _deal_hand(self.rng, self.hand_size)
        self.phase = "SELECTING_HAND"

    def _end_blind(self):
        self._apply_interest()
        self.shop = _generate_shop(self.rng, self.ante)
        self.reroll_cost = 5
        self.phase = "SHOP"

    # ------------------------------------------------------------------
    # Phase step functions
    # ------------------------------------------------------------------

    def step_blind_select(self, action: int) -> float:
        reward = 0.0
        if action == 1 and self.blind_type != "boss":
            # Skip — get tag value
            tag_val = int(self.rng.choice(TAG_VALUES))
            self.money += tag_val
            reward += 0.3
            self._apply_interest()
            self._advance_blind()
        else:
            self._start_blind()
        return reward

    def step_selecting_hand(self, action: int) -> float:
        reward = 0.0

        # Action 1: discard worst cards
        if action == 1 and self.discards_left > 0:
            self.discards_left -= 1
            self.hand = _deal_hand(self.rng, self.hand_size)
            return 0.0

        # Play hand
        play_cards, hand_type = _select_play_cards(self.hand, action)
        chips = _score_hand(
            play_cards, hand_type,
            self.hand_levels,
            self.joker_flat_mult,
            self.joker_xmult,
        )
        self.chips_scored += chips
        self.hands_left -= 1

        # Level up hand type (every 5 plays)
        self.hand_play_counts[hand_type] = self.hand_play_counts.get(hand_type, 0) + 1
        if self.hand_play_counts[hand_type] % 5 == 0:
            self.hand_levels[hand_type] = self.hand_levels.get(hand_type, 1) + 1
            reward += 0.05  # small reward for leveling up a hand

        # Redraw
        self.hand = _deal_hand(self.rng, self.hand_size)

        if self.chips_scored >= self.chips_needed:
            reward += 0.5
            self._end_blind()
        elif self.hands_left <= 0:
            self.done = True
            reward -= 0.5

        return reward

    def step_shop(self, action: int) -> float:
        reward = 0.0

        if action == 0:  # end shop
            self._advance_blind()
            return reward

        if action == 6:  # reroll
            if self.money >= self.reroll_cost:
                self.money -= self.reroll_cost
                self.reroll_cost += 1
                self.shop = _generate_shop(self.rng, self.ante)
            self._advance_blind()
            return reward

        buy_idx = int(action - 1)
        if buy_idx < len(self.shop):
            card = self.shop[buy_idx]
            cost = card["cost"]
            if self.money >= cost:
                self.money -= cost
                card_type = card["set"]

                if card_type == "JOKER" and self.joker_count < self.joker_limit:
                    self.joker_count += 1
                    self.joker_flat_mult += card["flat"]
                    # Combine Xmult multiplicatively
                    self.joker_xmult *= card["xmult"]
                    reward += 0.15

                elif card_type == "PLANET":
                    # Planet upgrades a random hand level
                    ht = str(self.rng.choice(list(BASE_HAND_SCORES.keys())))
                    self.hand_levels[ht] = self.hand_levels.get(ht, 1) + 1
                    reward += 0.05

                elif card_type == "VOUCHER":
                    # Voucher gives economy bonus
                    self.money += card["flat"]
                    reward += 0.05

        self._advance_blind()
        return reward

    def encode(self) -> np.ndarray:
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        obs[0] = min(self.ante, 8) / 8.0
        obs[1] = min((self.ante - 1) * 3 + self.blind_idx + 1, 24) / 24.0
        obs[2] = min(self.money, 100) / 100.0
        obs[3] = self.hands_left / 4.0
        obs[4] = self.discards_left / 4.0
        obs[5] = min(self.chips_scored, 300000) / 300000.0
        obs[6] = min(self.chips_needed, 300000) / 300000.0
        obs[7] = self.joker_count / 5.0
        obs[8] = self.joker_limit / 5.0
        obs[9] = min(self.reroll_cost, 10) / 10.0

        for i, (rank, suit) in enumerate(self.hand[:8]):
            obs[10 + i] = RANK_VALUES.get(rank, 0) / 12.0
            obs[18 + i] = SUIT_VALUES.get(suit, 0) / 3.0

        for i, card in enumerate(self.shop[:5]):
            obs[26 + i] = min(card["cost"], 20) / 20.0
            obs[31 + i] = 1.0 if card["set"] == "JOKER" else 0.0

        blind_type_map = {"small": 0.0, "big": 0.5, "boss": 1.0}
        obs[36] = blind_type_map.get(self.blind_type, 0.0)

        phase_map = {"SELECTING_HAND": 0.0, "SHOP": 0.5, "BLIND_SELECT": 1.0}
        obs[37] = phase_map.get(self.phase, 0.0)

        return obs


# ---------------------------------------------------------------------------
# Gymnasium env
# ---------------------------------------------------------------------------

class BalatroMockEnv(gym.Env):
    """
    High-fidelity mock Balatro environment for PPO training.
    Same obs/action space as BalatroEnv — policy transfers directly.

    Fidelity:
        - Hand level scaling (chips/mult grow as hand type is played)
        - Tiered joker effects (S+/S/A/common with Xmult + flat mult)
        - Realistic shop composition (jokers, planets, tarots, vouchers, packs)
        - Boss blind debuffs (hand size, forced discards)
        - Exact interest mechanic ($1 per $5, max $5)
        - Blind skip tag values (weighted distribution)
        - Red deck (+1 discard base)
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)
        self._rng = np.random.default_rng()
        self._game = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._game = MockGameState(self._rng)
        return self._game.encode(), {}

    def step(self, action: int):
        g = self._game
        action = int(action)

        if g.phase == "BLIND_SELECT":
            reward = g.step_blind_select(action)
        elif g.phase == "SELECTING_HAND":
            reward = g.step_selecting_hand(action)
        elif g.phase == "SHOP":
            reward = g.step_shop(action)
        else:
            reward = 0.0

        if g.won:
            reward += 10.0

        obs = g.encode()
        done = g.done or g.won
        info = {
            "ante":  g.ante,
            "round": g.blind_idx + 1,
            "won":   g.won,
        }

        return obs, reward, done, False, info

    def render(self):
        pass

    def close(self):
        pass