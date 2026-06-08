"""
balatro_mock_env.py
Fast mock Gymnasium environment for PPO training.

Simulates Balatro game logic in pure Python — no HTTP calls, no live game.
Trains at ~1000+ it/s vs ~1 it/s on the real env.

Workflow:
    1. Train on mock env (fast): python rl_bot.py --train --mock --timesteps 500000
    2. Evaluate on real env:     python rl_bot.py --run

The mock simulates:
    - Blind chip targets scaling with ante
    - Hand scoring (approximate chips per hand type)
    - Shop economy (joker buying, interest)
    - Blind skip/select decision
    - Game over / win conditions

Not simulated (approximated):
    - Specific joker effects
    - Boss blind special effects
    - Exact card draw probabilities
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import Counter

from balatro_env import OBS_DIM, N_ACTIONS, RANK_ORDER, RANK_VALUES, SUIT_VALUES

# ---------------------------------------------------------------------------
# Game constants
# ---------------------------------------------------------------------------

# Chip targets per blind per ante (approximate, scales like real game)
BLIND_CHIPS = {
    1: {"small": 300,  "big": 450,  "boss": 600},
    2: {"small": 800,  "big": 1200, "boss": 1600},
    3: {"small": 2000, "big": 3000, "boss": 4000},
    4: {"small": 5000, "big": 7500, "boss": 10000},
    5: {"small": 11000,"big": 16500,"boss": 22000},
    6: {"small": 20000,"big": 30000,"boss": 40000},
    7: {"small": 35000,"big": 52500,"boss": 70000},
    8: {"small": 60000,"big": 90000,"boss": 120000},
}

# Base chips + mult per hand type (approximate level-1 values)
HAND_SCORES = {
    "royal_flush":    (100, 8),
    "straight_flush": (100, 8),
    "five_of_a_kind": (120, 12),
    "flush_house":    (140, 14),
    "flush_five":     (160, 16),
    "four_of_a_kind": (60,  7),
    "full_house":     (40,  4),
    "flush":          (35,  4),
    "straight":       (30,  4),
    "three_of_a_kind":(30,  3),
    "two_pair":       (20,  2),
    "pair":           (10,  2),
    "high_card":      (5,   1),
}

RANK_CHIP_VALUES = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
    "7": 7, "8": 8, "9": 9, "T": 10,
    "J": 10, "Q": 10, "K": 10, "A": 11,
}

SUITS = ["S", "H", "D", "C"]
RANKS = list(RANK_ORDER)


# ---------------------------------------------------------------------------
# Mock card / hand logic
# ---------------------------------------------------------------------------

def _deal_hand(rng: np.random.Generator, n: int = 8) -> list[tuple[str, str]]:
    """Return n (rank, suit) tuples sampled without replacement."""
    deck = [(r, s) for r in RANKS for s in SUITS]
    indices = rng.choice(len(deck), size=min(n, len(deck)), replace=False)
    return [deck[i] for i in indices]


def _score_hand(cards: list[tuple[str, str]]) -> tuple[str, int]:
    """Detect best hand type and compute approximate chip score."""
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
        unique_vals[-1] - unique_vals[-5] == 4
    )

    if is_flush and is_straight:
        hand_type = "straight_flush"
    elif counts[0] == 4:
        hand_type = "four_of_a_kind"
    elif counts[0] == 3 and counts[1] == 2:
        hand_type = "full_house"
    elif is_flush:
        hand_type = "flush"
    elif is_straight:
        hand_type = "straight"
    elif counts[0] == 3:
        hand_type = "three_of_a_kind"
    elif counts[0] == 2 and counts[1] == 2:
        hand_type = "two_pair"
    elif counts[0] == 2:
        hand_type = "pair"
    else:
        hand_type = "high_card"

    base_chips, base_mult = HAND_SCORES[hand_type]
    card_chips = sum(RANK_CHIP_VALUES.get(r, 0) for r in ranks[:5])
    score = (base_chips + card_chips) * base_mult

    return hand_type, score


def _best_play(cards: list[tuple[str, str]], action: int) -> tuple[str, int]:
    """
    Simulate playing action on the given hand.
    Returns (hand_type, chips_scored).
    """
    if not cards:
        return "high_card", 0

    ranks = [r for r, s in cards]
    suits = [s for r, s in cards]
    rank_counts = Counter(ranks)
    suit_counts = Counter(suits)

    if action == 2:  # try flush
        best_suit = suit_counts.most_common(1)[0][0]
        flush_cards = [(r, s) for r, s in cards if s == best_suit]
        if len(flush_cards) >= 5:
            return _score_hand(flush_cards[:5])

    if action == 3:  # try straight
        rank_vals = sorted(set(RANK_VALUES[r] for r in ranks))
        for i in range(len(rank_vals) - 4):
            w = rank_vals[i:i+5]
            if w[-1] - w[0] == 4:
                straight_ranks = {RANK_ORDER[v] for v in w}
                straight_cards = [(r, s) for r, s in cards if r in straight_ranks][:5]
                return _score_hand(straight_cards)

    # Default: best pair hand
    sorted_by_count = sorted(rank_counts.items(), key=lambda x: (x[1], RANK_VALUES[x[0]]), reverse=True)
    top_rank = sorted_by_count[0][0]
    top_count = sorted_by_count[0][1]
    best_cards = [(r, s) for r, s in cards if r == top_rank][:top_count]

    if top_count == 2 and len(sorted_by_count) > 1 and sorted_by_count[1][1] == 2:
        r2 = sorted_by_count[1][0]
        best_cards += [(r, s) for r, s in cards if r == r2][:2]

    if top_count == 3 and len(sorted_by_count) > 1 and sorted_by_count[1][1] >= 2:
        r2 = sorted_by_count[1][0]
        best_cards += [(r, s) for r, s in cards if r == r2][:2]

    remaining = [(r, s) for r, s in cards if (r, s) not in best_cards]
    remaining.sort(key=lambda x: RANK_VALUES[x[0]], reverse=True)
    play_cards = (best_cards + remaining)[:5]

    return _score_hand(play_cards)


# ---------------------------------------------------------------------------
# Mock shop
# ---------------------------------------------------------------------------

def _generate_shop(rng: np.random.Generator, ante: int) -> list[dict]:
    """Generate a simple shop with 2 cards."""
    base_cost = 4 + ante
    shop = []
    for _ in range(2):
        is_joker = rng.random() < 0.5
        cost = base_cost + rng.integers(0, 3)
        shop.append({
            "set": "JOKER" if is_joker else "TAROT",
            "cost": cost,
        })
    return shop


# ---------------------------------------------------------------------------
# Mock game state
# ---------------------------------------------------------------------------

class MockGameState:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.ante = 1
        self.round = 1          # 1=small, 2=big, 3=boss within ante
        self.money = 4
        self.joker_count = 0
        self.joker_limit = 5
        self.joker_mult_bonus = 0  # flat mult from jokers
        self.hands_left = 4
        self.discards_left = 4    # Red deck: +1 discard
        self.chips_scored = 0
        self.phase = "BLIND_SELECT"  # BLIND_SELECT -> SELECTING_HAND -> SHOP
        self.blind_type = "small"
        self.shop = _generate_shop(rng, self.ante)
        self.hand = _deal_hand(rng)
        self.done = False
        self.won = False
        self.reroll_cost = 5

    @property
    def chips_needed(self) -> int:
        ante_blinds = BLIND_CHIPS.get(min(self.ante, 8), BLIND_CHIPS[8])
        blind_key = ["small", "big", "boss"][min(self.round - 1, 2)]
        return ante_blinds[blind_key]

    def _end_round(self):
        """Cash out, add interest, move to next blind."""
        # Interest
        interest = min(self.money // 5, 5)
        self.money += interest + 1  # $1 base per round

        # Advance blind
        if self.round == 3:
            self.ante += 1
            self.round = 1
            if self.ante > 8:
                self.won = True
                self.done = True
                return
        else:
            self.round += 1

        self.blind_type = ["small", "big", "boss"][self.round - 1]
        self.chips_scored = 0
        self.phase = "BLIND_SELECT"

    def _start_blind(self):
        self.hands_left = 4
        self.discards_left = 4
        self.chips_scored = 0
        self.hand = _deal_hand(self.rng)
        self.phase = "SELECTING_HAND"

    def _end_blind(self):
        self.shop = _generate_shop(self.rng, self.ante)
        self.phase = "SHOP"

    def step_blind_select(self, action: int) -> float:
        """action 0=select, 1=skip (small/big only)"""
        reward = 0.0
        if action == 1 and self.blind_type != "boss":
            # Skip: get $2 tag reward equivalent
            self.money += 2
            reward += 0.2
            self._end_round()  # advance without playing
        else:
            self._start_blind()
        return reward

    def step_selecting_hand(self, action: int) -> float:
        """action 0-4 map to play strategies, 1=discard_worst"""
        reward = 0.0

        if action == 1 and self.discards_left > 0:
            # Discard: redraw hand
            self.discards_left -= 1
            self.hand = _deal_hand(self.rng)
            return 0.0

        # Play
        hand_type, chips = _best_play(self.hand, action)
        # Apply joker bonus
        chips = int(chips * (1 + self.joker_mult_bonus * 0.1))
        self.chips_scored += chips
        self.hands_left -= 1
        self.hand = _deal_hand(self.rng)

        if self.chips_scored >= self.chips_needed:
            reward += 0.5  # round cleared
            self._end_blind()
        elif self.hands_left <= 0:
            # Failed blind
            self.done = True
            reward -= 0.5
        
        return reward

    def step_shop(self, action: int) -> float:
        """action 0=end, 1-5=buy card idx, 6=reroll"""
        reward = 0.0

        if action == 0:  # end shop
            self._end_round()
            return reward

        if action == 6:  # reroll
            if self.money >= self.reroll_cost:
                self.money -= self.reroll_cost
                self.reroll_cost += 1
                self.shop = _generate_shop(self.rng, self.ante)
            self._end_round()
            return reward

        buy_idx = action - 1
        if buy_idx < len(self.shop):
            card = self.shop[buy_idx]
            cost = card["cost"]
            if self.money >= cost and self.joker_count < self.joker_limit:
                self.money -= cost
                if card["set"] == "JOKER":
                    self.joker_count += 1
                    self.joker_mult_bonus += 1
                    reward += 0.1

        self._end_round()
        return reward

    def encode(self) -> np.ndarray:
        """Encode game state to observation vector (same layout as balatro_env.py)."""
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        obs[0] = min(self.ante, 8) / 8.0
        obs[1] = min((self.ante - 1) * 3 + self.round, 24) / 24.0
        obs[2] = min(self.money, 100) / 100.0
        obs[3] = self.hands_left / 4.0
        obs[4] = self.discards_left / 4.0
        obs[5] = min(self.chips_scored, 10000) / 10000.0
        obs[6] = min(self.chips_needed, 10000) / 10000.0
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
# Mock Gymnasium Env
# ---------------------------------------------------------------------------

class BalatroMockEnv(gym.Env):
    """
    Fast mock Balatro environment for PPO training.
    Same observation/action space as BalatroEnv — policy transfers directly.
    Runs at ~1000x speed vs real environment.
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

        # Ante progression bonus
        if g.won:
            reward += 10.0

        obs = g.encode()
        done = g.done or g.won
        info = {
            "ante": g.ante,
            "round": g.round,
            "won": g.won,
        }

        return obs, reward, done, False, info

    def render(self):
        pass

    def close(self):
        pass