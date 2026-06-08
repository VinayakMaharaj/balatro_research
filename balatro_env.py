"""
balatro_env.py
Gymnasium environment wrapping the Balatrobot HTTP JSON-RPC API.

Observation space: flat float32 vector encoding the game state.
Action space: Discrete — mapped differently per game state phase.

Used by rl_bot.py (PPO via Stable-Baselines3).

Install deps:
    pip install gymnasium stable-baselines3 --break-system-packages
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from balatro_client import BalatroClient, BalatroError

# ---------------------------------------------------------------------------
# Observation / action constants
# ---------------------------------------------------------------------------

# Observation vector layout (all normalized to [0, 1] where possible):
#   [0]  ante_num           / 8
#   [1]  round_num          / 24
#   [2]  money              / 100
#   [3]  hands_left         / 4
#   [4]  discards_left      / 4
#   [5]  chips_scored       / 10000
#   [6]  chips_needed       / 10000
#   [7]  joker_count        / 5
#   [8]  joker_limit        / 5
#   [9]  reroll_cost        / 10
#   [10-17] hand cards: 8 x rank_value/13  (0 if slot empty)
#   [18-25] hand cards: 8 x suit_onehot collapsed to suit_idx/4
#   [26-30] shop card costs (5 slots): cost/20  (0 if empty)
#   [31-35] shop card is_joker flag (5 slots): 0 or 1
#   [36]  blind_type: small=0, big=0.5, boss=1
#   [37]  state_phase: selecting_hand=0, shop=0.5, blind_select=1

OBS_DIM = 38

# Action spaces per phase
# SELECTING_HAND: 0=play_best, 1=discard_worst, 2=play_flush, 3=play_straight,
#                 4=play_pair_hand
N_HAND_ACTIONS = 5

# SHOP: 0=end_shop, 1=buy_card_0, 2=buy_card_1, 3=buy_card_2,
#       4=buy_card_3, 5=buy_card_4, 6=reroll
N_SHOP_ACTIONS = 7

# BLIND_SELECT: 0=select, 1=skip
N_BLIND_ACTIONS = 2

# Combined flat action space — agent outputs one of N_ACTIONS per step,
# and the env interprets it based on current phase.
# We use the largest phase size so the space is consistent.
N_ACTIONS = max(N_HAND_ACTIONS, N_SHOP_ACTIONS, N_BLIND_ACTIONS)

RANK_ORDER = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
RANK_VALUES = {r: i for i, r in enumerate(RANK_ORDER)}
SUIT_VALUES = {"S": 0, "H": 1, "D": 2, "C": 3}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rank_val(rank: str) -> float:
    return RANK_VALUES.get(rank, 0) / 12.0


def _suit_val(suit: str) -> float:
    return SUIT_VALUES.get(suit, 0) / 3.0


def _encode_obs(state: dict) -> np.ndarray:
    obs = np.zeros(OBS_DIM, dtype=np.float32)

    obs[0] = min(state.get("ante_num", 0), 8) / 8.0
    obs[1] = min(state.get("round_num", 0), 24) / 24.0
    obs[2] = min(state.get("money", 0), 100) / 100.0

    round_info = state.get("round", {})
    obs[3] = min(round_info.get("hands_left", 0), 4) / 4.0
    obs[4] = min(round_info.get("discards_left", 0), 4) / 4.0
    obs[5] = min(round_info.get("chips", 0), 10000) / 10000.0

    # chips needed from current blind
    chips_needed = 0
    blinds = state.get("blinds", {})
    for bk in ["small", "big", "boss"]:
        b = blinds.get(bk, {})
        if b.get("status") in ("SELECT", "CURRENT"):
            chips_needed = b.get("score", 0)
            break
    obs[6] = min(chips_needed, 10000) / 10000.0

    jokers = state.get("jokers", {})
    obs[7] = jokers.get("count", 0) / 5.0
    obs[8] = jokers.get("limit", 5) / 5.0

    obs[9] = min(round_info.get("reroll_cost", 5), 10) / 10.0

    # Hand cards (up to 8)
    hand_cards = state.get("hand", {}).get("cards", [])
    for i, card in enumerate(hand_cards[:8]):
        v = card.get("value", {})
        obs[10 + i] = _rank_val(v.get("rank", "2"))
        obs[18 + i] = _suit_val(v.get("suit", "S"))

    # Shop cards (up to 5)
    shop_cards = state.get("shop", {}).get("cards", [])
    for i, card in enumerate(shop_cards[:5]):
        cost = card.get("cost", {}).get("buy", 0)
        obs[26 + i] = min(cost, 20) / 20.0
        obs[31 + i] = 1.0 if card.get("set", "") == "JOKER" else 0.0

    # Blind type
    blind_type_map = {"small": 0.0, "big": 0.5, "boss": 1.0}
    for bk in ["small", "big", "boss"]:
        b = blinds.get(bk, {})
        if b.get("status") in ("SELECT", "CURRENT"):
            obs[36] = blind_type_map[bk]
            break

    # State phase
    state_name = state.get("state", "")
    phase_map = {"SELECTING_HAND": 0.0, "SHOP": 0.5, "BLIND_SELECT": 1.0}
    obs[37] = phase_map.get(state_name, 0.0)

    return obs


def _compute_reward(prev_state: dict, curr_state: dict, done: bool, won: bool) -> float:
    """
    Shaped reward function:
    - +1.0 per ante cleared (round_num increased by 3)
    - +0.5 per round survived
    - +0.1 per joker acquired
    - +10.0 for winning the run
    - -1.0 for game over (implicit — episode just ends)
    """
    reward = 0.0

    prev_round = prev_state.get("round_num", 0)
    curr_round = curr_state.get("round_num", 0)
    if curr_round > prev_round:
        reward += 0.5 * (curr_round - prev_round)

    prev_ante = prev_state.get("ante_num", 0)
    curr_ante = curr_state.get("ante_num", 0)
    if curr_ante > prev_ante:
        reward += 1.0 * (curr_ante - prev_ante)

    prev_jokers = prev_state.get("jokers", {}).get("count", 0)
    curr_jokers = curr_state.get("jokers", {}).get("count", 0)
    if curr_jokers > prev_jokers:
        reward += 0.1

    if won:
        reward += 10.0

    return reward


# ---------------------------------------------------------------------------
# Hand action helpers (mirrors heuristic_bots logic)
# ---------------------------------------------------------------------------

from collections import Counter


def _parse_card(card: dict):
    v = card.get("value", {})
    return v.get("rank", "2"), v.get("suit", "S")


def _find_flush(cards):
    suits = [_parse_card(c)[1] for c in cards]
    suit_counts = Counter(suits)
    for suit, count in suit_counts.items():
        if count >= 5:
            idxs = [i for i, c in enumerate(cards) if _parse_card(c)[1] == suit]
            idxs.sort(key=lambda i: RANK_VALUES.get(_parse_card(cards[i])[0], 0), reverse=True)
            return idxs[:5]
    return None


def _find_straight(cards):
    indexed = [(i, RANK_VALUES.get(_parse_card(c)[0], 0)) for i, c in enumerate(cards)]
    indexed.sort(key=lambda x: x[1])
    seen = {}
    for idx, rv in indexed:
        if rv not in seen:
            seen[rv] = idx
    unique = sorted(seen.keys())
    for start in range(len(unique) - 4):
        w = unique[start:start + 5]
        if w[-1] - w[0] == 4 and len(set(w)) == 5:
            return [seen[r] for r in w]
    return None


def _best_pair_hand(cards):
    ranks = [_parse_card(c)[0] for c in cards]
    rc = Counter(ranks)
    sorted_rc = sorted(rc.items(), key=lambda x: (x[1], RANK_VALUES.get(x[0], 0)), reverse=True)

    def idxs(rank, n):
        return [i for i, c in enumerate(cards) if _parse_card(c)[0] == rank][:n]

    counts = [c for _, c in sorted_rc]
    if counts[0] == 4:
        r = sorted_rc[0][0]
        return idxs(r, 4)[:5]
    if counts[0] == 3 and len(counts) > 1 and counts[1] >= 2:
        return idxs(sorted_rc[0][0], 3) + idxs(sorted_rc[1][0], 2)
    if counts[0] == 3:
        return idxs(sorted_rc[0][0], 3)
    if counts[0] == 2 and len(counts) > 1 and counts[1] == 2:
        return idxs(sorted_rc[0][0], 2) + idxs(sorted_rc[1][0], 2)
    if counts[0] == 2:
        return idxs(sorted_rc[0][0], 2)
    all_sorted = sorted(range(len(cards)), key=lambda i: RANK_VALUES.get(_parse_card(cards[i])[0], 0), reverse=True)
    return all_sorted[:5]


def _worst_cards(cards, n=3):
    """Return indices of n lowest-ranked cards."""
    sorted_idxs = sorted(range(len(cards)), key=lambda i: RANK_VALUES.get(_parse_card(cards[i])[0], 0))
    return sorted_idxs[:n]


# ---------------------------------------------------------------------------
# BalatroEnv
# ---------------------------------------------------------------------------

class BalatroEnv(gym.Env):
    """
    Gymnasium environment wrapping Balatrobot HTTP API.

    Each step corresponds to one decision point in the game:
    SELECTING_HAND, SHOP, or BLIND_SELECT.

    The action space is Discrete(N_ACTIONS=7). Valid actions depend on
    the current phase; invalid actions are masked to the closest valid one.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 12346,
        deck: str = "RED",
        stake: str = "WHITE",
        seed: str = "AAAAAAA",
    ):
        super().__init__()
        self.client = BalatroClient(host=host, port=port)
        self.deck = deck
        self.stake = stake
        self.game_seed = seed

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        self._state = {}
        self._prev_state = {}
        self._done = False

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        try:
            self.client.menu()
        except Exception:
            pass

        game_seed = options.get("seed", self.game_seed) if options else self.game_seed
        self._state = self.client.start(deck=self.deck, stake=self.stake, seed=game_seed)
        self._prev_state = dict(self._state)
        self._done = False

        # Advance through non-decision states
        self._state = self._advance_to_decision(self._state)
        obs = _encode_obs(self._state)
        return obs, {}

    def step(self, action: int):
        assert not self._done, "Call reset() before step()"

        self._prev_state = dict(self._state)
        state_name = self._state.get("state", "")

        try:
            if state_name == "SELECTING_HAND":
                next_state = self._do_hand_action(action)
            elif state_name == "SHOP":
                next_state = self._do_shop_action(action)
            elif state_name == "BLIND_SELECT":
                next_state = self._do_blind_action(action)
            else:
                next_state = self.client.gamestate()
        except BalatroError:
            # Invalid action — stay in current state, small penalty
            next_state = self.client.gamestate()

        # Advance through non-decision states (ROUND_EVAL, SMODS_BOOSTER_OPENED, etc.)
        next_state = self._advance_to_decision(next_state)
        self._state = next_state

        won = next_state.get("won", False)
        lost = next_state.get("state") == "GAME_OVER"
        self._done = won or lost

        reward = _compute_reward(self._prev_state, self._state, self._done, won)
        obs = _encode_obs(self._state)
        info = {
            "ante": self._state.get("ante_num", 0),
            "round": self._state.get("round_num", 0),
            "won": won,
        }

        return obs, reward, self._done, False, info

    def render(self):
        pass

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Action interpreters
    # ------------------------------------------------------------------

    def _do_hand_action(self, action: int) -> dict:
        cards = self._state.get("hand", {}).get("cards", [])
        discards_left = self._state.get("round", {}).get("discards_left", 0)

        if action == 0:  # play_best
            idxs = _best_pair_hand(cards)
        elif action == 1:  # discard_worst
            if discards_left > 0:
                return self.client.discard([int(i) for i in _worst_cards(cards, 3)])
            idxs = _best_pair_hand(cards)
        elif action == 2:  # play_flush
            idxs = _find_flush(cards) or _best_pair_hand(cards)
        elif action == 3:  # play_straight
            idxs = _find_straight(cards) or _best_pair_hand(cards)
        else:  # action 4: play_pair_hand
            idxs = _best_pair_hand(cards)

        idxs = [i for i in idxs if 0 <= i < len(cards)]
        if not idxs:
            idxs = list(range(min(5, len(cards))))
        return self.client.play([int(i) for i in idxs])

    def _do_shop_action(self, action: int) -> dict:
        shop_cards = self._state.get("shop", {}).get("cards", [])
        money = self._state.get("money", 0)
        reroll_cost = self._state.get("round", {}).get("reroll_cost", 5)

        if action == 0:  # end_shop
            return self.client.next_round()

        buy_idx = action - 1  # actions 1-5 map to card indices 0-4
        if action == 6:  # reroll
            if money >= reroll_cost:
                return self.client.reroll()
            return self.client.next_round()

        if buy_idx < len(shop_cards):
            cost = shop_cards[buy_idx].get("cost", {}).get("buy", 999)
            if cost <= money:
                try:
                    return self.client.buy(card=int(buy_idx))
                except BalatroError:
                    pass

        return self.client.next_round()

    def _do_blind_action(self, action: int) -> dict:
        blind_type = "boss"
        blinds = self._state.get("blinds", {})
        for bk in ["small", "big", "boss"]:
            b = blinds.get(bk, {})
            if b.get("status") in ("SELECT", "CURRENT"):
                blind_type = bk
                break

        if action == 1 and blind_type != "boss":
            return self.client.skip()
        return self.client.select()

    # ------------------------------------------------------------------
    # State advance helper
    # ------------------------------------------------------------------

    def _advance_to_decision(self, state: dict) -> dict:
        """
        Loop through non-decision states until we hit a state
        the agent needs to act on, or the game ends.
        """
        import time
        decision_states = {"SELECTING_HAND", "SHOP", "BLIND_SELECT", "GAME_OVER"}
        max_steps = 50

        for _ in range(max_steps):
            state_name = state.get("state", "UNKNOWN")

            if state_name in decision_states:
                return state
            if state.get("won", False):
                return state

            if state_name == "ROUND_EVAL":
                state = self.client.cash_out()
            elif state_name == "SMODS_BOOSTER_OPENED":
                state = self.client.pack(skip=True)
            else:
                time.sleep(0.2)
                state = self.client.gamestate()

        return state
