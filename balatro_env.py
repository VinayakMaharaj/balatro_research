"""
balatro_env.py
Gymnasium environment wrapping the Balatrobot HTTP JSON-RPC API.

Updated _encode_obs to fill all 72 dims matching balatro_mock_env.py
so trained MaskablePPO policy transfers correctly from mock to real game.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import Counter
from balatro_client import BalatroClient, BalatroError

# ---------------------------------------------------------------------------
# Constants — must match balatro_mock_env.py exactly
# ---------------------------------------------------------------------------

OBS_DIM = 72  # updated from 38 to match mock env training

N_HAND_ACTIONS  = 5
N_SHOP_ACTIONS  = 7
N_BLIND_ACTIONS = 2
N_ACTIONS = max(N_HAND_ACTIONS, N_SHOP_ACTIONS, N_BLIND_ACTIONS)

RANK_ORDER = ["2","3","4","5","6","7","8","9","T","J","Q","K","A"]
RANK_VALUES = {r: i for i, r in enumerate(RANK_ORDER)}
SUIT_VALUES = {"S": 0, "H": 1, "D": 2, "C": 3}

HAND_TYPES = [
    "straight_flush","four_of_a_kind","full_house","flush",
    "straight","three_of_a_kind","two_pair","pair","high_card",
]
HAND_TYPE_IDX = {h: i for i, h in enumerate(HAND_TYPES)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rank_val(rank: str) -> float:
    return RANK_VALUES.get(rank, 0) / 12.0

def _suit_val(suit: str) -> float:
    return SUIT_VALUES.get(suit, 0) / 3.0


def _detect_best_hand(hand_cards: list) -> str:
    """Detect best hand type from current hand cards."""
    if not hand_cards:
        return "high_card"
    ranks = [c.get("value", {}).get("rank", "2") for c in hand_cards]
    suits = [c.get("value", {}).get("suit", "S") for c in hand_cards]
    rc    = Counter(ranks)
    sc    = Counter(suits)
    counts = sorted(rc.values(), reverse=True)
    is_flush = max(sc.values()) >= 5
    rank_vals = sorted(set(RANK_VALUES.get(r, 0) for r in ranks))
    is_straight = (len(rank_vals) >= 5 and
                   any(rank_vals[i+4] - rank_vals[i] == 4
                       for i in range(len(rank_vals) - 4)))
    if is_flush and is_straight:                         return "straight_flush"
    if counts[0] == 4:                                   return "four_of_a_kind"
    if counts[0]==3 and len(counts)>1 and counts[1]==2:  return "full_house"
    if is_flush:                                         return "flush"
    if is_straight:                                      return "straight"
    if counts[0] == 3:                                   return "three_of_a_kind"
    if counts[0]==2 and len(counts)>1 and counts[1]==2:  return "two_pair"
    if counts[0] == 2:                                   return "pair"
    return "high_card"


def _encode_obs(state: dict) -> np.ndarray:
    """
    Encode real game state into 72-dim observation vector.
    Matches balatro_mock_env.py OBS_DIM=72 layout exactly.
    """
    obs = np.zeros(72, dtype=np.float32)

    # [0] ante
    ante = state.get("ante_num", 0)
    obs[0] = min(ante, 8) / 8.0

    # [1] round position
    blinds = state.get("blinds", {})
    blind_idx = 0
    blind_type = "small"
    chips_needed = 0
    for i, bk in enumerate(["small", "big", "boss"]):
        b = blinds.get(bk, {})
        if b.get("status") in ("SELECT", "CURRENT"):
            blind_idx = i
            blind_type = bk
            chips_needed = b.get("score", 0)
            break
    obs[1] = min((ante - 1) * 3 + blind_idx + 1, 24) / 24.0

    # [2] money
    obs[2] = min(state.get("money", 0), 100) / 100.0

    # [3-4] hands/discards left
    round_info = state.get("round", {})
    obs[3] = min(round_info.get("hands_left", 0), 4) / 4.0
    obs[4] = min(round_info.get("discards_left", 0), 4) / 4.0

    # [5] chips scored this round
    obs[5] = min(round_info.get("chips", 0), 300000) / 300000.0

    # [6] chips needed
    obs[6] = min(chips_needed, 300000) / 300000.0

    # [7-8] joker count/limit
    jokers = state.get("jokers", {})
    joker_count = jokers.get("count", 0)
    obs[7] = joker_count / 5.0
    obs[8] = jokers.get("limit", 5) / 5.0

    # [9] reroll cost
    obs[9] = min(round_info.get("reroll_cost", 5), 10) / 10.0

    # [10-17] hand card ranks, [18-25] hand card suits
    hand_cards = state.get("hand", {}).get("cards", [])
    for i, card in enumerate(hand_cards[:8]):
        v = card.get("value", {})
        obs[10 + i] = RANK_VALUES.get(v.get("rank", "2"), 0) / 12.0
        obs[18 + i] = SUIT_VALUES.get(v.get("suit", "S"), 0) / 3.0

    # [26-30] shop costs, [31-35] shop is_joker
    shop_cards = state.get("shop", {}).get("cards", [])
    for i, card in enumerate(shop_cards[:5]):
        cost = card.get("cost", {}).get("buy", 0)
        obs[26 + i] = min(cost, 20) / 20.0
        obs[31 + i] = 1.0 if card.get("set", "") == "JOKER" else 0.0

    # [36] blind type
    obs[36] = {"small": 0.0, "big": 0.5, "boss": 1.0}.get(blind_type, 0.0)

    # [37] state phase
    state_name = state.get("state", "")
    obs[37] = {"SELECTING_HAND": 0.0, "SHOP": 0.5, "BLIND_SELECT": 1.0}.get(state_name, 0.0)

    # [38-50] Rank counts in hand
    hand_ranks = Counter(c.get("value", {}).get("rank", "2") for c in hand_cards)
    for i, rank in enumerate(RANK_ORDER):
        obs[38 + i] = min(hand_ranks.get(rank, 0), 4) / 4.0

    # [51-54] Suit counts in hand
    hand_suits = Counter(c.get("value", {}).get("suit", "S") for c in hand_cards)
    for i, suit in enumerate(["S","H","D","C"]):
        obs[51 + i] = min(hand_suits.get(suit, 0), 8) / 8.0

    # [55-63] Dominant hand one-hot — best hand detectable from current cards
    dom = _detect_best_hand(hand_cards)
    obs[55 + HAND_TYPE_IDX.get(dom, 8)] = 1.0

    # [64] Hand level proxy — use joker count as rough proxy
    obs[64] = min(joker_count, 5) / 10.0

    # [65] Deck size — always full in real game
    obs[65] = 1.0

    # [66-69] Boss debuff encoding from blind effect text
    boss_b = blinds.get("boss", {})
    boss_effect = boss_b.get("effect", "").lower() if boss_b else ""
    suit_keywords = [("heart","H"),("spade","S"),("diamond","D"),("club","C")]
    debuff_suit = None
    for kw, code in suit_keywords:
        if kw in boss_effect:
            debuff_suit = code
            break
    obs[66] = 1.0 if debuff_suit else 0.0
    obs[67] = SUIT_VALUES.get(debuff_suit, 0) / 3.0 if debuff_suit else 0.0
    if any(x in boss_effect for x in ["1 card","needle","play 1"]):
        obs[68] = 1.0 / 5.0
    elif any(x in boss_effect for x in ["5 card","psychic","play 5"]):
        obs[68] = 5.0 / 5.0
    obs[69] = 1.0 if any(x in boss_effect for x in ["hand size","manacle","draw 1","fish"]) else 0.0

    # [70-71] Joker scaling proxies
    obs[70] = min(joker_count, 20) / 20.0
    obs[71] = min(joker_count * 0.5, 20) / 20.0

    return obs


def _compute_reward(prev_state: dict, curr_state: dict, done: bool, won: bool) -> float:
    reward = 0.0
    prev_round = prev_state.get("round_num", 0)
    curr_round = curr_state.get("round_num", 0)
    if curr_round > prev_round:
        reward += 0.5 * (curr_round - prev_round)
    prev_ante = prev_state.get("ante_num", 0)
    curr_ante = curr_state.get("ante_num", 0)
    if curr_ante > prev_ante:
        reward += 1.0 * (curr_ante - prev_ante)
    if won:
        reward += 10.0
    return reward


# ---------------------------------------------------------------------------
# Hand action helpers
# ---------------------------------------------------------------------------

def _parse_card(card: dict):
    v = card.get("value", {})
    return v.get("rank", "2"), v.get("suit", "S")


def _find_flush(cards):
    suits = [_parse_card(c)[1] for c in cards]
    suit_counts = Counter(suits)
    for suit, count in suit_counts.items():
        if count >= 5:
            idxs = [i for i,c in enumerate(cards) if _parse_card(c)[1] == suit]
            idxs.sort(key=lambda i: RANK_VALUES.get(_parse_card(cards[i])[0], 0), reverse=True)
            return idxs[:5]
    return None


def _find_straight(cards):
    indexed = [(i, RANK_VALUES.get(_parse_card(c)[0], 0)) for i,c in enumerate(cards)]
    indexed.sort(key=lambda x: x[1])
    seen = {}
    for idx, rv in indexed:
        if rv not in seen:
            seen[rv] = idx
    unique = sorted(seen.keys())
    for start in range(len(unique) - 4):
        w = unique[start:start+5]
        if w[-1]-w[0]==4 and len(set(w))==5:
            return [seen[r] for r in w]
    return None


def _best_pair_hand(cards):
    ranks = [_parse_card(c)[0] for c in cards]
    rc = Counter(ranks)
    sorted_rc = sorted(rc.items(), key=lambda x: (x[1], RANK_VALUES.get(x[0],0)), reverse=True)

    def idxs(rank, n):
        return [i for i,c in enumerate(cards) if _parse_card(c)[0] == rank][:n]

    counts = [c for _,c in sorted_rc]
    if counts[0] == 4:
        return idxs(sorted_rc[0][0], 4)[:5]
    if counts[0]==3 and len(counts)>1 and counts[1]>=2:
        return idxs(sorted_rc[0][0], 3) + idxs(sorted_rc[1][0], 2)
    if counts[0] == 3:
        return idxs(sorted_rc[0][0], 3)
    if counts[0]==2 and len(counts)>1 and counts[1]==2:
        return idxs(sorted_rc[0][0], 2) + idxs(sorted_rc[1][0], 2)
    if counts[0] == 2:
        return idxs(sorted_rc[0][0], 2)
    all_sorted = sorted(range(len(cards)), key=lambda i: RANK_VALUES.get(_parse_card(cards[i])[0], 0), reverse=True)
    return all_sorted[:5]


def _worst_cards(cards, n=3):
    sorted_idxs = sorted(range(len(cards)), key=lambda i: RANK_VALUES.get(_parse_card(cards[i])[0], 0))
    return sorted_idxs[:n]


# ---------------------------------------------------------------------------
# BalatroEnv
# ---------------------------------------------------------------------------

class BalatroEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, host="127.0.0.1", port=12346,
                 deck="RED", stake="WHITE", seed="AAAAAAA"):
        super().__init__()
        self.client    = BalatroClient(host=host, port=port)
        self.deck      = deck
        self.stake     = stake
        self.game_seed = seed

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space      = spaces.Discrete(N_ACTIONS)

        self._state      = {}
        self._prev_state = {}
        self._done       = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        try:
            self.client.menu()
        except Exception:
            pass
        game_seed    = options.get("seed", self.game_seed) if options else self.game_seed
        self._state  = self.client.start(deck=self.deck, stake=self.stake, seed=game_seed)
        self._prev_state = dict(self._state)
        self._done   = False
        self._state  = self._advance_to_decision(self._state)
        return _encode_obs(self._state), {}

    def step(self, action: int):
        assert not self._done
        self._prev_state = dict(self._state)
        state_name = self._state.get("state", "")
        try:
            if state_name == "SELECTING_HAND": next_state = self._do_hand_action(action)
            elif state_name == "SHOP":         next_state = self._do_shop_action(action)
            elif state_name == "BLIND_SELECT": next_state = self._do_blind_action(action)
            else:                              next_state = self.client.gamestate()
        except BalatroError:
            next_state = self.client.gamestate()

        next_state  = self._advance_to_decision(next_state)
        self._state = next_state
        won  = next_state.get("won", False)
        lost = next_state.get("state") == "GAME_OVER"
        self._done = won or lost
        reward = _compute_reward(self._prev_state, self._state, self._done, won)
        obs    = _encode_obs(self._state)
        return obs, reward, self._done, False, {
            "ante":  self._state.get("ante_num", 0),
            "round": self._state.get("round_num", 0),
            "won":   won,
        }

    def render(self): pass
    def close(self):  pass

    def _do_hand_action(self, action: int) -> dict:
        cards         = self._state.get("hand", {}).get("cards", [])
        discards_left = self._state.get("round", {}).get("discards_left", 0)
        if action == 1 and discards_left > 0:
            return self.client.discard([int(i) for i in _worst_cards(cards, 3)])
        if   action == 2: idxs = _find_flush(cards)    or _best_pair_hand(cards)
        elif action == 3: idxs = _find_straight(cards) or _best_pair_hand(cards)
        else:             idxs = _best_pair_hand(cards)
        idxs = [i for i in idxs if 0 <= i < len(cards)]
        if not idxs: idxs = list(range(min(5, len(cards))))
        return self.client.play([int(i) for i in idxs])

    def _do_shop_action(self, action: int) -> dict:
        shop_cards  = self._state.get("shop", {}).get("cards", [])
        money       = self._state.get("money", 0)
        reroll_cost = self._state.get("round", {}).get("reroll_cost", 5)
        if action == 0: return self.client.next_round()
        if action == 6:
            if money >= reroll_cost: return self.client.reroll()
            return self.client.next_round()
        buy_idx = action - 1
        if buy_idx < len(shop_cards):
            cost = shop_cards[buy_idx].get("cost", {}).get("buy", 999)
            if cost <= money:
                try:    return self.client.buy(card=int(buy_idx))
                except BalatroError: pass
        return self.client.next_round()

    def _do_blind_action(self, action: int) -> dict:
        blinds     = self._state.get("blinds", {})
        blind_type = "boss"
        for bk in ["small","big","boss"]:
            b = blinds.get(bk, {})
            if b.get("status") in ("SELECT","CURRENT"):
                blind_type = bk; break
        if action == 1 and blind_type != "boss":
            return self.client.skip()
        return self.client.select()

    def _advance_to_decision(self, state: dict) -> dict:
        import time
        decision_states = {"SELECTING_HAND","SHOP","BLIND_SELECT","GAME_OVER"}
        for _ in range(50):
            state_name = state.get("state", "UNKNOWN")
            if state_name in decision_states: return state
            if state.get("won", False):       return state
            if state_name == "ROUND_EVAL":         state = self.client.cash_out()
            elif state_name == "SMODS_BOOSTER_OPENED": state = self.client.pack(skip=True)
            else:
                time.sleep(0.2)
                state = self.client.gamestate()
        return state