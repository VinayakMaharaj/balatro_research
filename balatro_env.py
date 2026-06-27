"""
balatro_env.py
Gymnasium environment wrapping the Balatrobot HTTP JSON-RPC API.

Updated to OBS_DIM=148 matching balatro_mock_env.py v2.
New features: joker identity encoding, all hand levels, boss blind identity,
card enhancements/seals/debuffs, shop type breakdown, interest system.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import Counter
from balatro_client import BalatroClient, BalatroError

# ---------------------------------------------------------------------------
# Constants — must match balatro_mock_env.py exactly
# ---------------------------------------------------------------------------

OBS_DIM = 148

N_HAND_ACTIONS  = 5
N_SHOP_ACTIONS  = 7
N_BLIND_ACTIONS = 2
N_ACTIONS = max(N_HAND_ACTIONS, N_SHOP_ACTIONS, N_BLIND_ACTIONS)

RANK_ORDER  = ["2","3","4","5","6","7","8","9","T","J","Q","K","A"]
RANK_VALUES = {r: i for i, r in enumerate(RANK_ORDER)}
SUIT_VALUES = {"S": 0, "H": 1, "D": 2, "C": 3}
SUITS       = ["S","H","D","C"]

HAND_TYPES = [
    "straight_flush","four_of_a_kind","full_house","flush",
    "straight","three_of_a_kind","two_pair","pair","high_card",
]
HAND_TYPE_IDX = {h: i for i, h in enumerate(HAND_TYPES)}

ENHANCEMENTS    = ["none","glass","steel","gold","mult","bonus","wild","lucky"]
ENHANCEMENT_IDX = {e: i for i, e in enumerate(ENHANCEMENTS)}

SEALS    = ["none","red","blue","gold","purple"]
SEAL_IDX = {s: i for i, s in enumerate(SEALS)}

ALL_BOSS_BLINDS = [
    "The_Hook","The_Ox","The_House","The_Wall","The_Wheel",
    "The_Fish","The_Club","The_Tooth","The_Flint","The_Mark",
    "The_Psychic","The_Goad","The_Water","The_Eye","The_Plant",
    "The_Needle","The_Head","The_Serpent","The_Window","The_Manacle",
    "The_Arm","The_Pillar","The_Mouth",
]
BOSS_BLIND_IDX = {b: i for i, b in enumerate(ALL_BOSS_BLINDS)}

BOSS_CATEGORIES = {
    "The_Hook":    [0,0,0,0,1,0,0],
    "The_Ox":      [0,0,0,0,1,0,0],
    "The_House":   [1,0,0,0,0,0,0],
    "The_Wall":    [0,0,0,0,0,0,1],
    "The_Wheel":   [0,0,0,0,0,0,0],
    "The_Fish":    [0,1,0,0,0,0,0],
    "The_Club":    [1,0,0,0,0,0,0],
    "The_Tooth":   [0,0,0,0,0,0,1],
    "The_Flint":   [0,0,0,0,0,0,1],
    "The_Mark":    [0,0,0,0,0,1,0],
    "The_Psychic": [0,0,1,0,0,0,0],
    "The_Goad":    [1,0,0,0,0,0,0],
    "The_Water":   [0,0,0,0,0,0,0],
    "The_Eye":     [0,0,0,1,0,0,0],
    "The_Plant":   [1,0,0,0,0,0,0],
    "The_Needle":  [0,0,1,0,0,0,0],
    "The_Head":    [1,0,0,0,0,0,0],
    "The_Serpent": [0,0,0,0,0,0,0],
    "The_Window":  [1,0,0,0,0,0,0],
    "The_Manacle": [0,1,0,0,0,0,0],
    "The_Arm":     [0,0,0,0,0,0,1],
    "The_Pillar":  [0,0,0,0,0,0,0],
    "The_Mouth":   [0,0,0,1,0,0,0],
}

HAND_AFFINITY_IDX = {
    "none":0,"pair":1,"two_pair":2,"three_of_a_kind":3,
    "straight":4,"flush":5,"full_house":6,"four_of_a_kind":7,
}

# Map joker center keys to affinity/type info
JOKER_CENTER_KEY_MAP = {
    "j_jolly":      {"affinity":"pair",           "is_xmult":False,"is_economy":False},
    "j_zany":       {"affinity":"three_of_a_kind","is_xmult":False,"is_economy":False},
    "j_mad":        {"affinity":"two_pair",        "is_xmult":False,"is_economy":False},
    "j_crazy":      {"affinity":"straight",        "is_xmult":False,"is_economy":False},
    "j_droll":      {"affinity":"flush",           "is_xmult":False,"is_economy":False},
    "j_sly":        {"affinity":"pair",            "is_xmult":False,"is_economy":False},
    "j_wily":       {"affinity":"three_of_a_kind","is_xmult":False,"is_economy":False},
    "j_clever":     {"affinity":"straight",        "is_xmult":False,"is_economy":False},
    "j_devious":    {"affinity":"straight",        "is_xmult":False,"is_economy":False},
    "j_crafty":     {"affinity":"flush",           "is_xmult":False,"is_economy":False},
    "j_fibonacci":  {"affinity":"none",            "is_xmult":False,"is_economy":False},
    "j_baron":      {"affinity":"none",            "is_xmult":True, "is_economy":False},
    "j_ride_the_bus":{"affinity":"none",           "is_xmult":False,"is_economy":False},
    "j_cavendish":  {"affinity":"none",            "is_xmult":True, "is_economy":False},
    "j_even_steven":{"affinity":"none",            "is_xmult":False,"is_economy":False},
    "j_odd_todd":   {"affinity":"none",            "is_xmult":False,"is_economy":False},
    "j_scary_face": {"affinity":"none",            "is_xmult":False,"is_economy":False},
    "j_bull":       {"affinity":"none",            "is_xmult":False,"is_economy":True},
    "j_green_joker":{"affinity":"none",            "is_xmult":False,"is_economy":True},
    "j_hologram":   {"affinity":"none",            "is_xmult":True, "is_economy":False},
    "j_joker":      {"affinity":"none",            "is_xmult":False,"is_economy":False},
    "j_abstract":   {"affinity":"none",            "is_xmult":False,"is_economy":False},
    "j_smiley":     {"affinity":"none",            "is_xmult":False,"is_economy":False},
    "j_supernova":  {"affinity":"none",            "is_xmult":False,"is_economy":False},
    "j_blueprint":  {"affinity":"none",            "is_xmult":True, "is_economy":False},
    "j_swashbuckler":{"affinity":"none",           "is_xmult":False,"is_economy":True},
    "j_burglar":    {"affinity":"none",            "is_xmult":False,"is_economy":True},
}

PLANET_HAND_MAP = {
    "Mercury":"high_card","Venus":"three_of_a_kind","Earth":"full_house",
    "Mars":"four_of_a_kind","Jupiter":"flush","Saturn":"straight",
    "Uranus":"two_pair","Neptune":"straight_flush","Pluto":"pair",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_card(card: dict):
    v = card.get("value", {})
    return v.get("rank","2"), v.get("suit","S")


def _find_flush(cards):
    suits = [_parse_card(c)[1] for c in cards]
    sc    = Counter(suits)
    for suit, count in sc.most_common():
        if count >= 5:
            idxs = [i for i,c in enumerate(cards) if _parse_card(c)[1]==suit]
            idxs.sort(key=lambda i: RANK_VALUES.get(_parse_card(cards[i])[0],0), reverse=True)
            return idxs[:5]
    return None


def _find_straight(cards):
    indexed = [(i, RANK_VALUES.get(_parse_card(c)[0],0)) for i,c in enumerate(cards)]
    indexed.sort(key=lambda x: x[1])
    seen = {}
    for idx, rv in indexed:
        if rv not in seen: seen[rv] = idx
    unique = sorted(seen.keys())
    for start in range(len(unique)-4):
        w = unique[start:start+5]
        if w[-1]-w[0]==4 and len(set(w))==5:
            return [seen[r] for r in w]
    return None


def _find_straight_flush(cards):
    suits = [_parse_card(c)[1] for c in cards]
    sc    = Counter(suits)
    for suit, count in sc.items():
        if count < 5: continue
        suited = [(i,c) for i,c in enumerate(cards) if _parse_card(c)[1]==suit]
        vals   = [(i, RANK_VALUES.get(_parse_card(c)[0],0)) for i,c in suited]
        vals.sort(key=lambda x: x[1])
        seen = {}
        for idx, rv in vals:
            if rv not in seen: seen[rv] = idx
        unique = sorted(seen.keys())
        for start in range(len(unique)-4):
            w = unique[start:start+5]
            if w[-1]-w[0]==4 and len(set(w))==5:
                return [seen[r] for r in w]
    return None


def _find_four_of_a_kind(cards):
    ranks = [_parse_card(c)[0] for c in cards]
    rc    = Counter(ranks)
    for rank, count in rc.most_common():
        if count >= 4:
            idxs = [i for i,c in enumerate(cards) if _parse_card(c)[0]==rank][:4]
            # add kicker
            rest = [i for i in range(len(cards)) if i not in idxs]
            rest.sort(key=lambda i: RANK_VALUES.get(_parse_card(cards[i])[0],0), reverse=True)
            return idxs + rest[:1]
    return None


def _find_full_house(cards):
    ranks = [_parse_card(c)[0] for c in cards]
    rc    = Counter(ranks)
    threes = [r for r,c in rc.items() if c>=3]
    if not threes: return None
    three_rank = max(threes, key=lambda r: RANK_VALUES.get(r,0))
    pair_opts  = [r for r,c in rc.items() if c>=2 and r!=three_rank]
    if not pair_opts: return None
    pair_rank = max(pair_opts, key=lambda r: RANK_VALUES.get(r,0))
    idxs = ([i for i,c in enumerate(cards) if _parse_card(c)[0]==three_rank][:3] +
            [i for i,c in enumerate(cards) if _parse_card(c)[0]==pair_rank][:2])
    return idxs


def _best_pair_hand(cards):
    ranks = [_parse_card(c)[0] for c in cards]
    rc    = Counter(ranks)
    sorted_rc = sorted(rc.items(), key=lambda x:(x[1],RANK_VALUES.get(x[0],0)), reverse=True)

    def idxs(rank, n):
        return [i for i,c in enumerate(cards) if _parse_card(c)[0]==rank][:n]

    counts = [c for _,c in sorted_rc]
    if counts[0] == 2 and len(counts)>1 and counts[1]==2:
        return idxs(sorted_rc[0][0],2) + idxs(sorted_rc[1][0],2)
    if counts[0] == 2:
        return idxs(sorted_rc[0][0],2)
    all_sorted = sorted(range(len(cards)),
                        key=lambda i: RANK_VALUES.get(_parse_card(cards[i])[0],0), reverse=True)
    return all_sorted[:5]


def _worst_cards(cards, n=3):
    return sorted(range(len(cards)),
                  key=lambda i: RANK_VALUES.get(_parse_card(cards[i])[0],0))[:n]


def _detect_best_hand(hand_cards):
    if not hand_cards: return "high_card"
    ranks  = [_parse_card(c)[0] for c in hand_cards]
    suits  = [_parse_card(c)[1] for c in hand_cards]
    rc     = Counter(ranks)
    sc     = Counter(suits)
    vals   = sorted(RANK_VALUES.get(r,0) for r in ranks)
    counts = sorted(rc.values(), reverse=True)
    is_flush    = max(sc.values())>=5 if sc else False
    uv          = sorted(set(vals))
    is_straight = (len(uv)>=5 and any(uv[i+4]-uv[i]==4 for i in range(len(uv)-4)))
    if is_flush and is_straight:                         return "straight_flush"
    if counts[0]==4:                                     return "four_of_a_kind"
    if counts[0]==3 and len(counts)>1 and counts[1]==2:  return "full_house"
    if is_flush:                                         return "flush"
    if is_straight:                                      return "straight"
    if counts[0]==3:                                     return "three_of_a_kind"
    if counts[0]==2 and len(counts)>1 and counts[1]==2:  return "two_pair"
    if counts[0]==2:                                     return "pair"
    return "high_card"


def _parse_boss_blind_name(blinds: dict) -> str:
    """Extract boss blind name from blinds dict."""
    boss = blinds.get("boss", {})
    name = boss.get("name","") or boss.get("label","") or boss.get("effect","")
    # Normalise to our naming convention
    name = name.strip().replace(" ","_")
    if not name.startswith("The_"):
        name = "The_" + name
    return name if name in BOSS_BLIND_IDX else "none"


def _parse_joker_info(joker: dict) -> dict:
    """Extract affinity and type info from a real game joker."""
    center_key = joker.get("config",{}).get("center_key","")
    if not center_key:
        label = joker.get("label","").lower().replace(" ","_")
        center_key = f"j_{label}"
    info = JOKER_CENTER_KEY_MAP.get(center_key, {})
    return {
        "affinity":   info.get("affinity","none"),
        "is_xmult":   info.get("is_xmult",False),
        "is_economy": info.get("is_economy",False),
    }


def _encode_obs(state: dict, hand_levels: dict = None, dominant_hand: str = "pair",
                interest_earned: int = 0) -> np.ndarray:
    """
    Encode real game state into 148-dim observation vector.
    Matches balatro_mock_env.py v2 OBS_DIM=148 layout exactly.

    hand_levels and dominant_hand must be tracked externally
    since the API doesn't expose them directly.
    """
    obs = np.zeros(OBS_DIM, dtype=np.float32)

    if hand_levels is None:
        hand_levels = {ht:1 for ht in HAND_TYPES}

    # [0-9] Base game state
    ante = state.get("ante_num", 0)
    obs[0] = min(ante, 8) / 8.0

    blinds     = state.get("blinds", {})
    blind_idx  = 0
    blind_type = "small"
    chips_needed = 0
    for i, bk in enumerate(["small","big","boss"]):
        b = blinds.get(bk, {})
        if b.get("status") in ("SELECT","CURRENT"):
            blind_idx  = i
            blind_type = bk
            chips_needed = b.get("score",0) or b.get("chips",0)
            break

    obs[1]  = min((ante-1)*3 + blind_idx + 1, 24) / 24.0
    obs[2]  = min(state.get("money",0), 100) / 100.0
    round_info = state.get("round", {})
    obs[3]  = min(round_info.get("hands_left",0), 4) / 4.0
    obs[4]  = min(round_info.get("discards_left",0), 4) / 4.0
    obs[5]  = min(round_info.get("chips",0), 300000) / 300000.0
    obs[6]  = min(chips_needed, 300000) / 300000.0
    jokers  = state.get("jokers", {})
    jcount  = jokers.get("count",0) if isinstance(jokers,dict) else len(jokers)
    jlimit  = jokers.get("limit",5) if isinstance(jokers,dict) else 5
    obs[7]  = jcount / 5.0
    obs[8]  = jlimit / 5.0
    obs[9]  = min(round_info.get("reroll_cost",5), 10) / 10.0

    # [10-49] Hand cards: rank, suit, enhancement, seal, debuff
    hand_cards = state.get("hand",{}).get("cards",[])
    for i, card in enumerate(hand_cards[:8]):
        v    = card.get("value",{}) or card.get("base",{})
        rank = v.get("rank","2")
        suit = v.get("suit","S")
        obs[10+i] = RANK_VALUES.get(rank,0) / 12.0
        obs[18+i] = SUIT_VALUES.get(suit,0) / 3.0

        enh = card.get("enhancement","") or card.get("label","") or "none"
        enh = enh.lower().replace(" ","_")
        obs[26+i] = ENHANCEMENT_IDX.get(enh,0) / len(ENHANCEMENTS)

        seal = (card.get("seal","") or "none").lower()
        obs[34+i] = SEAL_IDX.get(seal,0) / len(SEALS)

        obs[42+i] = 1.0 if card.get("debuff",False) else 0.0

    # [50-59] Shop costs and types
    shop_cards  = state.get("shop",{}).get("cards",[])
    shop_type_enc = {
        "JOKER":0.0,"Joker":0.0,"joker":0.0,
        "PLANET":0.2,"Planet":0.2,"planet":0.2,
        "TAROT":0.4,"Tarot":0.4,"tarot":0.4,
        "VOUCHER":0.6,"Voucher":0.6,"voucher":0.6,
        "PACK":0.8,"Booster":0.8,"pack":0.8,
        "SPECTRAL":1.0,"Spectral":1.0,
    }
    for i, card in enumerate(shop_cards[:5]):
        cost    = card.get("cost",{}).get("buy",0) if isinstance(card.get("cost"),dict) else card.get("cost",0)
        card_set = card.get("set","") or card.get("ability",{}).get("set","")
        obs[50+i] = min(cost,20) / 20.0
        obs[55+i] = shop_type_enc.get(card_set, 0.0)

    # [60-61] Blind type and state phase
    obs[60] = {"small":0.0,"big":0.5,"boss":1.0}.get(blind_type, 0.0)
    state_name = state.get("state","")
    obs[61] = {"SELECTING_HAND":0.0,"SHOP":0.5,"BLIND_SELECT":1.0}.get(state_name, 0.0)

    # [62-74] Rank counts in hand
    hand_ranks = Counter(_parse_card(c)[0] for c in hand_cards)
    for i, rank in enumerate(RANK_ORDER):
        obs[62+i] = min(hand_ranks.get(rank,0),4) / 4.0

    # [75-78] Suit counts in hand
    hand_suits = Counter(_parse_card(c)[1] for c in hand_cards)
    for i, suit in enumerate(SUITS):
        obs[75+i] = min(hand_suits.get(suit,0),8) / 8.0

    # [79-87] Dominant hand one-hot
    di = HAND_TYPE_IDX.get(dominant_hand, 8)
    obs[79+di] = 1.0

    # [88-96] All hand levels
    for i, ht in enumerate(HAND_TYPES):
        obs[88+i] = min(hand_levels.get(ht,1), 10) / 10.0

    # [97] Deck size proxy
    obs[97] = 1.0  # real game always has full deck context

    # [98-104] Boss blind category flags
    boss_name = _parse_boss_blind_name(blinds)
    cat = BOSS_CATEGORIES.get(boss_name, [0]*7)
    for i, v in enumerate(cat):
        obs[98+i] = float(v)

    # [105-127] Boss blind identity one-hot
    boss_id = BOSS_BLIND_IDX.get(boss_name, -1)
    if boss_id >= 0:
        obs[105+boss_id] = 1.0

    # [128-129] Interest system
    money = state.get("money",0)
    obs[128] = min(interest_earned, 5) / 5.0
    obs[129] = (money % 5) / 4.0

    # [130-144] Joker slots 5x3
    joker_list = jokers.get("cards",[]) if isinstance(jokers,dict) else jokers
    for i, j in enumerate(joker_list[:5]):
        base = 130 + i*3
        info = _parse_joker_info(j)
        obs[base]   = HAND_AFFINITY_IDX.get(info["affinity"],0) / 7.0
        obs[base+1] = 1.0 if info["is_xmult"]   else 0.0
        obs[base+2] = 1.0 if info["is_economy"]  else 0.0

    # [145] Joker synergy score
    if joker_list:
        synergy = 0.0
        for j in joker_list:
            info = _parse_joker_info(j)
            aff  = info["affinity"]
            if aff == dominant_hand:   synergy += 1.0
            elif aff == "none":        synergy += 0.3 if info["is_xmult"] else 0.2
            else:                      synergy += 0.1
        obs[145] = min(synergy / max(len(joker_list),1), 1.0)

    # [146-147] Green joker / ride bus (tracked externally, default 0)
    obs[146] = 0.0
    obs[147] = 0.0

    return obs


def _compute_reward(prev_state: dict, curr_state: dict, done: bool, won: bool) -> float:
    reward = 0.0
    if curr_state.get("round_num",0) > prev_state.get("round_num",0):
        reward += 0.5
    if curr_state.get("ante_num",0) > prev_state.get("ante_num",0):
        reward += 1.0
    if won:
        reward += 10.0
    return reward


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

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)

        self._state        = {}
        self._prev_state   = {}
        self._done         = False
        self._hand_levels  = {ht:1 for ht in HAND_TYPES}
        self._hand_counts  = {ht:0 for ht in HAND_TYPES}
        self._dominant     = "pair"
        self._interest     = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._hand_levels = {ht:1 for ht in HAND_TYPES}
        self._hand_counts = {ht:0 for ht in HAND_TYPES}
        self._dominant    = "pair"
        self._interest    = 0
        try:
            self.client.menu()
        except Exception:
            pass
        game_seed    = options.get("seed",self.game_seed) if options else self.game_seed
        self._state  = self.client.start(deck=self.deck, stake=self.stake, seed=game_seed)
        self._prev_state = dict(self._state)
        self._done   = False
        self._state  = self._advance_to_decision(self._state)
        return _encode_obs(self._state, self._hand_levels, self._dominant, self._interest), {}

    def step(self, action: int):
        assert not self._done
        self._prev_state = dict(self._state)
        state_name = self._state.get("state","")
        try:
            if state_name == "SELECTING_HAND": next_state = self._do_hand_action(action)
            elif state_name == "SHOP":          next_state = self._do_shop_action(action)
            elif state_name == "BLIND_SELECT":  next_state = self._do_blind_action(action)
            else:                               next_state = self.client.gamestate()
        except BalatroError:
            next_state = self.client.gamestate()

        next_state  = self._advance_to_decision(next_state)
        self._state = next_state
        won  = next_state.get("won",False)
        lost = next_state.get("state") == "GAME_OVER"
        self._done = won or lost
        reward = _compute_reward(self._prev_state, self._state, self._done, won)
        obs    = _encode_obs(self._state, self._hand_levels, self._dominant, self._interest)
        return obs, reward, self._done, False, {
            "ante":  self._state.get("ante_num",0),
            "round": self._state.get("round_num",0),
            "won":   won,
        }

    def render(self): pass
    def close(self):  pass

    def _do_hand_action(self, action: int) -> dict:
        cards         = self._state.get("hand",{}).get("cards",[])
        discards_left = self._state.get("round",{}).get("discards_left",0)

        # Discard action
        if action == 1 and discards_left > 0:
            return self.client.discard([int(i) for i in _worst_cards(cards, 3)])

        # Try hands in proper priority order
        sf = _find_straight_flush(cards)
        if sf:   return self._play_and_track(sf, "straight_flush")
        foak = _find_four_of_a_kind(cards)
        if foak: return self._play_and_track(foak, "four_of_a_kind")
        fh = _find_full_house(cards)
        if fh:   return self._play_and_track(fh, "full_house")
        fl = _find_flush(cards)
        if fl:   return self._play_and_track(fl, "flush")
        st = _find_straight(cards)
        if st:   return self._play_and_track(st, "straight")
        # Pair/lower
        idxs = _best_pair_hand(cards)
        idxs = [i for i in idxs if 0<=i<len(cards)]
        if not idxs: idxs = list(range(min(5,len(cards))))
        return self._play_and_track(idxs, _detect_best_hand([cards[i] for i in idxs]))

    def _play_and_track(self, idxs: list, hand_type: str) -> dict:
        """Play cards and update hand level tracking."""
        self._hand_counts[hand_type] = self._hand_counts.get(hand_type,0) + 1
        # Level up if played 5 times
        if self._hand_counts[hand_type] % 5 == 0:
            self._hand_levels[hand_type] = self._hand_levels.get(hand_type,1) + 1
        # Update dominant hand
        self._dominant = max(self._hand_counts, key=self._hand_counts.get)
        return self.client.play([int(i) for i in idxs])

    def _do_shop_action(self, action: int) -> dict:
        shop_cards  = self._state.get("shop",{}).get("cards",[])
        money       = self._state.get("money",0)
        reroll_cost = self._state.get("round",{}).get("reroll_cost",5)

        if action == 0: return self.client.next_round()
        if action == 6:
            if money >= reroll_cost: return self.client.reroll()
            return self.client.next_round()

        buy_idx = action - 1
        if buy_idx < len(shop_cards):
            card = shop_cards[buy_idx]
            cost = card.get("cost",{}).get("buy",999) if isinstance(card.get("cost"),dict) else card.get("cost",999)
            if cost <= money and cost > 0:
                # Track planet card purchases for hand levels
                card_set = card.get("set","") or card.get("ability",{}).get("set","")
                if card_set in ("PLANET","Planet"):
                    label = card.get("label","")
                    ht    = PLANET_HAND_MAP.get(label, self._dominant)
                    self._hand_levels[ht] = self._hand_levels.get(ht,1) + 1
                try:
                    return self.client.buy(card=int(buy_idx))
                except BalatroError:
                    pass
        return self.client.next_round()

    def _do_blind_action(self, action: int) -> dict:
        blinds     = self._state.get("blinds",{})
        blind_type = "boss"
        for bk in ["small","big","boss"]:
            b = blinds.get(bk,{})
            if b.get("status") in ("SELECT","CURRENT"):
                blind_type = bk; break
        if action == 1 and blind_type != "boss":
            return self.client.skip()
        return self.client.select()

    def _advance_to_decision(self, state: dict) -> dict:
        import time
        decision_states = {"SELECTING_HAND","SHOP","BLIND_SELECT","GAME_OVER"}
        for _ in range(50):
            state_name = state.get("state","UNKNOWN")
            if state_name in decision_states: return state
            if state.get("won",False):        return state
            if state_name == "ROUND_EVAL":
                state = self.client.cash_out()
                # Track interest earned
                money_before = state.get("money",0)
                self._interest = min(money_before // 5, 5)
            elif state_name == "SMODS_BOOSTER_OPENED":
                state = self.client.pack(skip=True)
            else:
                time.sleep(0.2)
                state = self.client.gamestate()
        return state