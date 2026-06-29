"""
rl_bot.py  v3
MaskablePPO-based RL agent for Balatro.
Updated for OBS_DIM=148, dynamic hand selection, joker-aware shop policy.

Key fixes vs v2:
- Deterministic joker buying at ante 1-2 when joker count == 0
- Boss blind card filtering: skip debuffed cards in hand selection
- The Psychic: force 5-card play
- The Needle: force 1-card play with best card
- The Hook: no discards
- The Water: no discards
- Long-horizon ante awareness: track which cards were played this ante

Training:
    python rl_bot.py --train --mock --timesteps 120000000

Evaluation:
    python rl_bot.py --run
"""

import os
import time
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR      = Path("rl_model")
MODEL_PATH     = MODEL_DIR / "ppo_balatro"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
N_ENVS         = 16
MOCK_OBS_DIM   = 148

BENCHMARK_SEEDS = [f"SEED{str(i).zfill(3)}" for i in range(1, 101)]
RUNS_PER_SEED   = 1

CURRICULUM = [
    (0,          1),
    (40_000_000, 2),
    (60_000_000, 3),
    (80_000_000, 4),
    (100_000_000,5),
    (120_000_000,6),
    (140_000_000,7),
    (160_000_000,8),
]

HAND_TYPES = [
    "straight_flush","four_of_a_kind","full_house","flush",
    "straight","three_of_a_kind","two_pair","pair","high_card",
]

PLANET_HAND_MAP = {
    "Mercury":"high_card","Venus":"three_of_a_kind","Earth":"full_house",
    "Mars":"four_of_a_kind","Jupiter":"flush","Saturn":"straight",
    "Uranus":"two_pair","Neptune":"straight_flush","Pluto":"pair",
}

JOKER_AFFINITY = {
    "Jolly_Joker":"pair","Sly_Joker":"pair",
    "Zany_Joker":"three_of_a_kind","Wily_Joker":"three_of_a_kind",
    "Mad_Joker":"two_pair",
    "Crazy_Joker":"straight","Devious_Joker":"straight","Clever_Joker":"straight",
    "Droll_Joker":"flush","Crafty_Joker":"flush",
}

BOSS_NO_DISCARD  = {"The_Water"}
BOSS_PLAY_ONE    = {"The_Needle"}
BOSS_PLAY_FIVE   = {"The_Psychic"}
BOSS_HOOK        = {"The_Hook"}
BOSS_EYE         = {"The_Eye"}
BOSS_FACE_DEBUFF = {"The_Plant"}
BOSS_SUIT_DEBUFF = {
    "The_Club":"C","The_Goad":"S","The_Head":"H",
    "The_Window":"D","The_House":"C",
}

RANK_VALUES = {"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,
               "T":10,"J":11,"Q":12,"K":13,"A":14}
FACE_CARDS  = {"J","Q","K"}


def _get_curriculum_ante(total_steps):
    max_ante = 1
    for min_steps, ante in CURRICULUM:
        if total_steps >= min_steps:
            max_ante = ante
    return max_ante


# ---------------------------------------------------------------------------
# Diagnostic tracker
# ---------------------------------------------------------------------------

class DiagnosticTracker:
    def __init__(self):
        self.reset()
        self.all_games = []

    def reset(self):
        self.hand_decisions    = []
        self.shop_decisions    = []
        self.hand_types_played = Counter()
        self.missed_buys       = 0
        self.total_shop_visits = 0
        self.game_start_time   = time.time()
        self.flush_available_not_played = 0

    def log_hand(self, hand_type, chips_scored, chips_needed,
                 hands_left, discards_left, is_boss, ante):
        self.hand_decisions.append({
            "ante":ante,"hand_type":hand_type,
            "chips_scored":chips_scored,"chips_needed":chips_needed,
            "hands_left":hands_left,"discards_left":discards_left,
            "is_boss":is_boss,
        })
        self.hand_types_played[hand_type] += 1

    def log_shop(self, action, money, shop_cards, bought, rerolled, ante):
        affordable = [c for c in shop_cards
                      if c.get("cost",{}).get("buy",999) <= money
                      and c.get("cost",{}).get("buy",0) > 0]
        self.total_shop_visits += 1
        if len(affordable) > 0 and not bought and not rerolled:
            self.missed_buys += 1
        self.shop_decisions.append({
            "ante":ante,"money":money,
            "affordable":len(affordable),"bought":bought,
        })

    def classify_death(self, final_ante, final_chips, chips_needed,
                       hands_remaining, jokers_held, is_boss):
        if chips_needed and chips_needed > 0:
            pct = final_chips / chips_needed
            if is_boss and pct < 0.4:  return "boss_blind_crushed"
            if is_boss and pct < 0.8:  return "boss_blind_underscored"
            if pct < 0.5:              return "severe_chip_deficit"
            if pct < 0.8:              return "marginal_chip_deficit"
        if hands_remaining == 0:       return "ran_out_hands"
        if self.missed_buys > 2:       return "economy_failure"
        if jokers_held == 0 and final_ante >= 2: return "no_jokers_late"
        return "unknown"

    def finalize_game(self, seed, outcome, final_ante, final_round,
                      final_chips, chips_needed, hands_remaining,
                      discards_remaining, jokers_held, is_boss):
        death_reason = self.classify_death(
            final_ante, final_chips, chips_needed,
            hands_remaining, jokers_held, is_boss)
        total = sum(self.hand_types_played.values())
        flush_rate = (self.hand_types_played.get("flush",0) +
                      self.hand_types_played.get("straight_flush",0)) / max(total,1)
        summary = {
            "seed":seed,"outcome":outcome,
            "final_ante":final_ante,"final_round":final_round,
            "final_chips":final_chips,"chips_needed":chips_needed,
            "chip_deficit":max(0,(chips_needed or 0)-(final_chips or 0)),
            "hands_remaining":hands_remaining,
            "discards_remaining":discards_remaining,
            "jokers_held":jokers_held,"is_boss_blind":is_boss,
            "death_reason":death_reason,
            "hand_types":dict(self.hand_types_played),
            "flush_rate":round(flush_rate,3),
            "flush_available_not_played":self.flush_available_not_played,
            "missed_buys":self.missed_buys,
            "total_shop_visits":self.total_shop_visits,
            "duration_seconds":round(time.time()-self.game_start_time,1),
        }
        self.all_games.append(summary)
        return summary

    def print_full_report(self):
        if not self.all_games: return
        games     = self.all_games
        completed = [g for g in games if g["outcome"] in ("won","lost")]
        n         = len(completed)
        if n == 0: return

        print("\n" + "="*60)
        print("DIAGNOSTIC REPORT")
        print("="*60)

        death_counts = Counter(g["death_reason"] for g in completed if g["outcome"]=="lost")
        print(f"\nDEATH REASONS (n={sum(death_counts.values())}):")
        for reason, count in death_counts.most_common():
            print(f"  {reason:<35} {count:>3}  ({count/n*100:.1f}%)")

        ante_counts = Counter(g["final_ante"] for g in completed)
        print(f"\nDEATH BY ANTE:")
        for ante in sorted(ante_counts):
            count = ante_counts[ante]
            print(f"  Ante {ante}: {count:>3} ({count/n*100:.1f}%)")

        total_ht = Counter()
        for g in completed:
            for ht, cnt in g["hand_types"].items():
                total_ht[ht] += cnt
        total_h = sum(total_ht.values())
        print(f"\nHAND TYPE DISTRIBUTION (total={total_h}):")
        for ht, cnt in total_ht.most_common():
            print(f"  {ht:<25} {cnt:>5}  ({cnt/max(total_h,1)*100:.1f}%)")

        avg_flush      = sum(g["flush_rate"] for g in completed)/n
        avg_missed_buy = sum(g["missed_buys"] for g in completed)/n
        avg_jokers     = sum(g["jokers_held"] for g in completed)/n
        print(f"\nKEY METRICS:")
        print(f"  Avg flush/SF rate:        {avg_flush*100:.1f}%")
        print(f"  Avg missed buys/game:     {avg_missed_buy:.2f}")
        print(f"  Avg jokers at death:      {avg_jokers:.2f}")

        boss_deaths = [g for g in completed if g["outcome"]=="lost" and g.get("is_boss_blind")]
        print(f"\nBOSS BLIND DEATHS: {len(boss_deaths)}/{sum(1 for g in completed if g['outcome']=='lost')}")

        print("="*60)
        diag_path = Path("diagnostics.json")
        with open(diag_path,"w") as f:
            json.dump(games, f, indent=2)
        print(f"\nFull diagnostics saved to: {diag_path}")


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def _make_callbacks(checkpoint_dir, n_envs, vec_env):
    from stable_baselines3.common.callbacks import BaseCallback

    class CurriculumCallback(BaseCallback):
        def __init__(self, vec_env):
            super().__init__()
            self._vec_env      = vec_env
            self._current_ante = 1
            self._ep           = 0

        def _on_step(self):
            new_ante = _get_curriculum_ante(self.num_timesteps)
            if new_ante != self._current_ante:
                self._current_ante = new_ante
                for env_fn in self._vec_env.envs:
                    try:    env_fn.env.set_max_ante(new_ante)
                    except Exception:
                        try: env_fn.set_max_ante(new_ante)
                        except Exception: pass
                logger.info(f"Curriculum -> max_ante={new_ante} at {self.num_timesteps:,} steps")
                try:
                    import wandb
                    if wandb.run:
                        wandb.log({"curriculum/max_ante":new_ante,
                                   "curriculum/steps":self.num_timesteps})
                except ImportError: pass
            try:
                import wandb
                if wandb.run:
                    for info in self.locals.get("infos",[]):
                        if "episode" in info:
                            ep = info["episode"]
                            self._ep += 1
                            wandb.log({
                                "train/reward":          ep["r"],
                                "train/length":          ep["l"],
                                "train/ante":            info.get("ante",0),
                                "train/won":             int(info.get("won",False)),
                                "train/episodes":        self._ep,
                                "train/curriculum_ante": self._current_ante,
                            })
            except ImportError: pass
            return True

    class CheckpointCallback(BaseCallback):
        def __init__(self, checkpoint_freq, checkpoint_dir):
            super().__init__()
            self._freq      = checkpoint_freq
            self._dir       = Path(checkpoint_dir)
            self._last_save = 0
            self._dir.mkdir(parents=True, exist_ok=True)

        def _on_step(self):
            if self.num_timesteps - self._last_save >= self._freq:
                path = self._dir / f"ppo_balatro_{self.num_timesteps//1_000_000}M"
                self.model.save(str(path))
                self._last_save = self.num_timesteps
                logger.info(f"Checkpoint: {path}.zip")
            return True

    from stable_baselines3.common.callbacks import CallbackList
    return CallbackList([
        CurriculumCallback(vec_env),
        CheckpointCallback(10_000_000, checkpoint_dir),
    ])


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------

def _make_env():
    from balatro_mock_env import BalatroMockEnv
    from sb3_contrib.common.wrappers import ActionMasker
    env = BalatroMockEnv(max_ante=1)
    env = ActionMasker(env, lambda e: e.action_masks())
    return env


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(timesteps=120_000_000, seed="AAAAAAA", port=12346,
          use_mock=True, n_envs=N_ENVS):
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    logger.info(f"Training MaskablePPO v3 | {timesteps:,} steps | {n_envs} envs | OBS_DIM={MOCK_OBS_DIM}")

    vec_env = DummyVecEnv([_make_env for _ in range(n_envs)])

    try:
        import wandb
        wandb.init(
            project="balatro-research",
            name=f"rl_v3_{timesteps//1_000_000}Msteps_{n_envs}envs",
            config={
                "algorithm":"MaskablePPO","timesteps":timesteps,
                "n_envs":n_envs,"obs_dim":MOCK_OBS_DIM,
                "net_arch":[256,256,128],"curriculum":CURRICULUM,
                "version":"v3_joker_fix",
            },
            tags=["rl_bot","maskableppo","curriculum","v3","training"],
        )
    except Exception as e:
        logger.info(f"W&B skipped: {e}")

    model = MaskablePPO(
        "MlpPolicy", vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        policy_kwargs=dict(net_arch=[256,256,128]),
    )

    callbacks = _make_callbacks(CHECKPOINT_DIR, n_envs, vec_env)
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=True)

    MODEL_DIR.mkdir(exist_ok=True)
    model.save(str(MODEL_PATH))
    logger.info(f"Model saved to {MODEL_PATH}.zip")

    try:
        import wandb
        if wandb.run: wandb.finish()
    except ImportError: pass

    vec_env.close()
    return model


# ---------------------------------------------------------------------------
# Hand selection helpers
# ---------------------------------------------------------------------------

def _card_rank_value(card: dict) -> int:
    v = card.get("value",{})
    return RANK_VALUES.get(v.get("rank","2"), 2)

def _card_suit(card: dict) -> str:
    return card.get("value",{}).get("suit","S")

def _card_rank(card: dict) -> str:
    return card.get("value",{}).get("rank","2")

def _is_debuffed(card: dict) -> bool:
    return card.get("debuff", False)

def _filter_active(cards: list, debuff_suit: str = None, face_debuffed: bool = False) -> list:
    """Return indices of cards that are NOT debuffed by the current boss blind."""
    active = []
    for i, card in enumerate(cards):
        if _is_debuffed(card):
            continue
        if debuff_suit and _card_suit(card) == debuff_suit:
            continue
        if face_debuffed and _card_rank(card) in FACE_CARDS:
            continue
        active.append(i)
    return active

def _find_best_hand_from_indices(cards: list, indices: list, joker_aff: str = None):
    """Find best hand from a subset of card indices."""
    if not indices:
        return list(range(min(5, len(cards)))), "high_card"

    sub = [cards[i] for i in indices]
    from collections import Counter as C
    ranks = [_card_rank(c) for c in sub]
    suits = [_card_suit(c) for c in sub]
    rc    = C(ranks)
    sc    = C(suits)
    vals  = [RANK_VALUES.get(r,0) for r in ranks]
    uv    = sorted(set(vals))
    counts = sorted(rc.values(), reverse=True)

    is_flush    = max(sc.values()) >= 5 if sc else False
    is_straight = len(uv) >= 5 and any(uv[i+4]-uv[i]==4 for i in range(len(uv)-4))

    # Straight flush
    if is_flush and is_straight:
        for suit, cnt in sc.items():
            if cnt >= 5:
                suited_idx = [indices[j] for j,c in enumerate(sub) if _card_suit(c)==suit]
                sv = sorted(set(RANK_VALUES.get(_card_rank(cards[i]),0) for i in suited_idx))
                for s in range(len(sv)-4):
                    w = sv[s:s+5]
                    if w[-1]-w[0]==4:
                        play = [i for i in suited_idx if RANK_VALUES.get(_card_rank(cards[i]),0) in w][:5]
                        return play, "straight_flush"

    # Four of a kind
    if counts[0] >= 4:
        rank = rc.most_common(1)[0][0]
        play = [indices[j] for j,c in enumerate(sub) if _card_rank(c)==rank][:4]
        rest = [i for i in indices if i not in play]
        rest.sort(key=lambda i: _card_rank_value(cards[i]), reverse=True)
        return (play+rest[:1])[:5], "four_of_a_kind"

    # Full house
    threes = [r for r,c in rc.items() if c>=3]
    twos   = [r for r,c in rc.items() if c>=2]
    if threes:
        tr = max(threes, key=lambda r: RANK_VALUES.get(r,0))
        pr_opts = [r for r in twos if r!=tr]
        if pr_opts:
            pr = max(pr_opts, key=lambda r: RANK_VALUES.get(r,0))
            play = ([indices[j] for j,c in enumerate(sub) if _card_rank(c)==tr][:3] +
                    [indices[j] for j,c in enumerate(sub) if _card_rank(c)==pr][:2])
            return play, "full_house"

    # Flush
    if is_flush:
        best_suit = sc.most_common(1)[0][0]
        play = [indices[j] for j,c in enumerate(sub) if _card_suit(c)==best_suit]
        play.sort(key=lambda i: _card_rank_value(cards[i]), reverse=True)
        return play[:5], "flush"

    # Straight
    if is_straight:
        seen = {}
        for j, v in enumerate(vals):
            if v not in seen: seen[v] = indices[j]
        for s in range(len(uv)-4):
            w = uv[s:s+5]
            if w[-1]-w[0]==4 and len(w)==5:
                return [seen[v] for v in w], "straight"

    # Joker affinity override — prefer joker-synergy hand
    if joker_aff in ("pair","two_pair","three_of_a_kind"):
        pairs = sorted([r for r,c in rc.items() if c>=2],
                       key=lambda r: RANK_VALUES.get(r,0), reverse=True)
        if joker_aff == "three_of_a_kind" and counts[0] >= 3:
            rank = rc.most_common(1)[0][0]
            play = [indices[j] for j,c in enumerate(sub) if _card_rank(c)==rank][:3]
            return play, "three_of_a_kind"
        if joker_aff == "two_pair" and len(pairs) >= 2:
            play = ([indices[j] for j,c in enumerate(sub) if _card_rank(c)==pairs[0]][:2] +
                    [indices[j] for j,c in enumerate(sub) if _card_rank(c)==pairs[1]][:2])
            return play, "two_pair"
        if joker_aff == "pair" and pairs:
            play = [indices[j] for j,c in enumerate(sub) if _card_rank(c)==pairs[0]][:2]
            return play, "pair"

    # Three of a kind
    if counts[0] >= 3:
        rank = rc.most_common(1)[0][0]
        play = [indices[j] for j,c in enumerate(sub) if _card_rank(c)==rank][:3]
        return play, "three_of_a_kind"

    # Two pair
    pairs = sorted([r for r,c in rc.items() if c>=2],
                   key=lambda r: RANK_VALUES.get(r,0), reverse=True)
    if len(pairs) >= 2:
        play = ([indices[j] for j,c in enumerate(sub) if _card_rank(c)==pairs[0]][:2] +
                [indices[j] for j,c in enumerate(sub) if _card_rank(c)==pairs[1]][:2])
        return play, "two_pair"

    # Pair
    if pairs:
        play = [indices[j] for j,c in enumerate(sub) if _card_rank(c)==pairs[0]][:2]
        return play, "pair"

    # High card — best active card
    best = sorted(indices, key=lambda i: _card_rank_value(cards[i]), reverse=True)
    return best[:5], "high_card"


# ---------------------------------------------------------------------------
# RLBot
# ---------------------------------------------------------------------------

def _build_rl_bot(model_path, port, results_path, deck, stake):
    from base_bot import (BaseBot, get_hand_cards, get_discards_left,
                          get_money, get_shop_cards, get_blind_type, get_jokers)
    from balatro_env import (_encode_obs, HAND_TYPES, PLANET_HAND_MAP)

    try:
        from sb3_contrib import MaskablePPO as ModelCls
    except ImportError:
        from stable_baselines3 import PPO as ModelCls

    tracker = DiagnosticTracker()

    class _RLBot(BaseBot):
        BOT_TYPE   = "rl_bot"
        WANDB_TAGS = ["rl_bot","maskableppo","curriculum","v3"]

        def __init__(self):
            super().__init__(port=port,results_path=results_path,deck=deck,stake=stake)
            if not Path(f"{model_path}.zip").exists():
                raise FileNotFoundError(f"No model at {model_path}.zip")
            self.model = ModelCls.load(model_path)
            logger.info(f"Loaded model from {model_path}.zip")
            self._diag = tracker

            self._hand_levels  = {ht:1 for ht in HAND_TYPES}
            self._hand_counts  = {ht:0 for ht in HAND_TYPES}
            self._dominant     = "pair"
            self._interest     = 0

            self._last_ante          = 1
            self._last_round         = 1
            self._last_chips         = 0
            self._last_chips_needed  = 0
            self._last_hands_left    = 0
            self._last_discards_left = 0
            self._last_jokers        = 0
            self._last_is_boss       = False
            self._current_seed       = ""

        def _on_game_start(self, seed):
            self._diag.reset()
            self._current_seed      = seed
            self._hand_levels       = {ht:1 for ht in HAND_TYPES}
            self._hand_counts       = {ht:0 for ht in HAND_TYPES}
            self._dominant          = "pair"
            self._interest          = 0
            self._last_ante         = 1
            self._last_round        = 1
            self._last_chips        = 0
            self._last_chips_needed = 0
            self._last_hands_left   = 0
            self._last_discards_left= 0
            self._last_jokers       = 0
            self._last_is_boss      = False

        def _on_game_end(self, seed, outcome, state):
            summary = self._diag.finalize_game(
                seed=seed, outcome=outcome,
                final_ante=self._last_ante,
                final_round=self._last_round,
                final_chips=self._last_chips,
                chips_needed=self._last_chips_needed,
                hands_remaining=self._last_hands_left,
                discards_remaining=self._last_discards_left,
                jokers_held=self._last_jokers,
                is_boss=self._last_is_boss,
            )
            logger.info(
                f"[DIAG] {seed} {outcome} ante={summary['final_ante']} "
                f"reason={summary['death_reason']} flush={summary['flush_rate']:.0%} "
                f"jokers={summary['jokers_held']} missed_buys={summary['missed_buys']}"
            )
            try:
                import wandb
                if wandb.run:
                    wandb.log({
                        "diag/death_reason":   summary["death_reason"],
                        "diag/flush_rate":     summary["flush_rate"],
                        "diag/missed_buys":    summary["missed_buys"],
                        "diag/chip_deficit":   summary["chip_deficit"],
                        "diag/jokers_held":    summary["jokers_held"],
                        "diag/is_boss_death":  int(summary["is_boss_blind"]),
                    })
            except Exception: pass

        def _get_obs(self, state):
            obs = _encode_obs(state, self._hand_levels, self._dominant, self._interest)
            obs = obs.astype(np.float32)
            if len(obs) < MOCK_OBS_DIM:
                obs = np.concatenate([obs, np.zeros(MOCK_OBS_DIM-len(obs), dtype=np.float32)])
            return obs

        def _get_action(self, state):
            obs = self._get_obs(state)
            action, _ = self.model.predict(obs, deterministic=True)
            return int(action)

        def _get_boss_blind_name(self, state):
            blinds = state.get("blinds",{})
            boss   = blinds.get("boss",{})
            name   = boss.get("name","") or boss.get("label","") or ""
            name   = name.strip().replace(" ","_")
            if name and not name.startswith("The_"):
                name = "The_" + name
            return name

        def _get_joker_affinity(self, state):
            joker_list = get_jokers(state)
            affinities = Counter()
            for j in joker_list:
                label = j.get("label","")
                name  = label.replace(" ","_")
                aff   = JOKER_AFFINITY.get(name,"none")
                if aff != "none":
                    affinities[aff] += 1
            return affinities.most_common(1)[0][0] if affinities else None

        def select_hand_action(self, state):
            cards         = get_hand_cards(state)
            discards_left = get_discards_left(state)
            round_info    = state.get("round",{})
            hands_left    = round_info.get("hands_left",4)
            ante          = state.get("ante_num",1)
            blind_type    = get_blind_type(state)
            is_boss       = blind_type == "boss"
            chips         = round_info.get("chips",0)
            blinds        = state.get("blinds",{})
            chips_needed  = 0
            for bk in ["small","big","boss"]:
                b = blinds.get(bk,{})
                if b.get("status") in ("SELECT","CURRENT"):
                    chips_needed = b.get("score",0) or b.get("chips",0)
                    break

            self._last_ante          = ante
            self._last_round         = state.get("round_num",1)
            self._last_chips         = chips
            self._last_chips_needed  = chips_needed
            self._last_hands_left    = hands_left
            self._last_discards_left = discards_left
            self._last_jokers        = len(get_jokers(state))
            self._last_is_boss       = is_boss

            boss_name    = self._get_boss_blind_name(state) if is_boss else "none"
            joker_aff    = self._get_joker_affinity(state)
            debuff_suit  = BOSS_SUIT_DEBUFF.get(boss_name) if is_boss else None
            face_debuffed = is_boss and boss_name in BOSS_FACE_DEBUFF

            # --- Boss blind hard overrides ---

            # The Needle: play exactly 1 card — best non-debuffed card
            if is_boss and boss_name in BOSS_PLAY_ONE:
                active = _filter_active(cards, debuff_suit, face_debuffed)
                if not active:
                    active = list(range(len(cards)))
                best = max(active, key=lambda i: _card_rank_value(cards[i]))
                self._last_hand_type = "high_card"
                return "play", [best]

            # The Psychic: must play exactly 5 cards — use best 5 non-debuffed
            if is_boss and boss_name in BOSS_PLAY_FIVE:
                active = _filter_active(cards, debuff_suit, face_debuffed)
                if len(active) < 5:
                    active = list(range(len(cards)))  # fallback to all
                # Find best 5-card hand from active cards
                best_idxs, best_type = _find_best_hand_from_indices(cards, active, joker_aff)
                # Pad to exactly 5 if needed
                if len(best_idxs) < 5:
                    extras = [i for i in active if i not in best_idxs]
                    best_idxs = (best_idxs + extras)[:5]
                self._last_hand_type = best_type
                self._hand_counts[best_type] = self._hand_counts.get(best_type,0) + 1
                self._dominant = max(self._hand_counts, key=self._hand_counts.get)
                self._diag.log_hand(best_type, chips, chips_needed, hands_left, discards_left, is_boss, ante)
                return "play", [int(i) for i in best_idxs[:5]]

            # The Water / The Hook: no discards
            if is_boss and boss_name in BOSS_NO_DISCARD:
                discards_left = 0
            if is_boss and boss_name in BOSS_HOOK:
                discards_left = 0  # Hook discards after play, not before — don't waste discards

            # --- Find best hand from non-debuffed cards ---
            active = _filter_active(cards, debuff_suit, face_debuffed)
            if not active:
                active = list(range(len(cards)))  # fallback if all debuffed

            best_idxs, best_type = _find_best_hand_from_indices(cards, active, joker_aff)
            hand_strength = HAND_TYPES.index(best_type) if best_type in HAND_TYPES else 8

            # --- Discard logic ---
            save_last   = not is_boss and discards_left == 1
            can_discard = discards_left > 0 and not save_last and hands_left > 1

            if can_discard and hand_strength >= 5:  # three_of_a_kind or worse
                # Discard non-active cards first, then lowest ranked active cards
                non_active = [i for i in range(len(cards)) if i not in active]
                if non_active:
                    discard = non_active[:5]
                else:
                    discard = sorted(active, key=lambda i: _card_rank_value(cards[i]))[:3]
                self._last_hand_type = "discard"
                return "discard", [int(i) for i in discard]

            # --- Play best hand ---
            self._hand_counts[best_type] = self._hand_counts.get(best_type,0) + 1
            if self._hand_counts[best_type] % 5 == 0:
                self._hand_levels[best_type] = self._hand_levels.get(best_type,1) + 1
            self._dominant = max(self._hand_counts, key=self._hand_counts.get)

            self._diag.log_hand(best_type, chips, chips_needed, hands_left, discards_left, is_boss, ante)
            self._last_hand_type = best_type
            return "play", [int(i) for i in best_idxs]

        def select_shop_action(self, state):
            action     = min(self._get_action(state), 6)
            money      = get_money(state)
            shop_cards = get_shop_cards(state)
            rc         = state.get("round",{}).get("reroll_cost",5)
            ante       = state.get("ante_num",1)
            joker_count = len(get_jokers(state))

            logger.info(
                f"SHOP: ante={ante} money={money} jokers={joker_count} action={action} "
                f"cards={[(c.get('label','?'),c.get('cost',{}).get('buy','?'),c.get('set','?')) for c in shop_cards]}"
            )

            bought = rerolled = False

            # --- FIX: Deterministic joker buying at ante 1-2 when no jokers ---
            # Mock env failed to learn this — hardcode it
            # Without a joker, scoring is mathematically insufficient at ante 1+ boss
            if joker_count == 0 and ante <= 2:
                for i, card in enumerate(shop_cards):
                    card_set = card.get("set","") or card.get("ability",{}).get("set","")
                    cost_raw = card.get("cost",{})
                    cost     = cost_raw.get("buy",999) if isinstance(cost_raw,dict) else 999
                    if card_set in ("JOKER","Joker") and cost <= money and cost > 0:
                        logger.info(f"SHOP override: buying joker {card.get('label','?')} at ante {ante} (no jokers held)")
                        self._diag.log_shop(action, money, shop_cards, True, False, ante)
                        return [{"action":"buy_card","index":i},{"action":"end_shop"}]

            # --- Deterministic: always buy planet card for dominant hand ---
            for i, card in enumerate(shop_cards):
                card_set = card.get("set","") or card.get("ability",{}).get("set","")
                cost_raw = card.get("cost",{})
                cost     = cost_raw.get("buy",999) if isinstance(cost_raw,dict) else 999
                label    = card.get("label","")
                if card_set in ("PLANET","Planet") and cost <= money:
                    ht = PLANET_HAND_MAP.get(label,"")
                    if ht == self._dominant:
                        self._hand_levels[ht] = self._hand_levels.get(ht,1) + 1
                        self._diag.log_shop(action, money, shop_cards, True, False, ante)
                        return [{"action":"buy_card","index":i},{"action":"end_shop"}]

            # --- RL model controls remaining shop decisions ---
            money_mod5   = money % 5
            near_bracket = money_mod5 >= 4

            if action == 0:
                result = [{"action":"end_shop"}]
            elif action == 6:
                if money >= rc and not near_bracket:
                    rerolled = True
                    result   = [{"action":"reroll"},{"action":"end_shop"}]
                else:
                    result = [{"action":"end_shop"}]
            else:
                bi     = int(action-1)
                result = [{"action":"end_shop"}]
                if bi < len(shop_cards):
                    card     = shop_cards[bi]
                    card_set = card.get("set","") or card.get("ability",{}).get("set","")
                    cost_raw = card.get("cost",{})
                    cost     = cost_raw.get("buy",999) if isinstance(cost_raw,dict) else 999
                    if cost <= money and cost > 0 and card_set not in ("PACK","Booster"):
                        bought = True
                        if card_set in ("PLANET","Planet"):
                            label = card.get("label","")
                            ht    = PLANET_HAND_MAP.get(label, self._dominant)
                            self._hand_levels[ht] = self._hand_levels.get(ht,1) + 1
                        result = [{"action":"buy_card","index":bi},{"action":"end_shop"}]

            self._diag.log_shop(action, money, shop_cards, bought, rerolled, ante)
            return result

        def select_blind_action(self, state):
            action     = min(self._get_action(state), 1)
            blind_type = get_blind_type(state)
            ante       = state.get("ante_num",1)
            if ante <= 2:
                return "select"
            if action == 1 and blind_type != "boss":
                return "skip"
            return "select"

        def select_pack_action(self, state):
            return {"action":"skip","cards":[]}

    return _RLBot(), tracker


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",         action="store_true")
    parser.add_argument("--run",           action="store_true")
    parser.add_argument("--mock",          action="store_true")
    parser.add_argument("--timesteps",     type=int,  default=120_000_000)
    parser.add_argument("--n-envs",        type=int,  default=N_ENVS)
    parser.add_argument("--seeds",         nargs="+", default=BENCHMARK_SEEDS)
    parser.add_argument("--runs-per-seed", type=int,  default=RUNS_PER_SEED)
    parser.add_argument("--results",       default="results.csv")
    parser.add_argument("--port",          type=int,  default=12346)
    parser.add_argument("--deck",          default="RED")
    parser.add_argument("--stake",         default="WHITE")
    parser.add_argument("--model-path",    default=str(MODEL_PATH))
    args = parser.parse_args()

    if args.train:
        train(timesteps=args.timesteps, seed=args.seeds[0],
              port=args.port, use_mock=args.mock, n_envs=args.n_envs)

    if args.run:
        bot, tracker = _build_rl_bot(
            model_path=args.model_path, port=args.port,
            results_path=args.results, deck=args.deck, stake=args.stake,
        )
        if not bot.client.health():
            print("ERROR: Cannot connect to Balatro."); exit(1)

        print(f"Running RLBot v3 on {len(args.seeds)} seeds x {args.runs_per_seed} runs")
        results   = bot.run_experiment(seeds=args.seeds, runs_per_seed=args.runs_per_seed)
        completed = [r for r in results if r["outcome"] in ("won","lost")]
        if completed:
            print(f"\nSummary:")
            print(f"  Completed: {len(completed)}/{len(results)}")
            print(f"  Wins:      {sum(1 for r in completed if r['outcome']=='won')}")
            print(f"  Avg ante:  {sum(r['final_ante'] for r in completed)/len(completed):.2f}")
            print(f"  Avg round: {sum(r['final_round'] for r in completed)/len(completed):.2f}")
        tracker.print_full_report()