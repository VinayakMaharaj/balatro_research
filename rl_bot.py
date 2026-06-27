"""
rl_bot.py
MaskablePPO-based RL agent for Balatro.
Final training version — curriculum learning, checkpoints, deep network.

Training (48-60 hour run):
    python rl_bot.py --train --mock --timesteps 600000000

Evaluation:
    python rl_bot.py --run --runs-per-seed 1

Curriculum schedule:
    0-40M steps:   ante 1 only
    40-60M steps:  ante 1-2
    60-80M steps:  ante 1-3
    80-100M steps: ante 1-4
    100M+ steps:   full game

Checkpoints saved every 10M steps to rl_model/checkpoints/
"""

import os
import time
import logging
import argparse
import json
from collections import Counter, defaultdict
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR        = Path("rl_model")
MODEL_PATH       = MODEL_DIR / "ppo_balatro"
CHECKPOINT_DIR   = MODEL_DIR / "checkpoints"
N_ENVS           = 16
MOCK_OBS_DIM     = 72

BENCHMARK_SEEDS  = [f"SEED{str(i).zfill(3)}" for i in range(1, 101)]
RUNS_PER_SEED    = 1

# Curriculum schedule: (min_steps, max_ante)
CURRICULUM = [
    (0,          1),   # 0-40M: master ante 1 including boss blind
    (40_000_000, 2),   # 40-60M: add ante 2
    (60_000_000, 3),   # 60-80M: add ante 3
    (80_000_000, 4),
    (100_000_000,5),
    (120_000_000,6),
    (140_000_000,7),
    (160_000_000,8),
]


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
    """
    Tracks per-game and per-hand decisions to classify death causes
    and identify specific training gaps.
    """

    def __init__(self):
        self.reset()
        self.all_games = []

    def reset(self):
        """Call at start of each game."""
        self.hand_decisions     = []   # list of dicts per hand played
        self.shop_decisions     = []   # list of dicts per shop visit
        self.blind_decisions    = []   # list of dicts per blind
        self.chips_per_hand     = []   # chips scored each hand
        self.hand_types_played  = Counter()
        self.flush_available_not_played = 0
        self.missed_buys        = 0    # had money, skipped good joker
        self.total_shop_visits  = 0
        self.game_start_time    = time.time()

    def log_hand(self, cards_available, hand_type, card_indices,
                 chips_scored, chips_needed, hands_left, discards_left,
                 flush_was_available, straight_was_available, is_boss_blind, ante, round_num):
        self.hand_decisions.append({
            "ante":                   ante,
            "round":                  round_num,
            "hand_type":              hand_type,
            "chips_scored":           chips_scored,
            "chips_needed":           chips_needed,
            "hands_left_after":       hands_left,
            "discards_left":          discards_left,
            "flush_was_available":    flush_was_available,
            "straight_was_available": straight_was_available,
            "is_boss_blind":          is_boss_blind,
            "card_count":             len(cards_available),
        })
        self.hand_types_played[hand_type] += 1
        if flush_was_available and hand_type not in ("flush", "straight_flush", "royal_flush"):
            self.flush_available_not_played += 1

    def log_shop(self, action, money, shop_cards, bought, rerolled, ante):
        affordable = [c for c in shop_cards
                      if c.get("cost", {}).get("buy", 999) <= money
                      and c.get("cost", {}).get("buy", 0) > 0]
        self.total_shop_visits += 1
        skipped_affordable = len(affordable) > 0 and not bought and not rerolled
        if skipped_affordable:
            self.missed_buys += 1
        self.shop_decisions.append({
            "ante":               ante,
            "action":             action,
            "money":              money,
            "affordable_count":   len(affordable),
            "bought":             bought,
            "rerolled":           rerolled,
            "skipped_affordable": skipped_affordable,
        })

    def log_blind(self, ante, blind_type, decision, is_boss):
        self.blind_decisions.append({
            "ante":       ante,
            "blind_type": blind_type,
            "decision":   decision,
            "is_boss":    is_boss,
        })

    def classify_death(self, final_ante, final_round, final_chips,
                       chips_needed, hands_remaining, discards_remaining,
                       jokers_held, is_boss_blind):
        """
        Returns a string death reason tag for analysis.
        Priority order: most specific first.
        """
        deficit = (chips_needed - final_chips) if chips_needed else 0
        pct     = final_chips / chips_needed if chips_needed else 1.0

        if is_boss_blind and pct < 0.4:
            return "boss_blind_crushed"       # Boss blind wiped out scoring ability
        if is_boss_blind and pct < 0.8:
            return "boss_blind_underscored"   # Close but not enough vs boss
        if hands_remaining == 0 and pct < 1.0:
            return "ran_out_hands"            # Used all hands, still short
        if self.flush_available_not_played > 2:
            return "suboptimal_hand_selection" # Bot missed flushes repeatedly
        if self.missed_buys > 2:
            return "economy_failure"          # Had money, didn't buy jokers
        if jokers_held == 0 and final_ante >= 2:
            return "no_jokers_late"           # Survived without any joker economy
        if pct < 0.5:
            return "severe_chip_deficit"      # Far short regardless of cause
        if pct < 0.8:
            return "marginal_chip_deficit"    # Close but not close enough
        return "unknown"

    def finalize_game(self, seed, outcome, final_ante, final_round,
                      final_chips, chips_needed, hands_remaining,
                      discards_remaining, jokers_held, is_boss_blind):
        """Call at end of each game. Returns the game summary dict."""
        death_reason = self.classify_death(
            final_ante, final_round, final_chips, chips_needed,
            hands_remaining, discards_remaining, jokers_held, is_boss_blind
        )

        # Sim-to-real gap proxy: how often did hand type match what mock env
        # would predict (flush-first policy)? High flush% = good alignment.
        total_hands  = sum(self.hand_types_played.values())
        flush_rate   = (self.hand_types_played.get("flush", 0) +
                        self.hand_types_played.get("straight_flush", 0) +
                        self.hand_types_played.get("royal_flush", 0)) / max(total_hands, 1)

        game_summary = {
            "seed":                        seed,
            "outcome":                     outcome,
            "final_ante":                  final_ante,
            "final_round":                 final_round,
            "final_chips":                 final_chips,
            "chips_needed":                chips_needed,
            "chip_deficit":                max(0, (chips_needed or 0) - (final_chips or 0)),
            "hands_remaining":             hands_remaining,
            "discards_remaining":          discards_remaining,
            "jokers_held":                 jokers_held,
            "is_boss_blind":               is_boss_blind,
            "death_reason":                death_reason,
            "hand_types":                  dict(self.hand_types_played),
            "flush_rate":                  round(flush_rate, 3),
            "flush_available_not_played":  self.flush_available_not_played,
            "missed_buys":                 self.missed_buys,
            "total_shop_visits":           self.total_shop_visits,
            "duration_seconds":            round(time.time() - self.game_start_time, 1),
        }

        self.all_games.append(game_summary)
        return game_summary

    def print_full_report(self):
        """Print aggregate diagnostic report across all games."""
        if not self.all_games:
            return

        games      = self.all_games
        completed  = [g for g in games if g["outcome"] in ("won", "lost")]
        n          = len(completed)
        if n == 0:
            return

        print("\n" + "="*60)
        print("DIAGNOSTIC REPORT")
        print("="*60)

        # --- Death reason breakdown ---
        death_counts = Counter(g["death_reason"] for g in completed if g["outcome"] == "lost")
        print(f"\nDEATH REASONS (n={sum(death_counts.values())}):")
        for reason, count in death_counts.most_common():
            pct = count / n * 100
            print(f"  {reason:<35} {count:>3}  ({pct:.1f}%)")

        # --- Ante distribution ---
        ante_counts = Counter(g["final_ante"] for g in completed)
        print(f"\nDEATH BY ANTE:")
        for ante in sorted(ante_counts):
            count = ante_counts[ante]
            print(f"  Ante {ante}: {count:>3} games ({count/n*100:.1f}%)")

        # --- Hand selection ---
        total_hand_types = Counter()
        for g in completed:
            for ht, cnt in g["hand_types"].items():
                total_hand_types[ht] += cnt
        total_hands = sum(total_hand_types.values())
        print(f"\nHAND TYPE DISTRIBUTION (total hands={total_hands}):")
        for ht, cnt in total_hand_types.most_common():
            print(f"  {ht:<25} {cnt:>5}  ({cnt/max(total_hands,1)*100:.1f}%)")

        # --- Sim-to-real gap proxy ---
        avg_flush_rate = sum(g["flush_rate"] for g in completed) / n
        avg_missed_flush = sum(g["flush_available_not_played"] for g in completed) / n
        print(f"\nSIM-TO-REAL GAP PROXY:")
        print(f"  Avg flush/straight_flush rate:   {avg_flush_rate*100:.1f}%")
        print(f"  Avg flushes missed per game:     {avg_missed_flush:.2f}")
        print(f"  (High missed flushes = hand eval divergence from mock env)")

        # --- Economy ---
        avg_missed_buys = sum(g["missed_buys"] for g in completed) / n
        avg_jokers      = sum(g["jokers_held"] for g in completed) / n
        no_joker_games  = sum(1 for g in completed if g["jokers_held"] == 0)
        print(f"\nECONOMY:")
        print(f"  Avg missed buy opportunities:    {avg_missed_buys:.2f}")
        print(f"  Avg jokers held at death:        {avg_jokers:.2f}")
        print(f"  Games with 0 jokers at death:    {no_joker_games} ({no_joker_games/n*100:.1f}%)")

        # --- Boss blind specifically ---
        boss_deaths = [g for g in completed
                       if g["outcome"] == "lost" and g.get("is_boss_blind")]
        print(f"\nBOSS BLIND DEATHS: {len(boss_deaths)}/{sum(1 for g in completed if g['outcome']=='lost')} deaths at boss blind")
        if boss_deaths:
            avg_deficit = sum(g["chip_deficit"] for g in boss_deaths) / len(boss_deaths)
            avg_hands_left = sum(g["hands_remaining"] for g in boss_deaths) / len(boss_deaths)
            print(f"  Avg chip deficit at boss blind:  {avg_deficit:.0f}")
            print(f"  Avg hands remaining when dead:   {avg_hands_left:.2f}")

        # --- Targeted training recommendations ---
        print(f"\nTARGETED TRAINING RECOMMENDATIONS:")
        top_reason = death_counts.most_common(1)[0][0] if death_counts else None
        if top_reason == "boss_blind_crushed":
            print("  → Boss blind is crushing the agent. Increase boss blind penalty further.")
            print("    Consider adding boss blind modifier to obs space so agent can adapt.")
        if top_reason in ("boss_blind_crushed", "boss_blind_underscored"):
            print("  → Extend ante 1 curriculum even longer (try 60M steps).")
            print("    Agent needs more exposure to boss blind scenarios during training.")
        if top_reason == "suboptimal_hand_selection":
            print("  → Agent is ignoring available flushes. Option B hand selection may not")
            print("    be engaging — check if cards available in real game differ from mock.")
        if top_reason == "economy_failure":
            print("  → Agent skipping affordable jokers. Shop reward signal too weak.")
            print("    Consider adding immediate reward for buying jokers in mock env.")
        if top_reason == "ran_out_hands":
            print("  → Agent exhausting hand budget. Discard conservation not working.")
            print("    Increase reward for clearing blinds with hands remaining.")
        if avg_flush_rate < 0.3:
            print("  → Flush rate very low (<30%). Real game card distribution differs")
            print("    significantly from mock env. Sim-to-real gap is likely a ceiling.")
        if no_joker_games / n > 0.5:
            print("  → >50% of games end with 0 jokers. Shop policy is the primary failure.")

        print("="*60)

        # Save full diagnostic JSON
        diag_path = Path("diagnostics.json")
        with open(diag_path, "w") as f:
            json.dump(games, f, indent=2)
        print(f"\nFull per-game diagnostics saved to: {diag_path}")


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def _make_callbacks(checkpoint_dir, n_envs, vec_env):
    from stable_baselines3.common.callbacks import BaseCallback

    class CurriculumCallback(BaseCallback):
        """Updates max_ante on all envs as training progresses."""
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
                    try:
                        env_fn.env.set_max_ante(new_ante)
                    except Exception:
                        try:
                            env_fn.set_max_ante(new_ante)
                        except Exception:
                            pass
                logger.info(f"Curriculum advanced to max_ante={new_ante} at {self.num_timesteps:,} steps")
                try:
                    import wandb
                    if wandb.run:
                        wandb.log({"curriculum/max_ante": new_ante,
                                   "curriculum/steps": self.num_timesteps})
                except ImportError:
                    pass

            try:
                import wandb
                if wandb.run:
                    for info in self.locals.get("infos", []):
                        if "episode" in info:
                            ep = info["episode"]
                            self._ep += 1
                            wandb.log({
                                "train/reward":          ep["r"],
                                "train/length":          ep["l"],
                                "train/ante":            info.get("ante", 0),
                                "train/won":             int(info.get("won", False)),
                                "train/episodes":        self._ep,
                                "train/curriculum_ante": self._current_ante,
                            })
            except ImportError:
                pass
            return True

    class CheckpointCallback(BaseCallback):
        """Saves model every checkpoint_freq steps."""
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
                logger.info(f"Checkpoint saved: {path}.zip")
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

def train(timesteps=600_000_000, seed="AAAAAAA", port=12346,
          use_mock=True, n_envs=N_ENVS):

    if not use_mock:
        from stable_baselines3 import PPO
        from balatro_env import BalatroEnv
        from stable_baselines3.common.env_checker import check_env
        env = BalatroEnv(port=port, seed=seed)
        check_env(env, warn=True)
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4,
                    n_steps=2048, batch_size=512, n_epochs=10,
                    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.05,
                    policy_kwargs=dict(net_arch=[256,256,128]))
        model.learn(total_timesteps=timesteps, progress_bar=True)
        MODEL_DIR.mkdir(exist_ok=True)
        model.save(str(MODEL_PATH))
        env.close()
        return model

    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    logger.info(f"Training MaskablePPO | {timesteps:,} steps | {n_envs} envs | curriculum learning")
    logger.info(f"Curriculum: {CURRICULUM}")

    vec_env = DummyVecEnv([_make_env for _ in range(n_envs)])

    try:
        import wandb
        wandb.init(
            project="balatro-research",
            name=f"rl_curriculum_{timesteps//1_000_000}Msteps_{n_envs}envs",
            config={
                "algorithm":   "MaskablePPO",
                "timesteps":   timesteps,
                "n_envs":      n_envs,
                "obs_dim":     MOCK_OBS_DIM,
                "net_arch":    [256,256,128],
                "n_steps":     2048,
                "batch_size":  512,
                "ent_coef":    0.05,
                "curriculum":  CURRICULUM,
            },
            tags=["rl_bot","maskableppo","curriculum","training"],
        )
        logger.info("W&B initialized")
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
        policy_kwargs=dict(net_arch=[256, 256, 128]),
    )

    callbacks = _make_callbacks(CHECKPOINT_DIR, n_envs, vec_env)
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=True)

    MODEL_DIR.mkdir(exist_ok=True)
    model.save(str(MODEL_PATH))
    logger.info(f"Model saved to {MODEL_PATH}.zip")

    try:
        import wandb
        if wandb.run: wandb.finish()
    except ImportError:
        pass

    vec_env.close()
    return model


# ---------------------------------------------------------------------------
# RLBot evaluation
# ---------------------------------------------------------------------------

def _build_rl_bot(model_path, port, results_path, deck, stake):
    from base_bot import BaseBot, get_hand_cards, get_discards_left, get_money, get_shop_cards, get_blind_type
    from balatro_env import _encode_obs, _best_pair_hand, _find_flush, _find_straight, _worst_cards

    try:
        from sb3_contrib import MaskablePPO as ModelCls
    except ImportError:
        from stable_baselines3 import PPO as ModelCls

    tracker = DiagnosticTracker()

    class _RLBot(BaseBot):
        BOT_TYPE   = "rl_bot"
        WANDB_TAGS = ["rl_bot", "maskableppo", "curriculum"]

        def __init__(self):
            super().__init__(port=port, results_path=results_path, deck=deck, stake=stake)
            if not Path(f"{model_path}.zip").exists():
                raise FileNotFoundError(f"No model at {model_path}.zip")
            self.model = ModelCls.load(model_path)
            logger.info(f"Loaded model from {model_path}.zip")
            self._diag = tracker

        def _on_game_start(self, seed):
            """Reset diagnostics at start of each game."""
            self._diag.reset()
            self._current_seed    = seed
            self._last_chips      = 0
            self._last_chips_needed = 0
            self._last_hands_left = 0
            self._last_discards_left = 0
            self._last_jokers     = 0
            self._last_is_boss    = False
            self._last_ante       = 1
            self._last_round      = 1

        def _on_game_end(self, seed, outcome, state):
            """Finalize diagnostics at end of each game."""
            summary = self._diag.finalize_game(
                seed=seed,
                outcome=outcome,
                final_ante=self._last_ante,
                final_round=self._last_round,
                final_chips=self._last_chips,
                chips_needed=self._last_chips_needed,
                hands_remaining=self._last_hands_left,
                discards_remaining=self._last_discards_left,
                jokers_held=self._last_jokers,
                is_boss_blind=self._last_is_boss,
            )
            logger.info(
                f"[DIAG] seed={seed} outcome={outcome} ante={summary['final_ante']} "
                f"death_reason={summary['death_reason']} flush_rate={summary['flush_rate']:.0%} "
                f"missed_buys={summary['missed_buys']} jokers={summary['jokers_held']}"
            )
            # Log per-game diagnostics to W&B
            try:
                import wandb
                if wandb.run:
                    wandb.log({
                        "diag/death_reason":               summary["death_reason"],
                        "diag/flush_rate":                 summary["flush_rate"],
                        "diag/missed_buys":                summary["missed_buys"],
                        "diag/flush_available_not_played": summary["flush_available_not_played"],
                        "diag/chip_deficit":               summary["chip_deficit"],
                        "diag/jokers_held":                summary["jokers_held"],
                        "diag/is_boss_blind_death":        int(summary["is_boss_blind"]),
                    })
            except Exception:
                pass

        def _get_action(self, state):
            obs = _encode_obs(state).astype(np.float32)
            if len(obs) < MOCK_OBS_DIM:
                obs = np.concatenate([obs, np.zeros(MOCK_OBS_DIM - len(obs), dtype=np.float32)])
            action, _ = self.model.predict(obs, deterministic=True)
            return int(action)

        def _extract_state_info(self, state):
            """Pull key fields for diagnostic tracking."""
            round_info  = state.get("round", {})
            blind_info  = state.get("blind", {})
            jokers      = state.get("jokers", [])
            ante        = state.get("ante_num", 1)
            round_num   = state.get("round_num", 1)
            hands_left  = round_info.get("hands_left", 4)
            discards    = round_info.get("discards_left", 3)
            chips       = round_info.get("chips", 0)
            needed      = blind_info.get("chips", 0) or round_info.get("blind_chips", 0)
            blind_type  = get_blind_type(state)
            is_boss     = blind_type == "boss"
            return ante, round_num, hands_left, discards, chips, needed, len(jokers), is_boss

        def select_hand_action(self, state):
            cards         = get_hand_cards(state)
            discards_left = get_discards_left(state)
            hands_left    = state.get("round", {}).get("hands_left", 4)
            ante, round_num, _, _, chips, needed, joker_count, is_boss = self._extract_state_info(state)

            # Update running state for game-end summary
            self._last_ante          = ante
            self._last_round         = round_num
            self._last_chips         = chips
            self._last_chips_needed  = needed
            self._last_hands_left    = hands_left
            self._last_discards_left = discards_left
            self._last_jokers        = joker_count
            self._last_is_boss       = is_boss

            flush_idxs    = _find_flush(cards)
            straight_idxs = _find_straight(cards)
            flush_available    = flush_idxs is not None and len(flush_idxs) > 0
            straight_available = straight_idxs is not None and len(straight_idxs) > 0

            if flush_available:
                hand_type = "flush"
                chosen    = [int(i) for i in flush_idxs]
                action    = "play"
            elif straight_available:
                hand_type = "straight"
                chosen    = [int(i) for i in straight_idxs]
                action    = "play"
            elif discards_left > 0 and hands_left > 1:
                hand_type = "discard"
                chosen    = [int(i) for i in _worst_cards(cards, 3)]
                action    = "discard"
            else:
                idxs = _best_pair_hand(cards)
                idxs = [int(i) for i in idxs if 0 <= i < len(cards)]
                if not idxs:
                    idxs = list(range(min(5, len(cards))))
                hand_type = "pair_or_lower"
                chosen    = idxs
                action    = "play"

            # Log this hand decision
            self._diag.log_hand(
                cards_available       = cards,
                hand_type             = hand_type,
                card_indices          = chosen,
                chips_scored          = chips,
                chips_needed          = needed,
                hands_left            = hands_left,
                discards_left         = discards_left,
                flush_was_available   = flush_available,
                straight_was_available= straight_available,
                is_boss_blind         = is_boss,
                ante                  = ante,
                round_num             = round_num,
            )

            self._last_hand_type = "rl_play"
            return action, chosen

        def select_shop_action(self, state):
            action     = min(self._get_action(state), 6)
            money      = get_money(state)
            shop_cards = get_shop_cards(state)
            rc         = state.get("round", {}).get("reroll_cost", 5)
            ante       = state.get("ante_num", 1)

            logger.info(f"SHOP: action={action}, money={money}, cards={[(c.get('label','?'), c.get('cost',{}).get('buy','?'), c.get('set','?')) for c in shop_cards]}")

            bought   = False
            rerolled = False

            if action == 0:
                result = [{"action": "end_shop"}]
            elif action == 6:
                if money >= rc:
                    rerolled = True
                    result = [{"action": "reroll"}, {"action": "end_shop"}]
                else:
                    result = [{"action": "end_shop"}]
            else:
                bi   = int(action - 1)
                result = [{"action": "end_shop"}]
                if bi < len(shop_cards):
                    cost = shop_cards[bi].get("cost", {}).get("buy", 999)
                    if cost <= money and cost > 0:
                        bought = True
                        result = [{"action": "buy_card", "index": bi}, {"action": "end_shop"}]

            self._diag.log_shop(action, money, shop_cards, bought, rerolled, ante)
            return result

        def select_blind_action(self, state):
            action     = min(self._get_action(state), 1)
            blind_type = get_blind_type(state)
            ante       = state.get("ante_num", 1)
            is_boss    = blind_type == "boss"

            # Never skip at ante 1-2
            if ante <= 2:
                decision = "select"
            elif action == 1 and blind_type != "boss":
                decision = "skip"
            else:
                decision = "select"

            self._diag.log_blind(ante, blind_type, decision, is_boss)
            return decision

        def select_pack_action(self, state):
            return {"action": "skip", "cards": []}

    return _RLBot(), tracker


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",         action="store_true")
    parser.add_argument("--run",           action="store_true")
    parser.add_argument("--mock",          action="store_true")
    parser.add_argument("--timesteps",     type=int,  default=600_000_000)
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

        print(f"Running RLBot on {len(args.seeds)} seeds x {args.runs_per_seed} runs")
        results   = bot.run_experiment(seeds=args.seeds, runs_per_seed=args.runs_per_seed)
        completed = [r for r in results if r["outcome"] in ("won","lost")]
        if completed:
            print(f"\nSummary:")
            print(f"  Completed: {len(completed)}/{len(results)}")
            print(f"  Wins:      {sum(1 for r in completed if r['outcome']=='won')}")
            print(f"  Avg ante:  {sum(r['final_ante'] for r in completed)/len(completed):.2f}")
            print(f"  Avg round: {sum(r['final_round'] for r in completed)/len(completed):.2f}")

        # Print full diagnostic report
        tracker.print_full_report()