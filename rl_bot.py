"""
rl_bot.py
MaskablePPO-based RL agent for Balatro.
Final training version — curriculum learning, checkpoints, deep network.

Training (48-60 hour run):
    python rl_bot.py --train --mock --timesteps 600000000

Evaluation:
    python rl_bot.py --run --runs-per-seed 1

Curriculum schedule:
    0-25M steps:   ante 1 only
    25-50M steps:  ante 1-2
    50-100M steps: ante 1-3
    100-200M steps: ante 1-4
    200M+ steps:   full game

Checkpoints saved every 10M steps to rl_model/checkpoints/
"""

import os
import time
import logging
import argparse
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
    (0,          1),
    (25_000_000, 2),
    (50_000_000, 3),
    (100_000_000,4),
    (200_000_000,5),
    (300_000_000,6),
    (400_000_000,7),
    (500_000_000,8),
]


def _get_curriculum_ante(total_steps):
    max_ante = 1
    for min_steps, ante in CURRICULUM:
        if total_steps >= min_steps:
            max_ante = ante
    return max_ante


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
                # Update all envs
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

            # W&B episode logging
            try:
                import wandb
                if wandb.run:
                    for info in self.locals.get("infos", []):
                        if "episode" in info:
                            ep = info["episode"]
                            self._ep += 1
                            wandb.log({
                                "train/reward":        ep["r"],
                                "train/length":        ep["l"],
                                "train/ante":          info.get("ante", 0),
                                "train/won":           int(info.get("won", False)),
                                "train/episodes":      self._ep,
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
    env = BalatroMockEnv(max_ante=1)  # starts at stage 0
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

    class _RLBot(BaseBot):
        BOT_TYPE   = "rl_bot"
        WANDB_TAGS = ["rl_bot", "maskableppo", "curriculum"]

        def __init__(self):
            super().__init__(port=port, results_path=results_path, deck=deck, stake=stake)
            if not Path(f"{model_path}.zip").exists():
                raise FileNotFoundError(f"No model at {model_path}.zip")
            self.model = ModelCls.load(model_path)
            logger.info(f"Loaded model from {model_path}.zip")

        def _get_action(self, state):
            obs = _encode_obs(state).astype(np.float32)
            if len(obs) < MOCK_OBS_DIM:
                obs = np.concatenate([obs, np.zeros(MOCK_OBS_DIM - len(obs), dtype=np.float32)])
            action, _ = self.model.predict(obs, deterministic=True)
            return int(action)

        def select_hand_action(self, state):
            """
            Matches Option B training: flush->straight->discard->pairs.
            Model controls shop/blind — hand selection is deterministic.
            """
            cards         = get_hand_cards(state)
            discards_left = get_discards_left(state)
            hands_left    = state.get("round", {}).get("hands_left", 4)
            flush_idxs    = _find_flush(cards)
            if flush_idxs:
                self._last_hand_type = "rl_play"
                return "play", [int(i) for i in flush_idxs]
            straight_idxs = _find_straight(cards)
            if straight_idxs:
                self._last_hand_type = "rl_play"
                return "play", [int(i) for i in straight_idxs]
            if discards_left > 0 and hands_left > 1:
                return "discard", [int(i) for i in _worst_cards(cards, 3)]
            idxs = _best_pair_hand(cards)
            idxs = [int(i) for i in idxs if 0 <= i < len(cards)]
            if not idxs: idxs = list(range(min(5, len(cards))))
            self._last_hand_type = "rl_play"
            return "play", idxs

        def select_shop_action(self, state):
            action     = min(self._get_action(state), 6)
            money      = get_money(state)
            shop_cards = get_shop_cards(state)
            rc         = state.get("round", {}).get("reroll_cost", 5)
            logger.info(f"SHOP: action={action}, money={money}, cards={[(c.get('label','?'), c.get('cost',{}).get('buy','?'), c.get('set','?')) for c in shop_cards]}")
            if action == 0: return [{"action": "end_shop"}]
            if action == 6:
                if money >= rc: return [{"action": "reroll"}, {"action": "end_shop"}]
                return [{"action": "end_shop"}]
            bi = int(action - 1)
            if bi < len(shop_cards):
                cost = shop_cards[bi].get("cost", {}).get("buy", 999)
                if cost <= money and cost > 0:
                    return [{"action": "buy_card", "index": bi}, {"action": "end_shop"}]
            return [{"action": "end_shop"}]

        def select_blind_action(self, state):
            action     = min(self._get_action(state), 1)
            blind_type = get_blind_type(state)
            ante       = state.get("ante_num", 1)
            if ante <= 1: return "select"
            if action == 1 and blind_type != "boss": return "skip"
            return "select"

        def select_pack_action(self, state):
            return {"action": "skip", "cards": []}

    return _RLBot()


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
        bot = _build_rl_bot(
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