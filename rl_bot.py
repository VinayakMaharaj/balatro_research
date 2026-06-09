"""
rl_bot.py
MaskablePPO-based RL agent for Balatro using sb3_contrib + Gymnasium.

Training (9-hour overnight run):
    python rl_bot.py --train --mock --timesteps 50000000

Evaluation:
    python rl_bot.py --run --runs-per-seed 1

Install:
    pip install gymnasium stable-baselines3 sb3-contrib --break-system-packages
"""

import os
import time
import logging
import argparse
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR  = Path("rl_model")
MODEL_PATH = MODEL_DIR / "ppo_balatro"
N_ENVS     = 16

BENCHMARK_SEEDS = [f"SEED{str(i).zfill(3)}" for i in range(1, 101)]
RUNS_PER_SEED   = 1

# OBS_DIM from mock env (72). Real env still has 38; we pad to 72 at eval time.
MOCK_OBS_DIM = 72


# ---------------------------------------------------------------------------
# W&B callback
# ---------------------------------------------------------------------------

class BalatroTrainingCallback:
    """Thin wrapper — avoids importing BaseCallback at module level."""
    def __new__(cls):
        try:
            from stable_baselines3.common.callbacks import BaseCallback

            class _CB(BaseCallback):
                def __init__(self):
                    super().__init__()
                    self._ep = 0
                def _on_step(self):
                    try:
                        import wandb
                        if wandb.run is None: return True
                        for info in self.locals.get("infos", []):
                            if "episode" in info:
                                ep = info["episode"]
                                self._ep += 1
                                wandb.log({
                                    "train/reward":   ep["r"],
                                    "train/length":   ep["l"],
                                    "train/ante":     info.get("ante", 0),
                                    "train/won":      int(info.get("won", False)),
                                    "train/episodes": self._ep,
                                })
                    except ImportError:
                        pass
                    return True

            return _CB()
        except ImportError:
            return None


# ---------------------------------------------------------------------------
# Env factory — must be a top-level importable callable for SubprocVecEnv.
# We use DummyVecEnv (FIX #1: Windows-safe) with an inline factory.
# ---------------------------------------------------------------------------

def _make_env():
    """Returns a wrapped BalatroMockEnv. Safe to call in DummyVecEnv."""
    from balatro_mock_env import BalatroMockEnv
    from sb3_contrib.common.wrappers import ActionMasker
    env = BalatroMockEnv()
    env = ActionMasker(env, lambda e: e.action_masks())
    return env


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(timesteps=50_000_000, seed="AAAAAAA", port=12346,
          use_mock=True, n_envs=N_ENVS):

    if not use_mock:
        from stable_baselines3 import PPO
        from balatro_env import BalatroEnv
        from stable_baselines3.common.env_checker import check_env
        env = BalatroEnv(port=port, seed=seed)
        check_env(env, warn=True)
        logger.info(f"Training PPO on REAL env for {timesteps} steps (slow)")
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4,
                    n_steps=512, batch_size=64, n_epochs=10,
                    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01)
        model.learn(total_timesteps=timesteps, progress_bar=True)
        MODEL_DIR.mkdir(exist_ok=True)
        model.save(str(MODEL_PATH))
        env.close()
        return model

    from sb3_contrib import MaskablePPO
    # FIX #1: DummyVecEnv is Windows-safe; SubprocVecEnv requires picklable top-level factories
    from stable_baselines3.common.vec_env import DummyVecEnv
    logger.info(f"Training MaskablePPO on MOCK env | {timesteps} steps | {n_envs} envs (DummyVecEnv)")

    env = DummyVecEnv([_make_env for _ in range(n_envs)])

    try:
        import wandb
        wandb.init(
            project="balatro-research",
            name=f"rl_maskableppo_mock_{timesteps}steps_{n_envs}envs",
            config={"algorithm":"MaskablePPO","timesteps":timesteps,
                    "n_envs":n_envs,"obs_dim":MOCK_OBS_DIM,"env":"mock"},
            tags=["rl_bot","maskableppo","training"],
        )
    except Exception as e:
        logger.info(f"W&B skipped: {e}")

    model = MaskablePPO(
        "MlpPolicy", env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=256,    # 16 envs * 512 steps = 8192 rollout; 256 minibatch
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=str(MODEL_DIR / "tb_logs"),
    )

    cb = BalatroTrainingCallback()
    model.learn(total_timesteps=timesteps, callback=cb, progress_bar=True)

    MODEL_DIR.mkdir(exist_ok=True)
    model.save(str(MODEL_PATH))
    logger.info(f"Model saved to {MODEL_PATH}.zip")

    try:
        import wandb
        if wandb.run: wandb.finish()
    except ImportError:
        pass

    env.close()
    return model


# ---------------------------------------------------------------------------
# RLBot — real-env evaluation
# ---------------------------------------------------------------------------

class RLBot:
    """Thin eval wrapper; inherits BaseBot for CSV + W&B logging."""

    def __new__(cls, model_path=str(MODEL_PATH), port=12346, **kwargs):
        # Lazy import so rl_bot.py can be imported without sb3_contrib installed
        try:
            from sb3_contrib import MaskablePPO as _MPPO
            instance = object.__new__(cls)
            instance._mppo_cls = _MPPO
        except ImportError:
            from stable_baselines3 import PPO as _PPO
            instance = object.__new__(cls)
            instance._mppo_cls = _PPO
        return instance

    def __init__(self, model_path=str(MODEL_PATH), port=12346, **kwargs):
        from base_bot import BaseBot
        # Manually set up BaseBot attributes without calling __init__ twice
        # (we need to call it properly)
        super_cls = type("_RLBotBase", (self._get_base(),), {
            "BOT_TYPE":   "rl_bot",
            "WANDB_TAGS": ["rl_bot","maskableppo"],
        })
        self.__class__.__bases__ = (super_cls,)
        super_cls.__init__(self, port=port, **kwargs)

        if not Path(f"{model_path}.zip").exists():
            raise FileNotFoundError(f"No model at {model_path}.zip — run --train first.")
        self.model = self._mppo_cls.load(model_path)
        logger.info(f"Loaded model from {model_path}.zip")

    def _get_base(self):
        from base_bot import BaseBot
        return BaseBot

    def _get_action(self, state):
        from balatro_env import _encode_obs
        obs = _encode_obs(state).astype(np.float32)
        # FIX #2: pad real-env obs (38-dim) to mock obs (72-dim)
        if len(obs) < MOCK_OBS_DIM:
            obs = np.concatenate([obs, np.zeros(MOCK_OBS_DIM - len(obs), dtype=np.float32)])
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)

    def select_hand_action(self, state):
        from balatro_env import _best_pair_hand, _find_flush, _find_straight, _worst_cards
        from base_bot import get_hand_cards, get_discards_left
        action       = min(self._get_action(state), 4)
        cards        = get_hand_cards(state)
        discards_left = get_discards_left(state)
        if action == 1 and discards_left > 0:
            return "discard", [int(i) for i in _worst_cards(cards, 3)]
        if   action == 2: idxs = _find_flush(cards)    or _best_pair_hand(cards)
        elif action == 3: idxs = _find_straight(cards) or _best_pair_hand(cards)
        else:             idxs = _best_pair_hand(cards)
        idxs = [int(i) for i in idxs if 0 <= i < len(cards)]
        if not idxs: idxs = list(range(min(5, len(cards))))
        self._last_hand_type = "rl_play"
        return "play", idxs

    def select_shop_action(self, state):
        from base_bot import get_money, get_shop_cards
        action     = min(self._get_action(state), 6)
        money      = get_money(state)
        shop_cards = get_shop_cards(state)
        rc         = state.get("round", {}).get("reroll_cost", 5)
        if action == 0: return [{"action":"end_shop"}]
        if action == 6:
            if money >= rc: return [{"action":"reroll"},{"action":"end_shop"}]
            return [{"action":"end_shop"}]
        bi = int(action - 1)
        if bi < len(shop_cards):
            cost = shop_cards[bi].get("cost", {}).get("buy", 999)
            if cost <= money:
                return [{"action":"buy_card","index":bi},{"action":"end_shop"}]
        return [{"action":"end_shop"}]

    def select_blind_action(self, state):
        from base_bot import get_blind_type
        action     = min(self._get_action(state), 1)
        blind_type = get_blind_type(state)
        if action == 1 and blind_type != "boss": return "skip"
        return "select"


# ---------------------------------------------------------------------------
# Simpler RLBot that avoids metaclass gymnastics for BaseBot inheritance
# ---------------------------------------------------------------------------

def _build_rl_bot(model_path, port, results_path, deck, stake):
    """Returns a properly-constructed RLBot using simple class definition."""
    from base_bot import BaseBot, get_hand_cards, get_discards_left, get_money, get_shop_cards, get_blind_type
    from balatro_env import _encode_obs, _best_pair_hand, _find_flush, _find_straight, _worst_cards

    try:
        from sb3_contrib import MaskablePPO as ModelCls
    except ImportError:
        from stable_baselines3 import PPO as ModelCls

    class _RLBot(BaseBot):
        BOT_TYPE   = "rl_bot"
        WANDB_TAGS = ["rl_bot","maskableppo"]

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
            action        = min(self._get_action(state), 4)
            cards         = get_hand_cards(state)
            discards_left = get_discards_left(state)
            if action == 1 and discards_left > 0:
                return "discard", [int(i) for i in _worst_cards(cards, 3)]
            if   action == 2: idxs = _find_flush(cards)    or _best_pair_hand(cards)
            elif action == 3: idxs = _find_straight(cards) or _best_pair_hand(cards)
            else:             idxs = _best_pair_hand(cards)
            idxs = [int(i) for i in idxs if 0 <= i < len(cards)]
            if not idxs: idxs = list(range(min(5, len(cards))))
            self._last_hand_type = "rl_play"
            return "play", idxs

        def select_shop_action(self, state):
            action     = min(self._get_action(state), 6)
            money      = get_money(state)
            shop_cards = get_shop_cards(state)
            rc         = state.get("round", {}).get("reroll_cost", 5)
            if action == 0: return [{"action":"end_shop"}]
            if action == 6:
                if money >= rc: return [{"action":"reroll"},{"action":"end_shop"}]
                return [{"action":"end_shop"}]
            bi = int(action - 1)
            if bi < len(shop_cards):
                cost = shop_cards[bi].get("cost",{}).get("buy",999)
                if cost <= money:
                    return [{"action":"buy_card","index":bi},{"action":"end_shop"}]
            return [{"action":"end_shop"}]

        def select_blind_action(self, state):
            action     = min(self._get_action(state), 1)
            blind_type = get_blind_type(state)
            if action == 1 and blind_type != "boss": return "skip"
            return "select"

    return _RLBot()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",          action="store_true")
    parser.add_argument("--run",            action="store_true")
    parser.add_argument("--mock",           action="store_true")
    parser.add_argument("--timesteps",      type=int,   default=50_000_000)
    parser.add_argument("--n-envs",         type=int,   default=N_ENVS)
    parser.add_argument("--seeds",          nargs="+",  default=BENCHMARK_SEEDS)
    parser.add_argument("--runs-per-seed",  type=int,   default=RUNS_PER_SEED)
    parser.add_argument("--results",        default="results.csv")
    parser.add_argument("--port",           type=int,   default=12346)
    parser.add_argument("--deck",           default="RED")
    parser.add_argument("--stake",          default="WHITE")
    parser.add_argument("--model-path",     default=str(MODEL_PATH))
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