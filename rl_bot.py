"""
rl_bot.py
PPO-based RL agent for Balatro using Stable-Baselines3 + Gymnasium.

Two modes:
    --train        : train on real env (slow, ~1 it/s)
    --train --mock : train on mock env (fast, ~1000 it/s) — recommended
    --run          : evaluate trained model on real env

Install deps:
    pip install gymnasium stable-baselines3 --break-system-packages

Fast training (recommended):
    python rl_bot.py --train --mock --timesteps 500000

Real env training (slow):
    python rl_bot.py --train --timesteps 1000

Evaluation:
    python rl_bot.py --run --runs-per-seed 1
"""

import os
import time
import logging
import argparse
import numpy as np
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback

from balatro_env import BalatroEnv
from base_bot import (
    BaseBot, get_hand_cards, get_discards_left, get_money,
    get_shop_cards, get_shop_packs, get_blind_type, get_ante
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_DIR = Path("rl_model")
MODEL_PATH = MODEL_DIR / "ppo_balatro"

BENCHMARK_SEEDS = [f"SEED{str(i).zfill(3)}" for i in range(1, 101)]
RUNS_PER_SEED = 1


# ---------------------------------------------------------------------------
# W&B callback
# ---------------------------------------------------------------------------

class BalatroTrainingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        try:
            import wandb
            if wandb.run is None:
                return True
            infos = self.locals.get("infos", [])
            for info in infos:
                if "episode" in info:
                    ep = info["episode"]
                    wandb.log({
                        "train/episode_reward": ep["r"],
                        "train/episode_length": ep["l"],
                        "train/ante":           info.get("ante", 0),
                        "train/round":          info.get("round", 0),
                        "train/won":            int(info.get("won", False)),
                    })
        except ImportError:
            pass
        return True


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    timesteps: int = 500_000,
    seed: str = "AAAAAAA",
    port: int = 12346,
    use_mock: bool = True,
):
    if use_mock:
        from balatro_mock_env import BalatroMockEnv
        env = BalatroMockEnv()
        env_name = "mock"
        logger.info(f"Training PPO on MOCK env for {timesteps} timesteps")
    else:
        env = BalatroEnv(port=port, seed=seed)
        env_name = "real"
        logger.info(f"Training PPO on REAL env for {timesteps} timesteps (slow)")

    logger.info("Checking env...")
    check_env(env, warn=True)
    logger.info("Env check passed")

    try:
        import wandb
        wandb.init(
            project="balatro-research",
            name=f"rl_bot_ppo_{env_name}_{timesteps}steps",
            config={
                "bot_type":   "rl_bot",
                "algorithm":  "PPO",
                "timesteps":  timesteps,
                "env":        env_name,
                "seed":       seed,
                "policy":     "MlpPolicy",
            },
            tags=["rl_bot", "ppo", "training", env_name],
        )
        logger.info("W&B initialized")
    except Exception:
        logger.info("W&B init failed, training without W&B")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )

    model.learn(
        total_timesteps=timesteps,
        callback=BalatroTrainingCallback(),
        progress_bar=True,
    )

    MODEL_DIR.mkdir(exist_ok=True)
    model.save(str(MODEL_PATH))
    logger.info(f"Model saved to {MODEL_PATH}.zip")

    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass

    env.close()
    return model


# ---------------------------------------------------------------------------
# RLBot
# ---------------------------------------------------------------------------

class RLBot(BaseBot):
    BOT_TYPE = "rl_bot"
    WANDB_TAGS = ["rl_bot", "ppo"]

    def __init__(self, model_path: str = str(MODEL_PATH), port: int = 12346, **kwargs):
        super().__init__(port=port, **kwargs)
        if not Path(f"{model_path}.zip").exists():
            raise FileNotFoundError(
                f"No trained model at {model_path}.zip — run with --train first."
            )
        self.model = PPO.load(model_path)
        logger.info(f"Loaded PPO model from {model_path}.zip")

    def _get_action(self, state: dict) -> int:
        from balatro_env import _encode_obs
        obs = _encode_obs(state)
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)

    def select_hand_action(self, state: dict) -> tuple[str, list[int]]:
        from balatro_env import (
            _best_pair_hand, _find_flush, _find_straight,
            _worst_cards, N_HAND_ACTIONS
        )
        action = min(self._get_action(state), N_HAND_ACTIONS - 1)
        cards = get_hand_cards(state)
        discards_left = get_discards_left(state)

        if action == 1 and discards_left > 0:
            return "discard", [int(i) for i in _worst_cards(cards, 3)]

        if action == 2:
            idxs = _find_flush(cards) or _best_pair_hand(cards)
        elif action == 3:
            idxs = _find_straight(cards) or _best_pair_hand(cards)
        else:
            idxs = _best_pair_hand(cards)

        idxs = [int(i) for i in idxs if 0 <= i < len(cards)]
        if not idxs:
            idxs = list(range(min(5, len(cards))))
        self._last_hand_type = "rl_play"
        return "play", idxs

    def select_shop_action(self, state: dict) -> list[dict]:
        from balatro_env import N_SHOP_ACTIONS
        action = min(self._get_action(state), N_SHOP_ACTIONS - 1)

        money = get_money(state)
        shop_cards = get_shop_cards(state)
        reroll_cost = state.get("round", {}).get("reroll_cost", 5)

        if action == 0:
            return [{"action": "end_shop"}]

        if action == 6:
            if money >= reroll_cost:
                return [{"action": "reroll"}, {"action": "end_shop"}]
            return [{"action": "end_shop"}]

        buy_idx = int(action - 1)
        if buy_idx < len(shop_cards):
            cost = shop_cards[buy_idx].get("cost", {}).get("buy", 999)
            if cost <= money:
                return [{"action": "buy_card", "index": buy_idx}, {"action": "end_shop"}]

        return [{"action": "end_shop"}]

    def select_blind_action(self, state: dict) -> str:
        from balatro_env import N_BLIND_ACTIONS
        action = min(self._get_action(state), N_BLIND_ACTIONS - 1)
        blind_type = get_blind_type(state)
        if action == 1 and blind_type != "boss":
            return "skip"
        return "select"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO RL agent for Balatro")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--run",   action="store_true")
    parser.add_argument("--mock",  action="store_true", help="Train on fast mock env")
    parser.add_argument("--timesteps",    type=int, default=500_000)
    parser.add_argument("--seeds",        nargs="+", default=BENCHMARK_SEEDS)
    parser.add_argument("--runs-per-seed",type=int, default=RUNS_PER_SEED)
    parser.add_argument("--results",      default="results.csv")
    parser.add_argument("--port",         type=int, default=12346)
    parser.add_argument("--deck",         default="RED")
    parser.add_argument("--stake",        default="WHITE")
    parser.add_argument("--model-path",   default=str(MODEL_PATH))
    args = parser.parse_args()

    if args.train:
        train(
            timesteps=args.timesteps,
            seed=args.seeds[0],
            port=args.port,
            use_mock=args.mock,
        )

    if args.run:
        bot = RLBot(
            model_path=args.model_path,
            port=args.port,
            results_path=args.results,
            deck=args.deck,
            stake=args.stake,
        )

        if not bot.client.health():
            print("ERROR: Cannot connect to Balatro.")
            exit(1)

        print(f"Running RLBot on {len(args.seeds)} seeds x {args.runs_per_seed} runs")
        results = bot.run_experiment(seeds=args.seeds, runs_per_seed=args.runs_per_seed)

        completed = [r for r in results if r["outcome"] in ("won", "lost")]
        if completed:
            avg_round = sum(r["final_round"] for r in completed) / len(completed)
            avg_ante  = sum(r["final_ante"]  for r in completed) / len(completed)
            wins = sum(1 for r in results if r["outcome"] == "won")
            print(f"\nSummary:")
            print(f"  Completed: {len(completed)}/{len(results)}")
            print(f"  Wins: {wins}")
            print(f"  Avg round: {avg_round:.2f}")
            print(f"  Avg ante:  {avg_ante:.2f}")