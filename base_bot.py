"""
base_bot.py
Base class for all Balatro agent types.
"""

import csv
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Any
from balatro_client import BalatroClient, BalatroError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Game state helpers
# ---------------------------------------------------------------------------

def get_ante(state: dict) -> int:
    return state.get("ante_num", 0)

def get_round(state: dict) -> int:
    return state.get("round_num", 0)

def get_money(state: dict) -> int:
    return state.get("money", 0)

def get_hands_left(state: dict) -> int:
    return state.get("round", {}).get("hands_left", 0)

def get_discards_left(state: dict) -> int:
    return state.get("round", {}).get("discards_left", 0)

def get_hand_cards(state: dict) -> list[dict]:
    return state.get("hand", {}).get("cards", [])

def get_jokers(state: dict) -> list[dict]:
    return state.get("jokers", {}).get("cards", [])

def get_joker_count(state: dict) -> int:
    return state.get("jokers", {}).get("count", 0)

def get_joker_limit(state: dict) -> int:
    return state.get("jokers", {}).get("limit", 5)

def get_shop_cards(state: dict) -> list[dict]:
    return state.get("shop", {}).get("cards", [])

def get_shop_packs(state: dict) -> list[dict]:
    return state.get("packs", {}).get("cards", [])

def game_over(state: dict) -> bool:
    return state.get("state") == "GAME_OVER"

def game_won(state: dict) -> bool:
    return state.get("won", False)

def get_blind_type(state: dict) -> str:
    blinds = state.get("blinds", {})
    for blind_key in ["small", "big", "boss"]:
        if blinds.get(blind_key, {}).get("status") in ("SELECT", "CURRENT"):
            return blind_key
    return "small"


# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "timestamp", "bot_type", "seed", "deck", "stake",
    "outcome", "duration_seconds",
    "final_ante", "peak_ante", "final_round",
    "hands_played", "discards_used", "blinds_skipped",
    "final_dollars", "jokers_bought", "rerolls_used",
    "hands_by_type",
    "llm_calls", "tokens_used", "estimated_cost_usd",
    "notes",
]


# ---------------------------------------------------------------------------
# Base Bot
# ---------------------------------------------------------------------------

class BaseBot:
    BOT_TYPE = "base"
    WANDB_TAGS = ["base"]

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 12346,
        results_path: str = "results.csv",
        deck: str = "RED",
        stake: str = "WHITE",
        launch_wait: float = 5.0,
    ):
        self.client = BalatroClient(host=host, port=port)
        self.results_path = Path(results_path)
        self.deck = deck
        self.stake = stake
        self.launch_wait = launch_wait
        self._ensure_csv()

    def select_hand_action(self, state: dict) -> tuple[str, list[int]]:
        raise NotImplementedError

    def select_shop_action(self, state: dict) -> list[dict]:
        raise NotImplementedError

    def select_blind_action(self, state: dict) -> str:
        raise NotImplementedError

    def select_pack_action(self, state: dict) -> dict:
        return {"action": "skip", "cards": []}

    # -----------------------------------------------------------------------
    # Game loop
    # -----------------------------------------------------------------------

    def run_game(self, seed: str) -> dict:
        import json
        start_time = time.time()
        metrics = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "bot_type": self.BOT_TYPE, "seed": seed,
            "deck": self.deck, "stake": self.stake,
            "outcome": "incomplete", "duration_seconds": 0,
            "final_ante": 0, "peak_ante": 0, "final_round": 0,
            "hands_played": 0, "discards_used": 0, "blinds_skipped": 0,
            "final_dollars": 0, "jokers_bought": 0, "rerolls_used": 0,
            "hands_by_type": "{}", "llm_calls": 0, "tokens_used": 0,
            "estimated_cost_usd": 0.0, "notes": "",
        }
        hand_type_counts = {}

        try:
            try:
                self.client.menu()
                time.sleep(1.0)
            except Exception as e:
                logger.warning(f"Menu call failed: {e}")

            state = self.client.start(deck=self.deck, stake=self.stake, seed=seed)
            logger.info(f"Started seed={seed}")

            while True:
                current_state_name = state.get("state", "UNKNOWN")
                current_ante = get_ante(state)
                if current_ante > metrics["peak_ante"]:
                    metrics["peak_ante"] = current_ante

                if game_won(state):
                    metrics["outcome"] = "won"; break
                if game_over(state):
                    metrics["outcome"] = "lost"; break

                if current_state_name == "SELECTING_HAND":
                    action, cards = self.select_hand_action(state)
                    if action == "play":
                        metrics["hands_played"] += 1
                        hand_type = getattr(self, "_last_hand_type", None)
                        if hand_type:
                            hand_type_counts[hand_type] = hand_type_counts.get(hand_type, 0) + 1
                        state = self.client.play(cards)
                    elif action == "discard":
                        metrics["discards_used"] += 1
                        state = self.client.discard(cards)
                    else:
                        hand = get_hand_cards(state)
                        metrics["hands_played"] += 1
                        state = self.client.play(list(range(min(5, len(hand)))))

                elif current_state_name == "BLIND_SELECT":
                    decision = self.select_blind_action(state)
                    blind_type = get_blind_type(state)
                    if decision == "skip" and blind_type != "boss":
                        metrics["blinds_skipped"] += 1
                        state = self.client.skip()
                    else:
                        state = self.client.select()

                elif current_state_name == "ROUND_EVAL":
                    state = self.client.cash_out()

                elif current_state_name == "SHOP":
                    state = self._execute_shop_actions(state, metrics)

                elif current_state_name == "SMODS_BOOSTER_OPENED":
                    # FIX: always skip packs to avoid API hangs
                    state = self._execute_pack_action(state)

                elif current_state_name == "GAME_OVER":
                    metrics["outcome"] = "lost"; break

                else:
                    time.sleep(0.3)
                    state = self.client.gamestate()

        except BalatroError as e:
            logger.error(f"BalatroError: {e}")
            metrics["notes"] = f"BalatroError: {e.name} - {e.message}"
            metrics["outcome"] = "error"
        except Exception as e:
            logger.error(f"Error: {e}")
            metrics["notes"] = f"Error: {str(e)}"
            metrics["outcome"] = "error"
        finally:
            try:
                final_state = self.client.gamestate()
                metrics["final_ante"] = get_ante(final_state)
                metrics["final_round"] = get_round(final_state)
                metrics["final_dollars"] = get_money(final_state)
                if get_ante(final_state) > metrics["peak_ante"]:
                    metrics["peak_ante"] = get_ante(final_state)
            except Exception:
                pass

            if hasattr(self, "_total_llm_calls"):
                metrics["llm_calls"] = self._total_llm_calls
            if hasattr(self, "_total_tokens_used"):
                metrics["tokens_used"] = self._total_tokens_used
                cost_per_1m = getattr(self, "_cost_per_1m_tokens", 0.80)
                metrics["estimated_cost_usd"] = round(
                    (self._total_tokens_used / 1_000_000) * cost_per_1m, 6)

            metrics["hands_by_type"] = json.dumps(hand_type_counts)
            metrics["duration_seconds"] = round(time.time() - start_time, 1)
            self._write_row(metrics)

        logger.info(
            f"Game complete: seed={seed} outcome={metrics['outcome']} "
            f"ante={metrics['final_ante']} round={metrics['final_round']}"
        )
        return metrics

    def run_experiment(self, seeds: list[str], runs_per_seed: int = 1) -> list[dict]:
        all_results = []
        total = len(seeds) * runs_per_seed

        try:
            import wandb
            wandb.init(
                project="balatro-research",
                name=f"{self.BOT_TYPE}_{len(seeds)}seeds_x{runs_per_seed}",
                config={
                    "bot_type": self.BOT_TYPE, "seeds": seeds,
                    "runs_per_seed": runs_per_seed, "total_games": total,
                    "deck": self.deck, "stake": self.stake,
                },
                tags=self.WANDB_TAGS,
            )
            logger.info("W&B run initialized")
        except ImportError:
            logger.info("wandb not installed")

        for run_idx in range(runs_per_seed):
            for seed in seeds:
                game_num = run_idx * len(seeds) + seeds.index(seed) + 1
                logger.info(f"[{game_num}/{total}] seed={seed} run={run_idx+1}/{runs_per_seed}")

                if hasattr(self, "_total_llm_calls"):   self._total_llm_calls = 0
                if hasattr(self, "_total_tokens_used"): self._total_tokens_used = 0

                result = self.run_game(seed)
                all_results.append(result)
                time.sleep(2.0)

        try:
            import wandb
            if wandb.run is not None:
                completed = [r for r in all_results if r["outcome"] in ("won", "lost")]
                if completed:
                    wandb.summary["avg_final_round"]    = sum(r["final_round"] for r in completed) / len(completed)
                    wandb.summary["avg_final_ante"]     = sum(r["final_ante"] for r in completed) / len(completed)
                    wandb.summary["avg_peak_ante"]      = sum(r["peak_ante"] for r in completed) / len(completed)
                    wandb.summary["avg_jokers_bought"]  = sum(r["jokers_bought"] for r in completed) / len(completed)
                    wandb.summary["avg_blinds_skipped"] = sum(r["blinds_skipped"] for r in completed) / len(completed)
                    wandb.summary["avg_discards_used"]  = sum(r["discards_used"] for r in completed) / len(completed)
                    wandb.summary["win_rate"]           = sum(1 for r in completed if r["outcome"] == "won") / len(completed)
                    wandb.summary["games_completed"]    = len(completed)
                    if any(r["llm_calls"] > 0 for r in completed):
                        wandb.summary["total_llm_calls"]   = sum(r["llm_calls"] for r in all_results)
                        wandb.summary["total_tokens_used"] = sum(r["tokens_used"] for r in all_results)
                        wandb.summary["total_cost_usd"]    = sum(r["estimated_cost_usd"] for r in all_results)
                wandb.finish()
        except ImportError:
            pass

        return all_results

    # -----------------------------------------------------------------------
    # Pack execution — always skip
    # -----------------------------------------------------------------------

    def _execute_pack_action(self, state: dict) -> dict:
        pack_decision = self.select_pack_action(state)
        action = pack_decision.get("action", "skip")
        cards  = pack_decision.get("cards", [])
        pack_cards = state.get("pack_cards", {}).get("cards", [])
        choices    = state.get("pack_cards", {}).get("choose", 1)

        if action == "pick" and cards:
            picked = 0
            for card_idx in cards[:choices]:
                if 0 <= int(card_idx) < len(pack_cards):
                    try:
                        state  = self.client.pack(card=int(card_idx))
                        picked += 1
                    except BalatroError as e:
                        logger.warning(f"Pack pick {card_idx} failed: {e.name}")
            for _ in range(choices - picked):
                try:
                    state = self.client.pack(skip=True)
                except BalatroError:
                    state = self.client.gamestate(); break
        else:
            for _ in range(choices):
                try:
                    state = self.client.pack(skip=True)
                except BalatroError:
                    state = self.client.gamestate(); break

        return state

    # -----------------------------------------------------------------------
    # Shop execution
    # FIX: use planet/tarot consumables immediately after buying
    # FIX: never buy packs (causes SMODS_BOOSTER_OPENED hangs)
    # -----------------------------------------------------------------------

    def _execute_shop_actions(self, state: dict, metrics: dict) -> dict:
        actions = self.select_shop_action(state)

        for action_dict in actions:
            action = action_dict.get("action")
            try:
                if action == "buy_card":
                    state = self.client.buy(card=action_dict["index"])
                    time.sleep(2.0)
                    state = self.client.gamestate()
                    metrics["jokers_bought"] += 1
                    # FIX: use planet/tarot consumables immediately
                    consumables = state.get("consumables", {}).get("cards", [])
                    for i, cons in enumerate(consumables):
                        if cons.get("set") in ("PLANET", "TAROT"):
                            try:
                                state = self.client.use(consumable=i)
                                time.sleep(1.0)
                                state = self.client.gamestate()
                                logger.info(f"Used {cons.get('label','?')} from slot {i}")
                            except Exception as ce:
                                logger.warning(f"Could not use consumable {i}: {ce}")

                elif action == "buy_voucher":
                    state = self.client.buy(voucher=action_dict["index"])

                elif action == "buy_pack":
                    # FIX: never buy packs — causes SMODS_BOOSTER_OPENED hangs
                    logger.info("Skipping pack purchase to avoid API hang")

                elif action == "reroll":
                    state = self.client.reroll()
                    metrics["rerolls_used"] += 1

                elif action == "sell_joker":
                    state = self.client.sell(joker=action_dict["index"])

                elif action == "end_shop":
                    state = self.client.next_round()
                    return state

                else:
                    logger.warning(f"Unknown shop action: {action}")

            except BalatroError as e:
                logger.warning(f"Shop action {action} failed: {e.name} - {e.message}")
                try:
                    state = self.client.next_round()
                    return state
                except Exception:
                    pass
                break

        try:
            time.sleep(1.0)
            state = self.client.next_round()
        except Exception as e:
            logger.warning(f"next_round failed: {e}")
            state = self.client.gamestate()

        return state

    # -----------------------------------------------------------------------
    # CSV logging
    # -----------------------------------------------------------------------

    def _ensure_csv(self) -> None:
        if not self.results_path.exists():
            self.results_path.parent.mkdir(parents=True, exist_ok=True)
            with self.results_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writeheader()

    def _write_row(self, metrics: dict) -> None:
        with self.results_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            row = {k: metrics.get(k, "") for k in FIELDNAMES}
            writer.writerow(row)

        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    "final_ante":         metrics.get("final_ante", 0),
                    "peak_ante":          metrics.get("peak_ante", 0),
                    "final_round":        metrics.get("final_round", 0),
                    "hands_played":       metrics.get("hands_played", 0),
                    "discards_used":      metrics.get("discards_used", 0),
                    "final_dollars":      metrics.get("final_dollars", 0),
                    "jokers_bought":      metrics.get("jokers_bought", 0),
                    "blinds_skipped":     metrics.get("blinds_skipped", 0),
                    "rerolls_used":       metrics.get("rerolls_used", 0),
                    "llm_calls":          metrics.get("llm_calls", 0),
                    "tokens_used":        metrics.get("tokens_used", 0),
                    "estimated_cost_usd": metrics.get("estimated_cost_usd", 0.0),
                    "duration_seconds":   metrics.get("duration_seconds", 0),
                    "outcome":            1 if metrics.get("outcome") == "won" else 0,
                    "seed":               metrics.get("seed", ""),
                    "bot_type":           metrics.get("bot_type", ""),
                })
        except ImportError:
            pass