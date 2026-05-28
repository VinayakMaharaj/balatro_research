"""
base_bot.py
Base class for all Balatro agent types.

All three agent tiers inherit from this:
    - HeuristicBot  (flush bot, meta bot)
    - LLMBot        (zero-shot + RAG conditions)
    - RLBot         (PPO trained agent)

The base class handles:
    - Game loop state machine
    - CSV logging with full metrics
    - Run lifecycle (start, loop, detect end)
    - Consistent seed-based experiment protocol

Subclasses only need to implement:
    - select_hand_action(state)  -> ("play" | "discard", list[int])
    - select_shop_action(state)  -> list of shop actions or "end_shop"
    - select_blind_action(state) -> "select" | "skip"
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
    """Get current ante number directly from coder/balatrobot state."""
    return state.get("ante_num", 0)


def get_round(state: dict) -> int:
    """Get current round number directly from coder/balatrobot state."""
    return state.get("round_num", 0)


def get_money(state: dict) -> int:
    return state.get("money", 0)


def get_hands_left(state: dict) -> int:
    return state.get("round", {}).get("hands_left", 0)


def get_discards_left(state: dict) -> int:
    return state.get("round", {}).get("discards_left", 0)


def get_hand_cards(state: dict) -> list[dict]:
    """Return list of card dicts from current hand."""
    return state.get("hand", {}).get("cards", [])


def get_jokers(state: dict) -> list[dict]:
    return state.get("jokers", {}).get("cards", [])


def get_joker_count(state: dict) -> int:
    return state.get("jokers", {}).get("count", 0)


def get_joker_limit(state: dict) -> int:
    return state.get("jokers", {}).get("limit", 5)


def get_shop_cards(state: dict) -> list[dict]:
    """
    Returns shop card list. Each card has:
        label   : str  e.g. "Joker", "Greedy Joker"
        set     : str  e.g. "JOKER", "PLANET", "TAROT"
        cost    : dict { buy: int, sell: int }
    """
    return state.get("shop", {}).get("cards", [])


def get_shop_packs(state: dict) -> list[dict]:
    return state.get("packs", {}).get("cards", [])


def game_over(state: dict) -> bool:
    return state.get("state") == "GAME_OVER"


def game_won(state: dict) -> bool:
    return state.get("won", False)


def get_blind_type(state: dict) -> str:
    """Returns which blind is currently on deck: small, big, or boss."""
    blinds = state.get("blinds", {})
    for blind_key in ["small", "big", "boss"]:
        if blinds.get(blind_key, {}).get("status") in ("SELECT", "CURRENT"):
            return blind_key
    return "small"


# ---------------------------------------------------------------------------
# Metrics row structure
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "timestamp",
    "bot_type",
    "seed",
    "deck",
    "stake",
    "final_ante",
    "final_round",
    "hands_played",
    "discards_used",
    "final_dollars",
    "jokers_bought",
    "blinds_skipped",
    "outcome",
    "duration_seconds",
    "notes",
]


# ---------------------------------------------------------------------------
# Base Bot
# ---------------------------------------------------------------------------

class BaseBot:
    """
    Base class for all Balatro research agents.

    Subclass this and implement:
        select_hand_action(state) -> (action, cards)
            action: "play" or "discard"
            cards:  list of 0-based card indices

        select_shop_action(state) -> list of shop commands
            Each command is a dict like:
            {"action": "buy_card", "index": 0}
            {"action": "buy_pack", "index": 0}
            {"action": "reroll"}
            {"action": "end_shop"}

        select_blind_action(state) -> "select" or "skip"

    The base class will call these methods and handle everything else.
    """

    BOT_TYPE = "base"

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

    # -----------------------------------------------------------------------
    # Methods subclasses must override
    # -----------------------------------------------------------------------

    def select_hand_action(self, state: dict) -> tuple[str, list[int]]:
        """
        Decide what to do during SELECTING_HAND.

        Returns:
            ("play", [card_indices])   to play a hand
            ("discard", [card_indices]) to discard cards
        """
        raise NotImplementedError

    def select_shop_action(self, state: dict) -> list[dict]:
        """
        Decide what to do in the shop.

        Returns a list of actions to execute in order.
        Each action is a dict with an "action" key.
        Return [{"action": "end_shop"}] to leave immediately.

        Available actions:
            {"action": "buy_card",    "index": int}
            {"action": "buy_voucher", "index": int}
            {"action": "buy_pack",    "index": int}
            {"action": "reroll"}
            {"action": "sell_joker",  "index": int}
            {"action": "end_shop"}
        """
        raise NotImplementedError

    def select_blind_action(self, state: dict) -> str:
        """
        Decide whether to select or skip the current blind.

        Returns:
            "select" to play the blind
            "skip"   to skip it and collect the tag reward
                     (only valid for small and big blinds)
        """
        raise NotImplementedError

    # -----------------------------------------------------------------------
    # Game loop
    # -----------------------------------------------------------------------

    def run_game(self, seed: str) -> dict:
        """
        Run one complete game with the given seed.
        Returns a metrics dict.
        """
        start_time = time.time()
        metrics = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "bot_type": self.BOT_TYPE,
            "seed": seed,
            "deck": self.deck,
            "stake": self.stake,
            "final_ante": 0,
            "final_round": 0,
            "hands_played": 0,
            "discards_used": 0,
            "final_dollars": 0,
            "jokers_bought": 0,
            "blinds_skipped": 0,
            "outcome": "incomplete",
            "duration_seconds": 0,
            "notes": "",
        }

        try:
            # Return to menu and start fresh run
            try:
                self.client.menu()
                time.sleep(1.0)
            except Exception as e:
                logger.warning(f"Menu call failed: {e}")

            state = self.client.start(deck=self.deck, stake=self.stake, seed=seed)
            logger.info(f"Started game seed={seed} ante={get_ante(state)} round={get_round(state)}")

            # Main game loop
            while True:
                current_state_name = state.get("state", "UNKNOWN")
                logger.debug(f"State: {current_state_name} | ante={get_ante(state)} round={get_round(state)}")

                if game_won(state):
                    metrics["outcome"] = "won"
                    break

                if game_over(state):
                    metrics["outcome"] = "lost"
                    break

                if current_state_name == "SELECTING_HAND":
                    action, cards = self.select_hand_action(state)
                    if action == "play":
                        metrics["hands_played"] += 1
                        state = self.client.play(cards)
                    elif action == "discard":
                        metrics["discards_used"] += 1
                        state = self.client.discard(cards)
                    else:
                        logger.warning(f"Unknown hand action: {action}, falling back to play first 5")
                        hand = get_hand_cards(state)
                        play_count = min(5, len(hand))
                        metrics["hands_played"] += 1
                        state = self.client.play(list(range(play_count)))

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
                    # Default: skip all packs unless subclass overrides
                    state = self.client.pack(skip=True)

                elif current_state_name == "GAME_OVER":
                    metrics["outcome"] = "lost"
                    break

                else:
                    # Transitional states - just get current state
                    time.sleep(0.3)
                    state = self.client.gamestate()

        except BalatroError as e:
            logger.error(f"BalatroError during game: {e}")
            metrics["notes"] = f"BalatroError: {e.name} - {e.message}"
            metrics["outcome"] = "error"
        except Exception as e:
            logger.error(f"Unexpected error during game: {e}")
            metrics["notes"] = f"Error: {str(e)}"
            metrics["outcome"] = "error"
        finally:
            # Always capture final state metrics
            try:
                final_state = self.client.gamestate()
                metrics["final_ante"] = get_ante(final_state)
                metrics["final_round"] = get_round(final_state)
                metrics["final_dollars"] = get_money(final_state)
            except Exception:
                pass

            metrics["duration_seconds"] = round(time.time() - start_time, 1)
            self._write_row(metrics)

        logger.info(
            f"Game complete: seed={seed} outcome={metrics['outcome']} "
            f"ante={metrics['final_ante']} round={metrics['final_round']}"
        )
        return metrics

    def run_experiment(self, seeds: list[str], runs_per_seed: int = 1) -> list[dict]:
        """
        Run the full experiment across all seeds.

        Args:
            seeds: List of seed strings e.g. ["AAAAAAA", "BBBBBBB"]
            runs_per_seed: How many times to run each seed (default 1, use 20 for 100 total across 5 seeds)

        Returns:
            List of metrics dicts, one per game
        """
        all_results = []
        total = len(seeds) * runs_per_seed

        for run_idx in range(runs_per_seed):
            for seed in seeds:
                game_num = run_idx * len(seeds) + seeds.index(seed) + 1
                logger.info(f"[{game_num}/{total}] Running seed={seed} run={run_idx + 1}/{runs_per_seed}")

                result = self.run_game(seed)
                all_results.append(result)

                # Brief pause between games to let game reset cleanly
                time.sleep(2.0)

        return all_results

    # -----------------------------------------------------------------------
    # Shop execution helper
    # -----------------------------------------------------------------------

    def _execute_shop_actions(self, state: dict, metrics: dict) -> dict:
        """Execute the list of shop actions returned by select_shop_action."""
        actions = self.select_shop_action(state)

        for action_dict in actions:
            action = action_dict.get("action")
            try:
                if action == "buy_card":
                    state = self.client.buy(card=action_dict["index"])
                    metrics["jokers_bought"] += 1
                    logger.info(f"Bought card at index {action_dict['index']}")

                elif action == "buy_voucher":
                    state = self.client.buy(voucher=action_dict["index"])
                    logger.info(f"Bought voucher at index {action_dict['index']}")

                elif action == "buy_pack":
                    state = self.client.buy(pack=action_dict["index"])
                    logger.info(f"Bought pack at index {action_dict['index']}")
                    # After buying a pack we are in SMODS_BOOSTER_OPENED
                    # subclass can override pack handling, default skips
                    state = self.client.pack(skip=True)

                elif action == "reroll":
                    state = self.client.reroll()
                    logger.info("Rerolled shop")

                elif action == "sell_joker":
                    state = self.client.sell(joker=action_dict["index"])
                    logger.info(f"Sold joker at index {action_dict['index']}")

                elif action == "end_shop":
                    state = self.client.next_round()
                    return state

                else:
                    logger.warning(f"Unknown shop action: {action}")

            except BalatroError as e:
                logger.warning(f"Shop action {action} failed: {e.name} - {e.message}")
                # On any shop error, bail out to next round
                try:
                    state = self.client.next_round()
                    return state
                except Exception:
                    pass
                break

        # If we get here without an end_shop action, leave the shop
        try:
            state = self.client.next_round()
        except Exception as e:
            logger.warning(f"next_round failed: {e}")
            state = self.client.gamestate()

        return state

    # -----------------------------------------------------------------------
    # CSV logging
    # -----------------------------------------------------------------------

    def _ensure_csv(self) -> None:
        """Create results CSV with headers if it does not exist."""
        if not self.results_path.exists():
            self.results_path.parent.mkdir(parents=True, exist_ok=True)
            with self.results_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writeheader()

    def _write_row(self, metrics: dict) -> None:
        """Append one row to the results CSV."""
        with self.results_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            # Fill any missing fields with empty string
            row = {k: metrics.get(k, "") for k in FIELDNAMES}
            writer.writerow(row)
