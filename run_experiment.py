import sys
import time
import csv
import os
from datetime import datetime

sys.path.insert(0, r"C:\Users\vinay\AppData\Roaming\Balatro\Mods\balatrobot")

from bot import Bot, Actions, State

SEEDS = [
    "AAAA001", "AAAA002", "AAAA003", "AAAA004", "AAAA005",
    "AAAA006", "AAAA007", "AAAA008", "AAAA009", "AAAA010"
]
LOG_FILE = r"C:\projects\balatro_research\results.csv"


class LoggedFlushBot(Bot):

    def __init__(self, seed):
        super().__init__(
            deck="Red Deck",
            stake=1,
            seed=seed,
            challenge=None,
            bot_port=12346,
        )
        self.seed = seed
        self.rounds_survived = 0
        self.hands_played = 0
        self.discards_used = 0
        self.final_ante = 0
        self.final_dollars = 0

    def skip_or_select_blind(self, bot, G):
        return [Actions.SELECT_BLIND]

    def select_cards_from_hand(self, bot, G):
        # use game's own counter
        self.hands_played = G.get("num_hands_played", self.hands_played)
        self.rounds_survived = G.get("round", self.rounds_survived)
        self.final_dollars = G.get("dollars", self.final_dollars)
        self.final_ante = G.get("ante", {}).get("ante_number", self.final_ante)

        suit_count = {"Hearts": 0, "Diamonds": 0, "Clubs": 0, "Spades": 0}
        for card in G["hand"]:
            suit_count[card["suit"]] += 1

        most_common_suit = max(suit_count, key=suit_count.get)
        most_common_suit_count = suit_count[most_common_suit]

        if most_common_suit_count >= 5:
            flush_cards = [c for c in G["hand"] if c["suit"] == most_common_suit]
            flush_cards.sort(key=lambda x: x["value"], reverse=True)
            return [
                Actions.PLAY_HAND,
                [G["hand"].index(c) + 1 for c in flush_cards[:5]],
            ]

        discards = [c for c in G["hand"] if c["suit"] != most_common_suit]
        discards = sorted(discards, key=lambda x: x["value"], reverse=True)[:5]

        if discards:
            if G["current_round"]["discards_left"] > 0:
                self.discards_used += 1
                return [Actions.DISCARD_HAND, [G["hand"].index(c) + 1 for c in discards]]
            else:
                return [Actions.PLAY_HAND, [G["hand"].index(c) + 1 for c in discards]]

        return [Actions.PLAY_HAND, [1]]

    def select_shop_action(self, bot, G):
        return [Actions.END_SHOP]

    def select_booster_action(self, bot, G):
        return [Actions.SKIP_BOOSTER_PACK]

    def sell_jokers(self, bot, G):
        return [Actions.SELL_JOKER, []]

    def rearrange_jokers(self, bot, G):
        return [Actions.REARRANGE_JOKERS, []]

    def use_or_sell_consumables(self, bot, G):
        return [Actions.USE_CONSUMABLE, []]

    def rearrange_consumables(self, bot, G):
        return [Actions.REARRANGE_CONSUMABLES, []]

    def rearrange_hand(self, bot, G):
        return [Actions.REARRANGE_HAND, []]


def log_result(bot_name, seed, rounds, hands, discards, ante, dollars, outcome):
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "bot", "seed", "rounds_survived",
                "hands_played", "discards_used", "final_ante",
                "final_dollars", "outcome"
            ])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            bot_name, seed, rounds, hands, discards, ante, dollars, outcome
        ])
    print(f"Logged: seed={seed}, rounds={rounds}, hands={hands}, ante={ante}, dollars={dollars}, outcome={outcome}")


def run_single_game(seed):
    print(f"\nStarting game with seed: {seed}")
    bot = LoggedFlushBot(seed=seed)

    try:
        bot.start_balatro_instance()
        print("Balatro launching, waiting 35 seconds...")
        time.sleep(35)

        max_steps = 5000
        steps = 0

        while steps < max_steps:
            bot.run_step()
            steps += 1

            if bot.G is not None:
                state_val = bot.G.get("state")
                if state_val == State.GAME_OVER.value:
                    print(f"Game over at round {bot.rounds_survived}")
                    break

            time.sleep(0.05)

        outcome = "completed" if steps < max_steps else "timeout"
        log_result(
            "flush_bot", seed, bot.rounds_survived, bot.hands_played,
            bot.discards_used, bot.final_ante, bot.final_dollars, outcome
        )

    except Exception as e:
        print(f"Error during run: {e}")
        log_result(
            "flush_bot", seed, bot.rounds_survived, bot.hands_played,
            bot.discards_used, bot.final_ante, bot.final_dollars, "error"
        )
    finally:
        bot.stop_balatro_instance()
        print("Balatro instance stopped")
        time.sleep(15)


if __name__ == "__main__":
    print("Starting Balatro experiment — Flush Bot baseline")
    print(f"Running {len(SEEDS)} seeded games")
    print(f"Results will be logged to: {LOG_FILE}")

    for seed in SEEDS:
        run_single_game(seed)

    print(f"\nAll games complete. Check {LOG_FILE} for results.")