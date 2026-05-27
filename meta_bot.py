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

# Tiered joker priority based on community tier lists
# Source: Sportskeeda November 2024 tier list + reddit meta consensus
JOKERS_S_PLUS = [
    "blueprint", "brainstorm", "triboulet"
]

JOKERS_S = [
    "vampire", "cavendish", "the duo", "the trio", "the family",
    "campfire", "canio", "spare trousers"
]

JOKERS_A = [
    "hiker", "fortune teller", "rocket", "seltzer", "trading card",
    "bloodstone", "perkeo", "fibonacci", "onyx agate", "arrowhead",
    "sixth sense", "space joker", "burnt joker", "hologram",
    "drivers license", "steel joker", "ancient joker", "card sharp",
    "baseball card", "to do list", "business card", "mail-in rebate",
    "cloud 9", "golden joker", "to the moon", "dna", "green joker",
    "gros michel", "ramen", "ride the bus", "stuntman", "the tribe",
    "throwback", "vagabond"
]


def get_joker_priority(name):
    name_lower = name.lower()
    if any(j in name_lower for j in JOKERS_S_PLUS):
        return 0
    if any(j in name_lower for j in JOKERS_S):
        return 1
    if any(j in name_lower for j in JOKERS_A):
        return 2
    return 99


class MetaBot(Bot):

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
        self.jokers_bought = 0
        self.blinds_skipped = 0

    def get_ante_from_round(self, round_num):
        if round_num <= 0:
            return 0
        return ((round_num - 1) // 3) + 1

    def skip_or_select_blind(self, bot, G):
        round_num = G.get("round", 0)
        if round_num < self.rounds_survived and self.rounds_survived > 0:
            print(f"Run reset detected at blind select, stopping")
            self.running = False
            return [Actions.SELECT_BLIND]

        # skip small blind on ante 2+ to farm tag rewards
        blind_on_deck = G.get("ante", {}).get("blinds", {}).get("ondeck", "")
        ante = self.get_ante_from_round(round_num)
        if blind_on_deck == "Small" and ante >= 2:
            self.blinds_skipped += 1
            print(f"Skipping small blind at ante {ante} for tag reward")
            return [Actions.SKIP_BLIND]

        return [Actions.SELECT_BLIND]

    def get_best_hand(self, hand):
        suit_count = {"Hearts": 0, "Diamonds": 0, "Clubs": 0, "Spades": 0}
        value_count = {}

        for card in hand:
            suit_count[card["suit"]] += 1
            v = card["value"]
            value_count[v] = value_count.get(v, 0) + 1

        # check flush
        for suit, count in suit_count.items():
            if count >= 5:
                flush_cards = [c for c in hand if c["suit"] == suit]
                flush_cards.sort(key=lambda x: x["value"], reverse=True)
                return "Flush", flush_cards[:5]

        # check straight
        value_order = [
            "2", "3", "4", "5", "6", "7", "8", "9", "10",
            "Jack", "Queen", "King", "Ace"
        ]
        hand_values = sorted(set(
            [value_order.index(c["value"])
             for c in hand if c["value"] in value_order]
        ))
        for i in range(len(hand_values) - 4):
            window = hand_values[i:i+5]
            if window[-1] - window[0] == 4 and len(window) == 5:
                straight_vals = [value_order[v] for v in window]
                straight_cards = [
                    c for c in hand if c["value"] in straight_vals
                ][:5]
                return "Straight", straight_cards

        # check four of a kind
        for value, count in value_count.items():
            if count >= 4:
                four_cards = [c for c in hand if c["value"] == value][:4]
                remaining = [c for c in hand if c["value"] != value]
                remaining.sort(key=lambda x: x["value"], reverse=True)
                return "Four of a Kind", four_cards + remaining[:1]

        # check full house
        threes = [v for v, c in value_count.items() if c >= 3]
        twos = [v for v, c in value_count.items() if c >= 2 and v not in threes]
        if threes and twos:
            three_cards = [c for c in hand if c["value"] == threes[0]][:3]
            two_cards = [c for c in hand if c["value"] == twos[0]][:2]
            return "Full House", three_cards + two_cards

        # check three of a kind
        if threes:
            three_cards = [c for c in hand if c["value"] == threes[0]][:3]
            remaining = [c for c in hand if c["value"] != threes[0]]
            remaining.sort(key=lambda x: x["value"], reverse=True)
            return "Three of a Kind", three_cards + remaining[:2]

        # check two pair
        pairs = [v for v, c in value_count.items() if c >= 2]
        if len(pairs) >= 2:
            pair_cards = []
            for p in pairs[:2]:
                pair_cards += [c for c in hand if c["value"] == p][:2]
            remaining = [c for c in hand if c["value"] not in pairs[:2]]
            remaining.sort(key=lambda x: x["value"], reverse=True)
            return "Two Pair", pair_cards + remaining[:1]

        # check pair
        if len(pairs) == 1:
            pair_cards = [c for c in hand if c["value"] == pairs[0]][:2]
            remaining = [c for c in hand if c["value"] != pairs[0]]
            remaining.sort(key=lambda x: x["value"], reverse=True)
            return "Pair", pair_cards + remaining[:3]

        # high card fallback
        sorted_hand = sorted(
            hand,
            key=lambda x: value_order.index(x["value"])
            if x["value"] in value_order else 0,
            reverse=True
        )
        return "High Card", sorted_hand[:5]

    def select_cards_from_hand(self, bot, G):
        round_num = G.get("round", 0)

        if round_num < self.rounds_survived and self.rounds_survived > 0:
            print(f"Run reset detected at round {round_num}, stopping")
            self.running = False
            return [Actions.PLAY_HAND, [1]]

        self.rounds_survived = round_num
        self.hands_played = G.get("num_hands_played", self.hands_played)
        self.final_dollars = G.get("dollars", self.final_dollars)
        self.final_ante = self.get_ante_from_round(round_num)

        hand = G["hand"]
        hand_type, best_cards = self.get_best_hand(hand)

        # play immediately if strong hand
        if hand_type in ["Flush", "Straight", "Four of a Kind", "Full House"]:
            print(f"Playing strong hand: {hand_type}")
            return [
                Actions.PLAY_HAND,
                [hand.index(c) + 1 for c in best_cards]
            ]

        # discard if we can improve
        if G["current_round"]["discards_left"] > 0:
            discard_cards = [c for c in hand if c not in best_cards][:5]
            if discard_cards:
                self.discards_used += 1
                print(f"Discarding to improve from {hand_type}")
                return [
                    Actions.DISCARD_HAND,
                    [hand.index(c) + 1 for c in discard_cards]
                ]

        print(f"Playing best available: {hand_type}")
        return [
            Actions.PLAY_HAND,
            [hand.index(c) + 1 for c in best_cards]
        ]

    def select_shop_action(self, bot, G):
        dollars = G.get("dollars", 0)
        shop = G.get("shop", [])
        current_jokers = G.get("jokers", [])
        max_jokers = G.get("max_jokers", 5)

        if not shop:
            return [Actions.END_SHOP]

        # find best affordable joker in shop by tier
        best_item = None
        best_priority = 99
        best_index = -1

        for i, item in enumerate(shop):
            item_name = item.get("name", "")
            item_cost = item.get("cost", 999)
            item_type = item.get("type", "")

            if item_type == "Joker" and item_cost <= dollars:
                if len(current_jokers) < max_jokers:
                    priority = get_joker_priority(item_name)
                    if priority < best_priority:
                        best_priority = priority
                        best_item = item
                        best_index = i

        # buy S+, S, or A tier jokers
        if best_item is not None and best_priority <= 2:
            print(f"Buying {best_item.get('name')} "
                  f"(tier priority {best_priority}) "
                  f"for ${best_item.get('cost')}")
            self.jokers_bought += 1
            return [Actions.BUY_CARD, [best_index + 1]]

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


def log_result(bot_name, seed, rounds, hands, discards, ante,
               dollars, jokers_bought, blinds_skipped, outcome):
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "bot", "seed", "rounds_survived",
                "hands_played", "discards_used", "final_ante",
                "final_dollars", "jokers_bought", "blinds_skipped", "outcome"
            ])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            bot_name, seed, rounds, hands, discards, ante,
            dollars, jokers_bought, blinds_skipped, outcome
        ])
    print(f"Logged: seed={seed}, rounds={rounds}, ante={ante}, "
          f"jokers={jokers_bought}, skips={blinds_skipped}, outcome={outcome}")


def run_single_game(seed):
    print(f"\nStarting game with seed: {seed}")
    bot = MetaBot(seed=seed)

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
                if bot.G.get("state") == State.GAME_OVER.value:
                    print(f"Game over at round {bot.rounds_survived}")
                    break
                if not bot.running:
                    print(f"Bot stopped at round {bot.rounds_survived}")
                    break

            time.sleep(0.05)

        outcome = "completed" if steps < max_steps else "timeout"
        log_result(
            "meta_bot", seed, bot.rounds_survived, bot.hands_played,
            bot.discards_used, bot.final_ante, bot.final_dollars,
            bot.jokers_bought, bot.blinds_skipped, outcome
        )

    except Exception as e:
        print(f"Error during run: {e}")
        log_result(
            "meta_bot", seed, bot.rounds_survived, bot.hands_played,
            bot.discards_used, bot.final_ante, bot.final_dollars,
            bot.jokers_bought, bot.blinds_skipped, "error"
        )
    finally:
        bot.stop_balatro_instance()
        print("Balatro instance stopped")
        time.sleep(15)


if __name__ == "__main__":
    print("Starting Balatro experiment — Meta Bot")
    print(f"Running {len(SEEDS)} seeded games")
    print(f"Results will be logged to: {LOG_FILE}")

    for seed in SEEDS:
        run_single_game(seed)

    print(f"\nAll games complete. Check {LOG_FILE} for results.")