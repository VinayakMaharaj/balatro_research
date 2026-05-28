"""
heuristic_bots.py
Rule-based heuristic agents. Both inherit from BaseBot.
 
FlushBot    - Always plays flushes, never buys anything. Floor baseline.
MetaBot     - Meta-informed heuristics: joker priority, blind skipping,
              hand type hierarchy. Primary heuristic comparison agent.
 
Run with:
    python heuristic_bots.py
"""
 
import logging
from collections import Counter
from base_bot import (
    BaseBot,
    get_hand_cards,
    get_money,
    get_joker_count,
    get_joker_limit,
    get_shop_cards,
    get_ante,
    get_blind_type,
    get_hands_left,
    get_discards_left,
)
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
 
 
# ---------------------------------------------------------------------------
# Card parsing helpers
# ---------------------------------------------------------------------------
 
def parse_card(card: dict) -> tuple[str, str]:
    value = card.get("value", {})
    rank = value.get("rank", "?")
    suit = value.get("suit", "?")
    return rank, suit
 
 
def get_suits(cards: list[dict]) -> list[str]:
    return [parse_card(c)[1] for c in cards]
 
 
def get_ranks(cards: list[dict]) -> list[str]:
    return [parse_card(c)[0] for c in cards]
 
 
RANK_ORDER = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
RANK_VALUES = {r: i for i, r in enumerate(RANK_ORDER)}
 
 
def rank_value(rank: str) -> int:
    return RANK_VALUES.get(rank, 0)
 
 
# ---------------------------------------------------------------------------
# Hand detection helpers
# ---------------------------------------------------------------------------
 
def find_flush(cards: list[dict]) -> list[int] | None:
    suits = get_suits(cards)
    suit_counts = Counter(suits)
    for suit, count in suit_counts.items():
        if count >= 5:
            indices = [i for i, c in enumerate(cards) if parse_card(c)[1] == suit]
            indices.sort(key=lambda i: rank_value(parse_card(cards[i])[0]), reverse=True)
            return indices[:5]
    return None
 
 
def find_straight(cards: list[dict]) -> list[int] | None:
    indexed = [(i, rank_value(parse_card(c)[0])) for i, c in enumerate(cards)]
    indexed.sort(key=lambda x: x[1])
    seen_ranks = {}
    for idx, rv in indexed:
        if rv not in seen_ranks:
            seen_ranks[rv] = idx
 
    unique_ranks = sorted(seen_ranks.keys())
 
    for start in range(len(unique_ranks) - 4):
        window = unique_ranks[start:start + 5]
        if window[-1] - window[0] == 4 and len(set(window)) == 5:
            return [seen_ranks[r] for r in window]
 
    ace_indices = [i for i, c in enumerate(cards) if parse_card(c)[0] == "A"]
    low_ranks = {r for r in unique_ranks if r <= rank_value("5")}
    if ace_indices and len(low_ranks) >= 4:
        low_4 = sorted(low_ranks)[:4]
        if low_4 == [0, 1, 2, 3]:
            return [ace_indices[0]] + [seen_ranks[r] for r in low_4]
 
    return None
 
 
def find_best_pair_hand(cards: list[dict]) -> tuple[str, list[int]]:
    ranks = get_ranks(cards)
    rank_counts = Counter(ranks)
    sorted_by_count = sorted(rank_counts.items(), key=lambda x: (x[1], rank_value(x[0])), reverse=True)
 
    def indices_for_rank(rank: str, n: int) -> list[int]:
        found = [i for i, c in enumerate(cards) if parse_card(c)[0] == rank]
        return found[:n]
 
    counts = [count for _, count in sorted_by_count]
 
    if counts[0] == 4:
        rank = sorted_by_count[0][0]
        idxs = indices_for_rank(rank, 4)
        others = [i for i in range(len(cards)) if i not in idxs]
        if others:
            others.sort(key=lambda i: rank_value(parse_card(cards[i])[0]), reverse=True)
            idxs.append(others[0])
        return "four_of_a_kind", idxs[:5]
 
    if counts[0] == 3 and len(counts) > 1 and counts[1] >= 2:
        r1 = sorted_by_count[0][0]
        r2 = sorted_by_count[1][0]
        return "full_house", indices_for_rank(r1, 3) + indices_for_rank(r2, 2)
 
    if counts[0] == 3:
        rank = sorted_by_count[0][0]
        idxs = indices_for_rank(rank, 3)
        others = sorted(
            [i for i in range(len(cards)) if i not in idxs],
            key=lambda i: rank_value(parse_card(cards[i])[0]),
            reverse=True
        )
        return "three_of_a_kind", (idxs + others[:2])[:5]
 
    if counts[0] == 2 and len(counts) > 1 and counts[1] == 2:
        r1 = sorted_by_count[0][0]
        r2 = sorted_by_count[1][0]
        idxs = indices_for_rank(r1, 2) + indices_for_rank(r2, 2)
        others = sorted(
            [i for i in range(len(cards)) if i not in idxs],
            key=lambda i: rank_value(parse_card(cards[i])[0]),
            reverse=True
        )
        return "two_pair", (idxs + others[:1])[:5]
 
    if counts[0] == 2:
        rank = sorted_by_count[0][0]
        idxs = indices_for_rank(rank, 2)
        others = sorted(
            [i for i in range(len(cards)) if i not in idxs],
            key=lambda i: rank_value(parse_card(cards[i])[0]),
            reverse=True
        )
        return "pair", (idxs + others[:3])[:5]
 
    all_sorted = sorted(range(len(cards)), key=lambda i: rank_value(parse_card(cards[i])[0]), reverse=True)
    return "high_card", all_sorted[:5]
 
 
# ---------------------------------------------------------------------------
# FlushBot - Floor baseline
# ---------------------------------------------------------------------------
 
class FlushBot(BaseBot):
    BOT_TYPE = "flush_bot"
 
    def select_hand_action(self, state: dict) -> tuple[str, list[int]]:
        cards = get_hand_cards(state)
        if not cards:
            return "play", [0]
 
        flush = find_flush(cards)
        if flush:
            logger.debug(f"Playing flush: indices {flush}")
            return "play", flush
 
        hand_name, indices = find_best_pair_hand(cards)
        logger.debug(f"No flush, playing {hand_name}: indices {indices}")
        return "play", indices
 
    def select_shop_action(self, state: dict) -> list[dict]:
        return [{"action": "end_shop"}]
 
    def select_blind_action(self, state: dict) -> str:
        return "select"
 
 
# ---------------------------------------------------------------------------
# MetaBot - Meta-informed heuristic
# ---------------------------------------------------------------------------
 
JOKER_TIERS = {
    "S+": [
        "Blueprint", "Brainstorm", "Triboulet", "Chicot",
        "Perkeo", "Caino", "Yorick",
    ],
    "S": [
        "Vampire", "Cavendish", "Hologram", "Obelisk",
        "Campfire", "Ride the Bus", "Madness", "Glass Joker",
        "Constellation", "Fortune Teller", "Supernova",
    ],
    "A": [
        "Hack", "Smiley Face", "Scary Face", "Scholar",
        "Even Steven", "Odd Todd", "Fibonacci", "Baron",
        "Shoot the Moon", "Abstract Joker", "Bull",
        "Blue Joker", "Green Joker", "Bootstraps",
    ],
}
 
JOKER_PRIORITY = {}
for tier, jokers in JOKER_TIERS.items():
    priority = {"S+": 100, "S": 70, "A": 40}[tier]
    for j in jokers:
        JOKER_PRIORITY[j.lower()] = priority
 
 
def joker_score(label: str) -> int:
    return JOKER_PRIORITY.get(label.lower(), 10)
 
 
class MetaBot(BaseBot):
    BOT_TYPE = "meta_bot"
 
    def select_hand_action(self, state: dict) -> tuple[str, list[int]]:
        cards = get_hand_cards(state)
        if not cards:
            return "play", [0]
 
        discards_left = get_discards_left(state)
        hands_left = get_hands_left(state)
 
        flush = find_flush(cards)
        if flush:
            return "play", flush
 
        straight = find_straight(cards)
        if straight:
            return "play", straight
 
        if discards_left > 0 and hands_left > 1:
            discard_indices = self._find_discard_for_flush(cards)
            if discard_indices:
                return "discard", discard_indices
 
        hand_name, indices = find_best_pair_hand(cards)
        return "play", indices
 
    def _find_discard_for_flush(self, cards: list[dict]) -> list[int] | None:
        suits = get_suits(cards)
        suit_counts = Counter(suits)
        best_suit, best_count = suit_counts.most_common(1)[0]
 
        if best_count < 3:
            return None
 
        discard = [i for i, c in enumerate(cards) if parse_card(c)[1] != best_suit]
 
        if not discard:
            return None
 
        return discard[:5]
 
    def select_shop_action(self, state: dict) -> list[dict]:
        actions = []
        money = get_money(state)
        joker_count = get_joker_count(state)
        joker_limit = get_joker_limit(state)
        shop_cards = get_shop_cards(state)
 
        spendable = money - 2
 
        if joker_count < joker_limit and spendable > 0:
            best_idx = None
            best_score = -1
 
            for i, card in enumerate(shop_cards):
                label = card.get("label", "")
                card_set = card.get("set", "")
                cost = card.get("cost", {}).get("buy", 999)
 
                if card_set != "JOKER":
                    continue
                if cost > spendable:
                    continue
 
                score = joker_score(label)
                effective_min = 1 if joker_count == 0 else 40
 
                if score >= effective_min and score > best_score:
                    best_score = score
                    best_idx = i
 
            if best_idx is not None:
                logger.info(f"MetaBot buying joker at index {best_idx} (score={best_score}, jokers={joker_count})")
                actions.append({"action": "buy_card", "index": best_idx})
 
        actions.append({"action": "end_shop"})
        return actions
 
    def select_blind_action(self, state: dict) -> str:
        ante = get_ante(state)
        blind_type = get_blind_type(state)
 
        if blind_type == "boss":
            return "select"
        if ante >= 2:
            return "skip"
        return "select"
 
 
# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------
 
BENCHMARK_SEEDS = ["AAAAAAA", "BBBBBBB", "CCCCCCC", "DDDDDDD", "EEEEEEE"]
RUNS_PER_SEED = 20
 
 
if __name__ == "__main__":
    import argparse
 
    parser = argparse.ArgumentParser(description="Run heuristic Balatro bots")
    parser.add_argument("--bot", choices=["flush", "meta"], default="flush")
    parser.add_argument("--seeds", nargs="+", default=BENCHMARK_SEEDS)
    parser.add_argument("--runs-per-seed", type=int, default=RUNS_PER_SEED)
    parser.add_argument("--results", default="results.csv")
    parser.add_argument("--port", type=int, default=12346)
    parser.add_argument("--deck", default="RED")
    parser.add_argument("--stake", default="WHITE")
    args = parser.parse_args()
 
    BotClass = FlushBot if args.bot == "flush" else MetaBot
    bot = BotClass(
        port=args.port,
        results_path=args.results,
        deck=args.deck,
        stake=args.stake,
    )
 
    if not bot.client.health():
        print("ERROR: Cannot connect to Balatro. Make sure the game is running with the BalatroBot mod loaded.")
        exit(1)
 
    print(f"Running {BotClass.BOT_TYPE} on {len(args.seeds)} seeds x {args.runs_per_seed} runs")
    print(f"Total games: {len(args.seeds) * args.runs_per_seed}")
    print(f"Results: {args.results}")
 
    results = bot.run_experiment(seeds=args.seeds, runs_per_seed=args.runs_per_seed)
 
    completed = [r for r in results if r["outcome"] in ("won", "lost")]
    if completed:
        avg_round = sum(r["final_round"] for r in completed) / len(completed)
        avg_ante = sum(r["final_ante"] for r in completed) / len(completed)
        wins = sum(1 for r in results if r["outcome"] == "won")
        print(f"\nResults summary:")
        print(f"  Games completed: {len(completed)}/{len(results)}")
        print(f"  Wins: {wins}")
        print(f"  Avg final round: {avg_round:.2f}")
        print(f"  Avg final ante:  {avg_ante:.2f}")
 