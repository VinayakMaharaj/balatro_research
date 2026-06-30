"""
base_bot.py
Base class for all Balatro agent types.
"""

import csv
import time
import logging
from pathlib import Path
from datetime import datetime
from balatro_client import BalatroClient, BalatroError

logger = logging.getLogger(__name__)

STABLE_STATES = {
    "SELECTING_HAND","SHOP","BLIND_SELECT",
    "ROUND_EVAL","GAME_OVER","SMODS_BOOSTER_OPENED",
}

def get_ante(state): return state.get("ante_num", 0)
def get_round(state): return state.get("round_num", 0)
def get_money(state): return state.get("money", 0)
def get_hands_left(state): return state.get("round", {}).get("hands_left", 0)
def get_discards_left(state): return state.get("round", {}).get("discards_left", 0)
def get_hand_cards(state): return state.get("hand", {}).get("cards", [])
def get_jokers(state):
    j = state.get("jokers", {})
    if isinstance(j, dict): return j.get("cards", [])
    return j if isinstance(j, list) else []
def get_joker_count(state): return state.get("jokers", {}).get("count", 0)
def get_joker_limit(state): return state.get("jokers", {}).get("limit", 5)
def get_shop_cards(state): return state.get("shop", {}).get("cards", [])
def get_shop_packs(state): return state.get("packs", {}).get("cards", [])
def game_over(state): return state.get("state") == "GAME_OVER"
def game_won(state): return state.get("won", False)
def get_blind_type(state):
    blinds = state.get("blinds", {})
    for k in ["small","big","boss"]:
        if blinds.get(k,{}).get("status") in ("SELECT","CURRENT"):
            return k
    return "small"

FIELDNAMES = [
    "timestamp","bot_type","seed","deck","stake","outcome","duration_seconds",
    "final_ante","peak_ante","final_round","hands_played","discards_used","blinds_skipped",
    "final_dollars","jokers_bought","rerolls_used","hands_by_type",
    "llm_calls","tokens_used","estimated_cost_usd","notes",
]

class BaseBot:
    BOT_TYPE   = "base"
    WANDB_TAGS = ["base"]

    def __init__(self, host="127.0.0.1", port=12346, results_path="results.csv",
                 deck="RED", stake="WHITE", launch_wait=5.0):
        self.client       = BalatroClient(host=host, port=port)
        self.results_path = Path(results_path)
        self.deck         = deck
        self.stake        = stake
        self.launch_wait  = launch_wait
        self._ensure_csv()

    def select_hand_action(self, state): raise NotImplementedError
    def select_shop_action(self, state): raise NotImplementedError
    def select_blind_action(self, state): raise NotImplementedError
    def select_pack_action(self, state): return {"action":"skip","cards":[]}
    def _on_game_start(self, seed): pass
    def _on_game_end(self, seed, outcome, state): pass

    def _poll_until_stable(self, max_wait=12.0, interval=0.4):
        deadline = time.time() + max_wait
        while time.time() < deadline:
            try:
                state = self.client.gamestate()
                if state.get("state","") in STABLE_STATES or state.get("won",False):
                    return state
            except Exception as e:
                logger.debug(f"Poll error: {e}")
            time.sleep(interval)
        try:    return self.client.gamestate()
        except Exception: return {}

    def run_game(self, seed):
        import json
        start_time = time.time()
        metrics = {
            "timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "bot_type":self.BOT_TYPE,"seed":seed,"deck":self.deck,"stake":self.stake,
            "outcome":"incomplete","duration_seconds":0,
            "final_ante":0,"peak_ante":0,"final_round":0,
            "hands_played":0,"discards_used":0,"blinds_skipped":0,
            "final_dollars":0,"jokers_bought":0,"rerolls_used":0,
            "hands_by_type":"{}","llm_calls":0,"tokens_used":0,
            "estimated_cost_usd":0.0,"notes":"",
        }
        hand_type_counts = {}
        final_state      = {}

        try:
            try:
                self.client.menu()
                time.sleep(1.0)
            except Exception as e:
                logger.warning(f"Menu call failed: {e}")

            state = self.client.start(deck=self.deck, stake=self.stake, seed=seed)
            logger.info(f"Started seed={seed}")
            self._on_game_start(seed)

            while True:
                sname = state.get("state","UNKNOWN")
                ante  = get_ante(state)
                if ante > metrics["peak_ante"]: metrics["peak_ante"] = ante
                if game_won(state):
                    metrics["outcome"] = "won"; final_state = state; break
                if game_over(state):
                    metrics["outcome"] = "lost"; final_state = state; break

                if sname == "SELECTING_HAND":
                    action, cards = self.select_hand_action(state)
                    if action == "play":
                        metrics["hands_played"] += 1
                        ht = getattr(self,"_last_hand_type",None)
                        if ht: hand_type_counts[ht] = hand_type_counts.get(ht,0)+1
                        state = self.client.play(cards)
                    elif action == "discard":
                        metrics["discards_used"] += 1
                        state = self.client.discard(cards)
                    else:
                        hand = get_hand_cards(state)
                        metrics["hands_played"] += 1
                        state = self.client.play(list(range(min(5,len(hand)))))
                elif sname == "BLIND_SELECT":
                    decision   = self.select_blind_action(state)
                    blind_type = get_blind_type(state)
                    if decision == "skip" and blind_type != "boss":
                        metrics["blinds_skipped"] += 1
                        state = self.client.skip()
                        # Tag rewards from skipping can trigger pack opens or other states
                        # Poll until stable before continuing
                        state = self._poll_until_stable(max_wait=10.0, interval=0.4)
                    else:
                        state = self.client.select()
                        state = self._poll_until_stable(max_wait=10.0, interval=0.4)
                elif sname == "ROUND_EVAL":
                    state = self.client.cash_out()
                elif sname == "SHOP":
                    state = self._execute_shop_actions(state, metrics)
                elif sname == "SMODS_BOOSTER_OPENED":
                    state = self._execute_pack_action(state)
                    state = self._poll_until_stable(max_wait=10.0, interval=0.4)
                elif sname == "GAME_OVER":
                    metrics["outcome"] = "lost"; final_state = state; break
                else:
                    time.sleep(0.3)
                    state = self.client.gamestate()

        except BalatroError as e:
            logger.error(f"BalatroError: {e}")
            metrics["notes"]   = f"BalatroError: {e.name} - {e.message}"
            metrics["outcome"] = "error"
        except Exception as e:
            logger.error(f"Error: {e}")
            metrics["notes"]   = f"Error: {str(e)}"
            metrics["outcome"] = "error"
        finally:
            try:
                final_state = self.client.gamestate()
                metrics["final_ante"]    = get_ante(final_state)
                metrics["final_round"]   = get_round(final_state)
                metrics["final_dollars"] = get_money(final_state)
                if get_ante(final_state) > metrics["peak_ante"]:
                    metrics["peak_ante"] = get_ante(final_state)
            except Exception: pass
            if hasattr(self,"_total_llm_calls"):   metrics["llm_calls"] = self._total_llm_calls
            if hasattr(self,"_total_tokens_used"):
                metrics["tokens_used"] = self._total_tokens_used
                cpm = getattr(self,"_cost_per_1m_tokens",0.80)
                metrics["estimated_cost_usd"] = round((self._total_tokens_used/1_000_000)*cpm,6)
            self._on_game_end(seed, metrics["outcome"], final_state)
            metrics["hands_by_type"]    = json.dumps(hand_type_counts)
            metrics["duration_seconds"] = round(time.time()-start_time,1)
            self._write_row(metrics)

        logger.info(f"Game complete: seed={seed} outcome={metrics['outcome']} ante={metrics['final_ante']} round={metrics['final_round']}")
        return metrics

    def run_experiment(self, seeds, runs_per_seed=1):
        all_results = []
        total       = len(seeds)*runs_per_seed
        try:
            import wandb
            wandb.init(
                project="balatro-research",
                name=f"{self.BOT_TYPE}_{len(seeds)}seeds_x{runs_per_seed}",
                config={"bot_type":self.BOT_TYPE,"seeds":seeds,"runs_per_seed":runs_per_seed,
                        "total_games":total,"deck":self.deck,"stake":self.stake},
                tags=self.WANDB_TAGS,
            )
        except ImportError: pass

        for run_idx in range(runs_per_seed):
            for seed in seeds:
                gn = run_idx*len(seeds)+seeds.index(seed)+1
                logger.info(f"[{gn}/{total}] seed={seed} run={run_idx+1}/{runs_per_seed}")
                if hasattr(self,"_total_llm_calls"):   self._total_llm_calls   = 0
                if hasattr(self,"_total_tokens_used"): self._total_tokens_used = 0
                result = self.run_game(seed)
                all_results.append(result)
                time.sleep(2.0)

        try:
            import wandb
            if wandb.run is not None:
                completed = [r for r in all_results if r["outcome"] in ("won","lost")]
                if completed:
                    wandb.summary["avg_final_round"]    = sum(r["final_round"]    for r in completed)/len(completed)
                    wandb.summary["avg_final_ante"]     = sum(r["final_ante"]     for r in completed)/len(completed)
                    wandb.summary["avg_peak_ante"]      = sum(r["peak_ante"]      for r in completed)/len(completed)
                    wandb.summary["avg_jokers_bought"]  = sum(r["jokers_bought"]  for r in completed)/len(completed)
                    wandb.summary["avg_blinds_skipped"] = sum(r["blinds_skipped"] for r in completed)/len(completed)
                    wandb.summary["avg_discards_used"]  = sum(r["discards_used"]  for r in completed)/len(completed)
                    wandb.summary["win_rate"]           = sum(1 for r in completed if r["outcome"]=="won")/len(completed)
                    wandb.summary["games_completed"]    = len(completed)
                    if any(r["llm_calls"]>0 for r in completed):
                        wandb.summary["total_llm_calls"]   = sum(r["llm_calls"]         for r in all_results)
                        wandb.summary["total_tokens_used"] = sum(r["tokens_used"]        for r in all_results)
                        wandb.summary["total_cost_usd"]    = sum(r["estimated_cost_usd"] for r in all_results)
                wandb.finish()
        except ImportError: pass
        return all_results

    def _execute_pack_action(self, state):
        choices = max(state.get("pack_cards",{}).get("choose",1), 1)
        for _ in range(choices):
            try:
                state = self.client.pack(skip=True)
            except BalatroError:
                state = self.client.gamestate(); break
            except Exception:
                try:    state = self.client.gamestate()
                except Exception: pass
                break
        return state

    def _execute_shop_actions(self, state, metrics):
        actions = self.select_shop_action(state)
        for action_dict in actions:
            action = action_dict.get("action")
            try:
                if action == "buy_card":
                    state = self.client.buy(card=action_dict["index"])
                    # FIX: poll until stable — replaces hardcoded sleep, fixes hangs
                    state = self._poll_until_stable(max_wait=12.0, interval=0.4)
                    metrics["jokers_bought"] += 1
                    # Use planet/tarot consumables immediately
                    consumables = state.get("consumables",{}).get("cards",[])
                    for i, cons in enumerate(consumables):
                        cs = cons.get("set","") or cons.get("ability",{}).get("set","")
                        if cs in ("PLANET","Planet","TAROT","Tarot"):
                            try:
                                state = self.client.use(consumable=i)
                                state = self._poll_until_stable(max_wait=8.0, interval=0.4)
                                logger.info(f"Used {cons.get('label','?')} slot {i}")
                            except Exception as ce:
                                logger.warning(f"Could not use consumable {i}: {ce}")
                                try:    state = self.client.gamestate()
                                except Exception: pass

                elif action == "buy_voucher":
                    state = self.client.buy(voucher=action_dict["index"])
                    state = self._poll_until_stable(max_wait=8.0, interval=0.4)

                elif action == "buy_pack":
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
                except Exception: pass
                break

        try:
            time.sleep(0.5)
            state = self.client.next_round()
        except Exception as e:
            logger.warning(f"next_round failed: {e}")
            try:    state = self.client.gamestate()
            except Exception: state = {}
        return state

    def _ensure_csv(self):
        if not self.results_path.exists():
            self.results_path.parent.mkdir(parents=True, exist_ok=True)
            with self.results_path.open("w",newline="") as f:
                csv.DictWriter(f,fieldnames=FIELDNAMES).writeheader()

    def _write_row(self, metrics):
        with self.results_path.open("a",newline="") as f:
            csv.DictWriter(f,fieldnames=FIELDNAMES).writerow(
                {k:metrics.get(k,"") for k in FIELDNAMES})
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    "final_ante":metrics.get("final_ante",0),
                    "peak_ante":metrics.get("peak_ante",0),
                    "final_round":metrics.get("final_round",0),
                    "hands_played":metrics.get("hands_played",0),
                    "discards_used":metrics.get("discards_used",0),
                    "final_dollars":metrics.get("final_dollars",0),
                    "jokers_bought":metrics.get("jokers_bought",0),
                    "blinds_skipped":metrics.get("blinds_skipped",0),
                    "rerolls_used":metrics.get("rerolls_used",0),
                    "llm_calls":metrics.get("llm_calls",0),
                    "tokens_used":metrics.get("tokens_used",0),
                    "estimated_cost_usd":metrics.get("estimated_cost_usd",0.0),
                    "duration_seconds":metrics.get("duration_seconds",0),
                    "outcome":1 if metrics.get("outcome")=="won" else 0,
                    "seed":metrics.get("seed",""),
                    "bot_type":metrics.get("bot_type",""),
                })
        except ImportError: pass
