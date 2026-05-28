"""
balatro_client.py
HTTP JSON-RPC 2.0 client for coder/balatrobot.
Replaces the old UDP-based besteon client.

Usage:
    client = BalatroClient()
    state = client.call("gamestate")
    state = client.call("start", {"deck": "RED", "stake": "WHITE", "seed": "AAAAAAA"})
"""

import httpx
from dataclasses import dataclass, field
from typing import Any


class BalatroError(Exception):
    """Raised when the API returns a JSON-RPC error response."""

    def __init__(self, code: int, message: str, name: str) -> None:
        self.code = code
        self.message = message
        self.name = name
        super().__init__(f"[{name}] {message}")


@dataclass
class BalatroClient:
    """
    Synchronous JSON-RPC 2.0 HTTP client for coder/balatrobot.

    The coder fork communicates over HTTP POST to http://host:port/
    with standard JSON-RPC 2.0 request/response format.

    Game state fields now available (not in old besteon fork):
        - state         : current game state string e.g. "SELECTING_HAND"
        - ante_num      : current ante number (int)
        - round_num     : current round number (int)
        - won           : whether the run has been won (bool)
        - money         : current dollars (int)
        - shop          : shop cards with labels AND costs
        - jokers        : joker area with count and limit
        - hand          : current hand cards with indices
        - blinds        : small/big/boss blind info with scores
        - round         : hands_left, discards_left, reroll_cost
    """

    host: str = "127.0.0.1"
    port: int = 12346
    timeout: float = 30.0
    _request_id: int = field(default=0, init=False, repr=False)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Send a JSON-RPC 2.0 request and return the result dict.

        Args:
            method: API method name e.g. "gamestate", "play", "buy"
            params: Optional dict of parameters for the method

        Returns:
            The result dict from the game

        Raises:
            BalatroError: If the game returns an error response
            httpx.ConnectError: If Balatro is not running or mod not loaded
        """
        self._request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": self._request_id,
        }

        with httpx.Client(timeout=self.timeout) as http:
            response = http.post(self.url, json=payload)
            response.raise_for_status()
            data = response.json()

        if "error" in data:
            err = data["error"]
            raise BalatroError(
                code=err["code"],
                message=err["message"],
                name=err["data"]["name"],
            )

        return data["result"]

    def health(self) -> bool:
        """Check if the server is up. Returns True if healthy."""
        try:
            result = self.call("health")
            return result.get("status") == "ok"
        except Exception:
            return False

    def gamestate(self) -> dict[str, Any]:
        """Get current game state."""
        return self.call("gamestate")

    def menu(self) -> dict[str, Any]:
        """Return to main menu from any state."""
        return self.call("menu")

    def start(self, deck: str = "RED", stake: str = "WHITE", seed: str | None = None) -> dict[str, Any]:
        """Start a new run. Deck and stake must be uppercase enum strings."""
        params: dict[str, Any] = {"deck": deck, "stake": stake}
        if seed:
            params["seed"] = seed
        return self.call("start", params)

    def play(self, cards: list[int]) -> dict[str, Any]:
        """Play cards by 0-based index."""
        return self.call("play", {"cards": cards})

    def discard(self, cards: list[int]) -> dict[str, Any]:
        """Discard cards by 0-based index."""
        return self.call("discard", {"cards": cards})

    def select(self) -> dict[str, Any]:
        """Select the current blind."""
        return self.call("select")

    def skip(self) -> dict[str, Any]:
        """Skip current blind (Small or Big only)."""
        return self.call("skip")

    def cash_out(self) -> dict[str, Any]:
        """Cash out after round eval."""
        return self.call("cash_out")

    def next_round(self) -> dict[str, Any]:
        """Leave shop and advance to blind select."""
        return self.call("next_round")

    def buy(self, card: int | None = None, voucher: int | None = None, pack: int | None = None) -> dict[str, Any]:
        """
        Buy from shop. Provide exactly one of card, voucher, or pack
        as a 0-based index.
        """
        params: dict[str, Any] = {}
        if card is not None:
            params["card"] = card
        elif voucher is not None:
            params["voucher"] = voucher
        elif pack is not None:
            params["pack"] = pack
        else:
            raise ValueError("Must provide one of: card, voucher, pack")
        return self.call("buy", params)

    def sell(self, joker: int | None = None, consumable: int | None = None) -> dict[str, Any]:
        """Sell a joker or consumable by 0-based index."""
        params: dict[str, Any] = {}
        if joker is not None:
            params["joker"] = joker
        elif consumable is not None:
            params["consumable"] = consumable
        else:
            raise ValueError("Must provide one of: joker, consumable")
        return self.call("sell", params)

    def reroll(self) -> dict[str, Any]:
        """Reroll the shop."""
        return self.call("reroll")

    def use(self, consumable: int, cards: list[int] | None = None) -> dict[str, Any]:
        """Use a consumable, optionally targeting hand cards by 0-based index."""
        params: dict[str, Any] = {"consumable": consumable}
        if cards:
            params["cards"] = cards
        return self.call("use", params)

    def pack(self, card: int | None = None, skip: bool = False) -> dict[str, Any]:
        """Select or skip a card from an open booster pack."""
        params: dict[str, Any] = {}
        if skip:
            params["skip"] = True
        elif card is not None:
            params["card"] = card
        else:
            raise ValueError("Must provide card index or skip=True")
        return self.call("pack", params)
