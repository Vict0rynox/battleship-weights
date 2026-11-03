"""Public player SDK. No server or engine imports."""

from battleship_client.sdk import BattleshipClient, PlayerAPIError

__all__ = ["BattleshipClient", "PlayerAPIError"]
