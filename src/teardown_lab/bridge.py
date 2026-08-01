# ABOUTME: The Bridge transport interface between host and game (state out, commands in)
# ABOUTME: plus FakeBridge, the in-memory double every host-side test runs against.

from __future__ import annotations

from typing import Protocol, runtime_checkable

from teardown_lab.state import GameState


@runtime_checkable
class Bridge(Protocol):
    """Transport contract: read the latest game state, push commands, shut down."""

    def read_state(self, timeout: float = 1.0) -> GameState | None: ...

    def send(self, cmd: dict) -> None: ...

    def close(self) -> None: ...


class FakeBridge:
    """Replays a canned list of states; records commands; `reset` rewinds to the start."""

    def __init__(self, states: list[GameState]):
        self.states = list(states)
        self.sent: list[dict] = []
        self.closed = False
        self._index = 0

    def read_state(self, timeout: float = 1.0) -> GameState | None:
        if not self.states:
            return None
        index = min(self._index, len(self.states) - 1)
        self._index = index + 1
        return self.states[index]

    def send(self, cmd: dict) -> None:
        self.sent.append(cmd)
        if cmd.get("cmd") == "reset":
            self._index = 0

    def close(self) -> None:
        self.closed = True
