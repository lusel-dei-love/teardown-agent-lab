# ABOUTME: Turns a continuous Action into held keys, mouse buttons and relative mouse
# ABOUTME: motion through a swappable InputBackend (RecordingBackend in tests, X11 live).

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

WINDOW_CLASS = "steam_app_1167630"

# Deadzone: |axis| must exceed this before the matching key is held.
MOVE_THRESHOLD = 0.3

# Fixed iteration order so emitted call sequences are deterministic.
KEY_ORDER = ("w", "s", "a", "d")
BUTTON_ORDER = ("left", "right")


@dataclass(frozen=True)
class Action:
    """Agent output: look deltas and move axes in [-1, 1], plus two held buttons."""

    look_dx: float = 0.0
    look_dy: float = 0.0
    move_x: float = 0.0
    move_y: float = 0.0
    grab: bool = False
    swing: bool = False


@runtime_checkable
class InputBackend(Protocol):
    """Lowest-level input injection: relative mouse motion, key and button holds."""

    def move_mouse(self, dx: int, dy: int) -> None: ...

    def key(self, name: str, down: bool) -> None: ...

    def button(self, name: str, down: bool) -> None: ...


class RecordingBackend:
    """Test double: records every backend call as a tuple in `.calls`."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def move_mouse(self, dx: int, dy: int) -> None:
        self.calls.append(("move_mouse", dx, dy))

    def key(self, name: str, down: bool) -> None:
        self.calls.append(("key", name, down))

    def button(self, name: str, down: bool) -> None:
        self.calls.append(("button", name, down))

    @property
    def key_calls(self) -> list[tuple]:
        return [c for c in self.calls if c[0] == "key"]

    @property
    def button_calls(self) -> list[tuple]:
        return [c for c in self.calls if c[0] == "button"]


class Actuator:
    """Stateful mapper: diffs the desired hold set against what is currently held."""

    def __init__(self, backend: InputBackend, look_scale: int = 200):
        self.backend = backend
        self.look_scale = look_scale
        self._held_keys: set[str] = set()
        self._held_buttons: set[str] = set()

    def apply(self, action: Action) -> None:
        self.backend.move_mouse(
            int(round(action.look_dx * self.look_scale)),
            int(round(action.look_dy * self.look_scale)),
        )
        self._sync(
            KEY_ORDER, self._held_keys, self._wanted_keys(action), self.backend.key
        )
        self._sync(
            BUTTON_ORDER,
            self._held_buttons,
            self._wanted_buttons(action),
            self.backend.button,
        )

    def release_all(self) -> None:
        """Release exactly what is currently held, nothing else."""
        self._sync(KEY_ORDER, self._held_keys, set(), self.backend.key)
        self._sync(BUTTON_ORDER, self._held_buttons, set(), self.backend.button)

    @staticmethod
    def _wanted_keys(action: Action) -> set[str]:
        wanted = set()
        if action.move_y > MOVE_THRESHOLD:
            wanted.add("w")
        elif action.move_y < -MOVE_THRESHOLD:
            wanted.add("s")
        if action.move_x > MOVE_THRESHOLD:
            wanted.add("d")
        elif action.move_x < -MOVE_THRESHOLD:
            wanted.add("a")
        return wanted

    @staticmethod
    def _wanted_buttons(action: Action) -> set[str]:
        wanted = set()
        if action.swing:
            wanted.add("left")
        if action.grab:
            wanted.add("right")
        return wanted

    @staticmethod
    def _sync(order, held: set[str], wanted: set[str], emit) -> None:
        for name in order:
            if name in held and name not in wanted:
                emit(name, False)
                held.discard(name)
        for name in order:
            if name in wanted and name not in held:
                emit(name, True)
                held.add(name)


class X11Backend:
    """Live backend: pyautogui against a specific X display. Imported lazily so the
    rest of the module stays importable headless."""

    def __init__(self, display: str):
        self.display = display
        os.environ["DISPLAY"] = display
        import pyautogui

        pyautogui.FAILSAFE = False
        self._gui = pyautogui

    def move_mouse(self, dx: int, dy: int) -> None:
        if dx or dy:
            self._gui.moveRel(dx, dy, duration=0)

    def key(self, name: str, down: bool) -> None:
        if down:
            self._gui.keyDown(name)
        else:
            self._gui.keyUp(name)

    def button(self, name: str, down: bool) -> None:
        if down:
            self._gui.mouseDown(button=name)
        else:
            self._gui.mouseUp(button=name)


def find_game_window(display: str) -> int | None:
    """Window id of the Teardown window on `display`, matched by WM_CLASS, or None."""
    from Xlib import display as xdisplay

    conn = xdisplay.Display(display)
    try:
        root = conn.screen().root
        pending = [root]
        while pending:
            window = pending.pop()
            try:
                wm_class = window.get_wm_class()
            except Exception:
                wm_class = None
            if wm_class and WINDOW_CLASS in wm_class:
                return window.id
            pending.extend(window.query_tree().children)
        return None
    finally:
        conn.close()
