# ABOUTME: Turns a continuous Action into held keys, mouse buttons and relative mouse
# ABOUTME: motion through a swappable InputBackend (RecordingBackend in tests, uinput live).

from __future__ import annotations

import os
import time
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

    # 200 gives ~22 deg of yaw per unit command. Lowering it to 80 was tried and measured
    # WORSE for keeping the target in frame (0.127 vs 0.204): finer steps mean more frames
    # spent mid-turn. Left at 200.
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


class UinputBackend:
    """Live backend: kernel-level evdev events on /dev/uinput.

    Measured 2026-08-01: Teardown ignores XTest-synthesized input (pyautogui/xdotool)
    entirely - the pointer moves but clicks and keys never register - while uinput
    events do register. /dev/uinput is user-writable via ACL, so no sudo is needed.
    evdev is imported lazily so this module stays importable headless.
    """

    KEYS = {"w": "KEY_W", "a": "KEY_A", "s": "KEY_S", "d": "KEY_D"}
    BUTTONS = {"left": "BTN_LEFT", "right": "BTN_RIGHT"}

    # The game needs a moment to enumerate a freshly created input device.
    SETTLE_S = 1.5

    def __init__(self, settle_s: float = SETTLE_S, sleeper=time.sleep):
        from evdev import UInput, ecodes

        self._ecodes = ecodes
        key_codes = [getattr(ecodes, name) for name in self.KEYS.values()]
        button_codes = [getattr(ecodes, name) for name in self.BUTTONS.values()]
        self._device = UInput(
            {
                ecodes.EV_KEY: key_codes + button_codes,
                ecodes.EV_REL: [ecodes.REL_X, ecodes.REL_Y],
            },
            name="teardown-lab-input",
        )
        sleeper(settle_s)

    def move_mouse(self, dx: int, dy: int) -> None:
        if not dx and not dy:
            return
        if dx:
            self._device.write(self._ecodes.EV_REL, self._ecodes.REL_X, int(dx))
        if dy:
            self._device.write(self._ecodes.EV_REL, self._ecodes.REL_Y, int(dy))
        self._device.syn()

    def key(self, name: str, down: bool) -> None:
        self._emit(getattr(self._ecodes, self.KEYS[name]), down)

    def button(self, name: str, down: bool) -> None:
        self._emit(getattr(self._ecodes, self.BUTTONS[name]), down)

    def close(self) -> None:
        self._device.close()

    def _emit(self, code: int, down: bool) -> None:
        self._device.write(self._ecodes.EV_KEY, code, 1 if down else 0)
        self._device.syn()


class X11Backend:
    """Pointer-positioning backend: pyautogui against a specific X display.

    Only mouse MOTION lands in Teardown through XTest; its `key`/`button` events are
    silently ignored by the game (see UinputBackend). Kept for cursor placement and
    for driving non-game X clients. Imported lazily so the module stays headless-safe.
    """

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


def focus_game_window(display: str) -> bool:
    """Raise the game window and give it input focus. True if it was found.

    Teardown pauses when it loses focus, which stops the registry updating and starves
    the bridge - so any unattended run must be able to take focus back after another
    application steals it (a browser popping up mid-collection did exactly this).
    """
    from Xlib import X, protocol
    from Xlib import display as xdisplay

    conn = xdisplay.Display(display)
    try:
        root = conn.screen().root
        pending = [root]
        while pending:
            window = pending.pop()
            try:
                wm_class = window.get_wm_class()
                geometry = window.get_geometry()
            except Exception:
                pending.extend(_safe_children(window))
                continue
            if wm_class and WINDOW_CLASS in wm_class and geometry.width > 300:
                # Ask the window manager to activate the window via EWMH rather than
                # reordering it ourselves. Under a real WM (GNOME here) a direct
                # configure/set_input_focus is ignored or raises BadMatch, which left the
                # game paused UNDERNEATH another window - and, worse, left the frame
                # grabber capturing that other window as the agent's observation.
                mask = X.SubstructureRedirectMask | X.SubstructureNotifyMask

                # Switch to the workspace holding the game first. Opening another app can
                # move the desktop to a different workspace, leaving the game running
                # unfocused (so paused) and invisible - which silently feeds the frame
                # grabber whatever IS on screen instead of the game.
                current = root.get_full_property(
                    conn.intern_atom("_NET_CURRENT_DESKTOP"), X.AnyPropertyType
                )
                desktop = window.get_full_property(
                    conn.intern_atom("_NET_WM_DESKTOP"), X.AnyPropertyType
                )
                if (
                    current is not None
                    and desktop is not None
                    and current.value
                    and desktop.value
                ):
                    delta = int(desktop.value[0]) - int(current.value[0])
                    if delta:
                        _switch_workspace(delta)
                        conn.sync()

                # Ask to stay above other windows. Activation alone loses to GNOME's
                # focus-stealing prevention, which left another app covering the game.
                state_event = protocol.event.ClientMessage(
                    window=window,
                    client_type=conn.intern_atom("_NET_WM_STATE"),
                    data=(
                        32,
                        [
                            1,  # _NET_WM_STATE_ADD
                            conn.intern_atom("_NET_WM_STATE_ABOVE"),
                            0,
                            1,
                            0,
                        ],
                    ),
                )
                root.send_event(state_event, event_mask=mask)

                activate_event = protocol.event.ClientMessage(
                    window=window,
                    client_type=conn.intern_atom("_NET_ACTIVE_WINDOW"),
                    data=(32, [1, X.CurrentTime, 0, 0, 0]),
                )
                root.send_event(activate_event, event_mask=mask)
                conn.flush()
                conn.sync()
                return True
            pending.extend(_safe_children(window))
        return False
    finally:
        conn.close()


def game_window_visible(display: str) -> bool:
    """True if the game window can actually be read right now.

    Visibility is the only thing that matters for pixel capture, and it is the one thing
    window properties do not reliably report: a fullscreen game may carry no
    _NET_WM_DESKTOP at all. Attempting to read the window is the direct test - X refuses
    it when the window is not viewable.
    """
    from Xlib import X
    from Xlib import display as xdisplay

    conn = xdisplay.Display(display)
    try:
        window_id = find_game_window(display)
        if window_id is None:
            return False
        window = conn.create_resource_object("window", window_id)
        geometry = window.get_geometry()
        window.get_image(0, 0, 1, 1, X.ZPixmap, 0xFFFFFFFF)
        return geometry.width > 300
    except Exception:
        return False
    finally:
        conn.close()


def ensure_game_visible(display: str, attempts: int = 5) -> bool:
    """Cycle workspaces until the game window is readable. True if it ended visible.

    Another application opening can move the desktop to a different workspace. The game
    keeps running there (so the bridge still returns state and nothing looks broken) but
    the frame grabber captures whatever IS on screen - silently recording a browser as
    the agent's observation. This is the guard against that.
    """
    if game_window_visible(display):
        return True
    for _ in range(attempts):
        _switch_workspace(-1)
        if game_window_visible(display):
            return True
    return False


def _switch_workspace(delta: int) -> None:
    """Move `delta` workspaces using a synthetic Super+PageUp/PageDown.

    GNOME ignores _NET_CURRENT_DESKTOP from a non-pager client, so EWMH cannot move the
    view to the workspace holding the game. It does honour real input, and uinput events
    are indistinguishable from a physical keyboard - the same reason uinput is what
    drives the game at all. Without this, another app opening can strand the game on
    another workspace, where it pauses AND the frame grabber silently captures whatever
    is on screen instead.
    """
    import time

    from evdev import UInput
    from evdev import ecodes as e

    key = e.KEY_PAGEUP if delta < 0 else e.KEY_PAGEDOWN
    device = UInput({e.EV_KEY: [e.KEY_LEFTMETA, e.KEY_PAGEUP, e.KEY_PAGEDOWN]}, name="td-lab-ws")
    try:
        time.sleep(UinputBackend.SETTLE_S)
        for _ in range(abs(delta)):
            device.write(e.EV_KEY, e.KEY_LEFTMETA, 1)
            device.syn()
            time.sleep(0.05)
            device.write(e.EV_KEY, key, 1)
            device.syn()
            time.sleep(0.08)
            device.write(e.EV_KEY, key, 0)
            device.syn()
            time.sleep(0.05)
            device.write(e.EV_KEY, e.KEY_LEFTMETA, 0)
            device.syn()
            time.sleep(0.4)
    finally:
        device.close()


def _safe_children(window):
    try:
        return window.query_tree().children
    except Exception:
        return []


def find_game_window(display: str) -> int | None:
    """Window id of the real Teardown window on `display`, or None.

    The game creates several windows sharing WM_CLASS, including a 1x1 unmapped helper.
    Returning that one made the visibility check fail permanently: a 1x1 unmapped window
    can never be read, so the harness concluded the game was hidden while it was in fact
    running in plain sight. Pick the largest mapped window instead.
    """
    from Xlib import X
    from Xlib import display as xdisplay

    conn = xdisplay.Display(display)
    try:
        best_id, best_area = None, 0
        pending = [conn.screen().root]
        while pending:
            window = pending.pop()
            try:
                wm_class = window.get_wm_class()
                geometry = window.get_geometry()
                mapped = window.get_attributes().map_state == X.IsViewable
            except Exception:
                pending.extend(_safe_children(window))
                continue
            if wm_class and WINDOW_CLASS in wm_class and mapped and geometry.width > 300:
                area = geometry.width * geometry.height
                if area > best_area:
                    best_id, best_area = window.id, area
            pending.extend(_safe_children(window))
        return best_id
    finally:
        conn.close()
