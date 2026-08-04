# ABOUTME: Keeps the game process alive for long unattended runs: detects death,
# ABOUTME: relaunches Steam and Teardown, and drives the menus back into a sandbox level.

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from teardown_lab.actuator import ensure_game_visible, find_game_window, focus_game_window
from teardown_lab.xdisplay import detect_display

APP_ID = "1167630"
GAME_PROCESS = "teardown.exe"
STEAM_BIN = "/usr/games/steam"

# Menu positions as FRACTIONS of the game window, calibrated by screenshot at 1920x1080.
# Absolute pixels do not survive: this box has no monitor attached, so after a reboot the
# X screen came back as a virtual 3840x2160 and every hardcoded coordinate was 2x off
# (and offset, since the window is not at the origin). Fractions of the live window
# geometry are resolution-independent.
PLAY_BUTTON = (609 / 1920, 76 / 1080)
SANDBOX_BUTTON = (609 / 1920, 255 / 1080)
FIRST_LEVEL = (513 / 1920, 600 / 1080)
# NOT a click target. The character screen's "Select" sits at (1743, 90), which on the
# MAIN MENU is the Quit button - so whenever an earlier click failed to land, the blind
# sequence quit the game (twice). Confirm that screen with Enter instead: harmless on
# every screen it can reach.
CHARACTER_SELECT = (1743 / 1920, 90 / 1080)


def window_rect(display: str) -> tuple[int, int, int, int] | None:
    """(x, y, width, height) of the game window in root coordinates."""
    from Xlib import display as xdisplay

    window_id = find_game_window(display)
    if window_id is None:
        return None
    conn = xdisplay.Display(display)
    try:
        window = conn.create_resource_object("window", window_id)
        geometry = window.get_geometry()
        origin = window.translate_coords(conn.screen().root, 0, 0)
        # translate_coords returns the offset OF the root relative to the window, so the
        # window's own root-space origin is its negation.
        return (-origin.x, -origin.y, geometry.width, geometry.height)
    except Exception:
        return None
    finally:
        conn.close()


def to_pixels(display: str, fraction: tuple[float, float]) -> tuple[int, int] | None:
    rect = window_rect(display)
    if rect is None:
        return None
    x, y, width, height = rect
    return (int(x + fraction[0] * width), int(y + fraction[1] * height))


def game_running() -> bool:
    """True if the game process exists.

    Matches the command name exactly: `pgrep -f teardown` once matched this session's own
    shell and a kernel thread, reporting a dead game as alive for minutes.
    """
    out = subprocess.run(["ps", "-eo", "comm"], capture_output=True, text=True).stdout
    return any(line.strip() == GAME_PROCESS for line in out.splitlines())


def steam_running() -> bool:
    return subprocess.run(["pgrep", "-x", "steam"], capture_output=True).returncode == 0


def _child_env(display: str) -> dict[str, str]:
    """Minimal environment for Steam. HOME comes from the runtime, never hardcoded."""
    return {
        "DISPLAY": display,
        "HOME": str(Path.home()),
        "PATH": "/usr/bin:/bin:/usr/games",
    }


def _launch(display: str, args: list[str]) -> None:
    subprocess.Popen(
        args,
        env=_child_env(display),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def start_steam(display: str, wait_s: float = 120.0) -> bool:
    if steam_running():
        return True
    _launch(display, [STEAM_BIN])
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if steam_running():
            time.sleep(30)  # the client needs to finish logging in before it will launch
            return True
        time.sleep(5)
    return False


def start_game(display: str, wait_s: float = 300.0) -> bool:
    """Launch the game and wait for its process. Restarts Steam if it refuses.

    Steam frequently declines to launch the game after a crash; a full client restart is
    the only reliable remedy observed on this machine.
    """
    if game_running():
        return True
    if not start_steam(display):
        return False

    _launch(display, [STEAM_BIN, f"steam://rungameid/{APP_ID}"])
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if game_running():
            return True
        time.sleep(10)

    # Second attempt with a clean Steam client.
    subprocess.run(
        [STEAM_BIN, "-shutdown"],
        env=_child_env(display),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(30)
    if not start_steam(display):
        return False
    _launch(display, [STEAM_BIN, f"steam://rungameid/{APP_ID}"])
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if game_running():
            return True
        time.sleep(10)
    return False


def _click(display: str, xy: tuple[int, int]) -> None:
    """Position with XTest, click with uinput - the game ignores XTest button events."""
    import os

    os.environ["DISPLAY"] = display
    import pyautogui
    from evdev import UInput
    from evdev import ecodes as e

    pyautogui.moveTo(*xy)
    time.sleep(0.4)
    device = UInput(
        {e.EV_KEY: [e.BTN_LEFT], e.EV_REL: [e.REL_X, e.REL_Y]}, name="td-supervisor"
    )
    try:
        time.sleep(1.4)
        device.write(e.EV_REL, e.REL_X, 1)
        device.syn()
        time.sleep(0.15)
        device.write(e.EV_KEY, e.BTN_LEFT, 1)
        device.syn()
        time.sleep(0.08)
        device.write(e.EV_KEY, e.BTN_LEFT, 0)
        device.syn()
    finally:
        device.close()


# Key names used by the menu driver. Named here so a test can assert they exist in
# evdev rather than discovering a typo against the live game (KEY_RETURN does not exist;
# it is KEY_ENTER).
MENU_KEYS = ("KEY_SPACE", "KEY_ENTER", "KEY_ESC")


def _key(display: str, key_name: str) -> None:
    from evdev import UInput
    from evdev import ecodes as e

    code = getattr(e, key_name)
    device = UInput({e.EV_KEY: [code]}, name="td-supervisor-kbd")
    try:
        time.sleep(1.4)
        device.write(e.EV_KEY, code, 1)
        device.syn()
        time.sleep(0.08)
        device.write(e.EV_KEY, code, 0)
        device.syn()
    finally:
        device.close()


def in_level(bridge, timeout: float = 5.0) -> bool:
    """True if the mod is publishing FRESH live state, i.e. a level really is loaded.

    A single read is not enough. savegame.xml persists across sessions, so after a
    reboot it still holds the last payload of the previous run - a stale frame that
    reports phase=live and fooled this check into declaring the game ready while it sat
    at the main menu. Requiring the tick counter to advance is what distinguishes a
    running mod from a leftover file.
    """
    first = bridge.read_state(timeout=timeout)
    if first is None or not first.live:
        return False
    second = bridge.read_state(timeout=timeout)
    return second is not None and second.live and second.seq > first.seq


def drive_into_level(display: str, bridge, timeout: float = 240.0) -> bool:
    """Click through the menus until the mod reports a live level.

    Each step is followed by a state check rather than a fixed sleep: the only reliable
    signal that we are in a level is the mod publishing, and blind click sequences have
    previously landed on the wrong screen (once on Quit, killing the session).
    """
    deadline = time.monotonic() + timeout

    _key(display, "KEY_SPACE")  # dismiss the photosensitivity warning on first boot
    time.sleep(3)

    for step in (PLAY_BUTTON, SANDBOX_BUTTON, FIRST_LEVEL):
        if time.monotonic() > deadline:
            return False
        ensure_game_visible(display)
        point = to_pixels(display, step)
        if point is None:
            return False
        _click(display, point)
        time.sleep(6)
        if in_level(bridge, timeout=2.0):
            return True

    # Confirm character selection with the keyboard, never with the coordinate.
    for _ in range(3):
        if time.monotonic() > deadline:
            return False
        _key(display, "KEY_ENTER")
        time.sleep(8)
        if in_level(bridge, timeout=2.0):
            return True

    while time.monotonic() < deadline:
        if in_level(bridge, timeout=3.0):
            return True
        time.sleep(5)
    return False


def ensure_playable(display: str | None = None, bridge=None) -> bool:
    """Bring the game from any state (dead, menu, hidden) to a live level."""
    from teardown_lab.real_bridge import RealBridge

    display = display or detect_display()
    bridge = bridge or RealBridge()

    if not start_game(display):
        return False
    focus_game_window(display)
    ensure_game_visible(display)

    if in_level(bridge, timeout=5.0):
        return True
    return drive_into_level(display, bridge)
