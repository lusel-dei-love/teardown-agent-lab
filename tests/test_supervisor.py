# ABOUTME: Tests for the game supervisor's pure parts: key names and coordinate maths.
# ABOUTME: Nothing here needs an X display or the game.

import pytest

from teardown_lab.supervisor import (
    CHARACTER_SELECT,
    FIRST_LEVEL,
    MENU_KEYS,
    PLAY_BUTTON,
    SANDBOX_BUTTON,
)


def test_menu_key_names_exist_in_evdev():
    # A typo here only surfaces against the live game; KEY_RETURN does not exist.
    ecodes = pytest.importorskip("evdev.ecodes")
    for name in MENU_KEYS:
        assert hasattr(ecodes, name), name


@pytest.mark.parametrize(
    "target", [PLAY_BUTTON, SANDBOX_BUTTON, FIRST_LEVEL, CHARACTER_SELECT]
)
def test_menu_targets_are_fractions(target):
    # Absolute pixels silently break when the screen resolution changes, which it does
    # on this machine across reboots.
    x, y = target
    assert 0.0 < x < 1.0
    assert 0.0 < y < 1.0


def test_steam_binary_resolves_an_existing_launcher():
    # The 24.04 upgrade deleted /usr/games/steam while ~/.steam kept working, so this
    # must never be a hardcoded path.
    from pathlib import Path

    from teardown_lab.supervisor import STEAM_CANDIDATES, steam_binary

    assert len(STEAM_CANDIDATES) > 1
    resolved = steam_binary()
    assert Path(resolved).exists()
