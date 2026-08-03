# ABOUTME: Resolves which X display to drive. The display number is NOT stable across
# ABOUTME: reboots on this machine, so nothing in the harness may hardcode it.

from __future__ import annotations

import os
from pathlib import Path

X11_SOCKET_DIR = Path("/tmp/.X11-unix")


def detect_display(default: str = ":0") -> str:
    """The display to use: GAME_DISPLAY, else the live X socket, else `default`.

    A greeter login puts the session on :1 while GDM autologin puts it on :0, so the
    number changes across reboots. Hardcoding it meant that after one reboot every tool
    silently pointed at a display that no longer existed - the game looked unlaunchable
    and the visibility check could never pass.
    """
    override = os.environ.get("GAME_DISPLAY")
    if override:
        return override

    try:
        sockets = sorted(
            p.name for p in X11_SOCKET_DIR.iterdir() if p.name.startswith("X")
        )
    except OSError:
        return default

    for name in sockets:
        suffix = name[1:]
        if suffix.isdigit():
            return f":{suffix}"
    return default
