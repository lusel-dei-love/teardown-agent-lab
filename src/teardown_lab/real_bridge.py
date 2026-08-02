# ABOUTME: The live Bridge: reads state by polling Teardown's savegame.xml registry file
# ABOUTME: and sends commands by creating/removing files the mod polls with HasFile.

from __future__ import annotations

import time
from pathlib import Path

from teardown_lab.savegame import PayloadError, extract_payload, parse_payload
from teardown_lab.state import GameState

DEFAULT_PREFIX = Path.home() / ".steam/steam/steamapps/compatdata/1167630/pfx"
SAVEGAME_REL = "drive_c/users/steamuser/AppData/Local/Teardown/savegame.xml"
MODS_REL = "drive_c/users/steamuser/Documents/Teardown/mods"

RESET_FILE = "reset.txt"
HARD_RESET_FILE = "hardreset.txt"


class RealBridge:
    """Talks to the live game.

    State comes out through the registry: the mod writes one string key, the engine
    flushes the whole registry to savegame.xml every frame, and we poll that file.
    Commands go in through file existence, which the mod polls with HasFile - this
    needs no window focus and cannot collide with gameplay keybinds.
    """

    def __init__(
        self,
        mod: str = "teardownlab",
        prefix: Path | None = None,
        savegame: Path | None = None,
        mod_dir: Path | None = None,
        poll_interval: float = 0.005,
        sleeper=time.sleep,
    ):
        prefix = prefix or DEFAULT_PREFIX
        self.mod = mod
        self.savegame = savegame or (prefix / SAVEGAME_REL)
        self.mod_dir = mod_dir or (prefix / MODS_REL / mod)
        self.poll_interval = poll_interval
        self._sleep = sleeper
        self._last_seq: int | None = None
        self._last_mtime_ns: int = -1

    # -- state ---------------------------------------------------------------

    def _read_once(self) -> GameState | None:
        """One attempt: skip unchanged files, tolerate torn reads and stale frames."""
        try:
            mtime_ns = self.savegame.stat().st_mtime_ns
        except OSError:
            return None
        if mtime_ns == self._last_mtime_ns:
            return None
        self._last_mtime_ns = mtime_ns

        try:
            text = self.savegame.read_text(encoding="utf-8", errors="replace")
        except OSError:
            # The engine swaps this file by rename; a reader can hit the gap.
            return None

        payload = extract_payload(text, self.mod)
        if payload is None:
            return None

        try:
            state = parse_payload(payload)
        except PayloadError:
            return None

        if self._last_seq is not None and state.seq <= self._last_seq:
            return None  # same frame re-read
        self._last_seq = state.seq
        return state

    def read_state(self, timeout: float = 1.0) -> GameState | None:
        deadline = time.monotonic() + timeout
        while True:
            state = self._read_once()
            if state is not None:
                return state
            if time.monotonic() >= deadline:
                return None
            self._sleep(self.poll_interval)

    # -- commands ------------------------------------------------------------

    def send(self, cmd: dict) -> None:
        if cmd.get("cmd") != "reset":
            raise ValueError(f"unsupported command: {cmd!r}")
        self.reset(timeout=cmd.get("timeout", 10.0))

    def reset(self, timeout: float = 10.0) -> GameState | None:
        """Request a reset and block until the mod reports a new episode.

        The mod resets on the file's rising edge, so hold the file until the episode
        counter advances, then clear it - that is the whole handshake.
        """
        before = self.read_state(timeout=timeout)
        target_episode = (before.episode + 1) if before else None

        reset_path = self.mod_dir / RESET_FILE
        reset_path.write_text("1", encoding="utf-8")
        try:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                state = self._read_once()
                # Wait for a LIVE frame: mid-reset the tower is being rebuilt and its
                # blocks are still settling, so any displacement read now is spurious.
                if (
                    state is not None
                    and state.live
                    and (target_episode is None or state.episode >= target_episode)
                ):
                    return state
                self._sleep(self.poll_interval)
            return None
        finally:
            reset_path.unlink(missing_ok=True)
            # Let the mod observe the file disappear so the next reset is a rising edge.
            self._sleep(0.1)

    def hard_reset(self, timeout: float = 90.0) -> GameState | None:
        """Reload the level, restoring terrain the agent has destroyed.

        Soft resets respawn the tower but leave the world cratered; across a long run
        that silently changes the task. The level reload takes ~10-20 s and restarts the
        mod, so the episode counter goes back to zero - callers must not assume it only
        increases across a hard reset.
        """
        path = self.mod_dir / HARD_RESET_FILE
        path.write_text("1", encoding="utf-8")
        try:
            self._sleep(3.0)  # let the mod see the file and call Restart()
        finally:
            path.unlink(missing_ok=True)

        self._last_seq = None
        self._last_mtime_ns = -1
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self._read_once()
            if state is not None and state.live:
                return state
            self._sleep(0.25)
        return None

    def close(self) -> None:
        (self.mod_dir / RESET_FILE).unlink(missing_ok=True)
        (self.mod_dir / HARD_RESET_FILE).unlink(missing_ok=True)
