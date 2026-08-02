# ABOUTME: The non-privileged environment: the policy sees only pixels + its own
# ABOUTME: proprioception, declares when it thinks the task is done, and is scored on it.

from __future__ import annotations

import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from teardown_lab.actuator import Action, Actuator
from teardown_lab.bridge import Bridge
from teardown_lab.frames import FRAME_H, FRAME_W
from teardown_lab.referee import RewardConfig, reward as compute_reward, success
from teardown_lab.state import GameState

# Proprioception only: the agent's own yaw, pitch, and its own velocity (finite-
# differenced from consecutive player positions). Nothing here describes the world.
PROPRIO_DIM = 5
# Yaw accumulates unbounded in this engine, so allow plenty of headroom.
PROPRIO_LIMIT = 1.0e6

# Action layout: look_dx, look_dy, move_x, move_y, grab, swing, declare.
ACTION_DIM = 7
DECLARE_INDEX = 6


@dataclass(frozen=True)
class PixelEnvConfig:
    """Episode shape for the non-privileged env."""

    hz: float = 10.0
    timeout_s: float = 60.0
    k: int = 4
    threshold: float = 0.5
    # Reward shaping still uses privileged state - that is allowed, it never reaches
    # the policy's input. Only the observation is restricted.
    declare_correct_bonus: float = 10.0
    declare_wrong_penalty: float = -5.0


class TeardownPixelEnv(gym.Env):
    """Tower knockdown seen through pixels.

    The privileged bridge is still used - for reward, for scoring, and for the teacher -
    but its state never enters the observation. The agent's only view of the world is
    the downsampled frame; the only thing it knows about itself is proprioception.
    """

    metadata = {"render_modes": []}
    render_mode = None

    def __init__(
        self,
        bridge: Bridge,
        actuator: Actuator,
        frames,
        cfg: PixelEnvConfig | None = None,
        sleeper=time.sleep,
        refocus=None,
    ):
        super().__init__()
        self.bridge = bridge
        self.actuator = actuator
        self.frames = frames
        # Callable returning True if the game window was (re)focused; None disables
        # recovery, which is what tests want.
        self.refocus = refocus
        self.cfg = cfg or PixelEnvConfig()
        self.sleeper = sleeper
        self.reward_cfg = RewardConfig(k=self.cfg.k, threshold=self.cfg.threshold)

        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Box(
                    low=0, high=255, shape=(FRAME_H, FRAME_W, 3), dtype=np.uint8
                ),
                # Bounded rather than infinite: yaw/pitch are degrees and velocity is
                # metres per second, so this is generous, and it keeps the env checker
                # from warning about an unbounded space.
                "proprio": spaces.Box(
                    low=-PROPRIO_LIMIT,
                    high=PROPRIO_LIMIT,
                    shape=(PROPRIO_DIM,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32
        )

        self._state: GameState | None = None
        self._prev_state: GameState | None = None
        self._t0 = 0.0
        self._next_deadline: float | None = None

    def _pace(self) -> None:
        """Sleep only the remainder of the control period.

        Sleeping a full period and *then* doing the work (state read ~15 ms, frame grab
        ~30 ms) silently drops the real control rate - measured 156 ms per step against a
        100 ms target, i.e. 6.4 Hz while claiming 10 Hz. Deadline-based pacing keeps the
        rate honest and constant, which matters because the policy's actions are
        interpreted as per-tick commands.
        """
        period = 1.0 / self.cfg.hz
        now = time.monotonic()
        if self._next_deadline is None:
            self._next_deadline = now + period
            self.sleeper(period)
            return
        remaining = self._next_deadline - now
        if remaining > 0:
            self.sleeper(remaining)
            self._next_deadline += period
        else:
            # Overran the budget: resync rather than accumulate lateness forever.
            self._next_deadline = now + period

    # -- observation ---------------------------------------------------------

    def _proprio(self, state: GameState) -> np.ndarray:
        """Own yaw, pitch and velocity. Contains nothing about the world."""
        if self._prev_state is None:
            vel = np.zeros(3, dtype=np.float32)
        else:
            dt = max(state.t - self._prev_state.t, 1e-3)
            vel = (
                np.array(state.player_pos, dtype=np.float32)
                - np.array(self._prev_state.player_pos, dtype=np.float32)
            ) / dt
        return np.concatenate(
            [np.array([state.yaw, state.pitch], dtype=np.float32), vel]
        ).astype(np.float32)

    def _observe(self, state: GameState) -> dict:
        return {"pixels": self.frames.grab(), "proprio": self._proprio(state)}

    # -- gym api -------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.actuator.release_all()
        self.bridge.send({"cmd": "reset", "seed": seed})
        state = self._read()
        self._state = state
        self._prev_state = None
        self._t0 = state.t
        self._next_deadline = None
        return self._observe(state), self._info(state, declared=False)

    def step(self, action):
        vec = np.asarray(action, dtype=np.float32)
        declared = bool(vec[DECLARE_INDEX] > 0)

        self.actuator.apply(self._to_action(vec))
        self._pace()

        prev = self._state
        curr = self._read()
        self._prev_state = prev
        self._state = curr

        reward = compute_reward(prev, curr, self.reward_cfg)
        solved = success(curr, self.cfg.k, self.cfg.threshold)

        # The episode ends when the AGENT says so, not when the privileged referee
        # fires: an oracle termination would leak world state through episode structure
        # and would let the agent skip ever looking at the tower.
        terminated = declared
        if declared:
            reward += (
                self.cfg.declare_correct_bonus
                if solved
                else self.cfg.declare_wrong_penalty
            )

        truncated = (not terminated) and (curr.t - self._t0) >= self.cfg.timeout_s
        return (
            self._observe(curr),
            reward,
            terminated,
            truncated,
            self._info(curr, declared=declared),
        )

    def close(self) -> None:
        self.actuator.release_all()
        self.frames.close()
        self.bridge.close()

    # -- internals -----------------------------------------------------------

    def _read(self) -> GameState:
        state = self.bridge.read_state()
        if state is not None:
            return state

        # No fresh frame usually means the game paused because something else took
        # focus. Take it back and retry once before giving up, so an unattended run
        # survives a stray popup rather than dying mid-episode.
        if self.refocus is not None and self.refocus():
            state = self.bridge.read_state(timeout=5.0)
            if state is not None:
                return state
        raise RuntimeError("bridge returned no state (game paused or mod not running?)")

    def _info(self, state: GameState, declared: bool) -> dict:
        """Privileged truth for scoring and for the teacher. Never fed to the policy."""
        solved = success(state, self.cfg.k, self.cfg.threshold)
        return {
            "success": solved,
            "declared": declared,
            # A declaration made while the tower still stands - the metric the old
            # oracle-terminated design could not express.
            "false_declaration": declared and not solved,
            "t": state.t - self._t0,
            "privileged_state": state,
        }

    @staticmethod
    def _to_action(vec: np.ndarray) -> Action:
        return Action(
            look_dx=float(vec[0]),
            look_dy=float(vec[1]),
            move_x=float(vec[2]),
            move_y=float(vec[3]),
            grab=bool(vec[4] > 0),
            swing=bool(vec[5] > 0),
        )
