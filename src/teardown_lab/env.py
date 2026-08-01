# ABOUTME: Gymnasium environment wiring Bridge (state), Actuator (input) and referee
# ABOUTME: (reward) into a fixed-rate episode loop over the 9-block tower task.

from __future__ import annotations

import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from teardown_lab.actuator import Action, Actuator
from teardown_lab.bridge import Bridge
from teardown_lab.referee import RewardConfig, reward as compute_reward, success
from teardown_lab.state import GameState

# Finite bound on the observation box: world coordinates never come near it, and a
# bounded space keeps the gymnasium env checker quiet.
OBS_LIMIT = 1.0e4


@dataclass(frozen=True)
class EnvConfig:
    """Episode shape: control rate, wall-clock budget and the success rule."""

    hz: float = 10.0
    timeout_s: float = 60.0
    k: int = 5
    threshold: float = 0.5
    n_blocks: int = 9


class TeardownTowerEnv(gym.Env):
    """Privileged-state env: obs = player pose + block positions relative to the player."""

    metadata = {"render_modes": []}
    render_mode = None

    def __init__(
        self,
        bridge: Bridge,
        actuator: Actuator,
        cfg: EnvConfig,
        sleeper=time.sleep,
    ):
        super().__init__()
        self.bridge = bridge
        self.actuator = actuator
        self.cfg = cfg
        self.sleeper = sleeper
        self.reward_cfg = RewardConfig(k=cfg.k, threshold=cfg.threshold)

        n_obs = 5 + 3 * cfg.n_blocks
        self.observation_space = spaces.Box(
            low=-OBS_LIMIT, high=OBS_LIMIT, shape=(n_obs,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        self._state: GameState | None = None
        self._t0 = 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.actuator.release_all()
        self.bridge.send({"cmd": "reset", "seed": seed})
        state = self._read()
        self._state = state
        self._t0 = state.t
        return self._observe(state), self._info(state)

    def step(self, action):
        self.actuator.apply(self._to_action(np.asarray(action, dtype=np.float32)))
        self.sleeper(1.0 / self.cfg.hz)

        prev = self._state
        curr = self._read()
        self._state = curr

        reward = compute_reward(prev, curr, self.reward_cfg)
        terminated = success(curr, self.cfg.k, self.cfg.threshold)
        truncated = (not terminated) and (curr.t - self._t0) >= self.cfg.timeout_s
        return self._observe(curr), reward, terminated, truncated, self._info(curr)

    def close(self) -> None:
        self.actuator.release_all()
        self.bridge.close()

    def _read(self) -> GameState:
        state = self.bridge.read_state()
        if state is None:
            raise RuntimeError("bridge returned no state")
        return state

    def _info(self, state: GameState) -> dict:
        return {
            "success": success(state, self.cfg.k, self.cfg.threshold),
            "t": state.t - self._t0,
        }

    def _observe(self, state: GameState) -> np.ndarray:
        player = np.array(state.player_pos, dtype=np.float32)
        blocks = np.array([b.pos for b in state.blocks], dtype=np.float32) - player
        return np.concatenate(
            [player, np.array([state.yaw, state.pitch], dtype=np.float32), blocks.ravel()]
        ).astype(np.float32)

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
