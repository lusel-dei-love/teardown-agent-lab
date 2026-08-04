# ABOUTME: Tests for TeardownTowerEnv against FakeBridge + RecordingBackend + fake sleeper.
# ABOUTME: Covers obs layout, reset/seed wiring, termination, truncation and env_checker.

import gymnasium
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from teardown_lab.actuator import Actuator, RecordingBackend
from teardown_lab.bridge import FakeBridge
from teardown_lab.env import EnvConfig, TeardownTowerEnv
from teardown_lab.referee import RewardConfig, reward as referee_reward

from tests.helpers import state_with_displaced_blocks


class FakeSleeper:
    def __init__(self):
        self.slept: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.slept.append(seconds)


def build_env(states, cfg: EnvConfig | None = None):
    bridge = FakeBridge(states)
    backend = RecordingBackend()
    sleeper = FakeSleeper()
    env = TeardownTowerEnv(
        bridge, Actuator(backend), cfg or EnvConfig(), sleeper=sleeper
    )
    return env, bridge, backend, sleeper


def noop_action() -> np.ndarray:
    return np.zeros(6, dtype=np.float32)


def test_spaces():
    env, *_ = build_env([state_with_displaced_blocks(0, 0.0)])
    assert env.observation_space.shape == (32,)
    assert env.action_space.shape == (6,)
    assert np.all(env.action_space.low == -1.0)
    assert np.all(env.action_space.high == 1.0)


def test_reset_returns_obs_built_from_first_state():
    state = state_with_displaced_blocks(1, 2.0)
    env, *_ = build_env([state])
    obs, info = env.reset(seed=5)
    assert obs.shape == (32,)
    assert obs.dtype == np.float32
    assert obs in env.observation_space
    assert list(obs[0:3]) == list(state.player_pos)
    assert obs[3] == pytest.approx(state.yaw)
    assert obs[4] == pytest.approx(state.pitch)
    expected_rel = np.array(state.blocks[0].pos) - np.array(state.player_pos)
    assert obs[5:8] == pytest.approx(expected_rel)
    assert info["success"] is False


def test_reset_sends_seed_to_bridge():
    env, bridge, *_ = build_env([state_with_displaced_blocks(0, 0.0)])
    env.reset(seed=17)
    assert bridge.sent == [{"cmd": "reset", "seed": 17}]


def test_step_actuates_and_sleeps_on_the_hz_grid():
    states = [state_with_displaced_blocks(0, 0.0, t=i * 0.1) for i in range(3)]
    env, _, backend, sleeper = build_env(states, EnvConfig(hz=10.0))
    env.reset(seed=1)
    backend.calls.clear()
    env.step(np.array([0.5, -0.5, 0.0, 1.0, 1.0, -1.0], dtype=np.float32))
    scale = env.actuator.look_scale
    assert ("move_mouse", round(0.5 * scale), round(-0.5 * scale)) in backend.calls
    assert ("key", "w", True) in backend.calls
    assert ("button", "right", True) in backend.calls  # grab > 0
    assert ("button", "left", True) not in backend.calls  # swing < 0
    assert sleeper.slept == [pytest.approx(0.1)]


def test_success_terminates_and_pays_bonus():
    prev = state_with_displaced_blocks(4, 1.0, t=0.0)
    curr = state_with_displaced_blocks(5, 1.0, t=0.1)
    env, *_ = build_env([prev, curr], EnvConfig(k=5))
    env.reset(seed=1)
    obs, reward, terminated, truncated, info = env.step(noop_action())
    assert terminated is True
    assert truncated is False
    assert info["success"] is True
    assert info["t"] == pytest.approx(0.1)
    assert reward == pytest.approx(referee_reward(prev, curr, RewardConfig(k=5)))
    assert reward > 10.0


def test_timeout_truncates():
    states = [
        state_with_displaced_blocks(0, 0.0, t=0.0),
        state_with_displaced_blocks(0, 0.0, t=0.5),
        state_with_displaced_blocks(0, 0.0, t=1.5),
    ]
    env, *_ = build_env(states, EnvConfig(timeout_s=1.0))
    env.reset(seed=1)
    _, _, terminated, truncated, _ = env.step(noop_action())
    assert (terminated, truncated) == (False, False)
    _, _, terminated, truncated, _ = env.step(noop_action())
    assert terminated is False
    assert truncated is True


def test_close_releases_inputs_and_bridge():
    env, bridge, backend, _ = build_env([state_with_displaced_blocks(0, 0.0)])
    env.reset(seed=1)
    env.step(np.array([0, 0, 0, 1, 0, 0], dtype=np.float32))
    backend.calls.clear()
    env.close()
    assert ("key", "w", False) in backend.calls
    assert bridge.closed is True


def test_passes_gymnasium_env_checker():
    states = [state_with_displaced_blocks(i, 0.2, t=i * 0.1) for i in range(4)]
    env, *_ = build_env(states)
    assert isinstance(env, gymnasium.Env)
    check_env(env, skip_render_check=True)
