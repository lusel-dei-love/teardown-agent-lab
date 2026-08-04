# ABOUTME: Tests for the non-privileged env: observation carries no world state, the
# ABOUTME: agent controls termination via declare, and declarations are scored.

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from teardown_lab.actuator import Actuator, RecordingBackend
from teardown_lab.bridge import FakeBridge
from teardown_lab.frames import FakeFrameGrabber
from teardown_lab.pixel_env import ACTION_DIM, DECLARE_INDEX, PixelEnvConfig, TeardownPixelEnv
from tests.helpers import state_with_displaced_blocks


def make_env(states, **cfg_kwargs):
    return TeardownPixelEnv(
        bridge=FakeBridge(states),
        actuator=Actuator(RecordingBackend()),
        frames=FakeFrameGrabber(),
        cfg=PixelEnvConfig(**cfg_kwargs),
        sleeper=lambda _: None,
    )


def noop():
    return np.zeros(ACTION_DIM, dtype=np.float32)


def declare():
    a = noop()
    a[DECLARE_INDEX] = 1.0
    return a


def test_observation_contains_no_world_state():
    # The whole point of the amendment: block poses must not be reachable from the obs.
    # Displace the tower by a distinctive amount so any leak into proprio is visible.
    standing = state_with_displaced_blocks(n_displaced=9, dist=7.77, yaw=30.0, pitch=-4.0)
    env = make_env([standing])
    obs, _ = env.reset()

    assert set(obs) == {"pixels", "proprio"}
    assert obs["pixels"].shape == (126, 224, 3)
    # Proprioception is exactly yaw, pitch and own velocity - 5 numbers, no block data.
    assert obs["proprio"].shape == (5,)
    assert obs["proprio"].tolist() == pytest.approx([30.0, -4.0, 0.0, 0.0, 0.0])

    # No block coordinate (nor the displacement that produced it) appears anywhere.
    leaked = set(np.round(obs["proprio"], 3).tolist())
    assert 7.77 not in leaked
    for block in standing.blocks:
        assert not leaked & set(np.round(block.pos, 3).tolist()) - {0.0}


def test_episode_does_not_end_on_privileged_success():
    # A solved tower must NOT terminate the episode by itself: only the agent can.
    solved = state_with_displaced_blocks(n_displaced=9, dist=2.0)
    env = make_env([solved])
    env.reset()
    _, _, terminated, truncated, info = env.step(noop())
    assert info["success"] is True
    assert terminated is False
    assert truncated is False


def test_declare_terminates_and_is_rewarded_when_correct():
    solved = state_with_displaced_blocks(n_displaced=9, dist=2.0)
    env = make_env([solved], declare_correct_bonus=10.0)
    env.reset()
    _, reward, terminated, _, info = env.step(declare())
    assert terminated is True
    assert info["declared"] is True
    assert info["false_declaration"] is False
    assert reward >= 10.0


def test_false_declaration_is_penalised_and_flagged():
    standing = state_with_displaced_blocks(n_displaced=0, dist=0.0)
    env = make_env([standing], declare_wrong_penalty=-5.0)
    env.reset()
    _, reward, terminated, _, info = env.step(declare())
    assert terminated is True
    assert info["false_declaration"] is True
    assert reward <= -5.0


def test_timeout_truncates_without_declaration():
    standing = state_with_displaced_blocks(n_displaced=0, dist=0.0, t=99.0)
    env = make_env([standing], timeout_s=0.0)
    env.reset()
    _, _, terminated, truncated, _ = env.step(noop())
    assert terminated is False
    assert truncated is True


def test_proprio_reports_own_velocity():
    a = state_with_displaced_blocks(n_displaced=0, dist=0.0, t=0.0, player_pos=(0.0, 0.0, 0.0))
    b = state_with_displaced_blocks(n_displaced=0, dist=0.0, t=1.0, player_pos=(2.0, 0.0, 0.0))
    env = make_env([a, b])
    env.reset()
    obs, *_ = env.step(noop())
    assert obs["proprio"][2] == pytest.approx(2.0, abs=1e-3)


def test_env_checker_passes():
    standing = state_with_displaced_blocks(n_displaced=0, dist=0.0)
    env = make_env([standing])
    check_env(env, skip_render_check=True)
