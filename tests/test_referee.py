# ABOUTME: Tests for the pure referee: per-block displacement, success predicate and reward.
# ABOUTME: Runs headless against synthetic GameStates.

import numpy as np
import pytest

from teardown_lab import referee
from teardown_lab.referee import RewardConfig

from tests.helpers import state_with_displaced_blocks


def test_displacements_measures_distance_from_spawn():
    state = state_with_displaced_blocks(n_displaced=2, dist=1.5)
    disp = referee.displacements(state)
    assert isinstance(disp, np.ndarray)
    assert disp.shape == (9,)
    assert disp[0] == pytest.approx(1.5)
    assert disp[1] == pytest.approx(1.5)
    assert np.all(disp[2:] == 0.0)


def test_success_requires_k_blocks():
    s = state_with_displaced_blocks(n_displaced=4, dist=1.0)
    assert not referee.success(s, k=5, threshold=0.5)
    s = state_with_displaced_blocks(n_displaced=5, dist=1.0)
    assert referee.success(s, k=5, threshold=0.5)


def test_success_needs_blocks_past_threshold():
    s = state_with_displaced_blocks(n_displaced=9, dist=0.4)
    assert not referee.success(s, k=5, threshold=0.5)


def test_reward_displacement_delta():
    prev = state_with_displaced_blocks(0, 0.0)
    curr = state_with_displaced_blocks(1, 0.4)  # below threshold still rewards delta
    cfg = RewardConfig(approach_w=0.0)
    assert referee.reward(prev, curr, cfg) == pytest.approx(0.4)


def test_reward_scales_with_displace_weight():
    prev = state_with_displaced_blocks(0, 0.0)
    curr = state_with_displaced_blocks(2, 0.4)
    cfg = RewardConfig(approach_w=0.0, displace_w=2.0)
    assert referee.reward(prev, curr, cfg) == pytest.approx(1.6)


def test_reward_approach_term():
    prev = state_with_displaced_blocks(0, 0.0, player_pos=(0.0, 0.0, -5.0))
    curr = state_with_displaced_blocks(0, 0.0, player_pos=(0.0, 0.0, -4.0))
    cfg = RewardConfig(displace_w=0.0, approach_w=0.5)
    prev_dist = np.linalg.norm(np.array([0.0, 0.0, -5.0]) - referee.tower_centroid(prev))
    curr_dist = np.linalg.norm(np.array([0.0, 0.0, -4.0]) - referee.tower_centroid(curr))
    assert referee.reward(prev, curr, cfg) == pytest.approx(0.5 * (prev_dist - curr_dist))


def test_success_bonus_paid_once():
    before = state_with_displaced_blocks(4, 1.0)
    first = state_with_displaced_blocks(5, 1.0)
    again = state_with_displaced_blocks(5, 1.0)
    cfg = RewardConfig(approach_w=0.0, displace_w=0.0, success_bonus=10.0)
    assert referee.reward(before, first, cfg) == pytest.approx(10.0)
    assert referee.reward(first, again, cfg) == pytest.approx(0.0)


def test_reward_combines_terms():
    prev = state_with_displaced_blocks(4, 1.0, player_pos=(0.0, 0.0, -5.0))
    curr = state_with_displaced_blocks(5, 1.0, player_pos=(0.0, 0.0, -4.0))
    cfg = RewardConfig()
    prev_dist = np.linalg.norm(np.array(prev.player_pos) - referee.tower_centroid(prev))
    curr_dist = np.linalg.norm(np.array(curr.player_pos) - referee.tower_centroid(curr))
    expected = (
        cfg.displace_w * 1.0
        + cfg.approach_w * (prev_dist - curr_dist)
        + cfg.success_bonus
    )
    assert referee.reward(prev, curr, cfg) == pytest.approx(expected)
