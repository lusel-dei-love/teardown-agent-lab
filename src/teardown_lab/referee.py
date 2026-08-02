# ABOUTME: Pure reward and success rules for the tower-knockdown task, computed host-side
# ABOUTME: from GameState only: no I/O, no game imports, no hidden state between calls.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from teardown_lab.state import GameState


@dataclass(frozen=True)
class RewardConfig:
    """Weights and thresholds of the shaped reward. Defaults are the task's contract."""

    approach_w: float = 0.05
    displace_w: float = 1.0
    success_bonus: float = 10.0
    k: int = 4
    threshold: float = 0.5


def displacements(state: GameState) -> np.ndarray:
    """Per-block distance (metres) between the block's current pos and its spawn."""
    if not state.blocks:
        return np.zeros(0, dtype=float)
    pos = np.array([b.pos for b in state.blocks], dtype=float)
    spawn = np.array([b.spawn for b in state.blocks], dtype=float)
    return np.linalg.norm(pos - spawn, axis=1)


def tower_centroid(state: GameState) -> np.ndarray:
    """Mean of the blocks' current positions; the point the agent should approach."""
    if not state.blocks:
        return np.zeros(3, dtype=float)
    return np.array([b.pos for b in state.blocks], dtype=float).mean(axis=0)


def success(state: GameState, k: int = 5, threshold: float = 0.5) -> bool:
    """True once at least `k` blocks sit more than `threshold` metres off their spawn."""
    return int(np.count_nonzero(displacements(state) > threshold)) >= k


def reward(prev: GameState, curr: GameState, cfg: RewardConfig) -> float:
    """Displacement delta + approach shaping, plus the bonus on the step success starts."""
    displace = float(displacements(curr).sum() - displacements(prev).sum())

    prev_dist = float(np.linalg.norm(np.array(prev.player_pos) - tower_centroid(prev)))
    curr_dist = float(np.linalg.norm(np.array(curr.player_pos) - tower_centroid(curr)))
    approach = prev_dist - curr_dist

    became_successful = success(curr, cfg.k, cfg.threshold) and not success(
        prev, cfg.k, cfg.threshold
    )
    bonus = cfg.success_bonus if became_successful else 0.0

    return cfg.displace_w * displace + cfg.approach_w * approach + bonus
