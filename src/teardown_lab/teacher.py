# ABOUTME: Privileged teacher for distillation: solves the task using world state the
# ABOUTME: student never sees, and emits the 7-dim action (including declare) to imitate.

from __future__ import annotations

import math

import numpy as np

from teardown_lab.pixel_env import ACTION_DIM, DECLARE_INDEX
from teardown_lab.referee import success
from teardown_lab.state import GameState

SWING_RANGE = 2.2
AIM_TOLERANCE = 12.0

# After the tower falls, keep acting for a moment before declaring. The student copies
# this timing, so it learns to end the episode having actually looked at the result
# rather than declaring blind the instant it swings.
VERIFY_STEPS = 8


class PrivilegedTeacher:
    """Scripted expert over privileged state, producing student-shaped actions.

    Deliberately simple. Attempts to make it smarter - backing off when too close,
    a bearing deadzone, aiming only at blocks still standing - each looked reasonable
    and each measured WORSE on the live game (4/4 failures at one point), because they
    broke the sustained contact that actually topples the stack. Change this only
    against measured success rates, one variable at a time.
    """

    def __init__(self, k: int = 4, threshold: float = 0.5, verify_steps: int = VERIFY_STEPS):
        self.k = k
        self.threshold = threshold
        self.verify_steps = verify_steps
        self._solved_for = 0

    def reset(self) -> None:
        self._solved_for = 0

    def act(self, state: GameState) -> np.ndarray:
        action = np.zeros(ACTION_DIM, dtype=np.float32)

        if not state.blocks_local:
            return action

        centroid = np.asarray(state.blocks_local, dtype=np.float32).mean(axis=0)
        right, _, back = centroid
        forward = -back
        distance = math.hypot(float(right), float(forward))

        bearing = math.degrees(math.atan2(float(right), float(forward)))
        action[0] = float(np.clip(bearing / 45.0, -1.0, 1.0))

        if success(state, self.k, self.threshold):
            self._solved_for += 1
            # Turn toward the wreckage while verifying, then declare.
            if self._solved_for >= self.verify_steps:
                action[DECLARE_INDEX] = 1.0
            return action

        self._solved_for = 0

        if abs(bearing) > AIM_TOLERANCE:
            return action  # turn in place first
        if distance > SWING_RANGE:
            action[3] = 1.0  # walk forward
            return action

        action[3] = 0.4  # keep closing so the sledge actually reaches
        action[5] = 1.0  # swing
        return action
