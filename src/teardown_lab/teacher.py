# ABOUTME: Privileged teacher for distillation: solves the task using world state the
# ABOUTME: student never sees, and emits the 7-dim action (including declare) to imitate.

from __future__ import annotations

import math

import numpy as np

from teardown_lab.pixel_env import ACTION_DIM, DECLARE_INDEX
from teardown_lab.referee import success
from teardown_lab.state import GameState

# The band where a swing actually connects. Pressing forward inside it shoves the player
# onto the pile: player height climbs, the tower ends up underfoot, and the centroid
# bearing then swings wildly (measured: -97 deg to +74 deg between steps) because a
# direction to something at your feet is ill-conditioned. So there is a floor as well as
# a ceiling, and the teacher backs off below it.
SWING_RANGE = 2.2
TOO_CLOSE = 1.3
AIM_TOLERANCE = 12.0
# Below this distance the bearing is noise; stop steering rather than chase it.
BEARING_DEADZONE = 1.6

# After the tower falls, keep acting for a moment before declaring. The student copies
# this timing, so it learns to end the episode having actually looked at the result
# rather than declaring blind the instant it swings.
VERIFY_STEPS = 8


class PrivilegedTeacher:
    """Scripted expert over privileged state, producing student-shaped actions.

    Stateful across a single episode (it counts verification steps), so construct one
    per episode or call `reset()`.
    """

    def __init__(self, k: int = 5, threshold: float = 0.5, verify_steps: int = VERIFY_STEPS):
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
        # Close in, the bearing to the centroid is ill-conditioned; steering on it makes
        # the teacher spin in place instead of swinging.
        if distance > BEARING_DEADZONE:
            action[0] = float(np.clip(bearing / 45.0, -1.0, 1.0))

        solved = success(state, self.k, self.threshold)
        if solved:
            self._solved_for += 1
            # Turn toward the wreckage while verifying, then declare.
            if self._solved_for >= self.verify_steps:
                action[DECLARE_INDEX] = 1.0
            return action

        self._solved_for = 0

        if distance < TOO_CLOSE:
            # Standing on/inside the pile: back off to a range where a swing lands,
            # and keep swinging on the way out.
            action[3] = -1.0
            action[5] = 1.0
            return action

        if abs(bearing) > AIM_TOLERANCE and distance > BEARING_DEADZONE:
            return action  # turn in place first

        if distance > SWING_RANGE:
            action[3] = 1.0  # walk forward
            return action

        # In the band: keep a light forward press while swinging. The sledge does not
        # reach at the top of the band, so this press is load-bearing - removing it made
        # every episode fail. The TOO_CLOSE branch above is what stops it becoming a
        # climb.
        action[3] = 0.4
        action[5] = 1.0
        return action
