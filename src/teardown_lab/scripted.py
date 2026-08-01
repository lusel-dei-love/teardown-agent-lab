# ABOUTME: A hand-written policy that walks to the tower and swings, used to prove the
# ABOUTME: whole loop (bridge, actuator, referee) works on real physics before any RL.

from __future__ import annotations

import math

import numpy as np

from teardown_lab.env import TeardownTowerEnv

# Beyond this distance (metres) the policy walks; inside it, it swings.
SWING_RANGE = 2.2
# Yaw error (degrees) the policy tolerates before it walks rather than turns.
AIM_TOLERANCE = 12.0


def _tower_centroid(obs: np.ndarray, n_blocks: int) -> np.ndarray:
    """Mean block position, reconstructed from the observation's block-relative block."""
    blocks = obs[5 : 5 + 3 * n_blocks].reshape(n_blocks, 3)
    return blocks.mean(axis=0)


def scripted_topple(obs: np.ndarray, n_blocks: int = 9) -> np.ndarray:
    """Turn toward the tower, close the distance, then swing.

    Reads the tower centroid in camera space (+x right, -z forward), so the bearing is
    just atan2(right, forward) with no dependence on the engine's yaw convention.
    """
    rel = _tower_centroid(obs, n_blocks)
    right, _, back = rel
    forward = -back
    distance = math.hypot(right, forward)

    # Positive bearing => tower is to the right => look right.
    bearing = math.degrees(math.atan2(right, forward))
    look_dx = float(np.clip(bearing / 45.0, -1.0, 1.0))

    if abs(bearing) > AIM_TOLERANCE:
        # Turn in place first: walking while badly aimed wastes the episode clock.
        return np.array([look_dx, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    if distance > SWING_RANGE:
        return np.array([look_dx, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    # In range: keep pressing forward slightly and swing.
    return np.array([look_dx, 0.0, 0.0, 0.4, 0.0, 1.0], dtype=np.float32)


def run_episode(env: TeardownTowerEnv, max_steps: int = 600) -> dict:
    """Drive one episode with the scripted policy; returns a summary dict."""
    obs, _ = env.reset()
    total_reward = 0.0
    for step in range(max_steps):
        action = scripted_topple(obs, n_blocks=env.cfg.n_blocks)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            return {
                "steps": step + 1,
                "success": bool(info.get("success")),
                "t": float(info.get("t", 0.0)),
                "return": total_reward,
            }
    return {"steps": max_steps, "success": False, "t": 0.0, "return": total_reward}
