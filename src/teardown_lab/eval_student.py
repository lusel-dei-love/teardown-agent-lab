# ABOUTME: Fixed-protocol evaluation for the pixel student: identical seeds and budget
# ABOUTME: for every stage, reporting success AND false-declaration rates.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from teardown_lab.pixel_env import ACTION_DIM, DECLARE_INDEX

DEFAULT_SEEDS = list(range(1, 11))


def run_episode(env, policy, max_steps: int) -> dict:
    """One episode under `policy`, scored with privileged truth the policy never saw."""
    obs, info = env.reset()
    steps = 0
    for step in range(max_steps):
        action = policy(obs)
        obs, _reward, terminated, truncated, info = env.step(action)
        steps = step + 1
        if terminated or truncated:
            break
    return {
        "steps": steps,
        "success": bool(info["success"]),
        "declared": bool(info["declared"]),
        "false_declaration": bool(info["false_declaration"]),
        "t": float(info["t"]),
    }


def summarise(episodes: list[dict]) -> dict:
    n = max(len(episodes), 1)
    declared = [e for e in episodes if e["declared"]]
    return {
        "episodes": len(episodes),
        # Did the tower actually come down, regardless of what the agent claimed.
        "success_rate": sum(e["success"] for e in episodes) / n,
        # Declared complete while the tower still stood: only expressible because
        # termination is the agent's decision rather than an oracle's.
        "false_declaration_rate": sum(e["false_declaration"] for e in episodes) / n,
        "declaration_rate": len(declared) / n,
        # Of the times it claimed done, how often was it right.
        "declaration_precision": (
            sum(not e["false_declaration"] for e in declared) / len(declared)
            if declared
            else None
        ),
        "mean_steps": float(np.mean([e["steps"] for e in episodes])) if episodes else 0.0,
    }


def random_policy(rng):
    def act(_obs):
        action = rng.uniform(-1.0, 1.0, size=ACTION_DIM).astype(np.float32)
        action[DECLARE_INDEX] = -1.0  # a random agent never claims completion
        return action

    return act


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate student stages.")
    parser.add_argument("--stages", type=Path, default=Path("runs/student"))
    parser.add_argument("--out", type=Path, default=Path("runs/eval_student.json"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--display", default=":1")
    args = parser.parse_args()

    from teardown_lab.actuator import (
        Actuator,
        UinputBackend,
        ensure_game_visible,
        focus_game_window,
    )
    from teardown_lab.frames import FrameGrabber
    from teardown_lab.pixel_env import PixelEnvConfig, TeardownPixelEnv
    from teardown_lab.real_bridge import RealBridge
    from teardown_lab.student import StudentPolicy

    env = TeardownPixelEnv(
        bridge=RealBridge(),
        actuator=Actuator(UinputBackend()),
        frames=FrameGrabber(display=args.display),
        cfg=PixelEnvConfig(hz=10.0, timeout_s=40.0),
        refocus=lambda: focus_game_window(args.display),
    )

    candidates: list[tuple[str, object]] = [("random", random_policy(np.random.default_rng(0)))]
    for checkpoint in sorted(args.stages.glob("stage_*.pt")):
        policy = StudentPolicy.load(checkpoint)
        candidates.append((checkpoint.stem, policy.act))

    report = {}
    try:
        for name, policy in candidates:
            episodes = []
            for _ in range(args.episodes):
                focus_game_window(args.display)
                ensure_game_visible(args.display)
                episodes.append(run_episode(env, policy, args.max_steps))
            report[name] = summarise(episodes)
            report[name]["detail"] = episodes
            print(name, json.dumps(summarise(episodes)), flush=True)
    finally:
        env.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
