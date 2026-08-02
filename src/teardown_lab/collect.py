# ABOUTME: Collects the teacher demonstration dataset: frames + proprioception + the
# ABOUTME: action the privileged teacher would take, including off-policy scrambled states.

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from teardown_lab.pixel_env import ACTION_DIM, DECLARE_INDEX, TeardownPixelEnv
from teardown_lab.teacher import PrivilegedTeacher


@dataclass
class EpisodeBuffer:
    pixels: list = field(default_factory=list)
    proprio: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    # Whether the executed action was the teacher's (on-policy) or a scramble.
    on_policy: list = field(default_factory=list)

    def add(self, obs, action, on_policy: bool) -> None:
        self.pixels.append(obs["pixels"])
        self.proprio.append(obs["proprio"])
        self.actions.append(action)
        self.on_policy.append(on_policy)

    def __len__(self) -> int:
        return len(self.actions)


def scramble_action(rng: np.random.Generator) -> np.ndarray:
    """A random exploratory action - never declares.

    Weighted towards looking rather than travelling. What the pixel student needs is the
    tower seen from many angles and distances; what it does not need is the agent walking
    off into the level, which mostly produces frames with no tower in them and episodes
    the teacher spends its whole budget recovering from.
    """
    a = np.zeros(ACTION_DIM, dtype=np.float32)
    a[0] = rng.uniform(-1.0, 1.0)  # look yaw - the useful axis for viewpoint diversity
    a[1] = rng.uniform(-0.2, 0.2)  # look pitch
    a[2] = rng.uniform(-0.6, 0.6)  # strafe
    a[3] = rng.uniform(-0.6, 0.6)  # forward/back, symmetric so it does not drift away
    return a


def collect_episode(
    env: TeardownPixelEnv,
    teacher: PrivilegedTeacher,
    rng: np.random.Generator,
    scramble_steps: int,
    max_steps: int,
) -> tuple[EpisodeBuffer, dict]:
    """Run one episode, labelling every visited state with the teacher's action.

    The first `scramble_steps` steps EXECUTE random actions but still RECORD what the
    teacher would have done. A deterministic teacher from a fixed spawn would otherwise
    produce near-identical trajectories, and the student would never learn to recover
    from states it drifts into at test time - the classic behaviour-cloning failure.
    """
    buf = EpisodeBuffer()
    obs, info = env.reset()
    teacher.reset()

    result = {"steps": 0, "success": False, "declared": False, "false_declaration": False}

    for step in range(max_steps):
        label = teacher.act(info["privileged_state"])

        scrambling = step < scramble_steps
        if scrambling:
            executed = scramble_action(rng)
        else:
            executed = label

        buf.add(obs, label, on_policy=not scrambling)
        obs, _reward, terminated, truncated, info = env.step(executed)

        if terminated or truncated:
            result.update(
                steps=step + 1,
                success=bool(info["success"]),
                declared=bool(info["declared"]),
                false_declaration=bool(info["false_declaration"]),
            )
            break
    else:
        result["steps"] = max_steps

    return buf, result


def save_dataset(path: Path, buffers: list[EpisodeBuffer], results: list[dict]) -> dict:
    pixels = np.concatenate([np.asarray(b.pixels, dtype=np.uint8) for b in buffers])
    proprio = np.concatenate([np.asarray(b.proprio, dtype=np.float32) for b in buffers])
    actions = np.concatenate([np.asarray(b.actions, dtype=np.float32) for b in buffers])
    on_policy = np.concatenate([np.asarray(b.on_policy, dtype=bool) for b in buffers])

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, pixels=pixels, proprio=proprio, actions=actions, on_policy=on_policy
    )
    return {
        "samples": int(len(actions)),
        "episodes": len(buffers),
        "declare_positives": int((actions[:, DECLARE_INDEX] > 0).sum()),
        "on_policy_fraction": float(on_policy.mean()),
        "successes": sum(r["success"] for r in results),
        "path": str(path),
        "size_mb": round(path.stat().st_size / 1e6, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect teacher demonstrations.")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--scramble-max", type=int, default=25)
    parser.add_argument("--out", type=Path, default=Path("runs/teacher_dataset.npz"))
    parser.add_argument("--display", default=":1")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from teardown_lab.actuator import Actuator, UinputBackend, focus_game_window
    from teardown_lab.frames import FrameGrabber
    from teardown_lab.pixel_env import PixelEnvConfig
    from teardown_lab.real_bridge import RealBridge

    rng = np.random.default_rng(args.seed)
    env = TeardownPixelEnv(
        bridge=RealBridge(),
        actuator=Actuator(UinputBackend()),
        frames=FrameGrabber(display=args.display),
        cfg=PixelEnvConfig(hz=10.0, timeout_s=40.0),
        refocus=lambda: focus_game_window(args.display),
    )
    teacher = PrivilegedTeacher()

    buffers, results = [], []
    started = time.time()
    try:
        for ep in range(args.episodes):
            # Take focus back before every episode: the game pauses without it.
            focus_game_window(args.display)
            scramble = int(rng.integers(0, args.scramble_max + 1))
            buf, res = collect_episode(env, teacher, rng, scramble, args.max_steps)
            buffers.append(buf)
            results.append(res)
            print(
                f"ep {ep + 1:3d}/{args.episodes}  steps={res['steps']:3d} "
                f"scramble={scramble:2d} success={res['success']} "
                f"declared={res['declared']} samples={len(buf)}",
                flush=True,
            )
    finally:
        env.close()

    summary = save_dataset(args.out, buffers, results)
    summary["wall_minutes"] = round((time.time() - started) / 60, 1)
    print(summary)


if __name__ == "__main__":
    main()
