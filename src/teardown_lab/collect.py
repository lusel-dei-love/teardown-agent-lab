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
from teardown_lab.xdisplay import detect_display


@dataclass
class EpisodeBuffer:
    pixels: list = field(default_factory=list)
    proprio: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    # Whether the executed action was the teacher's (on-policy) or a scramble.
    on_policy: list = field(default_factory=list)
    solved: list = field(default_factory=list)

    def add(self, obs, action, on_policy: bool, solved: bool = False) -> None:
        self.pixels.append(obs["pixels"])
        self.proprio.append(obs["proprio"])
        self.actions.append(action)
        self.on_policy.append(on_policy)
        self.solved.append(solved)

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

        buf.add(obs, label, on_policy=not scrambling, solved=bool(info["success"]))
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

    # Widen the declare label. The teacher declares exactly once per episode, which left
    # 17 positives in 4097 samples - so little that the head learned either "always
    # declare" or "never declare" depending only on the class weight. Every frame where
    # the privileged referee already says solved is a moment at which declaring is
    # CORRECT (that is precisely how the env scores it), so all of them are valid
    # positives and the signal becomes learnable.
    if result["success"]:
        for i, was_solved in enumerate(buf.solved):
            if was_solved:
                buf.actions[i] = buf.actions[i].copy()
                buf.actions[i][DECLARE_INDEX] = 1.0

    return buf, result


def save_dataset(path: Path, buffers: list[EpisodeBuffer], results: list[dict]) -> dict:
    # A run that loses its first episode has nothing to concatenate; report that rather
    # than raising over the top of the original failure.
    buffers = [b for b in buffers if len(b)]
    if not buffers:
        return {"samples": 0, "episodes": 0, "path": str(path), "note": "no episodes collected"}

    pixels = np.concatenate([np.asarray(b.pixels, dtype=np.uint8) for b in buffers])
    proprio = np.concatenate([np.asarray(b.proprio, dtype=np.float32) for b in buffers])
    actions = np.concatenate([np.asarray(b.actions, dtype=np.float32) for b in buffers])
    on_policy = np.concatenate([np.asarray(b.on_policy, dtype=bool) for b in buffers])
    solved = np.concatenate([np.asarray(b.solved, dtype=bool) for b in buffers])

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pixels=pixels,
        proprio=proprio,
        actions=actions,
        on_policy=on_policy,
        solved=solved,
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
    parser.add_argument("--display", default=detect_display())
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument(
        "--hard-reset-every",
        type=int,
        default=25,
        help="Reload the level every N episodes to repair destroyed terrain (0 disables).",
    )
    args = parser.parse_args()

    from teardown_lab.actuator import (
        Actuator,
        UinputBackend,
        ensure_game_visible,
        focus_game_window,
    )
    from teardown_lab.frames import FrameGrabber
    from teardown_lab.pixel_env import PixelEnvConfig
    from teardown_lab.real_bridge import RealBridge
    from teardown_lab.supervisor import ensure_playable

    rng = np.random.default_rng(args.seed)
    env = TeardownPixelEnv(
        bridge=RealBridge(),
        actuator=Actuator(UinputBackend()),
        frames=FrameGrabber(display=args.display),
        cfg=PixelEnvConfig(hz=10.0, timeout_s=40.0),
        refocus=lambda: focus_game_window(args.display),
    )
    teacher = PrivilegedTeacher()

    # Make sure we are actually in a level before the first episode.
    if not ensure_playable(args.display, env.bridge):
        raise SystemExit("could not bring the game to a playable level")

    buffers, results = [], []
    started = time.time()
    try:
        for ep in range(args.episodes):
            # Take focus back before every episode, and make sure the game is actually
            # the visible workspace - otherwise the frame grabber records another app.
            focus_game_window(args.display)
            if not ensure_game_visible(args.display):
                print("-- WARNING: game window not visible; frames may be invalid", flush=True)

            if args.hard_reset_every and ep and ep % args.hard_reset_every == 0:
                print(f"-- hard reset at episode {ep} (repairing terrain)", flush=True)
                if env.bridge.hard_reset() is None:
                    print("-- hard reset timed out; continuing", flush=True)
                focus_game_window(args.display)
            scramble = int(rng.integers(0, args.scramble_max + 1))
            try:
                buf, res = collect_episode(env, teacher, rng, scramble, args.max_steps)
            except RuntimeError as exc:
                # The game can die mid-run (it did, at episode 41 of 200). Recover the
                # process and carry on rather than losing the rest of the run; the data
                # already collected is checkpointed either way.
                print(f"-- episode {ep + 1} lost: {exc}", flush=True)
                if not ensure_playable(args.display, env.bridge):
                    print("-- could not recover the game; stopping", flush=True)
                    break
                print("-- game recovered; continuing", flush=True)
                continue
            buffers.append(buf)
            results.append(res)

            # Checkpoint. Saving only at the end means a crash costs the whole run.
            if args.checkpoint_every and len(buffers) % args.checkpoint_every == 0:
                partial = save_dataset(args.out, buffers, results)
                print(f"-- checkpoint: {partial['samples']} samples saved", flush=True)
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
