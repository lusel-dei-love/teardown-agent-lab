# ABOUTME: DAgger loop: roll out the student, label the states IT visits with the
# ABOUTME: privileged teacher, aggregate, retrain. Fixes compounding error, not fit.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from teardown_lab.collect import EpisodeBuffer
from teardown_lab.pixel_env import ACTION_DIM, DECLARE_INDEX
from teardown_lab.student import PixelStudent, StudentPolicy, normalizer
from teardown_lab.teacher import PrivilegedTeacher
from teardown_lab.train_student import (
    binary_loss,  # noqa: F401  (re-exported for symmetry with the one-shot trainer)
    fit,
    load_dataset,
    tune_declare_threshold,
)
from teardown_lab.xdisplay import detect_display


def rollout_episode(
    env,
    teacher: PrivilegedTeacher,
    policy,
    beta: float,
    rng: np.random.Generator,
    max_steps: int,
) -> tuple[EpisodeBuffer, dict]:
    """Run one episode, EXECUTING a beta-mix of teacher and student, LABELLING with the
    teacher.

    This is the whole point of DAgger: behaviour cloning only ever sees the teacher's
    state distribution, so the student's own mistakes take it somewhere it was never
    taught, and the errors compound. Here the student drives (increasingly, as beta
    decays) while the teacher answers "what should you have done here?" for every state
    actually visited.
    """
    buf = EpisodeBuffer()
    obs, info = env.reset()
    teacher.reset()
    result = {"steps": 0, "success": False, "declared": False, "false_declaration": False}

    for step in range(max_steps):
        label = teacher.act(info["privileged_state"])
        executed = label if rng.random() < beta else policy(obs)

        buf.add(obs, label, on_policy=False, solved=bool(info["success"]))
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

    # Same widening as the teacher collector: any frame where the referee already says
    # solved is a moment at which declaring is correct.
    if result["success"]:
        for i, was_solved in enumerate(buf.solved):
            if was_solved:
                buf.actions[i] = buf.actions[i].copy()
                buf.actions[i][DECLARE_INDEX] = 1.0

    return buf, result


def concat_datasets(base: dict, buffers: list[EpisodeBuffer]) -> dict:
    """Append freshly labelled episodes to the aggregate dataset."""
    if not buffers:
        return base
    next_episode = int(base["episode"].max()) + 1 if "episode" in base else 0
    added = {
        "pixels": np.concatenate([np.asarray(b.pixels, dtype=np.uint8) for b in buffers]),
        "proprio": np.concatenate([np.asarray(b.proprio, dtype=np.float32) for b in buffers]),
        "actions": np.concatenate([np.asarray(b.actions, dtype=np.float32) for b in buffers]),
        "on_policy": np.concatenate([np.asarray(b.on_policy, dtype=bool) for b in buffers]),
        "solved": np.concatenate([np.asarray(b.solved, dtype=bool) for b in buffers]),
        "episode": np.concatenate(
            [np.full(len(b), next_episode + i, dtype=np.int32) for i, b in enumerate(buffers)]
        ),
    }
    return {k: np.concatenate([base[k], added[k]]) for k in added if k in base}


def main() -> None:
    parser = argparse.ArgumentParser(description="DAgger: aggregate on-policy states.")
    parser.add_argument("--dataset", type=Path, default=Path("runs/teacher_200.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/dagger"))
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--episodes-per-iter", type=int, default=25)
    parser.add_argument("--epochs-per-iter", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-pos-weight", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--display", default=detect_display())
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=Path("runs/student200bce/stage_100.pt"),
        help="Warm start from an existing student instead of training from scratch.",
    )
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
    from teardown_lab.supervisor import ensure_playable

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(args.dataset)
    print(f"base dataset: {len(data['actions'])} samples", flush=True)

    policy_wrapper = StudentPolicy.load(args.init_checkpoint)
    model = policy_wrapper.model
    model.train()

    env = TeardownPixelEnv(
        bridge=RealBridge(),
        actuator=Actuator(UinputBackend()),
        frames=FrameGrabber(display=args.display),
        cfg=PixelEnvConfig(hz=10.0, timeout_s=40.0),
        refocus=lambda: focus_game_window(args.display),
    )
    teacher = PrivilegedTeacher()

    if not ensure_playable(args.display, env.bridge):
        raise SystemExit("could not bring the game to a playable level")

    report = []
    try:
        for iteration in range(1, args.iterations + 1):
            # Classic beta decay: the teacher drives less each round, so the aggregated
            # states move steadily towards the student's own distribution.
            beta = 0.5 ** iteration
            mean, std = normalizer(data["proprio"])
            threshold = policy_wrapper.declare_threshold
            rollout_policy = StudentPolicy(model, mean, std, threshold).act

            buffers, results = [], []
            for _ in range(args.episodes_per_iter):
                focus_game_window(args.display)
                ensure_game_visible(args.display)
                try:
                    buf, res = rollout_episode(
                        env, teacher, rollout_policy, beta, rng, args.max_steps
                    )
                except RuntimeError as exc:
                    print(f"-- episode lost: {exc}", flush=True)
                    if not ensure_playable(args.display, env.bridge):
                        break
                    continue
                buffers.append(buf)
                results.append(res)

            successes = sum(r["success"] for r in results)
            data = concat_datasets(data, buffers)
            samples = len(data["actions"])
            print(
                f"iter {iteration}: beta={beta:.2f} episodes={len(results)} "
                f"student-driven successes={successes} aggregate={samples} samples",
                flush=True,
            )

            model.train()
            mean, std = normalizer(data["proprio"])
            idx = np.arange(samples)
            declare_targets = (data["actions"][:, DECLARE_INDEX] > 0).astype(np.float32)
            positives = max(declare_targets.sum(), 1.0)
            pos_weight = float(
                min((len(declare_targets) - positives) / positives, args.max_pos_weight)
            )
            history = fit(
                model,
                data,
                idx,
                idx[:: max(1, samples // 3000)],  # subsample the metric split; it is only for reporting
                mean,
                std,
                epochs=args.epochs_per_iter,
                lr=args.lr,
                batch_size=args.batch_size,
                pos_weight=pos_weight,
                rng=rng,
            )
            last = history[-1]
            print(
                f"iter {iteration}: control_mse={last['control_mse']:.4f} "
                f"declare P={last['declare_precision']:.2f} R={last['declare_recall']:.2f}",
                flush=True,
            )

            threshold = tune_declare_threshold(model, data, idx[::5], mean, std)
            policy_wrapper = StudentPolicy(model, mean, std, threshold)
            policy_wrapper.save(args.out_dir / f"dagger_{iteration}.pt")
            report.append(
                {
                    "iteration": iteration,
                    "beta": beta,
                    "episodes": len(results),
                    "successes": successes,
                    "samples": samples,
                    "declare_threshold": threshold,
                    **{k: last[k] for k in ("control_mse", "declare_precision", "declare_recall")},
                }
            )
            np.savez_compressed(args.out_dir / "aggregate.npz", **data)
    finally:
        env.close()

    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
