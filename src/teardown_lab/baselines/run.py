# ABOUTME: Runs the zero-shot baselines (VLA, world model) on the same task, seeds and
# ABOUTME: protocol as the trained student, and records the caveats alongside the scores.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from teardown_lab.baselines.models import LOADERS
from teardown_lab.baselines.protocol import TextActionPolicy, build_prompt
from teardown_lab.eval_student import run_episode, summarise
from teardown_lab.xdisplay import detect_display


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate zero-shot baselines.")
    parser.add_argument("--model", choices=sorted(LOADERS), required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--display", default=detect_display())
    parser.add_argument(
        "--no-slowmo",
        action="store_true",
        help="Do not slow the game clock (measures reaction speed, not decision quality).",
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

    bridge = RealBridge()
    env = TeardownPixelEnv(
        bridge=bridge,
        actuator=Actuator(UinputBackend()),
        frames=FrameGrabber(display=args.display),
        cfg=PixelEnvConfig(hz=10.0, timeout_s=40.0),
        refocus=lambda: focus_game_window(args.display),
    )
    if not ensure_playable(args.display, bridge):
        raise SystemExit("could not bring the game to a playable level")

    print(f"loading {args.model} ...", flush=True)
    responder = LOADERS[args.model]()
    policy = TextActionPolicy(responder)

    bridge.set_slowmo(not args.no_slowmo)
    episodes = []
    try:
        for i in range(args.episodes):
            focus_game_window(args.display)
            ensure_game_visible(args.display)
            episodes.append(run_episode(env, policy.act, args.max_steps))
            print(f"  episode {i + 1}: {episodes[-1]}", flush=True)
    finally:
        bridge.set_slowmo(False)
        env.close()

    report = summarise(episodes)
    report.update(
        model=args.model,
        # Everything a reader needs to judge the number, rather than the number alone.
        zero_shot=True,
        driven_as="VLM half via text-action protocol (native action heads are locked to "
        "robot embodiments and cannot emit game inputs)",
        parse_failure_rate=round(policy.parse_failure_rate, 3),
        # Does the model actually condition on the image? A high-scoring but CONSTANT
        # reply is a blind heuristic, not perception, and must not be read as competence.
        unique_reply_fraction=round(
            len(set(policy.replies)) / max(len(policy.replies), 1), 3
        ),
        distinct_replies=len(set(policy.replies)),
        total_decisions=len(policy.replies),
        median_decision_ms=round(responder.stats.median_ms),
        game_timescale=1.0 if args.no_slowmo else 0.1,
        max_steps=args.max_steps,
        sample_replies=policy.replies[:3],
    )
    print(json.dumps(report, indent=2))

    out = args.out or Path(f"runs/baseline_{args.model}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
