# ABOUTME: The shared text-action protocol every zero-shot baseline drives, so a VLA, a
# ABOUTME: world model and our trained student are measured through one identical interface.

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np

from teardown_lab.pixel_env import ACTION_DIM, DECLARE_INDEX

# Neither MolmoAct 2 nor Cosmos 3 Edge can emit game inputs: their action heads are
# hard-locked to robot embodiments (MolmoAct 2's predict_action even requires a joint
# state vector, and its normalisation tags are a closed set). So the honest zero-shot
# baseline drives their VLM half - image plus instruction in, structured text out - and
# maps that text onto the SAME action vector our student produces. Any result must be
# reported with this caveat: these models were not built to play a game.

TASK_INSTRUCTION = (
    "You are controlling a first-person character in a 3D voxel game. "
    "In front of you somewhere is a tower of nine white blocks. "
    "Your goal is to knock the tower down by walking to it and hitting it with the "
    "sledgehammer you are holding."
)

ACTION_VOCABULARY = """Reply with ONLY one JSON object, no prose, using exactly these keys:
{"turn": <float -1..1>, "pitch": <float -1..1>, "move": <float -1..1>, "strafe": <float -1..1>, "swing": <true|false>, "done": <true|false>}
where:
  turn:   -1 turn left hard, 0 keep heading, +1 turn right hard
  pitch:  -1 look up, 0 level, +1 look down
  move:   -1 walk backward, +1 walk forward
  strafe: -1 step left, +1 step right
  swing:  true to swing the sledgehammer this step
  done:   true only if the tower is already knocked down
Choose the single best action for THIS frame."""


def build_prompt() -> str:
    return f"{TASK_INSTRUCTION}\n\n{ACTION_VOCABULARY}"


@dataclass(frozen=True)
class ParsedAction:
    action: np.ndarray
    raw: str
    parsed_ok: bool


def _clip(value, low=-1.0, high=1.0) -> float:
    try:
        return float(np.clip(float(value), low, high))
    except (TypeError, ValueError):
        return 0.0


def parse_action(text: str) -> ParsedAction:
    """Map a model's text reply onto the env's 7-dim action vector.

    Tolerant by design: these models were never trained to emit this schema, so a strict
    parser would mostly measure formatting compliance rather than any grasp of the task.
    A reply we cannot read becomes a no-op, and `parsed_ok` records that so the failure
    is reported rather than hidden inside the success rate.
    """
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[DECLARE_INDEX] = -1.0

    match = re.search(r"\{.*?\}", text or "", re.S)
    if not match:
        return ParsedAction(action, text or "", False)

    try:
        blob = json.loads(match.group(0))
    except json.JSONDecodeError:
        return ParsedAction(action, text, False)
    if not isinstance(blob, dict):
        return ParsedAction(action, text, False)

    action[0] = _clip(blob.get("turn", 0.0))
    action[1] = _clip(blob.get("pitch", 0.0))
    action[2] = _clip(blob.get("strafe", 0.0))
    action[3] = _clip(blob.get("move", 0.0))
    action[4] = -1.0  # grab is unused by this task
    action[5] = 1.0 if blob.get("swing") is True else -1.0
    action[6] = 1.0 if blob.get("done") is True else -1.0
    return ParsedAction(action, text, True)


class TextActionPolicy:
    """Wraps any image+text -> text model as an env policy.

    `responder(frame_rgb, prompt) -> str` is the only thing a backend must provide, which
    is what keeps the two baselines honestly comparable: identical prompt, identical
    parsing, identical action space, identical actuator.
    """

    def __init__(self, responder, prompt: str | None = None):
        self.responder = responder
        self.prompt = prompt or build_prompt()
        self.replies: list[str] = []
        self.parse_failures = 0

    def act(self, obs: dict) -> np.ndarray:
        try:
            text = self.responder(obs["pixels"], self.prompt)
        except Exception as exc:  # a backend failure must not look like a task failure
            self.replies.append(f"<error: {type(exc).__name__}: {exc}>")
            self.parse_failures += 1
            action = np.zeros(ACTION_DIM, dtype=np.float32)
            action[DECLARE_INDEX] = -1.0
            return action

        parsed = parse_action(text)
        self.replies.append(parsed.raw)
        if not parsed.parsed_ok:
            self.parse_failures += 1
        return parsed.action

    @property
    def parse_failure_rate(self) -> float:
        return self.parse_failures / max(len(self.replies), 1)
