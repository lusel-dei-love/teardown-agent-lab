# ABOUTME: Tests for the shared baseline text-action protocol: parsing, clipping,
# ABOUTME: and that unreadable replies degrade to a recorded no-op rather than a crash.

import numpy as np

from teardown_lab.baselines.protocol import (
    TextActionPolicy,
    build_prompt,
    parse_action,
)
from teardown_lab.pixel_env import ACTION_DIM, DECLARE_INDEX


def obs():
    return {"pixels": np.zeros((126, 224, 3), dtype=np.uint8), "proprio": np.zeros(5, dtype=np.float32)}


def test_parses_a_well_formed_reply():
    p = parse_action('{"turn": 0.5, "pitch": -0.2, "move": 1.0, "strafe": 0, "swing": true, "done": false}')
    assert p.parsed_ok
    assert p.action.shape == (ACTION_DIM,)
    assert p.action[0] == np.float32(0.5)
    assert p.action[3] == np.float32(1.0)
    assert p.action[5] == 1.0          # swing engaged
    assert p.action[DECLARE_INDEX] == -1.0


def test_extracts_json_embedded_in_prose():
    # These models were never trained on this schema; they often wrap it in commentary.
    p = parse_action('Sure! Here is the action:\n{"turn": -1, "move": 0.5}\nHope that helps.')
    assert p.parsed_ok
    assert p.action[0] == np.float32(-1.0)


def test_out_of_range_values_are_clipped():
    p = parse_action('{"turn": 42, "move": -99}')
    assert p.action[0] == np.float32(1.0)
    assert p.action[3] == np.float32(-1.0)


def test_unreadable_reply_becomes_a_recorded_noop():
    p = parse_action("I cannot control a video game.")
    assert not p.parsed_ok
    # Every control axis is neutral - a no-op, never a random flail - and the model is
    # recorded as NOT declaring completion (-1), which is a decision, not a zero.
    assert not p.action[:DECLARE_INDEX].any()
    assert p.action[DECLARE_INDEX] == -1.0


def test_policy_records_parse_failure_rate():
    replies = iter(['{"move": 1}', "no idea", '{"turn": 0.1}'])
    policy = TextActionPolicy(lambda frame, prompt: next(replies))
    for _ in range(3):
        policy.act(obs())
    assert policy.parse_failure_rate == 1 / 3


def test_backend_exception_does_not_crash_the_episode():
    def broken(frame, prompt):
        raise RuntimeError("cuda oom")

    policy = TextActionPolicy(broken)
    action = policy.act(obs())
    assert not action[:DECLARE_INDEX].any()
    assert policy.parse_failure_rate == 1.0
    assert "cuda oom" in policy.replies[0]


def test_prompt_states_the_goal_and_the_schema():
    prompt = build_prompt()
    assert "tower" in prompt and "sledgehammer" in prompt
    for key in ("turn", "pitch", "move", "strafe", "swing", "done"):
        assert key in prompt
