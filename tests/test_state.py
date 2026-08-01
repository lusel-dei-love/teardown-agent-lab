# ABOUTME: Tests for the GameState/BlockState dataclasses and their JSON serialization.
# ABOUTME: Pure data, no game and no X display needed.

import json

from teardown_lab.state import BlockState, GameState

from tests.helpers import state_with_displaced_blocks


def test_json_round_trip():
    state = state_with_displaced_blocks(2, 0.75, t=3.5, seed=42, episode=7)
    restored = GameState.from_json(state.to_json())
    assert restored == state


def test_to_json_schema():
    state = state_with_displaced_blocks(0, 0.0, t=1.25, seed=3, episode=2)
    payload = json.loads(state.to_json())
    assert payload["t"] == 1.25
    assert payload["seed"] == 3
    assert payload["episode"] == 2
    assert payload["player_pos"] == [0.0, 0.0, -5.0]
    assert payload["yaw"] == 0.0
    assert payload["pitch"] == 0.0
    assert len(payload["blocks"]) == 9
    assert payload["blocks"][0] == {"pos": [-1.0, 0.0, 0.0], "spawn": [-1.0, 0.0, 0.0]}


def test_from_json_builds_tuples():
    raw = json.dumps(
        {
            "t": 0.5,
            "seed": 11,
            "episode": 1,
            "player_pos": [1, 2, 3],
            "yaw": 0.25,
            "pitch": -0.5,
            "blocks": [{"pos": [4, 5, 6], "spawn": [7, 8, 9]}],
        }
    )
    state = GameState.from_json(raw)
    assert state.player_pos == (1.0, 2.0, 3.0)
    assert isinstance(state.player_pos, tuple)
    assert state.blocks == [BlockState(pos=(4.0, 5.0, 6.0), spawn=(7.0, 8.0, 9.0))]
    assert isinstance(state.blocks[0].pos, tuple)
    assert state.yaw == 0.25
    assert state.pitch == -0.5
