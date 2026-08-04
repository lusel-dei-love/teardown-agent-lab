# ABOUTME: Tests for the DAgger aggregation logic (no game, no X display needed).
# ABOUTME: Covers the beta mix, teacher labelling, and dataset concatenation.

import numpy as np

from teardown_lab.collect import EpisodeBuffer
from teardown_lab.dagger import concat_datasets


def make_base(n=5):
    return {
        "pixels": np.zeros((n, 72, 128, 3), dtype=np.uint8),
        "proprio": np.zeros((n, 5), dtype=np.float32),
        "actions": np.zeros((n, 7), dtype=np.float32),
        "on_policy": np.ones(n, dtype=bool),
        "solved": np.zeros(n, dtype=bool),
        "episode": np.zeros(n, dtype=np.int32),
    }


def make_buffer(n=3):
    buf = EpisodeBuffer()
    for _ in range(n):
        obs = {
            "pixels": np.ones((72, 128, 3), dtype=np.uint8),
            "proprio": np.ones(5, dtype=np.float32),
        }
        buf.add(obs, np.ones(7, dtype=np.float32), on_policy=False, solved=False)
    return buf


def test_aggregate_appends_and_keeps_base():
    base = make_base(5)
    out = concat_datasets(base, [make_buffer(3)])
    assert len(out["actions"]) == 8
    # The base data must survive: DAgger aggregates, it does not replace.
    assert out["pixels"][:5].max() == 0
    assert out["pixels"][5:].min() == 1


def test_aggregate_assigns_new_episode_ids():
    base = make_base(5)
    out = concat_datasets(base, [make_buffer(2), make_buffer(2)])
    episodes = out["episode"]
    assert set(episodes[:5]) == {0}
    # Fresh rollouts must not reuse existing episode ids or slicing by episode breaks.
    assert sorted(set(episodes[5:])) == [1, 2]


def test_aggregate_with_no_buffers_is_identity():
    base = make_base(4)
    assert len(concat_datasets(base, [])["actions"]) == 4
