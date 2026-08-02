# ABOUTME: Tests for the pixel student: shapes, action bounds, declare thresholding,
# ABOUTME: and that a saved checkpoint round-trips through the safe torch loader.

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from teardown_lab.pixel_env import ACTION_DIM, DECLARE_INDEX, PROPRIO_DIM
from teardown_lab.student import (
    PixelStudent,
    StudentPolicy,
    normalizer,
    preprocess_pixels,
)


def fake_obs():
    return {
        "pixels": np.full((72, 128, 3), 120, dtype=np.uint8),
        "proprio": np.zeros(PROPRIO_DIM, dtype=np.float32),
    }


def test_preprocess_pixels_to_nchw_unit_range():
    batch = preprocess_pixels(np.full((72, 128, 3), 255, dtype=np.uint8))
    assert batch.shape == (1, 3, 72, 128)
    assert float(batch.max()) == pytest.approx(1.0)


def test_forward_shape_and_continuous_bounds():
    model = PixelStudent()
    out = model(preprocess_pixels(fake_obs()["pixels"]), torch.zeros(1, PROPRIO_DIM))
    assert out.shape == (1, ACTION_DIM)
    # Continuous controls are tanh-bounded to the env's action box.
    continuous = out[0, :DECLARE_INDEX]
    assert float(continuous.abs().max()) <= 1.0


def test_policy_emits_valid_env_action():
    mean = np.zeros(PROPRIO_DIM, dtype=np.float32)
    std = np.ones(PROPRIO_DIM, dtype=np.float32)
    policy = StudentPolicy(PixelStudent(), mean, std)
    action = policy.act(fake_obs())
    assert action.shape == (ACTION_DIM,)
    assert np.all(np.abs(action) <= 1.0)
    # Declare is a decision, so it must be one of the two extremes, never in between.
    assert action[DECLARE_INDEX] in (-1.0, 1.0)


def test_declare_threshold_controls_the_decision():
    mean = np.zeros(PROPRIO_DIM, dtype=np.float32)
    std = np.ones(PROPRIO_DIM, dtype=np.float32)
    model = PixelStudent()
    never = StudentPolicy(model, mean, std, declare_threshold=1.1)
    always = StudentPolicy(model, mean, std, declare_threshold=-0.1)
    assert never.act(fake_obs())[DECLARE_INDEX] == -1.0
    assert always.act(fake_obs())[DECLARE_INDEX] == 1.0


def test_checkpoint_round_trips_with_safe_loader(tmp_path):
    mean = np.arange(PROPRIO_DIM, dtype=np.float32)
    std = np.ones(PROPRIO_DIM, dtype=np.float32) * 2
    original = StudentPolicy(PixelStudent(), mean, std)
    path = tmp_path / "student.pt"
    original.save(path)

    restored = StudentPolicy.load(path)
    assert np.allclose(restored.mean, mean)
    assert np.allclose(restored.std, std)
    obs = fake_obs()
    assert np.allclose(original.act(obs), restored.act(obs))


def test_normalizer_avoids_divide_by_zero_on_constant_columns():
    proprio = np.ones((10, PROPRIO_DIM), dtype=np.float32)
    _, std = normalizer(proprio)
    assert np.all(std > 0)
