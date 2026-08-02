# ABOUTME: The pixel student: a small CNN over frames plus a proprioception branch,
# ABOUTME: trained to imitate the privileged teacher. It never sees world state.

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from teardown_lab.pixel_env import ACTION_DIM, DECLARE_INDEX, PROPRIO_DIM

CONTINUOUS_DIMS = list(range(DECLARE_INDEX))  # 0..5; index 6 is the declare logit


class PixelStudent(nn.Module):
    """Frames + proprioception -> action.

    Outputs the six continuous controls through a tanh (matching the env's [-1, 1] box)
    and the declare decision as a raw logit, since it is a binary choice trained with
    BCE rather than regression.
    """

    def __init__(self, proprio_dim: int = PROPRIO_DIM, action_dim: int = ACTION_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 5)),
            nn.Flatten(),
        )
        self.proprio = nn.Sequential(nn.Linear(proprio_dim, 64), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(64 * 3 * 5 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, pixels: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        visual = self.encoder(pixels)
        proprio_features = self.proprio(proprio)
        raw = self.head(torch.cat([visual, proprio_features], dim=1))
        continuous = torch.tanh(raw[:, CONTINUOUS_DIMS])
        declare_logit = raw[:, DECLARE_INDEX : DECLARE_INDEX + 1]
        return torch.cat([continuous, declare_logit], dim=1)


def preprocess_pixels(pixels: np.ndarray) -> torch.Tensor:
    """uint8 HWC (or NHWC) frames -> float NCHW in [0, 1]."""
    array = np.asarray(pixels)
    if array.ndim == 3:
        array = array[None]
    tensor = torch.from_numpy(np.ascontiguousarray(array)).float().div_(255.0)
    return tensor.permute(0, 3, 1, 2)


def normalizer(proprio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std for proprioception. Yaw is unbounded in this engine, so it must be
    standardised or it dominates the input scale."""
    mean = proprio.mean(axis=0)
    std = proprio.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


class StudentPolicy:
    """Inference wrapper: takes an env observation dict, returns an env action."""

    def __init__(self, model: PixelStudent, mean: np.ndarray, std: np.ndarray, declare_threshold: float = 0.5):
        self.model = model.eval()
        self.mean = mean
        self.std = std
        self.declare_threshold = declare_threshold

    @torch.no_grad()
    def act(self, obs: dict) -> np.ndarray:
        pixels = preprocess_pixels(obs["pixels"])
        proprio = torch.from_numpy(
            ((obs["proprio"] - self.mean) / self.std)[None].astype(np.float32)
        )
        out = self.model(pixels, proprio)[0].numpy()
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        action[CONTINUOUS_DIMS] = out[CONTINUOUS_DIMS]
        declare_prob = 1.0 / (1.0 + np.exp(-out[DECLARE_INDEX]))
        action[DECLARE_INDEX] = 1.0 if declare_prob > self.declare_threshold else -1.0
        return action

    def save(self, path) -> None:
        # Tensors only, so the checkpoint can be loaded with weights_only=True: that
        # loader refuses to unpickle arbitrary objects, which is what we want for
        # anything that might be copied between machines.
        torch.save(
            {
                "model": self.model.state_dict(),
                "mean": torch.from_numpy(np.asarray(self.mean, dtype=np.float32)),
                "std": torch.from_numpy(np.asarray(self.std, dtype=np.float32)),
                "declare_threshold": torch.tensor(self.declare_threshold),
            },
            path,
        )

    @classmethod
    def load(cls, path) -> StudentPolicy:
        blob = torch.load(path, map_location="cpu", weights_only=True)
        model = PixelStudent()
        model.load_state_dict(blob["model"])
        return cls(
            model,
            blob["mean"].numpy(),
            blob["std"].numpy(),
            float(blob["declare_threshold"]),
        )
