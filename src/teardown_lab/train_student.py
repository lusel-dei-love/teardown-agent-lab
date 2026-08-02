# ABOUTME: Trains the pixel student to imitate the privileged teacher (behaviour cloning
# ABOUTME: with a class-weighted declare head), and saves staged checkpoints.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from teardown_lab.pixel_env import DECLARE_INDEX
from teardown_lab.student import (
    CONTINUOUS_DIMS,
    PixelStudent,
    StudentPolicy,
    normalizer,
    preprocess_pixels,
)


def load_dataset(path: Path) -> dict:
    blob = np.load(path)
    return {k: blob[k] for k in ("pixels", "proprio", "actions", "on_policy")}


def split_indices(n: int, val_fraction: float, rng: np.random.Generator):
    order = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    return order[n_val:], order[:n_val]


def batches(indices: np.ndarray, batch_size: int, rng: np.random.Generator):
    shuffled = rng.permutation(indices)
    for start in range(0, len(shuffled), batch_size):
        yield shuffled[start : start + batch_size]


def evaluate(model, data, idx, mean, std, pos_weight) -> dict:
    model.eval()
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    with torch.no_grad():
        pixels = preprocess_pixels(data["pixels"][idx])
        proprio = torch.from_numpy(((data["proprio"][idx] - mean) / std).astype(np.float32))
        target = torch.from_numpy(data["actions"][idx])
        out = model(pixels, proprio)

        control_loss = mse(out[:, CONTINUOUS_DIMS], target[:, CONTINUOUS_DIMS])
        declare_target = (target[:, DECLARE_INDEX] > 0).float()
        declare_loss = bce(out[:, DECLARE_INDEX], declare_target)

        predicted = (torch.sigmoid(out[:, DECLARE_INDEX]) > 0.5).float()
        true_pos = float(((predicted == 1) & (declare_target == 1)).sum())
        false_pos = float(((predicted == 1) & (declare_target == 0)).sum())
        false_neg = float(((predicted == 0) & (declare_target == 1)).sum())
    model.train()
    return {
        "control_mse": float(control_loss),
        "declare_bce": float(declare_loss),
        "declare_precision": true_pos / max(true_pos + false_pos, 1.0),
        "declare_recall": true_pos / max(true_pos + false_neg, 1.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the pixel student.")
    parser.add_argument("--dataset", type=Path, default=Path("runs/teacher_dataset.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/student"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    data = load_dataset(args.dataset)
    n = len(data["actions"])
    train_idx, val_idx = split_indices(n, args.val_fraction, rng)
    mean, std = normalizer(data["proprio"][train_idx])

    # Declares are ~1 step per episode out of ~100, so without a positive weight the
    # head trivially learns "never declare" and the episode can only ever time out.
    declare_targets = (data["actions"][:, DECLARE_INDEX] > 0).astype(np.float32)
    positives = max(declare_targets.sum(), 1.0)
    pos_weight = float((len(declare_targets) - positives) / positives)

    model = PixelStudent()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Stage checkpoints mirror the showcase: untrained, then quarter/half/full training.
    stages = {0: "stage_000", args.epochs // 4: "stage_025", args.epochs // 2: "stage_050", args.epochs: "stage_100"}
    history = []

    print(
        f"samples={n} train={len(train_idx)} val={len(val_idx)} "
        f"declare_positives={int(positives)} pos_weight={pos_weight:.1f}",
        flush=True,
    )

    def save_stage(epoch: int) -> None:
        name = stages.get(epoch)
        if name:
            StudentPolicy(model, mean, std).save(args.out_dir / f"{name}.pt")

    save_stage(0)

    for epoch in range(1, args.epochs + 1):
        totals = []
        for batch in batches(train_idx, args.batch_size, rng):
            pixels = preprocess_pixels(data["pixels"][batch])
            proprio = torch.from_numpy(
                ((data["proprio"][batch] - mean) / std).astype(np.float32)
            )
            target = torch.from_numpy(data["actions"][batch])

            out = model(pixels, proprio)
            control_loss = mse(out[:, CONTINUOUS_DIMS], target[:, CONTINUOUS_DIMS])
            declare_loss = bce(out[:, DECLARE_INDEX], (target[:, DECLARE_INDEX] > 0).float())
            loss = control_loss + declare_loss

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            totals.append(float(loss))

        metrics = evaluate(model, data, val_idx, mean, std, pos_weight)
        metrics.update(epoch=epoch, train_loss=float(np.mean(totals)))
        history.append(metrics)
        print(
            f"epoch {epoch:3d} train_loss={metrics['train_loss']:.4f} "
            f"val_control_mse={metrics['control_mse']:.4f} "
            f"declare P={metrics['declare_precision']:.2f} R={metrics['declare_recall']:.2f}",
            flush=True,
        )
        save_stage(epoch)

    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(json.dumps(history[-1], indent=2))


if __name__ == "__main__":
    main()
