"""Train B1 history-stack or B2 GRU planner on a fixed paired BC cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from high_level_planner import (
    TemporalReferenceSequencePlanner,
    TemporalReferenceTrajectoryDataset,
    reference_imitation_loss,
)


def evaluate(model, loader, device, velocity_weight):
    model.eval()
    total_loss = 0.0
    position_errors, velocity_errors = [], []
    with torch.no_grad():
        for current, history, target in loader:
            current, history, target = current.to(device), history.to(device), target.to(device)
            prediction = model(current, history)
            loss, _ = reference_imitation_loss(prediction, target, velocity_weight=velocity_weight)
            total_loss += float(loss.item()) * current.shape[0]
            physical_prediction = model.denormalize(prediction)
            physical_target = model.denormalize(target)
            position_errors.extend(torch.linalg.vector_norm(physical_prediction[..., :3] - physical_target[..., :3], dim=-1).cpu().numpy().ravel())
            velocity_errors.extend(torch.linalg.vector_norm(physical_prediction[..., 3:] - physical_target[..., 3:], dim=-1).cpu().numpy().ravel())
    return {
        "loss": float(total_loss / len(loader.dataset)),
        "position_rmse_m": float(np.sqrt(np.mean(np.square(position_errors)))),
        "velocity_rmse_mps": float(np.sqrt(np.mean(np.square(velocity_errors)))),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/reference_planner/temporal_controlled_v1_temporal8")
    parser.add_argument("--variant", choices=("history_stack", "gru"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--velocity-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.epochs, args.batch_size) <= 0 or args.learning_rate <= 0.0:
        raise ValueError("Training hyperparameters must be positive.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device("cpu")
    train = TemporalReferenceTrajectoryDataset(args.dataset, "train")
    validation = TemporalReferenceTrajectoryDataset(args.dataset, "validation")
    test = TemporalReferenceTrajectoryDataset(args.dataset, "test")
    try:
        generator = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=generator)
        validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False)
        model = TemporalReferenceSequencePlanner(args.variant).to(device)
        model.set_target_normalization(train.target_mean, train.target_std)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        best_loss, best_epoch, history = float("inf"), 0, []
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss_sum = 0.0
            for current, temporal_history, target in train_loader:
                current, temporal_history, target = current.to(device), temporal_history.to(device), target.to(device)
                prediction = model(current, temporal_history)
                loss, _ = reference_imitation_loss(prediction, target, velocity_weight=args.velocity_weight)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                train_loss_sum += float(loss.item()) * current.shape[0]
            metrics = evaluate(model, validation_loader, device, args.velocity_weight)
            train_loss = float(train_loss_sum / len(train))
            history.append({"epoch": epoch, "train_loss": train_loss, **{f"validation_{key}": value for key, value in metrics.items()}})
            if metrics["loss"] < best_loss:
                best_loss, best_epoch = float(metrics["loss"]), epoch
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "variant": args.variant,
                    "input_dim": model.input_dim,
                    "num_points": 5,
                    "dataset": str(Path(args.dataset)),
                    "dataset_version": train.metadata["dataset_version"],
                    "velocity_weight": float(args.velocity_weight),
                    "epoch": epoch,
                    "validation": metrics,
                }, output)
            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                print(f"{args.variant} epoch={epoch}, train={train_loss:.5f}, val={metrics['loss']:.5f}, pos_rmse={metrics['position_rmse_m']:.4f}m")
        checkpoint = torch.load(output, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        report = {
            "variant": args.variant,
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "test": evaluate(model, test_loader, device, args.velocity_weight),
            "train_samples": len(train), "validation_samples": len(validation), "test_samples": len(test),
            "encoder_frozen": True, "actor_frozen": True, "pid_frozen": True,
            "history": history,
        }
        output.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"output": str(output), "variant": args.variant, "best_epoch": best_epoch, "test": report["test"]}, indent=2, sort_keys=True))
    finally:
        train.close(); validation.close(); test.close()


if __name__ == "__main__":
    main()
