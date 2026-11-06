"""Train only the latent-conditioned reference planner by teacher imitation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from high_level_planner import (
    ReferenceSequencePlanner,
    ReferenceTrajectoryDataset,
    reference_imitation_loss,
)


def evaluate(model, loader, device, velocity_weight):
    model.eval()
    totals = []
    position_errors = []
    velocity_errors = []
    with torch.no_grad():
        for planner_input, target in loader:
            planner_input = planner_input.to(device)
            target = target.to(device)
            prediction = model(planner_input)
            loss, _ = reference_imitation_loss(
                prediction,
                target,
                velocity_weight=velocity_weight,
            )
            physical_prediction = model.denormalize(prediction)
            physical_target = model.denormalize(target)
            totals.append(float(loss.item()) * planner_input.shape[0])
            position_errors.extend(
                torch.linalg.vector_norm(
                    physical_prediction[..., :3] - physical_target[..., :3],
                    dim=-1,
                ).cpu().numpy().reshape(-1).tolist()
            )
            velocity_errors.extend(
                torch.linalg.vector_norm(
                    physical_prediction[..., 3:] - physical_target[..., 3:],
                    dim=-1,
                ).cpu().numpy().reshape(-1).tolist()
            )
    sample_count = len(loader.dataset)
    return {
        "loss": float(sum(totals) / max(1, sample_count)),
        "position_rmse_m": float(np.sqrt(np.mean(np.square(position_errors)))),
        "velocity_rmse_mps": float(np.sqrt(np.mean(np.square(velocity_errors)))),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="data/reference_planner/latent_affordance_margin_v2",
    )
    parser.add_argument(
        "--output",
        default="checkpoints/reference_planner/latent_affordance_margin_v2/planner_final.pt",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--velocity-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0.0:
        raise ValueError("Training hyperparameters must be positive.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    train_data = ReferenceTrajectoryDataset(args.dataset, "train")
    validation_data = ReferenceTrajectoryDataset(args.dataset, "validation")
    test_data = ReferenceTrajectoryDataset(args.dataset, "test")
    try:
        generator = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
        )
        validation_loader = DataLoader(
            validation_data,
            batch_size=args.batch_size,
            shuffle=False,
        )
        test_loader = DataLoader(
            test_data,
            batch_size=args.batch_size,
            shuffle=False,
        )
        model = ReferenceSequencePlanner().to(device)
        model.set_target_normalization(train_data.target_mean, train_data.target_std)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        best_validation = float("inf")
        best_epoch = 0
        history = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss_sum = 0.0
            for planner_input, target in train_loader:
                planner_input = planner_input.to(device)
                target = target.to(device)
                prediction = model(planner_input)
                loss, _ = reference_imitation_loss(
                    prediction,
                    target,
                    velocity_weight=args.velocity_weight,
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                train_loss_sum += float(loss.item()) * planner_input.shape[0]
            validation = evaluate(
                model,
                validation_loader,
                device,
                args.velocity_weight,
            )
            train_loss = train_loss_sum / len(train_data)
            history.append({
                "epoch": epoch,
                "train_loss": float(train_loss),
                **{f"validation_{key}": value for key, value in validation.items()},
            })
            if validation["loss"] < best_validation:
                best_validation = validation["loss"]
                best_epoch = epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "input_dim": model.input_dim,
                        "num_points": model.num_points,
                        "dataset": str(Path(args.dataset)),
                        "dataset_version": train_data.metadata["dataset_version"],
                        "velocity_weight": float(args.velocity_weight),
                        "epoch": int(epoch),
                        "validation": validation,
                    },
                    output,
                )
            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                print(
                    f"epoch={epoch}, train={train_loss:.5f}, "
                    f"val={validation['loss']:.5f}, "
                    f"position_rmse={validation['position_rmse_m']:.4f}m"
                )

        checkpoint = torch.load(output, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = evaluate(model, test_loader, device, args.velocity_weight)
        report = {
            "best_epoch": int(best_epoch),
            "best_validation_loss": float(best_validation),
            "test": test_metrics,
            "train_samples": len(train_data),
            "validation_samples": len(validation_data),
            "test_samples": len(test_data),
            "encoder_frozen": True,
            "actor_frozen": True,
            "pid_frozen": True,
            "history": history,
        }
        report_path = output.with_suffix(".metrics.json")
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(
            f"saved {output}; best_epoch={best_epoch}, "
            f"test_position_rmse={test_metrics['position_rmse_m']:.4f}m, "
            f"test_velocity_rmse={test_metrics['velocity_rmse_mps']:.4f}m/s"
        )
    finally:
        train_data.close()
        validation_data.close()
        test_data.close()


if __name__ == "__main__":
    main()
