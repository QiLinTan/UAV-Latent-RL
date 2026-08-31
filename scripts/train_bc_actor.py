from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys

import numpy as np
import torch

from algos.td3.networks import Actor
from data.bc_metrics import evaluate_grouped_predictions
from data.behavior_cloning_dataset import BehaviorCloningDataset


def predict(actor, contexts, *, device, batch_size=8192):
    outputs = []
    actor.eval()
    with torch.no_grad():
        for start in range(0, contexts.shape[0], batch_size):
            batch = torch.as_tensor(
                contexts[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            outputs.append(actor(batch).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def action_rmse(prediction, target):
    return float(np.sqrt(np.mean(np.square(prediction - target))))


def validation_score(prediction, target, samples, indices):
    group_id = samples["sample_group_id"][indices]
    recovery = samples["recovery_flag"][indices]
    any_large_torque = np.logical_or.reduce(
        [
            samples["large_roll_torque_flag"][indices],
            samples["large_pitch_torque_flag"][indices],
            samples["large_yaw_torque_flag"][indices],
        ]
    )
    masks = {
        "overall": np.ones(indices.shape[0], dtype=bool),
        "nominal": group_id == 0,
        "initial_recovery": np.logical_and(group_id == 1, recovery),
        "impulse_recovery": np.logical_and(group_id == 2, recovery),
        "large_torque": any_large_torque,
    }
    values = {
        name: (
            action_rmse(prediction[mask], target[mask])
            if np.any(mask)
            else 0.0
        )
        for name, mask in masks.items()
    }
    score = (
        values["overall"]
        + values["nominal"]
        + 2.0 * values["initial_recovery"]
        + 2.0 * values["impulse_recovery"]
        + 2.0 * values["large_torque"]
    )
    return score, values


def build_sampling_probabilities(samples, indices):
    group_id = samples["sample_group_id"][indices]
    weights = np.ones(indices.shape[0], dtype=np.float64)
    for group in (0, 1, 2):
        mask = group_id == group
        if np.any(mask):
            weights[mask] *= indices.shape[0] / (3.0 * np.sum(mask))
    weights *= 1.0 + 3.0 * samples["recovery_flag"][indices].astype(np.float64)
    weights *= 1.0 + 2.0 * samples["recovery_early_flag"][indices].astype(np.float64)
    any_large_torque = np.logical_or.reduce(
        [
            samples["large_roll_torque_flag"][indices],
            samples["large_pitch_torque_flag"][indices],
            samples["large_yaw_torque_flag"][indices],
        ]
    )
    weights *= 1.0 + 2.0 * any_large_torque.astype(np.float64)
    action_norm = np.max(
        np.abs(samples["encoded_teacher_action"][indices]),
        axis=1,
    )
    weights *= 1.0 + 1.5 * (action_norm >= 0.20).astype(np.float64)
    weights = np.minimum(weights, np.percentile(weights, 99.5))
    return weights / np.sum(weights)


def main():
    parser = argparse.ArgumentParser(
        description="Train the plain four-output MLP by behavior cloning only."
    )
    parser.add_argument(
        "--dataset",
        default="data/behavior_cloning/asymmetric_rpm_v2",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints/behavior_cloning/asymmetric_rpm_v2_plain_mlp",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()

    output = pathlib.Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "actor_best.pt"
    if checkpoint_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing behavior-cloning checkpoint {checkpoint_path}."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = BehaviorCloningDataset(args.dataset)
    metadata = dataset.metadata
    samples = dataset.samples
    train_indices = dataset.split_sample_indices("train")
    validation_indices = dataset.split_sample_indices("validation")
    if np.intersect1d(
        dataset.split_episode_ids("train"),
        dataset.split_episode_ids("validation"),
    ).size:
        raise AssertionError("Training and validation episode IDs overlap.")

    context_dim = int(metadata["context_dim"])
    actor = Actor(context_dim, 4, 1.0).to(device)
    optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(args.epochs)),
        eta_min=float(args.learning_rate) * 0.05,
    )
    sampling_probabilities = build_sampling_probabilities(samples, train_indices)
    rng = np.random.default_rng(args.seed)
    steps_per_epoch = int(math.ceil(train_indices.shape[0] / args.batch_size))
    best_score = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    history = []

    validation_context = np.asarray(
        samples["context"][validation_indices],
        dtype=np.float32,
    )
    validation_target = np.asarray(
        samples["encoded_teacher_action"][validation_indices],
        dtype=np.float32,
    )

    for epoch in range(int(args.epochs)):
        actor.train()
        losses = []
        for _ in range(steps_per_epoch):
            weighted_count = args.batch_size // 2
            uniform_count = args.batch_size - weighted_count
            weighted = rng.choice(
                train_indices,
                size=weighted_count,
                replace=True,
                p=sampling_probabilities,
            )
            uniform = rng.choice(
                train_indices,
                size=uniform_count,
                replace=True,
            )
            batch_indices = np.concatenate([weighted, uniform])
            rng.shuffle(batch_indices)
            context = torch.as_tensor(
                samples["context"][batch_indices],
                dtype=torch.float32,
                device=device,
            )
            target = torch.as_tensor(
                samples["encoded_teacher_action"][batch_indices],
                dtype=torch.float32,
                device=device,
            )
            prediction = actor(context)
            loss = torch.mean(torch.square(prediction - target))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        scheduler.step()

        validation_prediction = predict(
            actor,
            validation_context,
            device=device,
        )
        score, validation_groups = validation_score(
            validation_prediction,
            validation_target,
            samples,
            validation_indices,
        )
        record = {
            "epoch": epoch + 1,
            "training_mse": float(np.mean(losses)),
            "validation_score": float(score),
            "validation_action_rmse": validation_groups,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        print(
            "[bc-train] "
            f"epoch={epoch + 1:03d} "
            f"train_mse={record['training_mse']:.8f} "
            f"val={score:.6f} "
            f"recovery={validation_groups['initial_recovery']:.6f}/"
            f"{validation_groups['impulse_recovery']:.6f}",
            file=sys.stderr,
            flush=True,
        )
        if score < best_score - 1e-7:
            best_score = score
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(
                {
                    "actor_state_dict": actor.state_dict(),
                    "architecture": {
                        "type": "plain_mlp",
                        "input_dim": context_dim,
                        "hidden_dims": [256, 256],
                        "output_dim": 4,
                        "output_activation": "tanh",
                        "max_action": 1.0,
                        "multi_step_action_history": 0,
                    },
                    "dataset_version": metadata["dataset_version"],
                    "dataset_samples_sha256": metadata["samples_sha256"],
                    "motor_action_codec": metadata["motor_action_codec"],
                    "teacher_reset_enabled": metadata["teacher_reset_enabled"],
                    "physical_parameters": metadata["motor_physical_parameters"],
                    "training_seed": args.seed,
                    "best_epoch": best_epoch,
                    "best_validation_score": best_score,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(args.patience):
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
    split_metrics = {}
    for split in (
        "train",
        "validation",
        "test_unseen_seed",
        "test_unseen_condition",
    ):
        indices = dataset.split_sample_indices(split)
        contexts = np.asarray(samples["context"][indices], dtype=np.float32)
        target = np.asarray(
            samples["encoded_teacher_action"][indices],
            dtype=np.float32,
        )
        prediction = predict(actor, contexts, device=device)
        split_metrics[split] = evaluate_grouped_predictions(
            prediction,
            target,
            samples,
            indices,
            metadata["motor_physical_parameters"],
        )

    training_summary = {
        "dataset": str(pathlib.Path(args.dataset)),
        "dataset_version": metadata["dataset_version"],
        "motor_action_codec": metadata["motor_action_codec"],
        "teacher_reset_enabled": metadata["teacher_reset_enabled"],
        "actor_bc_training_completed": True,
        "device": str(device),
        "architecture": checkpoint["architecture"],
        "training_seed": args.seed,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "training_episode_ids": dataset.split_episode_ids("train").tolist(),
        "validation_episode_ids": dataset.split_episode_ids("validation").tolist(),
        "history": history,
        "offline_metrics": split_metrics,
    }
    (output / "training_summary.json").write_text(
        json.dumps(training_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "offline_metrics.json").write_text(
        json.dumps(split_metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    dataset.close()
    print(
        json.dumps(
            {
                "actor_bc_training_completed": True,
                "checkpoint": str(checkpoint_path),
                "best_epoch": best_epoch,
                "best_validation_score": best_score,
                "offline_metrics": str(output / "offline_metrics.json"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
