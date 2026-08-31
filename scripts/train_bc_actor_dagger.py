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
from data.bc_metrics import evaluate_grouped_predictions, metric_block
from data.behavior_cloning_dataset import BehaviorCloningDataset
from scripts.train_bc_actor import (
    build_sampling_probabilities,
    predict,
    validation_score,
)


def dagger_masks(arrays, indices, thresholds):
    wrench = arrays["teacher_wrench"][indices]
    critical = np.logical_or.reduce(
        [
            np.abs(wrench[:, axis]) >= float(thresholds[name])
            for axis, name in enumerate(("roll", "pitch", "yaw"), start=1)
        ]
    )
    return {
        "overall": np.ones(indices.shape[0], dtype=bool),
        "recovery_active": arrays["recovery_active"][indices].astype(bool),
        "recovery_early": arrays["recovery_early"][indices].astype(bool),
        "critical_recovery_torque": np.logical_and(
            arrays["recovery_active"][indices].astype(bool),
            critical,
        ),
    }


def dagger_sampling_probabilities(arrays, indices, thresholds):
    masks = dagger_masks(arrays, indices, thresholds)
    weights = np.ones(indices.shape[0], dtype=np.float64)
    weights *= 1.0 + 4.0 * masks["recovery_active"].astype(np.float64)
    weights *= 1.0 + 3.0 * masks["recovery_early"].astype(np.float64)
    weights *= 1.0 + 4.0 * masks["critical_recovery_torque"].astype(np.float64)
    original_error = np.max(
        np.abs(
            arrays["actor_action"][indices]
            - arrays["teacher_action"][indices]
        ),
        axis=1,
    )
    weights *= 1.0 + 3.0 * (original_error >= 0.02).astype(np.float64)
    weights = np.minimum(weights, np.percentile(weights, 99.5))
    return weights / np.sum(weights)


def dagger_validation_score(prediction, target, arrays, indices, thresholds):
    masks = dagger_masks(arrays, indices, thresholds)
    scores = {}
    for name, mask in masks.items():
        scores[name] = (
            float(np.sqrt(np.mean(np.square(prediction[mask] - target[mask]))))
            if np.any(mask)
            else 0.0
        )
    score = (
        scores["overall"]
        + 2.0 * scores["recovery_active"]
        + 2.0 * scores["recovery_early"]
        + 2.0 * scores["critical_recovery_torque"]
    )
    return score, scores


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune the same plain four-output MLP on base BC data plus "
            "teacher labels from Actor-visited states."
        )
    )
    parser.add_argument(
        "--base-dataset",
        default="data/behavior_cloning/asymmetric_rpm_v2",
    )
    parser.add_argument(
        "--dagger-dataset",
        default="data/behavior_cloning/asymmetric_rpm_v2_dagger1",
    )
    parser.add_argument(
        "--initial-checkpoint",
        default=(
            "checkpoints/behavior_cloning/"
            "asymmetric_rpm_v2_plain_mlp_b4096/actor_best.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "checkpoints/behavior_cloning/"
            "asymmetric_rpm_v2_plain_mlp_dagger1_b4096"
        ),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    output = pathlib.Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "actor_best.pt"
    if checkpoint_path.exists():
        raise FileExistsError(f"Refusing to overwrite {checkpoint_path}.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = BehaviorCloningDataset(args.base_dataset)
    base_metadata = base.metadata
    base_train = base.split_sample_indices("train")
    base_validation = base.split_sample_indices("validation")
    # np.savez_compressed archives are expensive when the same member is
    # indexed thousands of times. Materialize once before minibatch sampling.
    base_samples = {
        name: np.asarray(base.samples[name])
        for name in base.samples.files
    }
    dagger_root = pathlib.Path(args.dagger_dataset)
    dagger_metadata = json.loads(
        (dagger_root / "metadata.json").read_text(encoding="utf-8")
    )
    if dagger_metadata["base_dataset_samples_sha256"] != base_metadata[
        "samples_sha256"
    ]:
        raise ValueError("DAgger and base dataset hashes do not match.")
    dagger_archive = np.load(dagger_root / "actor_visited_samples.npz")
    dagger = {
        name: np.asarray(dagger_archive[name])
        for name in dagger_archive.files
    }
    dagger_archive.close()
    if not np.array_equal(dagger["actor_action"], dagger["applied_action"]):
        raise AssertionError("DAgger data contains teacher takeover or action mismatch.")

    initial = torch.load(args.initial_checkpoint, map_location=device)
    if initial["dataset_samples_sha256"] != base_metadata["samples_sha256"]:
        raise ValueError("Initial Actor and base dataset hashes do not match.")
    context_dim = int(base_metadata["context_dim"])
    actor = Actor(context_dim, 4, 1.0).to(device)
    actor.load_state_dict(initial["actor_state_dict"])

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
    dagger_train = np.flatnonzero(dagger["split_id"] == 0)
    dagger_validation = np.flatnonzero(dagger["split_id"] == 1)
    thresholds = base_metadata["critical_recovery_torque_thresholds"]
    base_probabilities = build_sampling_probabilities(base_samples, base_train)
    dagger_probabilities = dagger_sampling_probabilities(
        dagger,
        dagger_train,
        thresholds,
    )
    rng = np.random.default_rng(args.seed)
    combined_train_size = base_train.shape[0] + dagger_train.shape[0]
    steps_per_epoch = int(math.ceil(combined_train_size / args.batch_size))

    base_val_context = np.asarray(
        base_samples["context"][base_validation],
        dtype=np.float32,
    )
    base_val_target = np.asarray(
        base_samples["encoded_teacher_action"][base_validation],
        dtype=np.float32,
    )
    dagger_val_context = np.asarray(
        dagger["context"][dagger_validation],
        dtype=np.float32,
    )
    dagger_val_target = np.asarray(
        dagger["teacher_action"][dagger_validation],
        dtype=np.float32,
    )

    best_score = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    history = []
    for epoch in range(int(args.epochs)):
        actor.train()
        losses = []
        for _ in range(steps_per_epoch):
            base_count = args.batch_size // 2
            dagger_count = args.batch_size - base_count
            base_batch = rng.choice(
                base_train,
                size=base_count,
                replace=True,
                p=base_probabilities,
            )
            dagger_batch = rng.choice(
                dagger_train,
                size=dagger_count,
                replace=True,
                p=dagger_probabilities,
            )
            contexts = np.concatenate(
                [
                    base_samples["context"][base_batch],
                    dagger["context"][dagger_batch],
                ],
                axis=0,
            )
            targets = np.concatenate(
                [
                    base_samples["encoded_teacher_action"][base_batch],
                    dagger["teacher_action"][dagger_batch],
                ],
                axis=0,
            )
            order = rng.permutation(args.batch_size)
            context_tensor = torch.as_tensor(
                contexts[order],
                dtype=torch.float32,
                device=device,
            )
            target_tensor = torch.as_tensor(
                targets[order],
                dtype=torch.float32,
                device=device,
            )
            prediction = actor(context_tensor)
            loss = torch.mean(torch.square(prediction - target_tensor))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        scheduler.step()

        base_prediction = predict(actor, base_val_context, device=device)
        base_score, base_groups = validation_score(
            base_prediction,
            base_val_target,
            base_samples,
            base_validation,
        )
        dagger_prediction = predict(actor, dagger_val_context, device=device)
        online_score, online_groups = dagger_validation_score(
            dagger_prediction,
            dagger_val_target,
            dagger,
            dagger_validation,
            thresholds,
        )
        score = base_score + online_score
        record = {
            "epoch": epoch + 1,
            "training_mse": float(np.mean(losses)),
            "combined_validation_score": float(score),
            "base_validation_score": float(base_score),
            "base_validation_action_rmse": base_groups,
            "dagger_validation_score": float(online_score),
            "dagger_validation_action_rmse": online_groups,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        print(
            "[bc-dagger] "
            f"epoch={epoch + 1:03d} train={record['training_mse']:.8f} "
            f"score={score:.6f} base_rec="
            f"{base_groups['initial_recovery']:.6f}/"
            f"{base_groups['impulse_recovery']:.6f} "
            f"online_rec={online_groups['recovery_active']:.6f}",
            file=sys.stderr,
            flush=True,
        )
        if score < best_score - 1e-7:
            best_score = float(score)
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(
                {
                    "actor_state_dict": actor.state_dict(),
                    "architecture": initial["architecture"],
                    "dataset_version": dagger_metadata["dataset_version"],
                    "base_dataset_version": base_metadata["dataset_version"],
                    "dataset_samples_sha256": base_metadata["samples_sha256"],
                    "dagger_samples_sha256": dagger_metadata["samples_sha256"],
                    "motor_action_codec": base_metadata["motor_action_codec"],
                    "teacher_reset_enabled": True,
                    "physical_parameters": base_metadata["motor_physical_parameters"],
                    "training_seed": args.seed,
                    "best_epoch": best_epoch,
                    "best_validation_score": best_score,
                    "initial_checkpoint": args.initial_checkpoint,
                    "training_algorithm": "behavior_cloning_dagger_aggregation",
                    "teacher_takeover_used_during_collection": False,
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
    base_metrics = {}
    for split in (
        "train",
        "validation",
        "test_unseen_seed",
        "test_unseen_condition",
    ):
        indices = base.split_sample_indices(split)
        prediction = predict(
            actor,
            np.asarray(base_samples["context"][indices], dtype=np.float32),
            device=device,
        )
        base_metrics[split] = evaluate_grouped_predictions(
            prediction,
            np.asarray(
                base_samples["encoded_teacher_action"][indices],
                dtype=np.float32,
            ),
            base_samples,
            indices,
            base_metadata["motor_physical_parameters"],
        )

    dagger_metrics = {}
    for split_name, split_id in (("train", 0), ("validation", 1)):
        indices = np.flatnonzero(dagger["split_id"] == split_id)
        prediction = predict(
            actor,
            np.asarray(dagger["context"][indices], dtype=np.float32),
            device=device,
        )
        target = np.asarray(dagger["teacher_action"][indices], dtype=np.float32)
        masks = dagger_masks(dagger, indices, thresholds)
        dagger_metrics[split_name] = {
            name: metric_block(
                prediction[mask],
                target[mask],
                base_metadata["motor_physical_parameters"],
            )
            for name, mask in masks.items()
        }

    summary = {
        "dataset_version": dagger_metadata["dataset_version"],
        "base_dataset_version": base_metadata["dataset_version"],
        "motor_action_codec": "asymmetric_rpm",
        "actor_bc_training_completed": True,
        "dagger_iteration": 1,
        "teacher_takeover_used": False,
        "architecture": checkpoint["architecture"],
        "initial_checkpoint": args.initial_checkpoint,
        "checkpoint": str(checkpoint_path),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "base_offline_metrics": base_metrics,
        "dagger_offline_metrics": dagger_metrics,
        "history": history,
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "offline_metrics.json").write_text(
        json.dumps(
            {
                "base_dataset": base_metrics,
                "dagger_actor_visited_dataset": dagger_metrics,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    base.close()
    print(
        json.dumps(
            {
                "actor_bc_training_completed": True,
                "dataset_version": dagger_metadata["dataset_version"],
                "checkpoint": str(checkpoint_path),
                "best_epoch": best_epoch,
                "best_validation_score": best_score,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
