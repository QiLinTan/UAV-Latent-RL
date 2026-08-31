from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np
import torch

from scripts.bc_scenarios import build_scenarios
from scripts.evaluate_bc_actor import merge_step_rows, run_episode


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Collect DAgger labels on states visited by the behavior-cloned Actor. "
            "The Actor always controls the vehicle; DSLPID only labels each state."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/behavior_cloning/"
            "asymmetric_rpm_v2_plain_mlp_b4096/actor_best.pt"
        ),
    )
    parser.add_argument(
        "--dataset-metadata",
        default="data/behavior_cloning/asymmetric_rpm_v2/metadata.json",
    )
    parser.add_argument(
        "--output-dir",
        default="data/behavior_cloning/asymmetric_rpm_v2_dagger1",
    )
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[5, 6, 7])
    parser.add_argument("--validation-seeds", nargs="+", type=int, default=[8])
    parser.add_argument("--duration", type=float, default=12.0)
    args = parser.parse_args()

    output = pathlib.Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    samples_path = output / "actor_visited_samples.npz"
    metadata_path = output / "metadata.json"
    if samples_path.exists() or metadata_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing DAgger data in {output}.")

    base_metadata = json.loads(
        pathlib.Path(args.dataset_metadata).read_text(encoding="utf-8")
    )
    assert base_metadata["motor_action_codec"] == "asymmetric_rpm"
    assert base_metadata["teacher_reset_enabled"] is True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if checkpoint["dataset_samples_sha256"] != base_metadata["samples_sha256"]:
        raise ValueError("Initial Actor and base dataset hashes do not match.")

    scenarios = [
        scenario for scenario in build_scenarios() if not scenario.holdout_condition
    ]
    seeds = [
        *(("train", seed) for seed in args.train_seeds),
        *(("validation", seed) for seed in args.validation_seeds),
    ]
    rows_by_episode = []
    split_ids = []
    manifests = []
    episode_uid = 0
    total = len(scenarios) * len(seeds)
    for scenario in scenarios:
        for split, seed in seeds:
            result, rows = run_episode(
                scenario,
                seed,
                float(args.duration),
                controller_mode="actor",
                checkpoint=checkpoint,
                device=device,
                physical_parameters=base_metadata["motor_physical_parameters"],
                torque_thresholds=base_metadata["large_torque_thresholds"],
                critical_torque_thresholds=base_metadata[
                    "critical_recovery_torque_thresholds"
                ],
                episode_uid=episode_uid,
            )
            if not result["full_horizon"]:
                raise RuntimeError(
                    "DAgger collection Actor became unstable: "
                    f"{scenario.name}, seed={seed}, reason={result['done_reason']}."
                )
            row_count = len(rows["step_id"])
            rows_by_episode.append(rows)
            split_ids.extend([0 if split == "train" else 1] * row_count)
            manifests.append(
                {
                    "episode_uid": episode_uid,
                    "scenario": scenario.name,
                    "category": scenario.category,
                    "reference_kind": scenario.reference_kind,
                    "seed": seed,
                    "split": split,
                    "steps": row_count,
                    "actor_full_horizon": result["full_horizon"],
                    "actor_recovery_time": result["recovery_time"],
                    "teacher_takeover_used": False,
                }
            )
            episode_uid += 1
            if episode_uid % 10 == 0 or episode_uid == total:
                print(
                    f"[dagger-collect] completed {episode_uid}/{total}",
                    file=sys.stderr,
                    flush=True,
                )

    arrays = merge_step_rows(rows_by_episode)
    arrays["split_id"] = np.asarray(split_ids, dtype=np.int8)
    if not np.array_equal(arrays["actor_action"], arrays["applied_action"]):
        raise AssertionError("Actor applied action differs from its recorded action.")
    np.savez_compressed(samples_path, **arrays)

    train_mask = arrays["split_id"] == 0
    validation_mask = arrays["split_id"] == 1
    metadata = {
        "dataset_version": "asymmetric_rpm_v2_dagger1",
        "base_dataset_version": base_metadata["dataset_version"],
        "motor_action_codec": "asymmetric_rpm",
        "teacher_controller": "DSLPIDControl",
        "teacher_reset_enabled": True,
        "collection_policy": "actor_best_from_asymmetric_rpm_v2_plain_mlp_b4096",
        "actor_controls_environment": True,
        "teacher_parallel_labels_only": True,
        "teacher_takeover_used": False,
        "teacher_action_noise": 0.0,
        "holdout_conditions_excluded": True,
        "train_seeds": args.train_seeds,
        "validation_seeds": args.validation_seeds,
        "episode_duration_seconds": float(args.duration),
        "episode_count": len(manifests),
        "train_episode_count": int(
            np.sum([item["split"] == "train" for item in manifests])
        ),
        "validation_episode_count": int(
            np.sum([item["split"] == "validation" for item in manifests])
        ),
        "sample_count": int(arrays["context"].shape[0]),
        "train_sample_count": int(np.sum(train_mask)),
        "validation_sample_count": int(np.sum(validation_mask)),
        "recovery_active_sample_count": int(np.sum(arrays["recovery_active"])),
        "recovery_early_sample_count": int(np.sum(arrays["recovery_early"])),
        "base_dataset_samples_sha256": base_metadata["samples_sha256"],
        "initial_checkpoint": args.checkpoint,
        "samples_sha256": None,
        "critical_recovery_torque_thresholds": base_metadata[
            "critical_recovery_torque_thresholds"
        ],
        "manifest": manifests,
    }
    metadata["samples_sha256"] = sha256(samples_path)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "dagger_dataset_collected": True,
                "dataset_version": metadata["dataset_version"],
                "episode_count": metadata["episode_count"],
                "sample_count": metadata["sample_count"],
                "train_sample_count": metadata["train_sample_count"],
                "validation_sample_count": metadata["validation_sample_count"],
                "samples_sha256": metadata["samples_sha256"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
