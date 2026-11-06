from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess

import torch


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state(root: pathlib.Path) -> dict:
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def main():
    parser = argparse.ArgumentParser(
        description="Freeze the dual lower-controller baselines and their provenance."
    )
    parser.add_argument(
        "--actor-checkpoint",
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
        "--output",
        default="configs/lower_baseline_freeze_v1.json",
    )
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parents[1]
    checkpoint_path = root / args.actor_checkpoint
    metadata_path = root / args.dataset_metadata
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen manifest {output}.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint["dataset_samples_sha256"] != metadata["samples_sha256"]:
        raise ValueError("Actor checkpoint and behavior-cloning dataset do not match.")
    if metadata["motor_action_codec"] != "asymmetric_rpm":
        raise ValueError("Frozen baselines require the asymmetric_rpm motor codec.")
    if metadata["teacher_reset_enabled"] is not True:
        raise ValueError("Frozen DSLPID baseline requires reset at every episode.")

    teacher_source = (
        root.parent
        / "gym-pybullet-drones"
        / "gym_pybullet_drones"
        / "control"
        / "DSLPIDControl.py"
    )
    manifest = {
        "freeze_version": "dual_lower_baseline_v1",
        "created_for_stage": "rule_upper_async_architecture_validation",
        "project": git_state(root),
        "shared_contract": {
            "motor_action_codec": "asymmetric_rpm",
            "motor_physical_parameters": metadata["motor_physical_parameters"],
            "control_frequency_hz": metadata["control_frequency"],
            "physics_frequency_hz": metadata["physics_frequency"],
            "reference_packet_sequence_length": 15,
            "reference_horizon_seconds": 1.0,
            "reference_frame": "world",
            "teacher_or_actor_takeover": False,
        },
        "L0_teacher_t3": {
            "controller": "DSLPIDControl",
            "source_path": str(teacher_source),
            "source_sha256": sha256(teacher_source),
            "reset_each_episode": True,
            "output_chain": (
                "DSLPID raw RPM -> asymmetric_rpm encode -> normalized motor "
                "action -> asymmetric_rpm decode -> environment RPM"
            ),
            "role": "traditional lower-controller reference baseline",
        },
        "L1_bc_mlp_t3": {
            "controller": "plain_mlp",
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "dataset_version": metadata["dataset_version"],
            "dataset_samples_sha256": metadata["samples_sha256"],
            "architecture": checkpoint["architecture"],
            "context_definition": metadata["context_definition"],
            "role": "learned direct-motor lower-controller baseline",
            "known_gates": {
                "actor_hover_gate_passed": True,
                "actor_dynamic_recovery_observed": True,
                "strict_teacher_imitation_fidelity_passed": False,
                "td3_training_allowed": False,
            },
        },
        "excluded_from_mainline": {
            "checkpoint": (
                "checkpoints/behavior_cloning/"
                "asymmetric_rpm_v2_plain_mlp_dagger1_b4096/actor_best.pt"
            ),
            "reason": "DAgger iteration introduced steady angular-rate oscillation.",
        },
        "change_control": {
            "frozen_means": (
                "No weight, input-layout, codec, controller-gain, or fallback "
                "change is allowed during dual-baseline comparisons."
            ),
            "new_version_required_for_changes": True,
        },
    }
    output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
