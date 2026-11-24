from __future__ import annotations

import json
from pathlib import Path

import numpy as np


EXPECTED_DATASET_VERSION = "asymmetric_rpm_v2"
EXPECTED_MOTOR_ACTION_CODEC = "asymmetric_rpm"

REQUIRED_METADATA_FIELDS = {
    "dataset_version",
    "motor_action_codec",
    "teacher_controller",
    "teacher_reset_enabled",
    "control_frequency",
    "physics_frequency",
    "reference_mode",
    "random_seed",
    "environment_commit",
    "project_commit",
    "context_dim",
    "sample_count",
    "episode_count",
}

REQUIRED_ARRAYS = {
    "episode_id",
    "step_id",
    "time",
    "state",
    "context",
    "current_reference",
    "lookahead_reference",
    "reference_age",
    "reference_valid",
    "previous_action",
    "raw_teacher_rpm",
    "encoded_teacher_action",
    "applied_action",
    "next_state",
    "disturbance_type_id",
    "disturbance_magnitude",
    "recovery_flag",
    "recovery_early_flag",
    "teacher_wrench",
    "large_roll_torque_flag",
    "large_pitch_torque_flag",
    "large_yaw_torque_flag",
    "sample_group_id",
}


class BehaviorCloningDataset:
    """Validated, version-locked behavior-cloning dataset."""

    def __init__(self, root):
        self.root = Path(root)
        self.metadata_path = self.root / "metadata.json"
        self.samples_path = self.root / "samples.npz"
        self.splits_path = self.root / "splits.npz"
        self.episode_manifest_path = self.root / "episode_manifest.jsonl"
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Missing dataset metadata: {self.metadata_path}")
        if not self.samples_path.exists():
            raise FileNotFoundError(f"Missing dataset samples: {self.samples_path}")
        if not self.splits_path.exists():
            raise FileNotFoundError(f"Missing dataset splits: {self.splits_path}")

        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        missing_metadata = REQUIRED_METADATA_FIELDS - set(self.metadata)
        if missing_metadata:
            raise ValueError(
                f"Dataset metadata is missing required fields: {sorted(missing_metadata)}"
            )

        # These explicit checks intentionally fail closed: legacy and mixed
        # datasets cannot silently enter the new training path.
        assert self.metadata["dataset_version"] == EXPECTED_DATASET_VERSION
        assert self.metadata["motor_action_codec"] == EXPECTED_MOTOR_ACTION_CODEC
        assert self.metadata["teacher_reset_enabled"] is True
        if self.metadata.get("quarantined", False):
            raise ValueError("Quarantined datasets cannot be used for training.")

        self.samples = np.load(self.samples_path, allow_pickle=False)
        missing_arrays = REQUIRED_ARRAYS - set(self.samples.files)
        if missing_arrays:
            raise ValueError(
                f"Dataset sample archive is missing arrays: {sorted(missing_arrays)}"
            )
        self.splits = np.load(self.splits_path, allow_pickle=False)
        for split in ("train", "validation", "test_unseen_seed", "test_unseen_condition"):
            if split not in self.splits.files:
                raise ValueError(f"Dataset is missing episode split {split!r}.")

        sample_count = int(self.metadata["sample_count"])
        for name in REQUIRED_ARRAYS:
            if int(self.samples[name].shape[0]) != sample_count:
                raise ValueError(
                    f"Array {name!r} has {self.samples[name].shape[0]} rows, "
                    f"expected {sample_count}."
                )
        if not np.array_equal(
            self.samples["encoded_teacher_action"],
            self.samples["applied_action"],
        ):
            difference = float(
                np.max(
                    np.abs(
                        self.samples["encoded_teacher_action"]
                        - self.samples["applied_action"]
                    )
                )
            )
            raise ValueError(
                "Teacher labels and applied actions differ; "
                f"maximum absolute difference is {difference}."
            )

    def split_episode_ids(self, split: str) -> np.ndarray:
        if split not in self.splits.files:
            raise KeyError(f"Unknown dataset split {split!r}.")
        return np.asarray(self.splits[split], dtype=np.int32)

    def split_sample_indices(self, split: str) -> np.ndarray:
        episode_ids = self.split_episode_ids(split)
        return np.flatnonzero(
            np.isin(self.samples["episode_id"], episode_ids)
        ).astype(np.int64)

    def close(self):
        self.samples.close()
        self.splits.close()
