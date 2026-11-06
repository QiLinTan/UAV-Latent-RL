"""Episode-separated augmented dataset for explicit temporal context."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .temporal_context import HISTORY_LENGTH, TEMPORAL_FRAME_DIM


DATASET_VERSION = "latent_reference_sequence_temporal_v1"
REQUIRED_ARRAYS = {
    "episode_id", "step_id", "planner_input", "environment_history",
    "history_padding_count", "reference_target",
}


def save_temporal_reference_dataset(
    root,
    *,
    episode_id,
    step_id,
    planner_input,
    environment_history,
    history_padding_count,
    reference_target,
    splits: dict,
    metadata: dict,
) -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    episode_id = np.asarray(episode_id, dtype=np.int32).reshape(-1)
    step_id = np.asarray(step_id, dtype=np.int32).reshape(-1)
    planner_input = np.asarray(planner_input, dtype=np.float32)
    history = np.asarray(environment_history, dtype=np.float32)
    padding = np.asarray(history_padding_count, dtype=np.int8).reshape(-1)
    target = np.asarray(reference_target, dtype=np.float32)
    count = int(episode_id.size)
    if step_id.shape != (count,) or planner_input.shape != (count, 27):
        raise ValueError("Temporal dataset needs [count] ids and [count,27] planner inputs.")
    if history.shape != (count, HISTORY_LENGTH, TEMPORAL_FRAME_DIM):
        raise ValueError("Temporal dataset history must have shape [count,8,16].")
    if padding.shape != (count,) or np.any(padding < 0) or np.any(padding >= HISTORY_LENGTH):
        raise ValueError("Invalid history padding counts.")
    if target.shape != (count, 5, 6):
        raise ValueError("reference_target must have shape [count,5,6].")
    if not all(np.all(np.isfinite(x)) for x in (planner_input, history, target)):
        raise ValueError("Temporal dataset arrays must be finite.")
    split_values = {name: np.asarray(values, dtype=np.int32).reshape(-1) for name, values in splits.items()}
    if set(split_values) != {"train", "validation", "test"}:
        raise ValueError("Temporal dataset requires train/validation/test episode splits.")
    train_mask = np.isin(episode_id, split_values["train"])
    target_mean = target[train_mask].mean(axis=0)
    target_std = np.maximum(target[train_mask].std(axis=0), 1e-4)
    result = {
        **dict(metadata),
        "dataset_version": DATASET_VERSION,
        "sample_count": count,
        "episode_count": int(np.unique(episode_id).size),
        "planner_input_dim": 27,
        "history_length": HISTORY_LENGTH,
        "temporal_frame_dim": TEMPORAL_FRAME_DIM,
        "temporal_frame_layout": "relative_goal_xyz+ranges8+normalized_linear_velocity_xyz+yaw_sin_cos",
        "history_padding": "repeat_first_available_high_level_frame_on_episode_start",
        "target_mean": target_mean.tolist(),
        "target_std": target_std.tolist(),
        "split_episode_counts": {name: int(values.size) for name, values in split_values.items()},
    }
    np.savez_compressed(
        root / "samples.npz",
        episode_id=episode_id,
        step_id=step_id,
        planner_input=planner_input,
        environment_history=history,
        history_padding_count=padding,
        reference_target=target,
    )
    np.savez_compressed(root / "splits.npz", **split_values)
    (root / "metadata.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


class TemporalReferenceTrajectoryDataset(Dataset):
    def __init__(self, root, split: str = "train"):
        self.root = Path(root)
        self.metadata = json.loads((self.root / "metadata.json").read_text(encoding="utf-8"))
        if self.metadata.get("dataset_version") != DATASET_VERSION:
            raise ValueError("Unsupported temporal reference dataset.")
        with np.load(self.root / "samples.npz", allow_pickle=False) as values:
            missing = REQUIRED_ARRAYS - set(values.files)
            if missing:
                raise ValueError(f"Dataset is missing arrays: {sorted(missing)}")
            self.episode_id = np.asarray(values["episode_id"], dtype=np.int32).copy()
            self.planner_input = np.asarray(values["planner_input"], dtype=np.float32).copy()
            self.environment_history = np.asarray(values["environment_history"], dtype=np.float32).copy()
            self.reference_target = np.asarray(values["reference_target"], dtype=np.float32).copy()
        with np.load(self.root / "splits.npz", allow_pickle=False) as values:
            self.splits = {name: np.asarray(values[name], dtype=np.int32).copy() for name in values.files}
        if split not in self.splits:
            raise KeyError(f"Unknown split {split!r}.")
        self.indices = np.flatnonzero(np.isin(self.episode_id, self.splits[split])).astype(np.int64)
        if not self.indices.size:
            raise ValueError(f"No samples in {split!r} split.")
        self.target_mean = np.asarray(self.metadata["target_mean"], dtype=np.float32)
        self.target_std = np.asarray(self.metadata["target_std"], dtype=np.float32)

    def __len__(self):
        return int(self.indices.size)

    def __getitem__(self, index):
        row = int(self.indices[int(index)])
        normalized_target = (self.reference_target[row] - self.target_mean) / self.target_std
        return (
            torch.from_numpy(self.planner_input[row]),
            torch.from_numpy(self.environment_history[row]),
            torch.from_numpy(normalized_target.astype(np.float32)),
        )

    def close(self):
        return None
