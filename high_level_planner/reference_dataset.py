"""Versioned teacher-trajectory dataset for the reference planner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


DATASET_VERSION = "latent_reference_sequence_v1"
REQUIRED_ARRAYS = {
    "episode_id",
    "step_id",
    "planner_input",
    "reference_target",
}


def _episode_splits(episode_ids, seed: int):
    unique = np.unique(np.asarray(episode_ids, dtype=np.int32))
    if unique.size < 3:
        raise ValueError("At least three successful episodes are required for leak-free splits.")
    shuffled = np.random.default_rng(int(seed)).permutation(unique)
    validation_count = max(1, int(round(0.15 * unique.size)))
    test_count = max(1, int(round(0.15 * unique.size)))
    if validation_count + test_count >= unique.size:
        validation_count = 1
        test_count = 1
    train_count = unique.size - validation_count - test_count
    return {
        "train": np.sort(shuffled[:train_count]),
        "validation": np.sort(shuffled[train_count : train_count + validation_count]),
        "test": np.sort(shuffled[train_count + validation_count :]),
    }


def save_reference_dataset(
    root,
    *,
    episode_id,
    step_id,
    planner_input,
    reference_target,
    metadata: dict,
    split_seed: int = 0,
) -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    episode_id = np.asarray(episode_id, dtype=np.int32).reshape(-1)
    step_id = np.asarray(step_id, dtype=np.int32).reshape(-1)
    planner_input = np.asarray(planner_input, dtype=np.float32)
    reference_target = np.asarray(reference_target, dtype=np.float32)
    count = int(episode_id.size)
    if step_id.shape != (count,) or planner_input.shape != (count, 27):
        raise ValueError("Dataset arrays do not satisfy the episode/step/27D contract.")
    if reference_target.shape != (count, 5, 6):
        raise ValueError("reference_target must have shape [samples, 5, 6].")
    if count == 0 or not np.all(np.isfinite(planner_input)) or not np.all(np.isfinite(reference_target)):
        raise ValueError("Reference dataset must be non-empty and finite.")

    splits = _episode_splits(episode_id, split_seed)
    train_mask = np.isin(episode_id, splits["train"])
    target_mean = reference_target[train_mask].mean(axis=0)
    target_std = reference_target[train_mask].std(axis=0)
    target_std = np.maximum(target_std, 1e-4)
    final_metadata = {
        **dict(metadata),
        "dataset_version": DATASET_VERSION,
        "sample_count": count,
        "episode_count": int(np.unique(episode_id).size),
        "planner_input_dim": 27,
        "num_future_points": 5,
        "target_layout": "relative_position_xyz_then_world_velocity_xyz",
        "future_times_seconds": [0.1, 0.2, 0.3, 0.4, 0.5],
        "target_mean": target_mean.tolist(),
        "target_std": target_std.tolist(),
        "split_episode_counts": {
            name: int(values.size) for name, values in splits.items()
        },
    }
    np.savez_compressed(
        root / "samples.npz",
        episode_id=episode_id,
        step_id=step_id,
        planner_input=planner_input,
        reference_target=reference_target,
    )
    np.savez_compressed(root / "splits.npz", **splits)
    (root / "metadata.json").write_text(
        json.dumps(final_metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return final_metadata


class ReferenceTrajectoryDataset(Dataset):
    """Load one episode-separated split and normalize its reference targets."""

    def __init__(self, root, split: str = "train"):
        self.root = Path(root)
        self.metadata = json.loads((self.root / "metadata.json").read_text(encoding="utf-8"))
        if self.metadata.get("dataset_version") != DATASET_VERSION:
            raise ValueError("Unsupported or missing reference dataset version.")
        with np.load(self.root / "samples.npz", allow_pickle=False) as samples:
            missing = REQUIRED_ARRAYS - set(samples.files)
            if missing:
                raise ValueError(f"Dataset is missing arrays: {sorted(missing)}")
            # NPZ members are compressed.  Keeping the archive lazy would
            # decompress a full member in every Dataset.__getitem__, making a
            # tiny MLP epoch take tens of seconds.  Materialize once instead.
            self.episode_id = np.asarray(samples["episode_id"], dtype=np.int32).copy()
            self.planner_input = np.asarray(samples["planner_input"], dtype=np.float32).copy()
            self.reference_target = np.asarray(samples["reference_target"], dtype=np.float32).copy()
        with np.load(self.root / "splits.npz", allow_pickle=False) as splits:
            self.splits = {
                name: np.asarray(splits[name], dtype=np.int32).copy()
                for name in splits.files
            }
        if split not in self.splits:
            raise KeyError(f"Unknown dataset split {split!r}.")
        split_ids = self.splits[split]
        self.indices = np.flatnonzero(
            np.isin(self.episode_id, split_ids)
        ).astype(np.int64)
        if self.indices.size == 0:
            raise ValueError(f"Dataset split {split!r} contains no samples.")
        self.target_mean = np.asarray(self.metadata["target_mean"], dtype=np.float32)
        self.target_std = np.asarray(self.metadata["target_std"], dtype=np.float32)
        if self.target_mean.shape != (5, 6) or self.target_std.shape != (5, 6):
            raise ValueError("Dataset normalization must have shape [5, 6].")

    def __len__(self):
        return int(self.indices.size)

    def __getitem__(self, index):
        row = int(self.indices[int(index)])
        planner_input = self.planner_input[row]
        target = self.reference_target[row]
        normalized_target = (target - self.target_mean) / self.target_std
        return torch.from_numpy(planner_input), torch.from_numpy(normalized_target)

    def close(self):
        # Kept for symmetry with larger memory-mapped dataset implementations.
        return None
