from __future__ import annotations

import numpy as np


class DepthSectorEncoder:
    """Compress a depth image into conservative grid-sector clearances."""

    def __init__(
        self,
        rows: int = 4,
        cols: int = 4,
        max_depth: float = 12.0,
        percentile: float = 10.0,
    ):
        if rows <= 0 or cols <= 0:
            raise ValueError("Depth sector rows and columns must be positive.")
        if max_depth <= 0.0:
            raise ValueError("max_depth must be positive.")
        if not 0.0 <= percentile <= 100.0:
            raise ValueError("percentile must be within [0, 100].")
        self.rows = int(rows)
        self.cols = int(cols)
        self.max_depth = float(max_depth)
        self.percentile = float(percentile)

    @property
    def output_dim(self) -> int:
        return self.rows * self.cols

    def encode(self, depth_images: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth_images, dtype=np.float32)
        if depth.ndim != 3:
            raise ValueError(
                f"Expected depth images with shape [num_envs, height, width], got {depth.shape}."
            )

        depth = np.nan_to_num(
            depth,
            nan=self.max_depth,
            posinf=self.max_depth,
            neginf=0.0,
        )
        depth = np.clip(depth, 0.0, self.max_depth)
        row_chunks = np.array_split(depth, self.rows, axis=1)
        features = []
        for row_chunk in row_chunks:
            for col_chunk in np.array_split(row_chunk, self.cols, axis=2):
                features.append(
                    np.percentile(
                        col_chunk,
                        self.percentile,
                        axis=(1, 2),
                    )
                )
        encoded = np.stack(features, axis=1) / self.max_depth
        return encoded.astype(np.float32)


def build_compact_observation(
    state_observation: np.ndarray,
    depth_images: np.ndarray,
    encoder: DepthSectorEncoder,
) -> np.ndarray:
    state = np.asarray(state_observation, dtype=np.float32)
    if state.ndim != 2:
        raise ValueError(f"Expected state observations [num_envs, state_dim], got {state.shape}.")
    depth_features = encoder.encode(depth_images)
    if state.shape[0] != depth_features.shape[0]:
        raise ValueError("State and depth observations have different environment counts.")
    observation = np.concatenate([state, depth_features], axis=1)
    if not np.isfinite(observation).all():
        raise FloatingPointError("AvoidBench compact observation contains NaN or Inf.")
    return observation.astype(np.float32)
