from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return scalar if math.isfinite(scalar) else None
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


def _normalize_obstacles(obstacle_specs):
    normalized = []
    for raw_spec in obstacle_specs or ():
        kind = str(raw_spec.get("kind", "cylinder")).lower()
        if "position" in raw_spec:
            position = np.asarray(raw_spec["position"], dtype=np.float32).reshape(3)
        elif "xy" in raw_spec:
            height = float(raw_spec["height"])
            position = np.array((*np.asarray(raw_spec["xy"], dtype=np.float32).reshape(2), height / 2.0))
        else:
            raise ValueError("Trajectory obstacle requires position or legacy xy metadata.")
        normalized.append(
            {
                "kind": kind,
                "name": str(raw_spec.get("name", kind)),
                "position": position,
                "radius": float(raw_spec.get("radius", np.nan)),
                "height": float(raw_spec.get("height", np.nan)),
                "half_extents": np.asarray(
                    raw_spec.get("half_extents", (np.nan, np.nan, np.nan)), dtype=np.float32
                ).reshape(3),
            }
        )
    return normalized


def _tree_arrays(obstacles):
    trees = [
        spec for spec in obstacles
        if spec["kind"] == "cylinder" and np.isfinite(spec["radius"]) and np.isfinite(spec["height"])
    ]
    if not trees:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    return (
        np.asarray([spec["position"][:2] for spec in trees], dtype=np.float32).reshape(-1, 2),
        np.asarray([spec["radius"] for spec in trees], dtype=np.float32),
        np.asarray([spec["height"] for spec in trees], dtype=np.float32),
    )


def export_upper_trajectory(
    output_dir,
    *,
    seed: int,
    training_step: int,
    trace: dict,
    tree_specs,
    start_position,
    target_position,
    episode_summary: dict,
    evaluation_summary: dict,
):
    """Write the latest recurrent-upper evaluation trace and two trajectory plots."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prefix = f"seed_{int(seed)}"
    trace_path = output / f"{prefix}_trace.npz"
    xy_path = output / f"{prefix}_xy.png"
    xyz_path = output / f"{prefix}_xyz.png"
    summary_path = output / "summary.json"

    arrays = {key: np.asarray(value) for key, value in trace.items()}
    obstacle_specs = _normalize_obstacles(tree_specs)
    tree_xy, tree_radius, tree_height = _tree_arrays(obstacle_specs)
    start = np.asarray(start_position, dtype=np.float32).reshape(3)
    target = np.asarray(target_position, dtype=np.float32).reshape(3)
    np.savez_compressed(
        trace_path,
        **arrays,
        tree_xy=tree_xy,
        tree_radius=tree_radius,
        tree_height=tree_height,
        obstacle_kind=np.asarray([spec["kind"] for spec in obstacle_specs], dtype="U16"),
        obstacle_position=np.asarray(
            [spec["position"] for spec in obstacle_specs], dtype=np.float32
        ).reshape(-1, 3),
        obstacle_half_extents=np.asarray(
            [spec["half_extents"] for spec in obstacle_specs], dtype=np.float32
        ).reshape(-1, 3),
        start_position=start,
        target_position=target,
        seed=np.asarray(int(seed), dtype=np.int64),
        training_step=np.asarray(int(training_step), dtype=np.int64),
    )

    matplotlib_config = Path(tempfile.gettempdir()) / "uav_latent_rl_matplotlib"
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", os.fspath(matplotlib_config))

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    actual = np.asarray(arrays["actual_position"], dtype=np.float32).reshape(-1, 3)
    reference = np.asarray(arrays["reference_position"], dtype=np.float32).reshape(-1, 3)
    endpoint = np.asarray(arrays["packet_endpoint"], dtype=np.float32).reshape(-1, 3)
    title_suffix = (
        f"step={int(training_step)}, seed={int(seed)}, "
        f"success={bool(episode_summary.get('success', False))}, "
        f"reason={episode_summary.get('done_reason', 'unknown')}"
    )

    figure, axis = plt.subplots(figsize=(8, 7))
    axis.plot(actual[:, 0], actual[:, 1], color="tab:blue", linewidth=2.0, label="UAV actual")
    axis.plot(
        reference[:, 0], reference[:, 1],
        color="tab:orange", linestyle="--", linewidth=1.5, label="Packet current ref",
    )
    axis.plot(
        endpoint[:, 0], endpoint[:, 1],
        color="tab:green", linestyle=":", linewidth=1.2, label="Packet horizon endpoint",
    )
    from matplotlib.patches import Rectangle
    for index, obstacle in enumerate(obstacle_specs):
        position = obstacle["position"]
        if obstacle["kind"] == "cylinder" and np.isfinite(obstacle["radius"]):
            axis.add_patch(
                Circle(
                    (float(position[0]), float(position[1])),
                    float(obstacle["radius"]),
                    color="saddlebrown",
                    alpha=0.35,
                    label="Cylinder / tree" if index == 0 else None,
                )
            )
        elif obstacle["kind"] == "box" and np.all(np.isfinite(obstacle["half_extents"][:2])):
            half = obstacle["half_extents"]
            axis.add_patch(
                Rectangle(
                    (float(position[0] - half[0]), float(position[1] - half[1])),
                    2.0 * float(half[0]),
                    2.0 * float(half[1]),
                    color="dimgray",
                    alpha=0.35,
                    label="Box obstacle" if index == 0 else None,
                )
            )
    axis.scatter(start[0], start[1], marker="o", s=80, color="black", label="Start", zorder=5)
    axis.scatter(target[0], target[1], marker="*", s=180, color="red", label="Target", zorder=5)
    axis.set_xlabel("world x [m]")
    axis.set_ylabel("world y [m]")
    axis.set_title(f"Upper DSLPID trajectory (XY)\n{title_suffix}")
    axis.axis("equal")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(xy_path, dpi=180)
    plt.close(figure)

    figure = plt.figure(figsize=(9, 7))
    axis3d = figure.add_subplot(111, projection="3d")
    axis3d.plot(actual[:, 0], actual[:, 1], actual[:, 2], color="tab:blue", linewidth=2.0, label="UAV actual")
    axis3d.plot(
        reference[:, 0], reference[:, 1], reference[:, 2],
        color="tab:orange", linestyle="--", linewidth=1.5, label="Packet current ref",
    )
    axis3d.plot(
        endpoint[:, 0], endpoint[:, 1], endpoint[:, 2],
        color="tab:green", linestyle=":", linewidth=1.2, label="Packet horizon endpoint",
    )
    for index, obstacle in enumerate(obstacle_specs):
        position = obstacle["position"]
        if obstacle["kind"] == "cylinder" and np.isfinite(obstacle["radius"]) and np.isfinite(obstacle["height"]):
            axis3d.plot(
                [float(position[0]), float(position[0])],
                [float(position[1]), float(position[1])],
                [float(position[2] - obstacle["height"] / 2.0), float(position[2] + obstacle["height"] / 2.0)],
                color="saddlebrown",
                linewidth=max(2.0, 16.0 * float(obstacle["radius"])),
                alpha=0.30,
                label="Cylinder / tree" if index == 0 else None,
            )
        elif obstacle["kind"] == "box" and np.all(np.isfinite(obstacle["half_extents"])):
            half = obstacle["half_extents"]
            axis3d.plot(
                [float(position[0]), float(position[0])],
                [float(position[1]), float(position[1])],
                [float(position[2] - half[2]), float(position[2] + half[2])],
                color="dimgray",
                linewidth=max(2.0, 12.0 * float(max(half[0], half[1]))),
                alpha=0.35,
                label="Box obstacle" if index == 0 else None,
            )
    axis3d.scatter(start[0], start[1], start[2], marker="o", s=60, color="black", label="Start")
    axis3d.scatter(target[0], target[1], target[2], marker="*", s=150, color="red", label="Target")
    axis3d.set_xlabel("world x [m]")
    axis3d.set_ylabel("world y [m]")
    axis3d.set_zlabel("world z [m]")
    axis3d.set_title(f"Upper DSLPID trajectory (3D)\n{title_suffix}")
    axis3d.legend(loc="best")
    figure.tight_layout()
    figure.savefig(xyz_path, dpi=180)
    plt.close(figure)

    summary = {
        "schema_version": "upper_dslpid_recurrent_trajectory_v1",
        "training_step": int(training_step),
        "seed": int(seed),
        "trace_samples": int(actual.shape[0]),
        "tree_count": int(tree_xy.shape[0]),
        "obstacle_count": int(len(obstacle_specs)),
        "files": {
            "trace": trace_path.name,
            "xy_plot": xy_path.name,
            "xyz_plot": xyz_path.name,
        },
        "episode": episode_summary,
        "evaluation": evaluation_summary,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(summary), handle, ensure_ascii=False, indent=2, allow_nan=False)

    return {
        "trace": os.fspath(trace_path),
        "xy": os.fspath(xy_path),
        "xyz": os.fspath(xyz_path),
        "summary": os.fspath(summary_path),
    }
