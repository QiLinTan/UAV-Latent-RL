"""Static figures for the frozen P0/B diagnostics and A3 Full witnesses."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

ROOT = Path("analysis_outputs/full_preimprovement_diagnostics_20261010")
FIGURES = ROOT / "figures"


def load(tag, seed, map_seed, mode="full"):
    stem = ROOT / tag / f"seed{seed}_map{map_seed}_{mode}" / "traces" / f"trace_seed{map_seed}"
    return np.load(stem.with_suffix(".npz")), json.loads(stem.with_suffix(".json").read_text())


def p0_figure(row):
    seed, map_seed, start = row["training_seed"], row["map_seed"], row["first_three_step_command_diff_step"]
    k1, meta = load("p0_baseline", seed, map_seed, "k1")
    full, _ = load("p0_baseline", seed, map_seed)
    intervene, _ = load(f"p0_intervention_step{start:03d}_len08", seed, map_seed)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for tree in meta["tree_specs"]:
        axes[0].add_patch(Circle(tree["xy"], tree["radius"] + 0.06,
                                 color="gray", alpha=0.16, linewidth=0))
    for data, label, color in ((k1, "K1", "tab:blue"), (full, "Full closed", "tab:red"),
                               (intervene, "Full 8-step K1", "tab:green")):
        p = data["upper_position"]
        axes[0].plot(p[:, 0], p[:, 1], color=color, label=label)
    for data, color in ((k1, "tab:blue"), (full, "tab:red")):
        pos = np.asarray(meta["initial_position"]) if start == 0 else data["upper_position"][start - 1]
        axes[0].scatter(pos[0], pos[1], color=color, s=28)
    axes[0].scatter(*np.asarray(meta["target_position"])[:2], marker="*", color="black")
    axes[0].set(xlabel="x (m)", ylabel="y (m)", title="Route and first sustained branch")
    axes[0].axis("equal")
    axes[0].legend(fontsize=8)
    t0 = max(0, start - 5)
    t1 = min(len(full["upper_command_velocity"]), start + 18)
    for data, prefix, style in ((k1, "K1", "-"), (full, "Full", "--")):
        t = np.arange(t0, t1) / 15.0
        for dim, color in ((0, "tab:blue"), (1, "tab:orange")):
            axes[1].plot(t, data["upper_command_velocity"][t0:t1, dim], style,
                         color=color, label=f"{prefix} v{'x' if dim == 0 else 'y'}")
    axes[1].axvline(start / 15.0, color="black", linestyle=":")
    axes[1].set(xlabel="time (s)", ylabel="upper velocity command (m/s)", title="Command direction")
    axes[1].legend(fontsize=7, ncol=2)
    for data, label, color in ((k1, "K1", "tab:blue"), (full, "Full", "tab:red")):
        horizontal_rays = data["upper_pre_observation"][start, 15 + 32:15 + 64]
        axes[2].plot(np.arange(32) * 11.25, horizontal_rays, color=color, label=label)
    axes[2].set(xlabel="world azimuth (deg)", ylabel="horizontal ray fraction",
                title="Current horizontal rays at branch", xlim=(0, 360), ylim=(0, 1.02))
    axes[2].legend()
    fig.suptitle(f"P0 seed {seed} / map {map_seed}: step {start}")
    fig.savefig(FIGURES / f"p0_seed{seed}_map{map_seed}.png", dpi=150)
    plt.close(fig)


def b_figure(row):
    map_seed = row["map_seed"]
    data12, meta = load("b_12s", 45, map_seed)
    data16, _ = load("b_16s", 45, map_seed)
    goal = np.asarray(meta["target_position"])
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    for data, label, color in ((data12, "12 s formal", "tab:red"), (data16, "16 s diagnostic", "tab:blue")):
        t = data["lower_time_seconds"]
        delta = goal - data["lower_position"]
        distance = np.linalg.norm(delta, axis=1)
        v = data["lower_velocity"]
        radial = np.sum(v * delta, axis=1) / np.maximum(distance, 1e-9)
        tangential = np.sqrt(np.maximum(0, np.sum(v * v, axis=1) - radial * radial))
        axes[0].plot(t, distance, color=color, label=label)
        axes[1].plot(t, delta[:, 2], color=color, label=label)
        axes[2].plot(t, radial, color=color, label=f"{label} radial")
        axes[2].plot(t, tangential, color=color, linestyle="--", alpha=0.75,
                     label=f"{label} tangential")
    axes[0].axhline(0.2, color="black", linestyle=":")
    for ax in axes:
        ax.axvline(12.0, color="gray", linestyle=":")
        ax.set_xlim(8.0, 12.3)
        ax.legend(fontsize=8)
    axes[0].set(ylabel="3D goal distance (m)")
    axes[1].set(ylabel="height error (m)")
    axes[2].set(xlabel="time (s)", ylabel="speed (m/s)")
    fig.suptitle(f"B seed 45 / Stage 0 map {map_seed}")
    fig.savefig(FIGURES / f"b_seed45_map{map_seed}.png", dpi=150)
    plt.close(fig)


def a3_figure(row):
    raw = np.load(row["full_trace"])
    p = raw["lower_position"]
    t = raw["lower_time_seconds"]
    clearance = raw["lower_clearance"]
    meta = json.loads(Path(row["full_trace"]).with_suffix(".json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for tree in meta["tree_specs"]:
        axes[0].add_patch(Circle(tree["xy"], tree["radius"] + 0.06,
                                 color="gray", alpha=0.16, linewidth=0))
    axes[0].plot(p[:, 0], p[:, 1], color="tab:blue", label="Full actual")
    path = np.asarray(row["path_xy"])
    axes[0].plot(path[:, 0], path[:, 1], "o--", color="tab:orange", label="truth waypoints")
    axes[0].scatter(*meta["initial_position"][:2], color="black", marker="s")
    axes[0].scatter(*meta["target_position"][:2], color="black", marker="*")
    axes[0].axis("equal")
    axes[0].set(xlabel="x (m)", ylabel="y (m)", title="Truth-assisted Full route")
    axes[0].legend(fontsize=8)
    axes[1].plot(t, clearance, color="tab:green")
    axes[1].axhline(0, color="black", linestyle=":")
    axes[1].set(xlabel="time (s)", ylabel="nearest tree clearance (m)", title="120 Hz actual clearance")
    fig.suptitle(f"A3 seed {row['training_seed']} / map {row['map_seed']}: success at {row['time_s']:.1f} s")
    fig.savefig(FIGURES / f"a3_seed{row['training_seed']}_map{row['map_seed']}.png", dpi=150)
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary = json.loads((ROOT / "analysis_summary.json").read_text())
    for row in summary["p0"]:
        p0_figure(row)
    for row in summary["b"]:
        b_figure(row)
    for row in json.loads((ROOT / "a3/summary.json").read_text()):
        a3_figure(row)
    print("wrote", len(list(FIGURES.glob("*.png"))), "figures to", FIGURES)


if __name__ == "__main__":
    main()
