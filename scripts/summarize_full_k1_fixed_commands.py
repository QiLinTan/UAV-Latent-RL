"""Summarize A2 fixed K1 command tapes against closed-loop K1 and Full."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SOURCE = Path("analysis_outputs/full_k1_paired_trajectories_20261010/all")
ROOT = Path("analysis_outputs/full_k1_fixed_command_a2_20261010")


def trace(path):
    return np.load(path)


def first_sustained(time, values, threshold, steps=2):
    above = values > threshold
    for index in range(len(values) - steps + 1):
        if np.all(above[index:index + steps]):
            return float(time[index])
    return None


def trajectory(data, start):
    return np.r_[0, data["upper_time_seconds"]], np.vstack([start, data["upper_position"]])


def vector_interp(t, values, query):
    return np.stack([np.interp(query, t, values[:, index]) for index in range(values.shape[1])], axis=1)


def case_summary(seed, map_seed, original):
    case = ROOT / f"seed{seed}_map{map_seed}"
    fixed_path = case / "full_tape" / "full" / "traces" / f"trace_seed{map_seed}.npz"
    if not fixed_path.exists():
        return None
    kpath = SOURCE / f"seed{seed}_k1" / "traces" / f"trace_seed{map_seed}.npz"
    cpath = SOURCE / f"seed{seed}_full" / "traces" / f"trace_seed{map_seed}.npz"
    k = trace(kpath)
    c = trace(cpath)
    f = trace(fixed_path)
    fixed_meta = json.loads(fixed_path.with_suffix(".json").read_text())
    kmeta = json.loads(kpath.with_suffix(".json").read_text())
    closed_meta = json.loads(cpath.with_suffix(".json").read_text())
    assert fixed_meta["initial_position"] == kmeta["initial_position"] == closed_meta["initial_position"]
    assert fixed_meta["initial_velocity"] == kmeta["initial_velocity"] == closed_meta["initial_velocity"]
    assert fixed_meta["target_position"] == kmeta["target_position"] == closed_meta["target_position"]
    assert fixed_meta["tree_specs"] == kmeta["tree_specs"] == closed_meta["tree_specs"]
    assert np.array_equal(k["upper_command_velocity"][:len(f["upper_command_velocity"])], f["upper_command_velocity"])
    assert np.array_equal(k["upper_action"][:len(f["upper_action"])], f["upper_action"])
    start = np.asarray(kmeta["initial_position"])
    kt, kp = trajectory(k, start)
    ct, cp = trajectory(c, start)
    ft, fp = trajectory(f, start)
    terminal_t = min(kt[-1], ct[-1], ft[-1])
    common_t = np.arange(0, terminal_t + 1e-8, 1 / 15)
    k_common = vector_interp(kt, kp, common_t)
    c_common = vector_interp(ct, cp, common_t)
    f_common = vector_interp(ft, fp, common_t)
    fixed_gap = np.linalg.norm(k_common - f_common, axis=1)
    closed_gap = np.linalg.norm(k_common - c_common, axis=1)
    decision_count = min(len(k["upper_command_velocity"]), len(c["upper_command_velocity"]))
    command_diff = np.linalg.norm(
        k["upper_command_velocity"][:decision_count] - c["upper_command_velocity"][:decision_count], axis=1
    )
    commands_at = c["upper_time_seconds"][:decision_count]
    fixed_tracking = np.linalg.norm(f["lower_position"] - f["lower_reference_position"], axis=1)
    closed_tracking = np.linalg.norm(c["lower_position"] - c["lower_reference_position"], axis=1)
    row = {
        "training_seed": seed,
        "map_seed": map_seed,
        "closed_full_done_reason": original["full_done_reason"],
        "fixed_full_done_reason": fixed_meta["done_reason"],
        "fixed_full_success": fixed_meta["success"],
        "source_k1_done_reason": kmeta["done_reason"],
        "closed_full_end_time_s": float(ct[-1]),
        "fixed_full_end_time_s": float(ft[-1]),
        "common_time_s": float(terminal_t),
        "first_sustained_command_difference_gt_0_2_mps_s": first_sustained(commands_at, command_diff, 0.2),
        "first_sustained_position_gap_gt_0_1_m_fixed_s": first_sustained(common_t, fixed_gap, 0.1, steps=4),
        "first_sustained_position_gap_gt_0_1_m_closed_s": first_sustained(common_t, closed_gap, 0.1, steps=4),
        "fixed_vs_k1_max_gap_common_m": float(np.max(fixed_gap)),
        "closed_vs_k1_max_gap_common_m": float(np.max(closed_gap)),
        "fixed_full_min_clearance_m": float(np.min(f["lower_clearance"])),
        "closed_full_min_clearance_m": float(np.min(c["lower_clearance"])),
        "k1_min_clearance_m": float(np.min(k["upper_clearance"])),
        "fixed_full_tracking_position_rmse_m": float(np.sqrt(np.mean(fixed_tracking**2))),
        "closed_full_tracking_position_rmse_m": float(np.sqrt(np.mean(closed_tracking**2))),
        "fixed_full_final_goal_distance_m": float(fixed_meta["final_goal_distance"]),
        "short_windows": {},
    }
    for steps in (15, 45):
        base = case / f"steps{steps:03d}"
        kshort_path = base / "k1" / "traces" / f"trace_seed{map_seed}.npz"
        fshort_path = base / "full" / "traces" / f"trace_seed{map_seed}.npz"
        if not (kshort_path.exists() and fshort_path.exists()):
            continue
        ks = trace(kshort_path)
        fs = trace(fshort_path)
        kshort_meta = json.loads(kshort_path.with_suffix(".json").read_text())
        fshort_meta = json.loads(fshort_path.with_suffix(".json").read_text())
        assert kshort_meta["initial_position"] == fshort_meta["initial_position"] == kmeta["initial_position"]
        assert kshort_meta["initial_velocity"] == fshort_meta["initial_velocity"] == kmeta["initial_velocity"]
        assert np.array_equal(k["upper_position"][:steps], ks["upper_position"])
        assert np.array_equal(ks["upper_command_velocity"], fs["upper_command_velocity"])
        row["short_windows"][str(steps)] = {
            "duration_s": steps / 15.0,
            "initial_position_gap_m": 0.0,
            "initial_velocity_gap_mps": 0.0,
            "final_position_gap_m": float(np.linalg.norm(ks["upper_position"][-1] - fs["upper_position"][-1])),
            "k1_min_clearance_m": float(np.min(ks["upper_clearance"])),
            "full_min_clearance_m": float(np.min(fs["lower_clearance"])),
            "full_position_tracking_rmse_m": float(np.sqrt(np.mean(np.sum((fs["lower_position"] - fs["lower_reference_position"])**2, axis=1)))),
        }
    make_plot(seed, map_seed, k, c, f, kp, cp, fp, kt, ct, ft, common_t, fixed_gap, closed_gap, command_diff, row)
    return row


def make_plot(seed, map_seed, k, c, f, kp, cp, fp, kt, ct, ft, common_t, fixed_gap, closed_gap, command_diff, row):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(kp[:, 0], kp[:, 1], label="K1 closed / source tape")
    ax.plot(cp[:, 0], cp[:, 1], label="Full closed")
    ax.plot(fp[:, 0], fp[:, 1], label="Full fixed K1 commands")
    ax.set(xlabel="x (m)", ylabel="y (m)", title="Same map, three executions")
    ax.axis("equal")
    ax.legend()
    ax = axes[0, 1]
    ax.plot(k["upper_time_seconds"], k["upper_clearance"], label="K1 closed")
    ax.plot(c["lower_time_seconds"], c["lower_clearance"], label="Full closed")
    ax.plot(f["lower_time_seconds"], f["lower_clearance"], label="Full fixed")
    ax.axhline(0, linestyle="--", color="red")
    ax.set(xlabel="time (s)", ylabel="signed tree clearance (m)", ylim=(-0.2, 1.0), title="Tree clearance")
    ax.legend()
    ax = axes[1, 0]
    ax.plot(common_t, fixed_gap, label="K1 vs Full fixed")
    ax.plot(common_t, closed_gap, label="K1 vs Full closed")
    ax.axhline(0.1, linestyle=":", color="gray")
    ax.set(xlabel="time (s)", ylabel="position gap (m)", title="Time-aligned trajectory gap")
    ax.legend()
    ax = axes[1, 1]
    ax.plot(c["upper_time_seconds"][:len(command_diff)], command_diff, label="K1 vs Full closed command")
    ax.axhline(0.2, linestyle=":", color="gray")
    ax.set(xlabel="time (s)", ylabel="3D command difference (m/s)", title="Actor feedback changes command")
    ax.legend()
    fig.suptitle(f"seed {seed} / map {map_seed}: Full closed {row['closed_full_done_reason']}, fixed {row['fixed_full_done_reason']}")
    directory = ROOT / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"seed{seed}_map{map_seed}_fixed_vs_closed.png", dpi=135)
    plt.close(fig)


def main():
    baseline = json.loads((SOURCE / "paired_summary.json").read_text())
    rows = [case_summary(row["training_seed"], row["map_seed"], row)
            for row in baseline["pairs"] if row["full_done_reason"] == "collision"]
    assert len(rows) == 12 and all(row is not None for row in rows)
    summary = {
        "schema_version": 1,
        "case_count": len(rows),
        "fixed_full_success_count": sum(row["fixed_full_success"] for row in rows),
        "fixed_full_collision_count": sum(row["fixed_full_done_reason"] == "collision" for row in rows),
        "fixed_full_timeout_count": sum(row["fixed_full_done_reason"] == "timeout" for row in rows),
        "rows": rows,
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("wrote", ROOT / "summary.json")


if __name__ == "__main__":
    main()
