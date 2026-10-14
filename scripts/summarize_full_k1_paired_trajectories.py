"""Summarize and plot frozen K1/Full paired Stage-3 diagnostic traces."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("analysis_outputs/full_k1_paired_trajectories_20261010/all")


def load(seed, mode, map_seed):
    base = ROOT / f"seed{seed}_{mode}" / "traces" / f"trace_seed{map_seed}"
    arrays = np.load(base.with_suffix(".npz"))
    meta = json.loads(base.with_suffix(".json").read_text())
    return arrays, meta


def trajectory(arrays, meta):
    return (
        np.r_[0.0, arrays["upper_time_seconds"]],
        np.vstack([meta["initial_position"], arrays["upper_position"]]),
    )


def interpolate_vector(t, values, query):
    return np.stack([np.interp(query, t, values[:, axis]) for axis in range(values.shape[1])], axis=1)


def first_sustained(time, values, threshold, steps=4):
    mask = values > threshold
    for index in range(len(mask) - steps + 1):
        if mask[index:index + steps].all():
            return float(time[index])
    return None


def lateral_flips(command, direction, *, min_magnitude=0.12):
    lateral_axis = np.array([-direction[1], direction[0]])
    lateral = np.asarray(command)[:, :2] @ lateral_axis
    active = np.sign(lateral[np.abs(lateral) >= min_magnitude])
    return int(np.sum(active[1:] != active[:-1])) if len(active) > 1 else 0


def summarize(seed, map_seed, *, plot=True):
    k, km = load(seed, "k1", map_seed)
    f, fm = load(seed, "full", map_seed)
    kt, kp = trajectory(k, km)
    ft, fp = trajectory(f, fm)
    query = np.arange(0, min(kt[-1], ft[-1]) + 1e-9, 1 / 15)
    gap = np.linalg.norm(interpolate_vector(kt, kp, query) - interpolate_vector(ft, fp, query), axis=1)
    divergence = {str(threshold): first_sustained(query, gap, threshold) for threshold in (0.05, 0.1, 0.15, 0.25, 0.35)}
    collision_index = np.flatnonzero(f["lower_collision"]) if "lower_collision" in f else np.array([])
    collision_time = float(f["lower_time_seconds"][collision_index[0]]) if len(collision_index) else None
    event_time = collision_time if collision_time is not None else float(f["lower_time_seconds"][-1])
    k1_position_at_event = interpolate_vector(kt, kp, np.array([event_time]))[0]
    full_position_at_event = f["lower_position"][collision_index[0]] if len(collision_index) else f["lower_position"][-1]
    k1_clearance_at_event = float(np.interp(event_time, k["upper_time_seconds"], k["upper_clearance"]))
    low_t = f["lower_time_seconds"]
    low_clearance = f["lower_clearance"]
    danger_index = np.flatnonzero(low_clearance < 0.25)
    danger_time = float(low_t[danger_index[0]]) if len(danger_index) else None
    track_position = np.linalg.norm(f["lower_position"] - f["lower_reference_position"], axis=1)
    track_velocity = np.linalg.norm(f["lower_velocity"] - f["lower_reference_velocity"], axis=1)
    command_speed = np.linalg.norm(f["lower_reference_velocity"], axis=1)
    actual_speed = np.linalg.norm(f["lower_velocity"], axis=1)
    valid_angle = (command_speed > 0.1) & (actual_speed > 0.1)
    cos_angle = np.sum(f["lower_velocity"] * f["lower_reference_velocity"], axis=1) / np.maximum(command_speed * actual_speed, 1e-9)
    angle = np.full(len(low_t), np.nan)
    angle[valid_angle] = np.degrees(np.arccos(np.clip(cos_angle[valid_angle], -1, 1)))
    motor_saturated = np.any(np.abs(f["lower_motor_action"]) > 0.99, axis=1)
    packet_boundary = np.diff(f["lower_packet_version"]) != 0
    packet_velocity_jump = np.linalg.norm(np.diff(f["lower_reference_velocity"], axis=0), axis=1)
    packet_position_jump = np.linalg.norm(np.diff(f["lower_reference_position"], axis=0), axis=1)
    rpm_difference = np.linalg.norm(f["lower_commanded_rpm_clipped"] - f["lower_executed_rpm"], axis=1)
    pre_window = (low_t >= (collision_time if collision_time is not None else low_t[-1]) - 1.0)
    inward = None
    if collision_time is not None and fm["tree_specs"]:
        event_position = np.asarray(full_position_at_event)
        tree = min(fm["tree_specs"], key=lambda spec: np.linalg.norm(event_position[:2] - np.asarray(spec["xy"])) - spec["radius"])
        center = np.asarray(tree["xy"], dtype=float)
        terminal_window = low_t >= collision_time - 0.25
        relative = f["lower_position"][terminal_window, :2] - center
        outward = relative / np.maximum(np.linalg.norm(relative, axis=1, keepdims=True), 1e-9)
        inward = {
            "tree_xy": center.tolist(),
            "actual_mean_mps": float(np.mean(-np.sum(f["lower_velocity"][terminal_window, :2] * outward, axis=1))),
            "reference_mean_mps": float(np.mean(-np.sum(f["lower_reference_velocity"][terminal_window, :2] * outward, axis=1))),
            "command_mean_mps": float(np.mean(-np.sum(f["lower_command_velocity"][terminal_window, :2] * outward, axis=1))),
            "actual_at_collision_mps": float(-np.dot(f["lower_velocity"][collision_index[0], :2], outward[-1])),
            "reference_at_collision_mps": float(-np.dot(f["lower_reference_velocity"][collision_index[0], :2], outward[-1])),
        }
    direction = np.asarray(fm["target_position"])[:2] - np.asarray(fm["initial_position"])[:2]
    direction /= np.linalg.norm(direction)
    full_cmd = f["upper_command_velocity"]
    k1_cmd = k["upper_command_velocity"]
    full_upper_t = f["upper_time_seconds"]
    k1_upper_t = k["upper_time_seconds"]
    window_end = collision_time if collision_time is not None else float(low_t[-1])
    pre_full = (full_upper_t >= window_end - 1) & (full_upper_t <= window_end)
    pre_k1 = (k1_upper_t >= window_end - 1) & (k1_upper_t <= window_end)
    row = {
        "training_seed": seed,
        "map_seed": map_seed,
        "k1_done_reason": km["done_reason"],
        "full_done_reason": fm["done_reason"],
        "k1_end_time_s": float(kt[-1]),
        "full_end_time_s": float(low_t[-1]),
        "collision_time_s": collision_time,
        "first_clearance_below_0_25_s": danger_time,
        "divergence_time_s_by_gap_m": divergence,
        "divergence_to_collision_s": None if collision_time is None or divergence["0.25"] is None else collision_time - divergence["0.25"],
        "position_gap_at_event_m": float(np.linalg.norm(k1_position_at_event - full_position_at_event)),
        "inward_velocity_nearest_tree_last_0_25_s": inward,
        "k1_clearance_at_event_m": k1_clearance_at_event,
        "full_min_clearance_m": float(np.min(low_clearance)),
        "k1_min_clearance_m": float(np.min(k["upper_clearance"])),
        "full_tracking_position_rmse_m": float(np.sqrt(np.mean(track_position**2))),
        "full_tracking_velocity_rmse_mps": float(np.sqrt(np.mean(track_velocity**2))),
        "full_pre_event_position_p95_m": float(np.percentile(track_position[pre_window], 95)),
        "full_pre_event_velocity_p95_mps": float(np.percentile(track_velocity[pre_window], 95)),
        "full_pre_event_velocity_angle_median_deg": float(np.nanmedian(angle[pre_window])) if np.any(np.isfinite(angle[pre_window])) else None,
        "full_motor_saturation_fraction": float(np.mean(motor_saturated)),
        "full_max_packet_boundary_velocity_jump_mps": float(np.max(packet_velocity_jump[packet_boundary])) if np.any(packet_boundary) else None,
        "full_max_packet_boundary_position_jump_m": float(np.max(packet_position_jump[packet_boundary])) if np.any(packet_boundary) else None,
        "full_max_clipped_command_to_executed_rpm_norm": float(np.max(rpm_difference)),
        "full_pre_event_motor_saturation_fraction": float(np.mean(motor_saturated[pre_window])),
        "full_lateral_command_flips": lateral_flips(full_cmd, direction),
        "k1_lateral_command_flips": lateral_flips(k1_cmd, direction),
        "full_pre_event_lateral_command_flips": lateral_flips(full_cmd[pre_full], direction),
        "k1_pre_event_lateral_command_flips": lateral_flips(k1_cmd[pre_k1], direction),
        "full_max_abs_roll_pitch_rad": float(np.max(np.abs(f["lower_rpy"][:, :2]))),
        "full_max_angular_velocity_radps": float(np.max(np.linalg.norm(f["lower_angular_velocity"], axis=1))),
        "position_gap_at_last_shared_time_m": float(gap[-1]),
    }
    if plot:
        make_plot(seed, map_seed, k, km, f, fm, kt, kp, ft, fp, query, gap, track_position, track_velocity, motor_saturated, row)
    return row


def make_plot(seed, map_seed, k, km, f, fm, kt, kp, ft, fp, query, gap, track_p, track_v, motor_sat, row):
    fig = plt.figure(figsize=(17, 13), constrained_layout=True)
    axes = [fig.add_subplot(3, 2, i + 1, projection="3d" if i == 1 else None) for i in range(6)]
    ax = axes[0]
    for tree in fm["tree_specs"]:
        ax.add_patch(plt.Circle(tree["xy"], tree["radius"], color="gray", alpha=0.12))
    ax.plot(kp[:, 0], kp[:, 1], label="K1", color="tab:blue")
    ax.plot(fp[:, 0], fp[:, 1], label="Full", color="tab:orange")
    ax.scatter([kp[0, 0], fm["target_position"][0]], [kp[0, 1], fm["target_position"][1]], c=["black", "green"], s=20)
    ax.set(xlabel="x (m)", ylabel="y (m)", title="Top view; circles are trees")
    ax.axis("equal")
    ax.legend()
    ax = axes[1]
    ax.plot(kp[:, 0], kp[:, 1], kp[:, 2], label="K1")
    ax.plot(fp[:, 0], fp[:, 1], fp[:, 2], label="Full")
    ax.set(xlabel="x", ylabel="y", zlabel="z", title="3D trajectory")
    ax.legend()
    ax = axes[2]
    ax.plot(query, gap, color="black", label="position gap")
    ax.axhline(0.25, linestyle="--", color="gray")
    if row["divergence_time_s_by_gap_m"]["0.25"] is not None:
        ax.axvline(row["divergence_time_s_by_gap_m"]["0.25"], color="red", linestyle=":")
    ax.set(xlabel="time (s)", ylabel="gap (m)", title="Time-aligned position divergence")
    ax.legend()
    ax = axes[3]
    ax.plot(k["upper_time_seconds"], k["upper_clearance"], label="K1 15 Hz")
    ax.plot(f["lower_time_seconds"], f["lower_clearance"], label="Full 120 Hz")
    ax.axhline(0, color="red", linestyle="--")
    ax.axhline(0.25, color="gray", linestyle=":")
    ax.set(xlabel="time (s)", ylabel="signed clearance (m)", title="Nearest tree clearance", ylim=(-0.2, 2.0))
    ax.legend()
    ax = axes[4]
    for axis, color in ((0, "tab:red"), (1, "tab:purple")):
        ax.step(f["upper_time_seconds"], f["upper_command_velocity"][:, axis], where="post", color=color, label=f"cmd {'xy'[axis]}")
        ax.plot(f["lower_time_seconds"], f["lower_velocity"][:, axis], color=color, alpha=0.55, label=f"actual {'xy'[axis]}")
    ax.set(xlabel="time (s)", ylabel="velocity (m/s)", title="Full command and actual velocity")
    ax.legend(ncol=2)
    ax = axes[5]
    ax.plot(f["lower_time_seconds"], track_p, label="position error (m)")
    ax.plot(f["lower_time_seconds"], track_v, label="velocity error (m/s)")
    ax.plot(f["lower_time_seconds"], motor_sat.astype(float), label="any motor at limit", alpha=0.5)
    ax.set(xlabel="time (s)", title="Full tracking and motor limit")
    ax.legend()
    fig.suptitle(f"seed {seed}, map {map_seed}: K1 {km['done_reason']} / Full {fm['done_reason']}")
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"seed{seed}_map{map_seed}_overview.png", dpi=130)
    plt.close(fig)

    # Align by geometric progress without pretending that same progress is
    # the same dynamic state (obstacle detours can make progress nonmonotonic).
    start = np.asarray(fm["initial_position"])
    target = np.asarray(fm["target_position"])
    route = target[:2] - start[:2]
    route /= np.linalg.norm(route)
    lateral = np.array([-route[1], route[0]])
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for pos, label in ((kp, "K1"), (fp, "Full")):
        rel = pos[:, :2] - start[:2]
        progress = rel @ route / np.linalg.norm(target[:2] - start[:2])
        ax.plot(progress, rel @ lateral, label=label)
    ax.set(xlabel="normalized start-to-goal projection", ylabel="lateral offset (m)", title="Geometry-aligned detour; progress may be nonmonotonic")
    ax.legend()
    fig.savefig(out / f"seed{seed}_map{map_seed}_progress.png", dpi=130)
    plt.close(fig)


def main():
    pairs = []
    for seed in (42, 43, 44, 45):
        directory = ROOT / f"seed{seed}_k1" / "traces"
        for path in sorted(directory.glob("trace_seed*.json")):
            map_seed = int(path.stem.removeprefix("trace_seed"))
            pairs.append(summarize(seed, map_seed))
    assert len(pairs) == 15, len(pairs)
    summary = {
        "schema_version": 1,
        "pair_count": len(pairs),
        "collision_pair_count": sum(row["full_done_reason"] == "collision" for row in pairs),
        "timeout_pair_count": sum(row["full_done_reason"] == "timeout" for row in pairs),
        "pairs": pairs,
    }
    (ROOT / "paired_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("wrote", ROOT / "paired_summary.json")


if __name__ == "__main__":
    main()
