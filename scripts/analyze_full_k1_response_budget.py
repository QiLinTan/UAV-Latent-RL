"""A2 supplement: split command/reference/tracking lag and audit turn timing."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


A1 = Path("analysis_outputs/full_k1_paired_trajectories_20261010/all")
A2 = Path("analysis_outputs/full_k1_fixed_command_a2_20261010")
OUT = A2 / "response_budget"
REFERENCE_AXIS_ACCEL_MPS2 = 1.0  # Frozen noroute preset, per horizontal axis.
MIN_OUTWARD_MPS = 0.05
MIN_OUTWARD_SAMPLES = 16  # Two 15 Hz decisions at the 120 Hz Full rate.
RESPONSE_DROP_MPS = 0.10


def _first_run_time(time, values, *, start, end, threshold, below, samples=3):
    for index in range(start, max(start, end - samples + 1)):
        part = values[index:index + samples]
        if len(part) == samples and (np.all(part <= threshold) if below else np.all(part >= threshold)):
            return float(time[index] - time[start])
    return None


def _runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return [(int(start), int(end - 1)) for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def directional_series(trace, center, tree_radius, drone_radius):
    positions = np.asarray(trace["lower_position"])
    relative = positions[:, :2] - np.asarray(center, dtype=np.float64)
    radius = np.linalg.norm(relative, axis=1)
    outward = relative / np.maximum(radius[:, None], 1e-9)
    vectors = {
        "cmd": np.asarray(trace["lower_command_velocity"]),
        "ref": np.asarray(trace["lower_reference_velocity"]),
        "actual": np.asarray(trace["lower_velocity"]),
    }
    inward = {key: -np.sum(value[:, :2] * outward, axis=1) for key, value in vectors.items()}
    return {
        "time": np.asarray(trace["lower_time_seconds"], dtype=np.float64),
        "clearance_to_collision_tree": radius - float(tree_radius) - float(drone_radius),
        "nearest_clearance": np.asarray(trace["lower_clearance"], dtype=np.float64),
        "outward": outward,
        "inward": inward,
        "e_gen_3d": np.linalg.norm(vectors["cmd"] - vectors["ref"], axis=1),
        "e_track_3d": np.linalg.norm(vectors["ref"] - vectors["actual"], axis=1),
        "e_gen_inward": inward["ref"] - inward["cmd"],
        "e_track_inward": inward["actual"] - inward["ref"],
    }


def turn_events(series, *, collision_time):
    t = series["time"]
    u = series["inward"]
    clearance = series["clearance_to_collision_tree"]
    mask = (
        (u["cmd"] <= -MIN_OUTWARD_MPS)
        & (clearance > 0)
        & (clearance < 0.5)
        & (t >= collision_time - 2.0)
        & (t <= collision_time)
    )
    events = []
    for start, end in _runs(mask):
        if end - start + 1 < MIN_OUTWARD_SAMPLES or u["actual"][start] <= 0.1:
            continue
        # Response attribution stops at the end of this outward command run;
        # an Actor reversal makes later response time ambiguous.
        stop = end + 1
        ref_drop = _first_run_time(t, u["ref"][start] - u["ref"], start=start, end=stop,
                                   threshold=RESPONSE_DROP_MPS, below=False)
        actual_drop = _first_run_time(t, u["actual"][start] - u["actual"], start=start, end=stop,
                                      threshold=RESPONSE_DROP_MPS, below=False)
        ref_zero = _first_run_time(t, u["ref"], start=start, end=stop, threshold=0.0, below=True)
        actual_zero = _first_run_time(t, u["actual"], start=start, end=stop, threshold=0.0, below=True)
        available = float(clearance[start] / u["actual"][start])
        normal = series["outward"][start]
        reference_radial_accel_bound = REFERENCE_AXIS_ACCEL_MPS2 * float(np.sum(np.abs(normal)))
        ref_stop = float(max(0.0, u["ref"][start]) ** 2 / (2 * reference_radial_accel_bound))
        actual_stop_at_ref_bound = float(u["actual"][start] ** 2 / (2 * reference_radial_accel_bound))
        sample_after = min(end, start + 18)  # Up to 0.15 s of observed response.
        elapsed = float(t[sample_after] - t[start])
        empirical_radial_decel = (
            float((u["actual"][start] - u["actual"][sample_after]) / elapsed)
            if elapsed > 0 else None
        )
        empirical_stop = (
            float(u["actual"][start] ** 2 / (2 * empirical_radial_decel))
            if empirical_radial_decel is not None and empirical_radial_decel > 0.05
            else None
        )
        events.append({
            "time_s": float(t[start]),
            "time_to_collision_s": float(collision_time - t[start]),
            "command_outward_duration_s": float((end - start + 1) / 120.0),
            "clearance_m": float(clearance[start]),
            "actual_inward_mps": float(u["actual"][start]),
            "reference_inward_mps": float(u["ref"][start]),
            "command_inward_mps": float(u["cmd"][start]),
            "available_time_s": available,
            "reference_radial_accel_optimistic_bound_mps2": reference_radial_accel_bound,
            "reference_stop_distance_optimistic_m": ref_stop,
            "actual_stop_distance_at_reference_bound_m": actual_stop_at_ref_bound,
            "reference_stop_distance_over_clearance": float(ref_stop / clearance[start]),
            "actual_stop_distance_at_reference_bound_over_clearance": float(actual_stop_at_ref_bound / clearance[start]),
            "observed_actual_radial_deceleration_first_0_15_s_mps2": empirical_radial_decel,
            "observed_actual_stop_distance_constant_decel_m": empirical_stop,
            "reference_drop_0_1_s": ref_drop,
            "actual_drop_0_1_s": actual_drop,
            "reference_noninward_s": ref_zero,
            "actual_noninward_s": actual_zero,
            "reference_noninward_censored": ref_zero is None,
            "actual_noninward_censored": actual_zero is None,
        })
    return events


def outward_run_lengths(series, *, collision_time):
    t = series["time"]
    mask = (
        (series["inward"]["cmd"] <= -MIN_OUTWARD_MPS)
        & (series["clearance_to_collision_tree"] > 0)
        & (series["clearance_to_collision_tree"] < 0.5)
        & (t >= collision_time - 2.0)
        & (t <= collision_time)
    )
    return [end - start + 1 for start, end in _runs(mask)]


def summarize_window(series, end_time, width):
    t = series["time"]
    mask = (t > end_time - width) & (t <= end_time)
    if not np.any(mask):
        return None
    e_gen = series["e_gen_3d"][mask]
    e_track = series["e_track_3d"][mask]
    return {
        "samples": int(np.sum(mask)),
        "e_gen_mean_mps": float(np.mean(e_gen)),
        "e_gen_p95_mps": float(np.percentile(e_gen, 95)),
        "e_track_mean_mps": float(np.mean(e_track)),
        "e_track_p95_mps": float(np.percentile(e_track, 95)),
        "e_gen_inward_mean_mps": float(np.mean(series["e_gen_inward"][mask])),
        "e_track_inward_mean_mps": float(np.mean(series["e_track_inward"][mask])),
        "cmd_inward_mean_mps": float(np.mean(series["inward"]["cmd"][mask])),
        "ref_inward_mean_mps": float(np.mean(series["inward"]["ref"][mask])),
        "actual_inward_mean_mps": float(np.mean(series["inward"]["actual"][mask])),
        "minimum_collision_tree_clearance_m": float(np.min(series["clearance_to_collision_tree"][mask])),
    }


def make_plot(seed, map_seed, closed, fixed, collision_time, events):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    for data, style, name in ((closed, "-", "closed"), (fixed, "--", "fixed K1 tape")):
        t = data["time"]
        axes[0].plot(t, data["clearance_to_collision_tree"], style, label=name)
        for key, color in (("cmd", "tab:blue"), ("ref", "tab:orange"), ("actual", "tab:red")):
            axes[1].plot(t, data["inward"][key], style, color=color,
                         label=f"{name} {key}" if name == "closed" else None,
                         alpha=1.0 if name == "closed" else 0.55)
        axes[2].plot(t, data["e_gen_3d"], style, color="tab:purple",
                     label=f"{name} command-to-ref")
        axes[2].plot(t, data["e_track_3d"], style, color="tab:green",
                     label=f"{name} ref-to-actual")
    for event in events:
        for ax in axes:
            ax.axvline(event["time_s"], color="black", alpha=0.3, linestyle=":")
    for ax in axes:
        ax.axvline(collision_time, color="red", linestyle="--", alpha=0.7)
        ax.legend(ncol=2, fontsize=8)
        ax.set_xlim(max(0, collision_time - 1.5), collision_time + 0.25)
    axes[0].axhline(0, color="black", alpha=0.4)
    axes[0].set(ylabel="clearance to collision tree (m)")
    axes[1].axhline(0, color="black", alpha=0.4)
    axes[1].set(ylabel="inward velocity (m/s)")
    axes[2].set(xlabel="time (s)", ylabel="3D velocity gap (m/s)")
    fig.suptitle(f"seed {seed}, map {map_seed}: command -> reference -> actual")
    fig.savefig(OUT / f"seed{seed}_map{map_seed}_response.png", dpi=140)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paired = json.loads((A1 / "paired_summary.json").read_text())
    rows = []
    for pair in paired["pairs"]:
        if pair["full_done_reason"] != "collision":
            continue
        seed, map_seed = pair["training_seed"], pair["map_seed"]
        source_path = A1 / f"seed{seed}_full" / "traces" / f"trace_seed{map_seed}"
        fixed_path = A2 / f"seed{seed}_map{map_seed}" / "full_tape" / "full" / "traces" / f"trace_seed{map_seed}"
        meta = json.loads(source_path.with_suffix(".json").read_text())
        fixed_meta = json.loads(fixed_path.with_suffix(".json").read_text())
        assert meta["initial_position"] == fixed_meta["initial_position"]
        assert meta["tree_specs"] == fixed_meta["tree_specs"]
        center = np.asarray(pair["inward_velocity_nearest_tree_last_0_25_s"]["tree_xy"])
        tree = next(spec for spec in meta["tree_specs"] if np.allclose(spec["xy"], center))
        with np.load(source_path.with_suffix(".npz")) as raw_closed, np.load(fixed_path.with_suffix(".npz")) as raw_fixed:
            # At impact this tree is the nearest one; infer the collision-body
            # radius from logged signed clearance to avoid a duplicate constant.
            drone_radius = float(np.linalg.norm(raw_closed["lower_position"][-1, :2] - center)
                                 - tree["radius"] - raw_closed["lower_clearance"][-1])
            assert 0.04 < drone_radius < 0.08, drone_radius
            closed = directional_series(raw_closed, center, tree["radius"], drone_radius)
            fixed = directional_series(raw_fixed, center, tree["radius"], drone_radius)
        collision_time = float(pair["collision_time_s"])
        events = turn_events(closed, collision_time=collision_time)
        fixed_window_end = min(collision_time, float(fixed["time"][-1]))
        row = {
            "training_seed": seed,
            "map_seed": map_seed,
            "closed_full_done_reason": meta["done_reason"],
            "fixed_full_done_reason": fixed_meta["done_reason"],
            "collision_time_s": collision_time,
            "collision_tree_xy": center.tolist(),
            "collision_tree_radius_m": float(tree["radius"]),
            "drone_collision_radius_m": drone_radius,
            "closed_last_1_s": summarize_window(closed, collision_time, 1.0),
            "closed_last_0_25_s": summarize_window(closed, collision_time, 0.25),
            "fixed_common_horizon_last_0_25_s": summarize_window(fixed, fixed_window_end, 0.25),
            "fixed_common_horizon_end_s": fixed_window_end,
            "sustained_outward_events": events,
            "decisive_outward_event": events[-1] if events else None,
            "no_sustained_outward_command_in_final_2_s": not bool(events),
            "outward_run_lengths_lower_samples": outward_run_lengths(closed, collision_time=collision_time),
        }
        rows.append(row)
        make_plot(seed, map_seed, closed, fixed, collision_time, events)
    assert len(rows) == 12
    summary = {
        "schema_version": 1,
        "definition": {
            "inward_positive": True,
            "outward_command_threshold_mps": -MIN_OUTWARD_MPS,
            "outward_sustained_samples": MIN_OUTWARD_SAMPLES,
            "outward_search_last_seconds": 2.0,
            "outward_event_clearance_range_m": [0.0, 0.5],
            "response_drop_mps": RESPONSE_DROP_MPS,
            "reference_axis_acceleration_limit_mps2": REFERENCE_AXIS_ACCEL_MPS2,
            "response_times_censored_at_command_reversal_or_collision": True,
            "stopping_distance_is_local_constant_deceleration_proxy": True,
        },
        "collision_cases": len(rows),
        "cases_with_sustained_outward_command": sum(bool(row["sustained_outward_events"]) for row in rows),
        "cases_with_any_outward_command_run": sum(bool(row["outward_run_lengths_lower_samples"]) for row in rows),
        "rows": rows,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("wrote", OUT / "summary.json")


if __name__ == "__main__":
    main()
