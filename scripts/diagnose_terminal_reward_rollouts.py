"""No-training reward audit using the production K1 transition and rewards.

The obstacle-free scene isolates the terminal contract; it is not a policy
evaluation or a substitute for the forest benchmark. Both reward versions are
scored on identical physical transitions. No networks/checkpoints are loaded.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from scripts.change_td3 import _make_argparser
from scripts.run_change_td3 import build_mode_cmd
from envs.ForestAviary import CustomForestAviary
from envs.forest.rewards import (
    RateIntegratedOutcomeForestReward,
    TerminalGuidanceRateIntegratedReward,
)
from envs.learned_upper_dslpid import KinematicUpperEnv
from envs.map_atlas import MapCatalog
from gym_pybullet_drones.utils.enums import ActionType, ObservationType


PRESET = "upper_dslpid_recurrent_multitree_terminal_v2_h8_k1"
ROOT = Path(__file__).resolve().parents[1]
BEHAVIORS = (
    "scheduled_capture", "fast_to_goal", "brake_to_stop", "hold_outside",
    "flyby", "orbit", "height_then_capture",
)
TRACE_COLUMNS = (
    "time", "x", "y", "z", "vx", "vy", "vz", "d_xy", "z_error", "d_3d",
    "raw_v1", "raw_v2", "progress", "tracking_cost", "terminal_outcome",
    "discounted_v1", "discounted_v2", "cmd_x", "cmd_y", "cmd_z",
)


def preset_args(name=PRESET):
    return _make_argparser().parse_args(build_mode_cmd(name)[3:])


def reward_model(args):
    kwargs = dict(
        integration_dt=args.high_level_interval / args.ctrl_freq,
        route_progress_reward_weight=args.route_progress_reward_weight,
        goal_progress_reward_weight=args.goal_progress_reward_weight,
        lateral_penalty_scale=args.lateral_penalty_scale,
        speed_penalty_weight=args.speed_penalty_weight,
        safety_boundary_penalty_weight=args.safety_boundary_penalty_weight,
        time_penalty_weight=args.time_penalty_weight,
        proximity_penalty_weight=args.proximity_penalty_weight,
        near_goal_radius=args.near_goal_radius,
        near_goal_speed_penalty_weight=args.near_goal_speed_penalty_weight,
        near_goal_away_penalty_weight=args.near_goal_away_penalty_weight,
        route_overshoot_penalty_weight=args.route_overshoot_penalty_weight,
        success_terminal_reward=args.success_terminal_reward,
        collision_terminal_penalty=args.collision_terminal_penalty,
        bound_terminal_penalty=args.bound_terminal_penalty,
        timeout_terminal_penalty=args.timeout_terminal_penalty,
    )
    if args.reward_variant == "terminal_guidance_v5":
        for key in (
            "near_goal_guidance_radius", "near_goal_guidance_ramp_width",
            "near_goal_capture_speed", "near_goal_cruise_speed",
            "near_goal_velocity_tracking_weight",
        ):
            kwargs[key] = getattr(args, key)
        return TerminalGuidanceRateIntegratedReward(**kwargs)
    return RateIntegratedOutcomeForestReward(**kwargs)


def make_env(args, model):
    spec = MapCatalog(args.map_catalog).get(args.eval_map_id)
    with contextlib.redirect_stdout(io.StringIO()):
        base = CustomForestAviary(
            obs=ObservationType.KIN, act=ActionType.RPM,
            initial_xyzs=np.asarray([spec.start_pos], dtype=np.float32),
            target_pos=spec.target_pos, forest_half_extent=spec.xy_half_extent,
            obstacle_sensor_range=spec.obstacle_sensor_range,
            pyb_freq=args.pyb_freq, ctrl_freq=args.ctrl_freq, gui=False,
            num_trees=0, route_blocking_tree=False, curriculum=False,
            range_azimuth_count=args.range_azimuth_count,
            range_elevation_degrees=args.range_elevation_degrees,
            reward_model=model,
        )
        env = KinematicUpperEnv(
            base, upper_control_interval=args.high_level_interval,
            max_reference_speed=args.max_reference_speed,
            max_reference_acceleration=args.max_reference_acceleration,
            max_reference_vertical_speed=args.max_reference_vertical_speed,
            max_reference_vertical_acceleration=args.max_reference_vertical_acceleration,
            semantic_history_length=args.semantic_history_length,
            kinematic_model="k1",
            kinematic_planar_acceleration=args.kinematic_planar_acceleration,
            kinematic_vertical_acceleration=args.kinematic_vertical_acceleration,
        )
        env.reset(seed=0)
    assert len(base._tree_ids) == 0
    return env


def initialize(env, position, velocity, elapsed=0.0):
    """Inject a diagnostic state, preserving the original route and clock rule."""
    with contextlib.redirect_stdout(io.StringIO()):
        env.reset(seed=0)
    assert not env._place_kinematic_state(position, velocity)
    base = env.base_env
    base._prev_pos = np.asarray(position, dtype=np.float32).copy()
    base._prev_goal_dist = float(np.linalg.norm(base.TARGET_POS - position))
    base.step_counter = int(round(elapsed * base.PYB_FREQ))
    env.runtime_lower_step = int(round(elapsed * base.CTRL_FREQ))
    env.reference_position = np.asarray(position, dtype=np.float32).copy()
    env.reference_velocity = np.asarray(velocity, dtype=np.float32).copy()
    env.command_velocity = np.asarray(velocity, dtype=np.float32).copy()


def feasible(command, env):
    command = np.asarray(command, dtype=np.float64).copy()
    speed = np.linalg.norm(command[:2])
    if speed > env.max_reference_speed:
        command[:2] *= env.max_reference_speed / speed
    command[2] = np.clip(command[2], -env.max_reference_vertical_speed,
                         env.max_reference_vertical_speed)
    return command


def command_for(behavior, env, initial, t):
    pos = env.base_env.pos[0]
    goal = env.base_env.TARGET_POS
    vector = goal - pos
    distance = float(np.linalg.norm(vector))
    direction = vector / max(distance, 1e-9)
    model = env.base_env._reward_model
    ratio = np.clip((distance - env.GOAL_TOLERANCE) /
                    (model.near_goal_guidance_radius - env.GOAL_TOLERANCE), 0, 1)
    speed = model.near_goal_capture_speed + ratio * (
        model.near_goal_cruise_speed - model.near_goal_capture_speed)
    scheduled = direction * speed
    if behavior.startswith("delay_"):
        return np.zeros(3) if t < float(behavior.split("_")[1]) else feasible(scheduled, env)
    if behavior == "scheduled_capture":
        command = scheduled
    elif behavior == "fast_to_goal":
        command = direction * env.max_reference_speed
    elif behavior == "brake_to_stop":
        command = np.zeros(3)
    elif behavior == "hold_outside":
        anchor = goal + np.array([-1.3, 0, 0])
        command = 1.5 * (anchor - pos)
    elif behavior == "flyby":
        # Miss the ball at y=0.4, then hold beyond the goal plane. Captures
        # caused by initial inertia still count as successes without override.
        waypoint = goal + np.array([0.0 if pos[0] < goal[0] else 0.5, 0.4, 0])
        delta = waypoint - pos
        command = 2 * delta
    elif behavior == "orbit":
        offset = pos[:2] - goal[:2]
        radius = max(np.linalg.norm(offset), 1e-9)
        radial = offset / radius
        target_radius = max(float(initial["d_xy"]), 0.35)
        command = np.r_[0.4 * np.array([-radial[1], radial[0]])
                        - 1.5 * (radius - target_radius) * radial,
                        1.5 * (initial["position"][2] - pos[2])]
    elif behavior == "height_then_capture":
        command = (np.array([0.0, 0.0, 1.5 * vector[2]])
                   if abs(vector[2]) > 0.08 else scheduled)
    else:
        raise ValueError(behavior)
    return feasible(command, env)


def score_step(env, model_v1, command):
    base = env.base_env
    previous_position = base.pos[0].copy()
    previous_distance = float(np.linalg.norm(base.TARGET_POS - previous_position))
    previous_velocity = base.vel[0].copy()
    action = command / np.array([env.max_reference_speed, env.max_reference_speed,
                                 env.max_reference_vertical_speed])
    assert np.linalg.norm(action[:2]) <= 1 + 1e-6 and abs(action[2]) <= 1 + 1e-6
    _, reward_v2, terminated, truncated, info = env.step(action)
    state = base._getDroneStateVector(0)
    reward_v1, terms_v1, _ = model_v1.compute(
        state=state, prev_goal_dist=previous_distance, prev_pos=previous_position,
        start_pos=base.START_POS, target_pos=base.TARGET_POS,
        goal_tolerance=base.GOAL_TOLERANCE, safe_distance=base.SAFE_DISTANCE,
        clearance=base._computeNearestTreeClearance(state[:3]),
        collision=bool(info["collision"]), terminal_reason=base._last_done_reason,
    )
    # Check real acceleration/velocity limits and exact reward accounting.
    assert np.max(np.abs(base.vel[0][:2] - previous_velocity[:2])) <= (
        env.kinematic_planar_acceleration * env.CTRL_TIMESTEP + 2e-6)
    assert abs(base.vel[0][2] - previous_velocity[2]) <= (
        env.kinematic_vertical_acceleration * env.CTRL_TIMESTEP + 2e-6)
    assert np.linalg.norm(base.vel[0][:2]) <= env.max_reference_speed + 2e-6
    assert abs(base.vel[0][2]) <= env.max_reference_vertical_speed + 2e-6
    terms_v2 = base._last_reward_terms.copy()
    for reward, terms in ((reward_v1, terms_v1), (reward_v2, terms_v2)):
        expected = (terms["terminal_outcome_reward"] if terminated or truncated else
                    terms["progress_reward"] - terms["continuous_penalty_total"])
        assert np.isclose(reward, expected, atol=2e-5)
    return float(reward_v1), float(reward_v2), terms_v1, terms_v2, terminated or truncated


def rollout(env, model_v1, args, case, behavior):
    initialize(env, case["position"], case["velocity"], case["elapsed"])
    rows = []
    totals = {"v1": {}, "v2": {}}
    for step in range(200):
        command = command_for(behavior, env, case, step * env.CTRL_TIMESTEP)
        r1, r2, terms1, terms2, done = score_step(env, model_v1, command)
        pos, vel = env.base_env.pos[0], env.base_env.vel[0]
        delta = env.base_env.TARGET_POS - pos
        multiplier = args.reward_scale * args.gamma ** step
        rows.append([
            (step + 1) * env.CTRL_TIMESTEP, *pos, *vel,
            np.linalg.norm(delta[:2]), abs(delta[2]), np.linalg.norm(delta),
            r1, r2, terms2["progress_reward"],
            terms2["near_goal_velocity_tracking_penalty"],
            terms2["terminal_outcome_reward"], multiplier * r1, multiplier * r2,
            *command,
        ])
        for name, terms in (("v1", terms1), ("v2", terms2)):
            for key, value in terms.items():
                totals[name][key] = totals[name].get(key, 0.0) + float(value)
        if done:
            break
    else:
        raise AssertionError("Production terminal rule did not end rollout")
    trace = np.asarray(rows, dtype=np.float64)
    outcome = env.base_env._last_done_reason
    if outcome == "success":
        assert trace[-1, 9] < env.GOAL_TOLERANCE
    result = {
        "case_id": case["id"], "behavior": behavior,
        "initial_xy_distance": case["d_xy"], "initial_height_error": case["z_error"],
        "initial_velocity": case["velocity"], "initial_velocity_label": case["velocity_label"],
        "elapsed_at_start": case["elapsed"], "outcome": outcome,
        "steps": len(trace), "duration": float(trace[-1, 0]),
        "min_3d_distance": float(trace[:, 9].min()),
        "final_3d_distance": float(trace[-1, 9]),
        "guidance_region_steps": int(np.sum((trace[:, 9] < 1.2) &
                                             (trace[:, 9] >= env.GOAL_TOLERANCE))),
        "tracking_cost_total": float(trace[:, 13].sum()),
        "raw_return_v1": float(trace[:, 10].sum()),
        "raw_return_v2": float(trace[:, 11].sum()),
        "G_v1": float(trace[:, 15].sum()), "G_v2": float(trace[:, 16].sum()),
        "reward_totals": totals,
    }
    for horizon in (0.5, 1, 2, 4, 8):
        mask = trace[:, 0] <= horizon + 1e-8
        result[f"G_v2_first_{horizon}s"] = float(trace[mask, 16].sum())
    return result, trace


def cases(env):
    goal = env.base_env.TARGET_POS
    velocities = {"rest": [0, 0, 0], "toward_fast": [0.8, 0, 0],
                  "tangent_fast": [0, 0.8, 0], "away_fast": [-0.8, 0, 0]}
    result = []
    for xy in (0.3, 0.5, 0.8, 1.0, 1.1, 1.3):
        for z in (0.0, 0.3, 0.8, 1.2):
            for label, velocity in velocities.items():
                result.append(dict(
                    id=f"xy{xy:.1f}_z{z:.1f}_{label}_t0", d_xy=xy, z_error=z,
                    position=(goal + [-xy, 0, z]).tolist(), velocity=velocity,
                    velocity_label=label, elapsed=0.0,
                ))
    # Remaining-time sensitivity: identical states with 6 or 2 seconds left.
    for elapsed in (6.0, 10.0):
        for xy, z in ((1.3, 0.0), (0.5, 0.0), (0.5, 1.2)):
            result.append(dict(
                id=f"xy{xy:.1f}_z{z:.1f}_rest_t{elapsed:g}", d_xy=xy, z_error=z,
                position=(goal + [-xy, 0, z]).tolist(), velocity=[0, 0, 0],
                velocity_label="rest", elapsed=elapsed,
            ))
    return result


def entry_scan(env, model_v1):
    rows = []
    goal = env.base_env.TARGET_POS
    for speed in (0.0, 0.2, 0.8, -0.8):
        for distance in sorted(set(np.linspace(0.95, 1.3, 36).tolist() +
                                   [1.1999, 1.2, 1.2001])):
            # Constant feasible velocity makes the post-step distance exact.
            start = goal + [-distance - speed * env.CTRL_TIMESTEP, 0, 0]
            initialize(env, start, [speed, 0, 0])
            r1, r2, _, terms, done = score_step(env, model_v1, [speed, 0, 0])
            assert not done
            rows.append(dict(distance=distance, radial_speed=speed, reward_v1=r1,
                             reward_v2=r2, tracking_cost=terms["near_goal_velocity_tracking_penalty"],
                             progress=terms["progress_reward"]))
    return rows


def summarize(results):
    summary = {"rollouts": len(results), "by_behavior": {}, "paired_rankings": {}}
    for behavior in sorted({r["behavior"] for r in results}):
        group = [r for r in results if r["behavior"] == behavior and r["elapsed_at_start"] == 0]
        summary["by_behavior"][behavior] = {
            "count": len(group),
            "outcomes": {reason: sum(r["outcome"] == reason for r in group)
                         for reason in sorted({r["outcome"] for r in group})},
            "mean_G_v2": float(np.mean([r["G_v2"] for r in group])) if group else None,
        }
    groups = {}
    for row in results:
        groups.setdefault(row["case_id"], []).append(row)
    for version in ("v1", "v2"):
        margins = []
        failures = []
        no_success = []
        for case, group in groups.items():
            success = [r for r in group if r["outcome"] == "success"]
            failure = [r for r in group if r["outcome"] != "success"]
            if not success:
                no_success.append(case)
                continue
            if not failure:
                continue
            best_success = max(success, key=lambda r: r[f"G_{version}"])
            best_failure = max(failure, key=lambda r: r[f"G_{version}"])
            margin = best_success[f"G_{version}"] - best_failure[f"G_{version}"]
            margins.append(margin)
            if margin <= 0:
                failures.append(dict(case=case, margin=margin,
                                     successful_behavior=best_success["behavior"],
                                     unsuccessful_behavior=best_failure["behavior"]))
        summary["paired_rankings"][version] = dict(
            comparable_cases=len(margins), success_better=sum(m > 0 for m in margins),
            minimum_margin=min(margins) if margins else None,
            counterexamples=failures, cases_without_sampled_success=no_success,
        )
    pair_margins = []
    for case, group in groups.items():
        by_name = {r["behavior"]: r for r in group}
        scheduled = by_name["scheduled_capture"]
        if scheduled["outcome"] != "success":
            continue
        for other in group:
            if other["outcome"] != "success":
                pair_margins.append(dict(case=case, alternative=other["behavior"],
                                        margin=scheduled["G_v2"] - other["G_v2"]))
    summary["scheduled_vs_unsuccessful"] = {
        "comparisons": len(pair_margins),
        "positive": sum(p["margin"] > 0 for p in pair_margins),
        "minimum_margin": min((p["margin"] for p in pair_margins), default=None),
        "counterexamples": [p for p in pair_margins if p["margin"] <= 0],
    }
    return summary


def plots(output, results, traces, scan):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for speed in (0.0, 0.2, 0.8, -0.8):
        points = [r for r in scan if r["radial_speed"] == speed]
        for ax, key in zip(axes, ("reward_v2", "tracking_cost")):
            ax.plot([r["distance"] for r in points], [r[key] for r in points],
                    label=f"radial velocity {speed:g} m/s")
            ax.set_xlabel("Post-step 3-D goal distance (m)")
            ax.axvline(1.2, color="gray", linestyle="--", linewidth=1)
            ax.grid(alpha=0.25)
    axes[0].set_ylabel("Raw environment reward per step")
    axes[1].set_ylabel("Integrated terminal tracking cost per step")
    axes[0].legend(fontsize=8)
    fig.savefig(output / "entry_scan.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    selected = [r for r in results if r["case_id"] == "xy1.3_z0.0_rest_t0"]
    for row in selected:
        if row["behavior"].startswith("delay_"):
            continue
        trace = traces[row["trace_key"]]
        label = row["behavior"]
        axes[0, 0].plot(trace[:, 0], trace[:, 9], label=label)
        axes[0, 1].plot(trace[:, 0], trace[:, 11])
        axes[1, 0].plot(trace[:, 0], np.cumsum(trace[:, 16]))
        axes[1, 1].plot(trace[:, 1], trace[:, 2])
    axes[0, 0].axhline(1.2, color="gray", linestyle="--")
    axes[0, 0].axhline(0.2, color="black", linestyle=":")
    axes[0, 0].set_ylabel("3-D goal distance (m)")
    axes[0, 1].set_ylabel("Raw reward per step")
    axes[1, 0].set_ylabel("Cumulative discounted scaled return")
    axes[1, 1].set_ylabel("Y (m)")
    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.set_xlabel("Time since diagnostic start (s)")
    axes[1, 1].set_xlabel("X (m)")
    axes[0, 0].legend(fontsize=7)
    fig.savefig(output / "trajectory_comparison.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "analysis_outputs/terminal_v2_reward_rollout_audit")
    parser.add_argument("--smoke", action="store_true")
    cli = parser.parse_args()
    output = cli.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    args = preset_args()
    old_args = preset_args(PRESET.replace("terminal_v2", "terminal_v1"))
    model_v1 = reward_model(old_args)
    env = make_env(args, reward_model(args))
    results, traces = [], {}
    try:
        selected_cases = cases(env)
        if cli.smoke:
            selected_cases = selected_cases[:1]
        metadata = {
            "preset": PRESET, "gamma": args.gamma, "reward_scale": args.reward_scale,
            "decision_dt": env.CTRL_TIMESTEP, "episode_seconds": env.EPISODE_LEN_SEC,
            "start_pos": env.START_POS.tolist(), "target_pos": env.TARGET_POS.tolist(),
            "success_radius": env.GOAL_TOLERANCE, "num_obstacles": len(env._tree_ids),
            "observation_shape": list(env.observation_space.shape),
            "limits": {key: getattr(env, key) for key in (
                "max_reference_speed", "max_reference_vertical_speed",
                "kinematic_planar_acceleration", "kinematic_vertical_acceleration")},
            "v1_reward": vars(model_v1), "v2_reward": vars(env.base_env._reward_model),
            "initial_states": selected_cases, "trace_columns": TRACE_COLUMNS,
            "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                              for name in ("envs/forest/rewards.py", "envs/forest/core.py",
                                           "envs/learned_upper_dslpid.py", "scripts/run_change_td3.py",
                                           "scripts/diagnose_terminal_reward_rollouts.py")},
            "scope": "Obstacle-free diagnostic, scripted controllers, production K1 and reward; no training or learned-policy evaluation.",
        }
        for index, case in enumerate(selected_cases):
            behaviors = list(BEHAVIORS)
            if case["velocity_label"] == "rest" and case["elapsed"] == 0:
                behaviors += ["delay_1", "delay_3", "delay_6"]
            for behavior in behaviors:
                result, trace = rollout(env, model_v1, args, case, behavior)
                key = f"trace_{len(results):04d}"
                result["trace_key"] = key
                traces[key] = trace
                results.append(result)
            if (index + 1) % 12 == 0:
                print(f"Completed {index + 1}/{len(selected_cases)} states, {len(results)} trajectories", flush=True)
        scan = entry_scan(env, model_v1)
    finally:
        env.close()
    summary = summarize(results)
    summary["metadata"] = metadata
    with (output / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    with (output / "rollouts.jsonl").open("w") as stream:
        for row in results:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    for filename, rows in (("rollouts.csv", [{k: v for k, v in r.items()
                                             if k not in ("reward_totals", "initial_velocity")}
                                            for r in results]), ("entry_scan.csv", scan)):
        with (output / filename).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    np.savez_compressed(output / "traces.npz", **traces)
    plots(output, results, traces, scan)
    print(json.dumps({key: value for key, value in summary.items() if key != "metadata"}, indent=2))
    print(f"Artifacts: {output}", flush=True)


if __name__ == "__main__":
    main()
