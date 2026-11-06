"""Truth-assisted Full reachability witnesses for three residual A3 cases.

This is a bounded diagnostic controller search, not an Actor evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys
from unittest.mock import patch

import numpy as np

from benchmarks.forest_benchmark import grid_route
from scripts import change_td3
from scripts.run_change_td3 import build_mode_cmd
from trainers.paired_trajectory_trace import PairedTrajectoryTrace


BASE = "upper_dslpid_recurrent_multitree_terminal_v2_noroute_h8_k1"
MAP = "procedural_forest_multitree_validation_random_v1"
ROOT = Path("analysis_outputs/full_preimprovement_diagnostics_20261010/a3")
CASES = ((42, 24), (43, 21), (43, 24))
CONFIGS = (
    (0.22, 0.20, 2.0, 0.30),
    (0.30, 0.20, 2.0, 0.30),
    (0.38, 0.20, 2.0, 0.30),
    (0.30, 0.35, 2.0, 0.30),
    (0.38, 0.35, 2.0, 0.30),
    (0.30, 0.20, 1.4, 0.50),
    (0.38, 0.20, 1.4, 0.50),
)


def truth_path(env, extra_clearance):
    trees = env.base_env._tree_specs
    record = {"start": env.pos[0].copy(), "goal": env.TARGET_POS.copy(), "trees": trees}
    inflation = float(env.COLLISION_R) + extra_clearance
    path = grid_route(record, inflation=inflation, bound=max(env.FOREST_HALF_EXTENT, 4.0))
    if path is None:
        return None
    centers = np.asarray([tree["xy"] for tree in trees])
    radii = np.asarray([tree["radius"] + inflation for tree in trees])

    def visible(start, end):
        delta = end - start
        scale = np.clip(((centers - start) @ delta) / max(float(delta @ delta), 1e-9), 0, 1)
        d = np.linalg.norm(centers - (start + scale[:, None] * delta), axis=1)
        return bool(np.all(d > radii))

    smooth, index = [path[0]], 0
    while index < len(path) - 1:
        candidate = len(path) - 1
        while candidate > index + 1 and not visible(path[index], path[candidate]):
            candidate -= 1
        smooth.append(path[candidate])
        index = candidate
    return np.asarray(smooth[1:])


def command(env, path, waypoint_index, *, lookahead, gain, damping):
    pos = np.asarray(env.pos[0], dtype=float)
    vel = np.asarray(env.vel[0], dtype=float)
    goal = np.asarray(env.TARGET_POS, dtype=float)
    while waypoint_index < len(path) - 1 and np.linalg.norm(path[waypoint_index] - pos[:2]) < lookahead:
        waypoint_index += 1
    waypoint = path[waypoint_index]
    xy = gain * (waypoint - pos[:2]) - damping * vel[:2]
    if waypoint_index == len(path) - 1:
        # Slow capture near the 0.2 m success ball.
        distance = float(np.linalg.norm(goal[:2] - pos[:2]))
        if distance < 0.9:
            xy = gain * (goal[:2] - pos[:2]) - damping * vel[:2]
    speed = float(np.linalg.norm(xy))
    if speed > env.max_reference_speed:
        xy *= env.max_reference_speed / speed
    z = np.clip(1.8 * (goal[2] - pos[2]) - 0.45 * vel[2],
                -env.max_reference_vertical_speed, env.max_reference_vertical_speed)
    return np.r_[xy, z], waypoint_index


def run_case(env, seed, map_seed, index, cfg):
    extra, lookahead, gain, damping = cfg
    _, info = env.reset(seed=map_seed)
    assert info["curriculum_stage"] == 3, info
    source = Path("analysis_outputs/full_k1_paired_trajectories_20261010/all") / f"seed{seed}_full" / "traces" / f"trace_seed{map_seed}.json"
    meta = json.loads(source.read_text())
    np.testing.assert_allclose(env.pos[0], meta["initial_position"], atol=1e-6)
    np.testing.assert_allclose(env.TARGET_POS, meta["target_position"], atol=1e-6)
    assert len(env.base_env._tree_specs) == len(meta["tree_specs"])
    for actual, expected in zip(env.base_env._tree_specs, meta["tree_specs"]):
        np.testing.assert_allclose(actual["xy"], expected["xy"], atol=1e-6)
        assert abs(float(actual["radius"]) - expected["radius"]) < 1e-6
    path = truth_path(env, extra)
    row = {"training_seed": seed, "map_seed": map_seed, "config_index": index,
           "extra_clearance_m": extra, "lookahead_m": lookahead,
           "gain": gain, "damping": damping, "path_available": path is not None}
    if path is None:
        return row
    positions, velocities, commands, clearances, distances, times = [], [], [], [], [], []
    diagnostic_trace = PairedTrajectoryTrace(env, map_seed)
    diagnostic_trace.install()
    waypoint_index = 0
    for step in range(250):
        cmd, waypoint_index = command(env, path, waypoint_index,
                                       lookahead=lookahead, gain=gain, damping=damping)
        action = cmd / np.array([env.max_reference_speed, env.max_reference_speed,
                                 env.max_reference_vertical_speed])
        _, reward, terminated, truncated, info = env.step(action.reshape(1, -1))
        diagnostic_trace.record_upper(action, action, reward, info)
        positions.append(np.asarray(env.pos[0], dtype=float).copy())
        velocities.append(np.asarray(env.vel[0], dtype=float).copy())
        commands.append(cmd.copy())
        clearances.append(float(info["min_tree_clearance"]))
        distances.append(float(info["goal_distance"]))
        times.append((step + 1) / 15.0)
        if terminated or truncated:
            break
    diagnostic_trace.uninstall()
    trace_dir = ROOT / f"seed{seed}_map{map_seed}" / f"candidate_{index:02d}_full_trace"
    diagnostic_trace.save(trace_dir, done_reason=info["done_reason"],
                          success=info["done_reason"] == "success",
                          final_goal_distance=distances[-1])
    row.update({
        "path_xy": path.tolist(), "done_reason": info["done_reason"],
        "time_s": times[-1], "final_goal_distance_m": distances[-1],
        "minimum_goal_distance_m": min(distances),
        "minimum_tree_clearance_m": min(clearances),
        "steps": len(times),
        "minimum_120hz_tree_clearance_m": float(min(sample["clearance"] for sample in diagnostic_trace.lower)),
        "full_trace": str(trace_dir / f"trace_seed{map_seed}.npz"),
    })
    directory = ROOT / f"seed{seed}_map{map_seed}"
    directory.mkdir(parents=True, exist_ok=True)
    trace_path = directory / f"candidate_{index:02d}.npz"
    np.savez_compressed(trace_path, time_seconds=times, position=positions,
                        velocity=velocities, command_velocity=commands,
                        clearance=clearances, goal_distance=distances, path_xy=path)
    row["trace"] = str(trace_path)
    return row


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--all-configs", action="store_true", help="Continue search after first success")
    options = cli.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    class AuditCallback(change_td3.EvalCallback):
        def on_step(self, trainer, *, force=False):
            env = self.env_cls(**self.env_kwargs, gui=False)
            env.set_curriculum_stage_override(3)
            try:
                for seed, map_seed in CASES:
                    for index, cfg in enumerate(CONFIGS):
                        row = run_case(env, seed, map_seed, index, cfg)
                        rows.append(row)
                        (ROOT / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
                        print(seed, map_seed, index, row.get("done_reason", "no_path"),
                              row.get("final_goal_distance_m"), flush=True)
                        if row.get("done_reason") == "success" and not options.all_configs:
                            break
            finally:
                env.close()
    cmd = build_mode_cmd(BASE)[3:]
    for option, value in (("--seed", 42), ("--eval_map_id", MAP),
                          ("--upper_execution_model", "full"),
                          ("--log_dir", ROOT / "loader")):
        cmd[cmd.index(option) + 1] = str(value)
    cmd += ["--evaluate_only", "true", "--curriculum_stage_override", "3",
            "--load_checkpoint", "checkpoints/startobscurr_v1_selected_20261010/seed42/model_final"]
    with patch.object(change_td3, "EvalCallback", AuditCallback), patch.object(sys, "argv", [__file__] + cmd):
        change_td3.main()


if __name__ == "__main__":
    main()
