"""Reproduce the archived Seed43 opening failures and Stage0 goal overshoots."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path

import numpy as np
import pybullet as p

from evaluate import make_validation_env, realtime_decisions
from experiments.rv1_range_cnn.environment import RangeImageEnv
from sensor import use_parallel_lidar
from t0_eval import APERTURES, follow, initial_info, paired_map, through
from t2_diag_traces import xy_bounds
from t2_eval import read_json, write_json
import t3_eval as evaluator
from train import sha


def run_trace(env, seed, expected, wall=None):
    obs, info = env.reset(seed=seed)
    if initial_info(env, info) != expected["init"]:
        raise RuntimeError("opening/overshoot start state mismatch")
    b = env.base_env
    positions = [b.pos[0].tolist()]
    contacts = []
    original_step = env.step

    def track(action):
        result = original_step(action)
        positions.extend(tick["position"] for tick in env.last_ticks)
        if result[2] or result[3]:
            contacts.extend(dict(body=int(x[2]), point_on_body_m=list(x[6]), distance_m=float(x[8]))
                            for x in p.getContactPoints(bodyA=int(b.DRONE_IDS[0]), physicsClientId=b.CLIENT))
        return result

    env.step = track
    with realtime_decisions():
        row, trace = follow(env, evaluator.agent, evaluator.builder, obs, info, wall)
    if ((row["reason"], row["upper_steps"]) != (expected["reason"], expected["upper_steps"])
            or abs(row["final_goal_distance_m"] - expected["final_goal_distance_m"]) > 1e-6):
        raise RuntimeError("frozen result did not reproduce")
    return row, trace, np.asarray(positions), contacts


def opening(case, expected):
    side = expected["side"]
    wall, offset, width = APERTURES[case]
    env = use_parallel_lidar(RangeImageEnv(map_spec=paired_map(case, side)))
    try:
        row, trace, path, contacts = run_trace(env, 1000 + case, expected, wall)
        passed = through(trace, wall, side * offset, width)
        if passed != expected["passed"]:
            raise RuntimeError("opening passage did not reproduce")
        crossings = [dict(upper_step=tick["step"], position=tick["next_position"],
                          signed_y_from_opening_m=tick["next_position"][1] - side * offset)
                     for tick in trace if tick["position"][0] < wall + .09 <= tick["next_position"][0]]
        panels = [r for r in contacts if r["body"] in env.base_env._tree_ids]
        inner_edges = (side * offset - width / 2, side * offset + width / 2)
        jamb_gap = min((abs(r["point_on_body_m"][1] - edge) for r in panels for edge in inner_edges), default=None)
        jamb_contact = jamb_gap is not None and jamb_gap <= .06
        if passed:
            category = "passed_then_" + row["reason"]
        elif panels and row["reason"] == "collision":
            category = "opening_jamb_contact" if jamb_contact else "wall_face_contact"
        elif crossings and abs(crossings[0]["position"][1]) > 1.45:
            category = "went_outside_wall_edge"
        elif crossings:
            category = "crossed_wall_plane_outside_opening_pass_criterion"
        else:
            category = "did_not_cross_wall_plane"
        return dict(case=case, side=side, passed=passed, category=category, reason=row["reason"],
                    panel_contact=bool(panels), jamb_contact=jamb_contact, min_contact_to_jamb_y_m=jamb_gap,
                    ground_contact=any(r["body"] == int(env.base_env.PLANE_ID) for r in contacts),
                    terminal_contacts=contacts,
                    wall_rear_plane_x=wall + .09, opening_center_y=side * offset, opening_width=width,
                    crossings=crossings, tail=trace[-15:], final_position_m=path[-1].tolist())
    finally:
        env.close()


def overshoot(expected):
    env = make_validation_env(evaluator.contract)
    try:
        env.obstacle_curriculum.stage = expected["stage"]
        row, _, path, _ = run_trace(env, expected["case"], expected)
        b = env.base_env
        goal = np.asarray(b.TARGET_POS)
        distance = np.linalg.norm(path - goal, axis=1)
        best = int(distance.argmin())
        direction = np.diff(path, axis=0)
        denom = (direction * direction).sum(axis=1)
        t = np.divide(((goal - path[:-1]) * direction).sum(axis=1), denom,
                      out=np.zeros_like(denom), where=denom > 0).clip(0, 1)
        minimum_segment = float(np.linalg.norm(path[:-1] + direction * t[:, None] - goal, axis=1).min())
        bounds = xy_bounds(b).tolist()
        return dict(seed=expected["case"], target_xyz_m=goal.tolist(), success_radius_m=float(b.GOAL_TOLERANCE),
                    closest_control_state_m=path[best].tolist(), closest_control_distance_m=float(distance[best]),
                    closest_piecewise_linear_distance_m=minimum_segment, below_goal_radius=bool(distance.min() < b.GOAL_TOLERANCE),
                    final_xyz_m=path[-1].tolist(), xy_limit_m=bounds, terminal_reason=row["reason"],
                    overshot_goal_x_and_missed_sphere=bool(path[-1, 0] > bounds[0] > goal[0]
                                                          and minimum_segment >= b.GOAL_TOLERANCE))
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    path = args.run / "train/checkpoint_240000.pt"
    forest = read_json(args.results / "t3_240000.json")
    e12 = read_json(args.results / "e12_240000.json")
    digest = sha(path)
    if digest != forest["checkpoint_sha256"] or digest != e12["checkpoint_sha256"]:
        raise RuntimeError("240k checksum mismatch")
    fingerprint = read_json(args.run / "train/metadata.json")["contract"]["pool_fingerprint"]
    cases = [(entry["case"], row) for entry in e12["cases"] for row in entry["real"]
             if not row["passed"] or not row["success"]]
    over = [r for r in forest["rows"] if r["stage"] == 0 and r["case"] in (0, 16)]
    with ProcessPoolExecutor(max_workers=8, mp_context=multiprocessing.get_context("spawn"),
                             initializer=evaluator.init_worker, initargs=("t3", path, fingerprint)) as pool:
        opening_jobs = [pool.submit(opening, case, row) for case, row in cases]
        over_jobs = [pool.submit(overshoot, row) for row in over]
        failures = [job.result() for job in opening_jobs]
        overshoots = [job.result() for job in over_jobs]
    if len(failures) != 11 or sum(not r["passed"] for r in failures) != 10 or len(overshoots) != 2:
        raise RuntimeError("unexpected number of opening/overshoot cases")
    write_json(args.results / "opening_and_overshoot_check.json",
               dict(checkpoint_sha256=digest, opening_failures=failures, stage0_overshoot=overshoots,
                    jamb_definition="Terminal wall contact within 0.06 m laterally of either opening inner edge."))
    print("Verified 11 opening failures and both Stage0 overshoots.", flush=True)


if __name__ == "__main__":
    main()
