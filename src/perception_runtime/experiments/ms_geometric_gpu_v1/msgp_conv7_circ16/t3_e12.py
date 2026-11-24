"""E12 closed-loop far-only intervention and physical aperture evaluation."""
from contextlib import ExitStack
from dataclasses import replace
from math import comb

import numpy as np
import pybullet as pyb
import torch

from evaluate import realtime_decisions
from perception import NEAR_WIDTH
from sensor import use_parallel_lidar
from t0_eval import APERTURES, follow, initial_info, paired_map, through

from experiments.rv1_range_cnn.environment import RangeImageEnv

GEOMETRY = slice(NEAR_WIDTH, NEAR_WIDTH + 3 * 16)
HORIZONS = (15, 45)


def from_shadow(shadow, physical, obs, builder):
    ghost = shadow.base_env
    scan = shadow.lidar.scan(client_id=ghost.CLIENT, position=physical.pos[0],
                             quaternion_xyzw=physical.quat[0], timestamp_s=physical.physics_ticks / 240,
                             self_ids=tuple(map(int, ghost.DRONE_IDS)),
                             ground_ids=(int(ghost.PLANE_ID),), obstacle_ids=tuple(ghost._tree_ids))
    return builder(dict(range_m=scan.range_m, hit_mask=scan.hit_mask,
                        scanned_mask=scan.scanned_mask, state=obs["state"]))


def counterfactual(case, agent, builder):
    reference_map = replace(paired_map(case, 1), map_id=f"e12_cf_{case}", obstacles=())
    chosen, initial, far_first = {}, [], []
    with ExitStack() as stack:
        references = {}
        for side in (-1, 1):
            env = use_parallel_lidar(RangeImageEnv(map_spec=paired_map(case, side)))
            stack.callback(env.close)
            env.reset(seed=1000 + case)
            ghost = env.base_env
            # The static diagnostic ray worlds have no drone in the sensed volume.
            for body in ghost.DRONE_IDS:
                pyb.resetBasePositionAndOrientation(int(body), (50, 50, 10), (0, 0, 0, 1),
                                                    physicsClientId=ghost.CLIENT)
            references[side] = env
        for side in (-1, 1):
            env = use_parallel_lidar(RangeImageEnv(map_spec=reference_map))
            stack.callback(env.close)
            obs, info = env.reset(seed=1000 + case)
            initial.append(initial_info(env, info))
            origin_y = float(env.base_env.pos[0, 1])
            outcomes, signal_steps = {}, []
            with realtime_decisions():
                for tick in range(1, HORIZONS[-1] + 1):
                    base = builder(obs)
                    hybrid = {}
                    for variant, shadow in references.items():
                        from_far = from_shadow(shadow, env.base_env, obs, builder)
                        combined = base.clone()
                        combined[GEOMETRY] = from_far[GEOMETRY]
                        hybrid[variant] = combined
                    if (not torch.equal(hybrid[-1][:NEAR_WIDTH], hybrid[1][:NEAR_WIDTH])
                            or not torch.equal(hybrid[-1][GEOMETRY.stop:], hybrid[1][GEOMETRY.stop:])):
                        raise RuntimeError("E12 changed near field, alignment, or State18")
                    if tick == 1:
                        far_first.append(hybrid[1][GEOMETRY] - hybrid[-1][GEOMETRY])
                    if float((hybrid[1][GEOMETRY] - hybrid[-1][GEOMETRY]).norm()) > 1e-6:
                        signal_steps.append(tick)
                    action = agent.action(hybrid[side])
                    obs, _, terminal, truncated, _ = env.step(action)
                    if tick in HORIZONS:
                        outcomes[str(tick // 15)] = float(env.base_env.pos[0, 1]) - origin_y
                    if terminal or truncated:
                        break
            chosen[side] = (outcomes, signal_steps)
    if initial[0] != initial[1] or len(far_first) != 2 or not torch.equal(*far_first):
        raise RuntimeError("the counterfactual closed loops started from different states")
    far_difference = float(far_first[0].norm())
    horizons = {}
    for duration in ("1", "3"):
        left, right = (chosen[side][0].get(duration) for side in (-1, 1))
        complete = left is not None and right is not None
        delta = right - left if complete else None
        exposed = any(tick <= int(duration) * 15 for side in (-1, 1) for tick in chosen[side][1])
        horizons[duration] = dict(negative_open_y_m=left, positive_open_y_m=right, delta_y_m=delta,
                                  complete=complete, far_signal_observed=exposed,
                                  correct_switch=complete and exposed and delta > 1e-9,
                                  both_sides_correct=complete and exposed and left < -1e-9 and right > 1e-9)
    return dict(far_geometry_delta_l2=far_difference, horizons=horizons, initial=initial[0])


def real_apertures(case, agent, builder):
    wall, offset, width = APERTURES[case]
    starts, rollouts = [], []
    for side in (-1, 1):
        env = use_parallel_lidar(RangeImageEnv(map_spec=paired_map(case, side)))
        try:
            obs, info = env.reset(seed=1000 + case)
            starts.append(builder(obs))
            init = initial_info(env, info)
            with realtime_decisions():
                row, trace = follow(env, agent, builder, obs, info, wall=wall)
            rollouts.append(dict(side=side, init=init, passed=through(trace, wall, side * offset, width),
                                 wall_m=wall, offset_m=offset, aperture_width_m=width, **row))
        finally:
            env.close()
    if (rollouts[0]["init"] != rollouts[1]["init"]
            or not torch.equal(starts[0][:NEAR_WIDTH], starts[1][:NEAR_WIDTH])
            or not torch.equal(starts[0][GEOMETRY.stop:], starts[1][GEOMETRY.stop:])):
        raise RuntimeError("real E12 aperture pair changed initial near field or State18")
    delta = float((starts[1][GEOMETRY] - starts[0][GEOMETRY]).norm())
    return dict(initial_far_geometry_delta_l2=delta, real=rollouts)


def measure(case, agent, builder):
    return dict(case=case, intervention=counterfactual(case, agent, builder),
                **real_apertures(case, agent, builder))


def summarize(cases):
    if [r["case"] for r in cases] != list(range(12)):
        raise RuntimeError("E12 requires all 12 paired layouts")
    outcomes = {}
    for horizon in ("1", "3"):
        results = [r["intervention"]["horizons"][horizon] for r in cases]
        verified = [r for r in results if r["complete"]]
        exposed = [r for r in verified if r["far_signal_observed"]]
        deltas = [r["delta_y_m"] for r in exposed]
        correct = sum(r["correct_switch"] for r in verified)
        nonzero = sum(abs(d) > 1e-9 for d in deltas)
        positive = sum(d > 1e-9 for d in deltas)
        p = sum(comb(nonzero, k) for k in range(positive, nonzero + 1)) / 2 ** nonzero if nonzero else None
        outcomes[horizon] = dict(pairs=len(results), complete=len(verified), signal_pairs=len(exposed),
                                 correct_switch=correct,
                                 both_sides_correct=sum(r["both_sides_correct"] for r in verified),
                                 median_delta_y_m=float(np.median(deltas)) if deltas else None,
                                 one_sided_p=p, nonzero_pairs=nonzero)
    physical = [t for row in cases for t in row["real"]]
    return dict(closed_loop=outcomes,
                initially_distinguishable_pairs=sum(r["intervention"]["far_geometry_delta_l2"] > 1e-6 for r in cases),
                actual_apertures=dict(episodes=len(physical),
                passed=sum(r["passed"] for r in physical), arrived=sum(r["success"] for r in physical),
                passed_and_arrived=sum(r["passed"] and r["success"] for r in physical)),
                method="Same empty reference physics for both closed loops; at each visited pose only the first three far-sector channels are read from mirrored static ray worlds. Actual passage is measured separately in the wall scenes.")
