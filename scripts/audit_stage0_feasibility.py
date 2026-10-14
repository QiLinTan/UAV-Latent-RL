"""Read historical Stage-0 episode logs and probe the unchanged K1 train map.

Historical TensorBoard records contain terminal, not per-step, distances and
reward terms. Physical probes use the production preset, reward, K1 wrapper,
and a forced Stage-0 procedural map. No network updates or training occur.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from envs.learned_upper_dslpid import KinematicUpperForestEnv
from envs.map_atlas import MapCatalog
from gym_pybullet_drones.utils.enums import ActionType, ObservationType
from scripts.audit_spatial_terminal_consistency import planned_path
from scripts.diagnose_terminal_reward_rollouts import feasible, preset_args, reward_model

BASE = 'upper_dslpid_recurrent_multitree_terminal_v2_noroute_h8_k1'
PENALTIES = (
    'time_penalty', 'height_penalty', 'vertical_speed_penalty',
    'lateral_penalty', 'proximity_penalty', 'attitude_penalty',
    'speed_penalty', 'near_goal_speed_penalty', 'near_goal_away_penalty',
    'near_goal_velocity_tracking_penalty',
    'route_overshoot_penalty', 'safety_boundary_penalty',
)
TERMINAL_REASONS = ('success', 'collision', 'xy_bound', 'height_bound', 'attitude_bound', 'timeout')


def historical_episodes():
    output = {}
    for train_seed in (42, 43, 44, 45):
        name = BASE if train_seed == 42 else f'{BASE}_seed{train_seed}'
        files = sorted(Path('runs', name).glob('events.out.tfevents.*'))
        rows = defaultdict(dict)
        tags = ['env/curriculum_stage', 'env/goal_distance', 'episode/return',
                'episode/length'] + [f'env/done_reason/{reason}' for reason in TERMINAL_REASONS]
        for file in files:
            event = EventAccumulator(str(file), size_guidance={'scalars': 0})
            event.Reload()
            available = event.Tags()['scalars']
            read_tags = tags + [tag for tag in available
                if tag.startswith('env/training_episode/')]
            for tag in read_tags:
                if tag in available:
                    for scalar in event.Scalars(tag):
                        rows[scalar.step][tag] = float(scalar.value)
        stage0 = [row for row in rows.values() if row.get('env/curriculum_stage') == 0.
                  and 'env/goal_distance' in row]
        distances = [row['env/goal_distance'] for row in stage0]
        outcomes = Counter(next((reason for reason in TERMINAL_REASONS
            if row.get(f'env/done_reason/{reason}', 0.) > .5), 'unknown') for row in stage0)
        exact_rows = [row for row in stage0
            if 'env/training_episode/min_goal_distance' in row]
        exact_available = len(exact_rows) == len(stage0) and len(stage0) > 0
        reward_sum_tags = sorted({tag for row in exact_rows for tag in row
            if tag.startswith('env/training_episode/reward_sum/')})
        output[str(train_seed)] = dict(
            event_files=[str(path) for path in files],
            stage0_episode_count=len(stage0),
            stage0_outcomes=dict(outcomes),
            stage0_final_distance_m=dict(
                min=float(min(distances)), median=float(np.median(distances)),
                mean=float(np.mean(distances)), max=float(max(distances)),
                less_than_1p2=sum(d < 1.2 for d in distances),
                less_than_2=sum(d < 2. for d in distances),
                less_than_3=sum(d < 3. for d in distances),
            ) if distances else {},
            guidance_entry_count_lower_bound=sum(d < 1.2 for d in distances),
            historical_minimum_distance_available=exact_available,
            historical_guidance_entry_fraction_available=exact_available,
            historical_cumulative_reward_terms_available=bool(reward_sum_tags),
            exact_episode_guidance_entries=(
                sum(row['env/training_episode/min_goal_distance'] < 1.2
                    for row in exact_rows) if exact_available else None),
            exact_episode_minimum_distance_m=(
                dict(min=float(min(row['env/training_episode/min_goal_distance']
                    for row in exact_rows)),
                    median=float(np.median([row['env/training_episode/min_goal_distance']
                        for row in exact_rows]))) if exact_available else None),
            mean_cumulative_reward_terms=(
                {tag.split('reward_sum/', 1)[1]:float(np.mean([row[tag]
                    for row in exact_rows if tag in row])) for tag in reward_sum_tags}
                if reward_sum_tags else None),
            caveat=('Exact training minimum/entry and cumulative reward require '
                'env/training_episode/* tags; older event files only have last-step values.'),
        )
    return output


def make_stage0_env(args, env_class=KinematicUpperForestEnv):
    spec = MapCatalog(args.map_catalog).get(args.map_id)
    env = env_class(
        obs=ObservationType.KIN, act=ActionType.RPM,
        initial_xyzs=np.asarray([spec.start_pos], dtype=np.float32),
        initial_rpys=np.zeros((1, 3), dtype=np.float32),
        pyb_freq=args.pyb_freq, ctrl_freq=args.ctrl_freq, gui=False,
        curriculum=args.curriculum, curriculum_milestones=args.curriculum_milestones,
        curriculum_success_gated=args.curriculum_success_gated,
        curriculum_success_window=args.curriculum_success_window,
        curriculum_success_thresholds=args.curriculum_success_thresholds,
        curriculum_max_collision_rates=args.curriculum_max_collision_rates,
        curriculum_minimum_stage_episodes=args.curriculum_minimum_stage_episodes,
        route_count_curriculum=args.route_count_curriculum,
        route_count_curriculum_counts=args.route_count_curriculum_counts,
        curriculum_simple_mix_probability=args.curriculum_simple_mix_probability,
        num_trees=args.num_trees, route_blocking_tree=args.route_blocking_tree,
        route_tree_fraction=args.route_tree_fraction,
        route_blocking_tree_count=args.route_blocking_tree_count,
        route_tree_lateral_range=args.route_tree_lateral_range,
        route_tree_layout=args.route_tree_layout,
        range_azimuth_count=args.range_azimuth_count,
        range_elevation_degrees=args.range_elevation_degrees,
        target_pos=spec.target_pos, map_spec=spec, reward_model=reward_model(args),
        upper_control_interval=args.high_level_interval,
        reference_sequence_length=args.reference_sequence_length,
        reference_horizon_seconds=args.reference_horizon_seconds,
        max_reference_speed=args.max_reference_speed,
        max_reference_acceleration=args.max_reference_acceleration,
        max_reference_vertical_speed=args.max_reference_vertical_speed,
        max_reference_vertical_acceleration=args.max_reference_vertical_acceleration,
        semantic_history_length=args.semantic_history_length,
        planar_only=args.upper_planar_only,
        height_hold_gain=args.upper_height_hold_gain,
        use_local_occupancy=args.use_local_occupancy,
        local_occupancy_voxel_size=args.local_occupancy_voxel_size,
        local_occupancy_shape=args.local_occupancy_shape,
        kinematic_model=args.upper_execution_model,
        kinematic_planar_acceleration=args.kinematic_planar_acceleration,
        kinematic_vertical_acceleration=args.kinematic_vertical_acceleration,
    )
    env.set_curriculum_stage_override(0)
    return env


def map_fingerprint(env):
    return [(tuple(round(float(x), 6) for x in tree['xy']),
             round(float(tree['radius']), 6)) for tree in env.base_env._tree_specs]


def run_episode(env, seed, behavior):
    _, info = env.reset(seed=seed)
    assert info['curriculum_stage'] == 0
    start = env.pos[0].copy()
    goal = env.TARGET_POS.copy()
    path = planned_path(env) if behavior == 'planned_p' else None
    if behavior == 'planned_p' and path is None:
        return dict(seed=seed, behavior=behavior, planner_unavailable=True)
    waypoint = 0
    distances = [float(np.linalg.norm(goal - start))]
    rewards = []
    terms_sum = defaultdict(float)
    z_max = float(start[2])
    z_min = float(start[2])
    for step in range(200):
        pos = env.pos[0].copy()
        if behavior == 'hold':
            command = np.zeros(3)
        elif behavior == 'climb':
            command = np.array([0., 0., env.max_reference_vertical_speed])
        elif behavior == 'direct_p':
            command = feasible(1.5 * (goal - pos), env)
        elif behavior == 'planned_p':
            while waypoint < len(path)-1 and np.linalg.norm(path[waypoint] - pos[:2]) < .12:
                waypoint += 1
            command = feasible(np.r_[2.5 * (path[waypoint] - pos[:2]),
                                      1.5 * (goal[2] - pos[2])], env)
        else:
            raise ValueError(behavior)
        action = command / [env.max_reference_speed, env.max_reference_speed,
                            env.max_reference_vertical_speed]
        _, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        terms = env.base_env._last_reward_terms
        for key in ('progress_reward', 'goal_progress', 'route_progress',
                    'continuous_penalty_total', 'terminal_outcome_reward', *PENALTIES):
            terms_sum[key] += float(terms.get(key, 0.))
        distances.append(float(info['goal_distance']))
        z_max = max(z_max, float(env.pos[0][2]))
        z_min = min(z_min, float(env.pos[0][2]))
        if terminated or truncated:
            break
    assert terminated or truncated
    total = float(sum(rewards))
    identity = terms_sum['progress_reward'] - terms_sum['continuous_penalty_total'] + terms_sum['terminal_outcome_reward']
    assert np.isclose(total, identity, atol=5e-4), (total, identity)
    return dict(seed=seed, behavior=behavior, planner_unavailable=False,
        stage=int(info['curriculum_stage']), tree_count=int(info['num_trees']),
        route_blocking_tree_count=int(info['route_blocking_tree_count']),
        corridor_protected=bool(info['corridor_protected']),
        map_fingerprint=map_fingerprint(env), start=start.tolist(), target=goal.tolist(),
        initial_distance_m=distances[0], initial_height_error_m=float(goal[2]-start[2]),
        steps=len(rewards), outcome=info['done_reason'], final_distance_m=distances[-1],
        minimum_distance_m=float(min(distances)), entered_1p2=bool(min(distances) < 1.2),
        entered_success_ball=bool(min(distances) < env.GOAL_TOLERANCE),
        min_z=z_min, max_z=z_max, return_raw=total,
        reward_terms={key:float(value) for key, value in terms_sum.items()})


def aggregate(rows):
    out = {}
    for behavior in ('hold', 'climb', 'direct_p', 'planned_p'):
        selection = [r for r in rows if r['behavior'] == behavior]
        valid = [r for r in selection if not r['planner_unavailable']]
        totals = defaultdict(float)
        for row in valid:
            for key, value in row['reward_terms'].items():
                totals[key] += value
        out[behavior] = dict(
            evaluated=len(valid), planner_unavailable=len(selection)-len(valid),
            outcomes=dict(Counter(r['outcome'] for r in valid)),
            guidance_entries=sum(r['entered_1p2'] for r in valid),
            successes=sum(r['outcome'] == 'success' for r in valid),
            minimum_distance_m=dict(min=float(min(r['minimum_distance_m'] for r in valid)),
                median=float(np.median([r['minimum_distance_m'] for r in valid])),
                mean=float(np.mean([r['minimum_distance_m'] for r in valid]))) if valid else {},
            mean_raw_return=float(np.mean([r['return_raw'] for r in valid])) if valid else None,
            mean_reward_terms={key:float(value/len(valid)) for key, value in totals.items()} if valid else {},
        )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--output-dir', type=Path,
        default=Path('analysis_outputs/stage0_feasibility_20261008'))
    cli = parser.parse_args()
    if cli.seeds <= 0:
        raise ValueError('--seeds must be positive')
    args = preset_args(BASE)
    historical = historical_episodes()
    env = make_stage0_env(args)
    rows = []
    try:
        for seed in range(cli.seeds):
            per_seed = []
            for behavior in ('hold', 'climb', 'direct_p', 'planned_p'):
                row = run_episode(env, seed, behavior)
                rows.append(row)
                per_seed.append(row)
            fingerprints = [r['map_fingerprint'] for r in per_seed if 'map_fingerprint' in r]
            assert all(f == fingerprints[0] for f in fingerprints)
            print(f'seed {seed}: ' + ', '.join(
                f"{r['behavior']}={r.get('outcome', 'planner_unavailable')}"
                for r in per_seed), flush=True)
    finally:
        env.close()
    initial = [r for r in rows if r['behavior'] == 'hold']
    result = dict(config=dict(preset=BASE, map_id=args.map_id, stage=0,
        execution_model=args.upper_execution_model, reward_variant=args.reward_variant,
        reward_scale=args.reward_scale, note='K1 bypasses the actual DSLPID motor-control loop.'),
        historical_training=historical,
        stage0_initial=dict(count=len(initial),
            goal_distance_m=dict(min=float(min(r['initial_distance_m'] for r in initial)),
                max=float(max(r['initial_distance_m'] for r in initial)),
                mean=float(np.mean([r['initial_distance_m'] for r in initial]))),
            target_minus_initial_z_m=dict(min=float(min(r['initial_height_error_m'] for r in initial)),
                max=float(max(r['initial_height_error_m'] for r in initial)),
                mean=float(np.mean([r['initial_height_error_m'] for r in initial]))),
            route_blocking_tree_counts=dict(Counter(r['route_blocking_tree_count'] for r in initial)),
            corridor_protected_count=sum(r['corridor_protected'] for r in initial)),
        hand_policy=aggregate(rows))
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    (cli.output_dir/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (cli.output_dir/'episodes.json').write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['stage0_initial'], indent=2))
    print(json.dumps(result['hand_policy'], indent=2))


if __name__ == '__main__':
    main()
