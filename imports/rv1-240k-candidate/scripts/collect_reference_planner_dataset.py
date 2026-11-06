"""Collect successful frozen-policy rollouts for reference-sequence imitation."""

from __future__ import annotations

import argparse
import contextlib
import heapq
import io
from pathlib import Path
import sys

import numpy as np
import pybullet as p
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.gym_pybullet_compat import ensure_gym_pybullet_envs_compat

ensure_gym_pybullet_envs_compat()

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

from algos.td3.td3_latent_affordance import TD3LatentAffordance
from envs.ForestAviary import CustomForestAviary
from envs.preprocess import preprocess_state
from high_level_planner import FUTURE_TIMES, extract_planner_input, save_reference_dataset
from models.motor_action_codec import ASYMMETRIC_RPM, MotorActionCodec, MotorPhysicalLimits


SCENARIO_CONFIGS = {
    "clear": {"num_trees": 0, "route_blocking_tree": False},
    "single": {
        "num_trees": 0,
        "route_blocking_tree": True,
        "route_blocking_tree_count": 1,
        "route_tree_fraction": 0.5,
    },
    "forest24": {
        "num_trees": 24,
        "route_blocking_tree": True,
        "route_blocking_tree_count": 1,
    },
}


def make_env(scenario: str, episode_len_sec: float):
    if scenario not in SCENARIO_CONFIGS:
        raise KeyError(f"Unknown scenario {scenario!r}.")
    with contextlib.redirect_stdout(io.StringIO()):
        return CustomForestAviary(
            obs=ObservationType.KIN,
            act=ActionType.RPM,
            pyb_freq=240,
            ctrl_freq=120,
            gui=False,
            curriculum=False,
            episode_len_sec=float(episode_len_sec),
            **SCENARIO_CONFIGS[scenario],
        )


def load_teacher(checkpoint_prefix: str, state_dim: int, device):
    agent = TD3LatentAffordance(
        state_dim=state_dim,
        action_dim=4,
        max_action=1.0,
        latent_dim=16,
        target_pos=(3.5, 0.0, 1.0),
        device=device,
    )
    agent.encoder.load_state_dict(
        torch.load(checkpoint_prefix + "_encoder", map_location=device)
    )
    agent.actor.load_state_dict(
        torch.load(checkpoint_prefix + "_actor", map_location=device)
    )
    agent.encoder.eval()
    agent.actor.eval()
    return agent


def latent_and_action(agent, state):
    tensor = torch.as_tensor(
        state, dtype=torch.float32, device=agent.device
    ).reshape(1, -1)
    with torch.no_grad():
        latent = agent.encoder(tensor)
        action = agent.actor(agent._policy_input(latent))
    return (
        latent[0].cpu().numpy().astype(np.float32),
        action[0].cpu().numpy().astype(np.float32),
    )


def make_motor_codec(env):
    codec = MotorActionCodec(
        MotorPhysicalLimits(
            min_rpm=0.0,
            hover_rpm=float(env.HOVER_RPM),
            max_rpm=float(env.MAX_RPM),
            kf=float(env.KF),
        ),
        mode=ASYMMETRIC_RPM,
    )
    env.set_motor_action_codec(codec)
    return codec


def _oracle_path(env, *, resolution: float = 0.10, safety_margin: float = 0.32):
    """Build a collision-inflated 2D A* path for teacher data only."""

    start = np.asarray(env.START_POS[:2], dtype=np.float64)
    goal = np.asarray(env.TARGET_POS[:2], dtype=np.float64)
    lower = np.array(
        [min(start[0], goal[0]) - 0.35, -float(env.FOREST_HALF_EXTENT) - 0.35]
    )
    upper = np.array(
        [max(start[0], goal[0]) + 0.35, float(env.FOREST_HALF_EXTENT) + 0.35]
    )
    shape = np.floor((upper - lower) / float(resolution)).astype(int) + 1

    def to_index(point):
        return tuple(np.rint((np.asarray(point) - lower) / resolution).astype(int))

    def to_point(index):
        return lower + np.asarray(index, dtype=np.float64) * resolution

    def blocked(index):
        if index[0] < 0 or index[1] < 0 or index[0] >= shape[0] or index[1] >= shape[1]:
            return True
        point = to_point(index)
        for spec in env._tree_specs:
            required = (
                float(spec["radius"])
                + float(env.COLLISION_R)
                + float(safety_margin)
            )
            if np.linalg.norm(point - np.asarray(spec["xy"])) <= required:
                return True
        return False

    start_index = to_index(start)
    goal_index = to_index(goal)
    frontier = [(0.0, start_index)]
    came_from = {start_index: None}
    cost = {start_index: 0.0}
    moves = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal_index:
            break
        for move in moves:
            following = (current[0] + move[0], current[1] + move[1])
            if blocked(following):
                continue
            step_cost = float(np.hypot(move[0], move[1]))
            candidate = cost[current] + step_cost
            if candidate < cost.get(following, float("inf")):
                cost[following] = candidate
                heuristic = float(np.linalg.norm(np.asarray(following) - np.asarray(goal_index)))
                heapq.heappush(frontier, (candidate + heuristic, following))
                came_from[following] = current
    if goal_index not in came_from:
        raise RuntimeError("Oracle teacher could not find a collision-free grid path.")
    indices = []
    current = goal_index
    while current is not None:
        indices.append(current)
        current = came_from[current]
    points = np.asarray([to_point(index) for index in reversed(indices)])
    points[0] = start
    points[-1] = goal

    def segment_clear(first, second):
        distance = float(np.linalg.norm(second - first))
        samples = max(2, int(np.ceil(distance / (0.5 * resolution))) + 1)
        for alpha in np.linspace(0.0, 1.0, samples):
            point = (1.0 - alpha) * first + alpha * second
            for spec in env._tree_specs:
                required = (
                    float(spec["radius"])
                    + float(env.COLLISION_R)
                    + float(safety_margin)
                )
                if np.linalg.norm(point - np.asarray(spec["xy"])) <= required:
                    return False
        return True

    simplified = [points[0]]
    index = 0
    while index < len(points) - 1:
        candidate = len(points) - 1
        while candidate > index + 1 and not segment_clear(points[index], points[candidate]):
            candidate -= 1
        simplified.append(points[candidate])
        index = candidate
    return np.asarray(simplified, dtype=np.float64)


class OracleReferencePIDTeacher:
    """Follow an oracle collision-free path with a smooth moving PID reference."""

    def __init__(self, env, *, cruise_speed: float = 0.75, acceleration: float = 1.2):
        self.env = env
        self.path_xy = _oracle_path(env)
        self.points = np.column_stack(
            [self.path_xy, np.full(len(self.path_xy), float(env.TARGET_POS[2]))]
        )
        self.segment_lengths = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        self.cumulative = np.concatenate([[0.0], np.cumsum(self.segment_lengths)])
        self.total_length = float(self.cumulative[-1])
        self.distance = 0.0
        self.speed = 0.0
        self.cruise_speed = float(cruise_speed)
        self.acceleration = float(acceleration)
        self.controller = DSLPIDControl(drone_model=env.DRONE_MODEL)
        self.controller.reset()
        self.codec = make_motor_codec(env)

    def _sample_path(self, distance):
        value = float(np.clip(distance, 0.0, self.total_length))
        upper = int(np.searchsorted(self.cumulative, value, side="right"))
        upper = min(max(upper, 1), len(self.points) - 1)
        lower = upper - 1
        segment_length = max(float(self.segment_lengths[lower]), 1e-9)
        alpha = (value - float(self.cumulative[lower])) / segment_length
        position = (1.0 - alpha) * self.points[lower] + alpha * self.points[upper]
        tangent = (self.points[upper] - self.points[lower]) / segment_length
        return position, tangent

    def action(self):
        dt = float(self.env.CTRL_TIMESTEP)
        remaining = max(0.0, self.total_length - self.distance)
        braking_speed = np.sqrt(max(0.0, 2.0 * self.acceleration * remaining))
        desired_speed = min(self.cruise_speed, braking_speed)
        speed_delta = np.clip(
            desired_speed - self.speed,
            -self.acceleration * dt,
            self.acceleration * dt,
        )
        self.speed = max(0.0, self.speed + float(speed_delta))
        self.distance = min(self.total_length, self.distance + self.speed * dt)
        target_position, tangent = self._sample_path(self.distance)
        target_velocity = tangent * self.speed
        rpm, _, _ = self.controller.computeControl(
            control_timestep=dt,
            cur_pos=np.asarray(self.env.pos[0], dtype=np.float64),
            cur_quat=np.asarray(p.getQuaternionFromEuler(self.env.rpy[0]), dtype=np.float64),
            cur_vel=np.asarray(self.env.vel[0], dtype=np.float64),
            cur_ang_vel=np.asarray(self.env.ang_v[0], dtype=np.float64),
            target_pos=target_position,
            target_vel=target_velocity,
        )
        return self.codec.rpm_to_normalized_action(rpm).astype(np.float32)


def rollout_episode(
    env,
    agent,
    seed: int,
    high_level_interval: int,
    teacher_mode: str,
):
    observation, _ = env.reset(seed=int(seed))
    state = preprocess_state(observation.reshape(-1))
    oracle_teacher = (
        OracleReferencePIDTeacher(env) if teacher_mode == "oracle_pid" else None
    )
    physical_positions = [np.asarray(env.pos[0], dtype=np.float32).copy()]
    physical_velocities = [np.asarray(env.vel[0], dtype=np.float32).copy()]
    decisions = []
    step = 0
    info = {}
    terminated = truncated = False
    while not (terminated or truncated):
        latent, actor_action = latent_and_action(agent, state)
        if step % int(high_level_interval) == 0:
            decisions.append(
                {
                    "step": int(step),
                    "input": extract_planner_input(latent, state),
                    "position": physical_positions[-1].copy(),
                }
            )
        action = (
            oracle_teacher.action()
            if oracle_teacher is not None
            else actor_action
        )
        observation, _, terminated, truncated, info = env.step(action.reshape(1, -1))
        state = preprocess_state(observation.reshape(-1))
        physical_positions.append(np.asarray(env.pos[0], dtype=np.float32).copy())
        physical_velocities.append(np.asarray(env.vel[0], dtype=np.float32).copy())
        step += 1
    return {
        "success": bool(info.get("success", False)),
        "collision": bool(info.get("collision", False)),
        "done_reason": str(info.get("done_reason", "unknown")),
        "decisions": decisions,
        "positions": np.asarray(physical_positions, dtype=np.float32),
        "velocities": np.asarray(physical_velocities, dtype=np.float32),
    }


def append_successful_episode(arrays, rollout, episode_id: int, ctrl_freq: int):
    offsets = np.rint(FUTURE_TIMES * int(ctrl_freq)).astype(np.int32)
    added = 0
    for decision in rollout["decisions"]:
        indices = int(decision["step"]) + offsets
        if int(indices[-1]) >= rollout["positions"].shape[0]:
            continue
        future_positions = rollout["positions"][indices]
        future_velocities = rollout["velocities"][indices]
        target = np.concatenate(
            [
                future_positions - decision["position"][None, :],
                future_velocities,
            ],
            axis=1,
        ).astype(np.float32)
        arrays["episode_id"].append(int(episode_id))
        arrays["step_id"].append(int(decision["step"]))
        arrays["planner_input"].append(decision["input"])
        arrays["reference_target"].append(target)
        added += 1
    return added


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/latent_only_affordance_dim16/model_final",
    )
    parser.add_argument(
        "--output",
        default="data/reference_planner/latent_affordance_margin_v2",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=tuple(SCENARIO_CONFIGS),
        default=("clear", "single", "forest24"),
    )
    parser.add_argument("--successful-episodes-per-scenario", type=int, default=20)
    parser.add_argument("--max-attempts-per-scenario", type=int, default=200)
    parser.add_argument("--episode-len-sec", type=float, default=12.0)
    parser.add_argument("--high-level-interval", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--teacher",
        choices=("oracle_pid", "frozen_actor"),
        default="oracle_pid",
        help="DSLPID path teacher is the default; frozen_actor is a diagnostic baseline.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.successful_episodes_per_scenario <= 0 or args.max_attempts_per_scenario <= 0:
        raise ValueError("Episode and attempt counts must be positive.")
    checkpoint = str(Path(args.checkpoint))
    for suffix in ("_encoder", "_actor"):
        if not Path(checkpoint + suffix).is_file():
            raise FileNotFoundError(checkpoint + suffix)
    device = torch.device("cpu")
    arrays = {name: [] for name in (
        "episode_id", "step_id", "planner_input", "reference_target"
    )}
    scenario_stats = {}
    episode_id = 0
    teacher = None
    for scenario_index, scenario in enumerate(args.scenarios):
        env = make_env(scenario, args.episode_len_sec)
        try:
            observation, _ = env.reset(seed=args.seed)
            state_dim = preprocess_state(observation.reshape(-1)).size
            if teacher is None:
                teacher = load_teacher(checkpoint, int(state_dim), device)
            success_count = 0
            attempts = 0
            collisions = 0
            samples = 0
            while (
                success_count < args.successful_episodes_per_scenario
                and attempts < args.max_attempts_per_scenario
            ):
                seed = int(args.seed) + scenario_index * 100000 + attempts
                rollout = rollout_episode(
                    env,
                    teacher,
                    seed,
                    args.high_level_interval,
                    args.teacher,
                )
                attempts += 1
                collisions += int(rollout["collision"])
                if rollout["success"]:
                    added = append_successful_episode(
                        arrays,
                        rollout,
                        episode_id,
                        env.CTRL_FREQ,
                    )
                    if added > 0:
                        success_count += 1
                        episode_id += 1
                        samples += added
                print(
                    f"{scenario}: attempt={attempts}, success={success_count}/"
                    f"{args.successful_episodes_per_scenario}, last={rollout['done_reason']}"
                )
            scenario_stats[scenario] = {
                "attempted_episodes": int(attempts),
                "successful_episodes": int(success_count),
                "collision_episodes": int(collisions),
                "samples": int(samples),
            }
            if success_count < args.successful_episodes_per_scenario:
                raise RuntimeError(
                    f"Teacher produced only {success_count} successful {scenario} episodes "
                    f"in {attempts} attempts; dataset was not written."
                )
        finally:
            env.close()

    metadata = save_reference_dataset(
        args.output,
        episode_id=arrays["episode_id"],
        step_id=arrays["step_id"],
        planner_input=arrays["planner_input"],
        reference_target=arrays["reference_target"],
        metadata={
            "teacher_checkpoint": checkpoint,
            "teacher_controller": (
                "oracle_astar_reference_plus_dslpid"
                if args.teacher == "oracle_pid"
                else "frozen_latent_only_affordance_dim16_direct_motor_actor"
            ),
            "encoder_frozen": True,
            "actor_frozen": True,
            "control_frequency_hz": 120,
            "high_level_frequency_hz": 120.0 / args.high_level_interval,
            "scenario_stats": scenario_stats,
            "successful_episodes_only": True,
        },
        split_seed=args.seed,
    )
    print(
        f"wrote {args.output}: samples={metadata['sample_count']}, "
        f"episodes={metadata['episode_count']}"
    )


if __name__ == "__main__":
    main()
