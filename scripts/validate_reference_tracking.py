from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from utils.gym_pybullet_compat import ensure_gym_pybullet_envs_compat

ensure_gym_pybullet_envs_compat()

from gym_pybullet_drones.utils.enums import ActionType, ObservationType

from algos.td3.td3_reference_tracking import TD3ReferenceTracking
from envs.ForestAviary import CustomForestAviary
from envs.preprocess import preprocess_state


def make_env():
    return CustomForestAviary(
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        initial_xyzs=np.array([[-3.5, 0.0, 1.0]], dtype=np.float32),
        pyb_freq=240,
        ctrl_freq=120,
        gui=False,
        curriculum=False,
        num_trees=0,
        route_blocking_tree=False,
        target_pos=[3.5, 0.0, 1.0],
    )


def make_agent(
    state_dim: int,
    reference_mode: str,
    actor_structure: str,
    motor_action_mode: str,
):
    return TD3ReferenceTracking(
        state_dim=state_dim,
        action_dim=4,
        max_action=1.0,
        target_position=[3.5, 0.0, 1.0],
        ctrl_freq=120,
        sequence_length=15,
        reference_horizon_seconds=1.0,
        high_level_interval=8,
        max_reference_speed=0.6,
        max_reference_acceleration=1.5,
        max_reference_vertical_speed=0.4,
        reference_mode=reference_mode,
        actor_structure=actor_structure,
        motor_action_mode=motor_action_mode,
    )


def main():
    parser = argparse.ArgumentParser(description="Gate test for the low-level reference tracker.")
    parser.add_argument("--controller", choices=("teacher", "actor"), required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--reference-mode", choices=("line", "hover"), default="line")
    parser.add_argument(
        "--actor-structure",
        choices=("plain", "residual", "structured"),
        default="plain",
    )
    parser.add_argument(
        "--motor-action-mode",
        choices=("asymmetric_rpm", "asymmetric_thrust", "legacy_projected"),
        default="asymmetric_rpm",
    )
    args = parser.parse_args()
    if args.controller == "actor" and not args.checkpoint:
        parser.error("--checkpoint is required for actor validation.")

    episode_metrics = []
    for seed in range(args.episodes):
        env = make_env()
        agent = make_agent(
            int(np.prod(env.observation_space.shape)),
            args.reference_mode,
            args.actor_structure,
            args.motor_action_mode,
        )
        if args.checkpoint:
            agent.load(args.checkpoint)
        obs, _ = env.reset(seed=seed)
        agent.reset_episode()
        agent.configure_motor_action_interface(env)
        state = preprocess_state(obs.reshape(-1))
        done = False
        steps = 0
        tracking_errors = []
        max_attitude = 0.0
        saturated = 0
        actions = 0
        info = {}
        while not done:
            if args.controller == "teacher":
                agent.prepare_runtime_context(state)
                action = agent.teacher_action_from_env(env)
                agent.record_executed_action(action)
                agent.advance_runtime_step()
            else:
                action = agent.select_action(state)
            obs, _, terminated, truncated, info = env.step(action.reshape(1, -1))
            state = preprocess_state(obs.reshape(-1))
            done = terminated or truncated
            steps += 1
            actions += action.size
            saturated += int(np.sum(np.abs(action) >= agent.max_action - 1e-3))
            tracking_errors.append(
                float(
                    np.linalg.norm(
                        env.pos[0] - agent.last_reference_sample["lookahead_position"]
                    )
                )
            )
            max_attitude = max(
                max_attitude,
                abs(float(env.rpy[0][0])),
                abs(float(env.rpy[0][1])),
            )
        episode_metrics.append(
            {
                "steps": steps,
                "done_reason": str(info.get("done_reason", "unknown")),
                "goal_distance": float(info.get("goal_distance", np.nan)),
                "tracking_rmse": float(np.sqrt(np.mean(np.square(tracking_errors)))),
                "max_roll_pitch": max_attitude,
                "motor_saturation_fraction": saturated / max(1, actions),
            }
        )
        env.close()

    unstable_reasons = {"attitude_bound", "height_bound", "collision", "xy_bound"}
    summary = {
        "controller": args.controller,
        "checkpoint": args.checkpoint,
        "reference_mode": args.reference_mode,
        "actor_structure": args.actor_structure,
        "motor_action_mode": args.motor_action_mode,
        "episodes": args.episodes,
        "mean_steps": float(np.mean([m["steps"] for m in episode_metrics])),
        "mean_tracking_rmse": float(
            np.mean([m["tracking_rmse"] for m in episode_metrics])
        ),
        "mean_max_roll_pitch": float(
            np.mean([m["max_roll_pitch"] for m in episode_metrics])
        ),
        "instability_rate": float(
            np.mean([m["done_reason"] in unstable_reasons for m in episode_metrics])
        ),
        "motor_saturation_fraction": float(
            np.mean([m["motor_saturation_fraction"] for m in episode_metrics])
        ),
        "episode_metrics": episode_metrics,
    }
    summary["gate_passed"] = bool(
        summary["mean_steps"] >= 0.9 * 1442
        and summary["mean_tracking_rmse"] <= 0.5
        and summary["instability_rate"] <= 0.02
        and summary["motor_saturation_fraction"] <= 0.05
    )
    rendered = json.dumps(summary, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        output = pathlib.Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
