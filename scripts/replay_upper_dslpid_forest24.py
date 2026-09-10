"""Replay the frozen learned-upper DSLPID checkpoint in the forest GUI.

This utility performs inference only.  It does not train, save checkpoints, or
write TensorBoard logs, so it can be used as a reproducible demonstration of
the patent main embodiment.
"""

from __future__ import annotations

import argparse

import numpy as np

from gym_pybullet_drones.utils.enums import ActionType, ObservationType

from algos.td3.td3_plain import TD3Plain
from envs.forest.rewards import BaselineForestReward
from envs.learned_upper_dslpid import LearnedUpperDSLPIDForestEnv
from envs.preprocess import preprocess_state


DEFAULT_CHECKPOINT = "checkpoints/upper_dslpid_five_trees_forest24_v1/model_best"


def make_environment(*, gui: bool) -> LearnedUpperDSLPIDForestEnv:
    """Build exactly the forest24 main-embodiment evaluation environment."""

    environment = LearnedUpperDSLPIDForestEnv(
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        initial_xyzs=np.array([[-3.5, 0.0, 1.0]], dtype=np.float32),
        initial_rpys=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        pyb_freq=240,
        ctrl_freq=120,
        gui=gui,
        curriculum=False,
        num_trees=24,
        route_blocking_tree=True,
        route_blocking_tree_count=5,
        route_tree_lateral_range=0.55,
        route_tree_layout="fixed_safe_five",
        target_pos=[3.5, 0.0, 1.0],
        reward_model=BaselineForestReward(
            distance_reward_weight=0.0,
            time_penalty_weight=0.01,
        ),
        upper_control_interval=8,
        reference_sequence_length=15,
        reference_horizon_seconds=1.0,
        max_reference_speed=0.8,
        max_reference_acceleration=1.0,
        max_reference_vertical_speed=0.3,
        max_reference_vertical_acceleration=0.5,
    )
    environment.set_curriculum_stage_override(0)
    return environment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay the frozen upper-DSLPID forest24 policy without training."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    environment = make_environment(gui=not args.headless)
    try:
        agent = TD3Plain(
            state_dim=int(np.prod(environment.observation_space.shape)),
            action_dim=int(environment.action_space.shape[-1]),
            max_action=1.0,
        )
        agent.load(args.checkpoint)

        observation, _ = environment.reset(seed=args.seed)
        state = preprocess_state(observation.reshape(-1))
        total_reward = 0.0
        steps = 0
        while True:
            action = agent.select_action(state)
            observation, reward, terminated, truncated, info = environment.step(
                action.reshape(1, -1)
            )
            state = preprocess_state(observation.reshape(-1))
            total_reward += float(reward)
            steps += 1
            if terminated or truncated:
                print(
                    {
                        "seed": args.seed,
                        "success": bool(info["success"]),
                        "done_reason": info["done_reason"],
                        "goal_distance": float(info["goal_distance"]),
                        "steps_15hz": steps,
                        "return": total_reward,
                    }
                )
                break
    finally:
        environment.close()


if __name__ == "__main__":
    main()
