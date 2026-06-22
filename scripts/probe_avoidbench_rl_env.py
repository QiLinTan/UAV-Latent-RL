from __future__ import annotations

import argparse

import numpy as np

from envs.avoidbench.rl_env import AvoidBenchRLEnv


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a minimal reset/step smoke test for the ROS-backed AvoidBenchRLEnv."
    )
    parser.add_argument("--namespace", default="/hummingbird", help="ROS namespace to use.")
    parser.add_argument("--steps", type=int, default=20, help="Number of random steps to execute.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for action sampling.")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive.")

    env = AvoidBenchRLEnv(namespace=args.namespace)
    rng = np.random.default_rng(args.seed)

    obs, info = env.reset()
    print("=== avoidbench rl env probe ===")
    print(f"reset_obs_shape: {obs.shape}")
    print(f"reset_obs_dtype: {obs.dtype}")
    print(f"reset_info: {info}")

    total_reward = 0.0
    done = False
    for step_idx in range(args.steps):
        action = env.sample_random_action(rng)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        print(
            f"step={step_idx:02d} "
            f"action={np.round(action, 4).tolist()} "
            f"reward={reward:.4f} done={done} "
            f"distance={info['distance']:.4f} "
            f"position={np.round(info['position'], 4)} "
            f"collision={info['collision']} "
            f"autopilot_state={info['autopilot_state']}"
        )
        if done:
            break

    print(f"final_obs_shape: {obs.shape}")
    print(f"total_reward: {total_reward:.4f}")
    print(f"final_done: {done}")
    print(f"final_info: {info}")
    print("PROBE_EXIT=0")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
