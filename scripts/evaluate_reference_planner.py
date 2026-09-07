"""Compare the legacy latent-Actor adapter with the learned sequence planner."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import pybullet as p
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from async_control import (
    AsyncBuffer,
    AsyncScheduler,
    AsyncSchedulerConfig,
    LowLevelReferenceSelector,
    ReferenceGenerator,
)
from high_level_planner import (
    FUTURE_TIMES,
    ReferenceSequenceDecision,
    ReferenceSequencePacketGenerator,
    ReferenceSequencePlanner,
    extract_planner_input,
)
from utils.gym_pybullet_compat import ensure_gym_pybullet_envs_compat

ensure_gym_pybullet_envs_compat()

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

from envs.ForestAviary import CustomForestAviary
from envs.preprocess import preprocess_state
from scripts.evaluate_latent_async_pid import (
    FrozenLatentPolicyReferenceAdapter,
    load_frozen_agent,
    make_motor_codec,
)


SCENARIOS = {
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
CONTROLLERS = ("actor_adapter", "reference_planner")


class FrozenLatentReferencePlannerPolicy:
    """Frozen Encoder followed by the separately trained sequence planner."""

    def __init__(self, latent_agent, planner):
        self.latent_agent = latent_agent
        self.planner = planner

    def decision(self, state, env):
        tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.latent_agent.device,
        ).reshape(1, -1)
        with torch.no_grad():
            latent = self.latent_agent.encoder(tensor)[0]
        latent_value = latent.cpu().numpy().astype(np.float32)
        planner_input = extract_planner_input(latent_value, state)
        prediction = self.planner.predict_sequence(planner_input)
        return ReferenceSequenceDecision(
            position_offsets=prediction[:, :3],
            velocities=prediction[:, 3:],
            relative_times=FUTURE_TIMES,
            current_velocity=np.asarray(env.vel[0], dtype=np.float32),
            latent=latent_value,
        )


def load_planner(path, device):
    checkpoint = torch.load(path, map_location=device)
    planner = ReferenceSequencePlanner(
        input_dim=int(checkpoint.get("input_dim", 27)),
        num_points=int(checkpoint.get("num_points", 5)),
    ).to(device)
    planner.load_state_dict(checkpoint["model_state_dict"])
    planner.eval()
    return planner, checkpoint


def make_env(scenario, episode_len_sec):
    with contextlib.redirect_stdout(io.StringIO()):
        return CustomForestAviary(
            obs=ObservationType.KIN,
            act=ActionType.RPM,
            pyb_freq=240,
            ctrl_freq=120,
            gui=False,
            curriculum=False,
            episode_len_sec=float(episode_len_sec),
            **SCENARIOS[scenario],
        )


def run_episode(
    *,
    scenario,
    controller,
    seed,
    checkpoint_prefix,
    planner_checkpoint,
    episode_len_sec,
    delay_max_seconds,
    device,
):
    env = make_env(scenario, episode_len_sec)
    try:
        observation, _ = env.reset(seed=int(seed))
        state = preprocess_state(observation.reshape(-1))
        latent_agent = load_frozen_agent(
            checkpoint_prefix,
            int(state.size),
            device,
        )
        if controller == "actor_adapter":
            upper_policy = FrozenLatentPolicyReferenceAdapter(latent_agent)
            packet_generator = ReferenceGenerator(
                horizon_seconds=0.5,
                sequence_length=16,
                max_speed=0.6,
                max_acceleration=1.2,
                max_vertical_speed=0.35,
            )
        elif controller == "reference_planner":
            planner, _ = load_planner(planner_checkpoint, device)
            upper_policy = FrozenLatentReferencePlannerPolicy(latent_agent, planner)
            packet_generator = ReferenceSequencePacketGenerator(
                max_speed=0.8,
                max_acceleration=2.0,
                max_vertical_speed=0.4,
            )
        else:
            raise ValueError(f"Unknown controller {controller!r}.")

        codec = make_motor_codec(env)
        pid = DSLPIDControl(drone_model=env.DRONE_MODEL)
        pid.reset()
        buffer = AsyncBuffer()
        scheduler = AsyncScheduler(
            AsyncSchedulerConfig(
                high_frequency_hz=15.0,
                delay_min_seconds=0.0,
                delay_max_seconds=float(delay_max_seconds),
                skip_probability=0.0,
                seed=int(seed),
            ),
            packet_generator,
            buffer,
        )
        lower = LowLevelReferenceSelector(env.pos[0])
        step = 0
        reward_sum = 0.0
        terminated = truncated = False
        info = {}
        tracking_errors = []
        path_length = 0.0
        minimum_clearance = float("inf")
        previous_position = np.asarray(env.pos[0], dtype=np.float64).copy()
        while not (terminated or truncated):
            now = step * float(env.CTRL_TIMESTEP)
            state_snapshot = state.copy()
            scheduler.advance(
                now,
                lambda _t_generate: upper_policy.decision(state_snapshot, env),
                fallback_position=lower.last_position,
                fallback_velocity=np.zeros(3, dtype=np.float32),
            )
            lower.select(buffer, now)
            tracking_errors.append(
                float(np.linalg.norm(np.asarray(env.pos[0]) - lower.last_position))
            )
            rpm, _, _ = pid.computeControl(
                control_timestep=float(env.CTRL_TIMESTEP),
                cur_pos=np.asarray(env.pos[0], dtype=np.float64),
                cur_quat=np.asarray(p.getQuaternionFromEuler(env.rpy[0]), dtype=np.float64),
                cur_vel=np.asarray(env.vel[0], dtype=np.float64),
                cur_ang_vel=np.asarray(env.ang_v[0], dtype=np.float64),
                target_pos=np.asarray(lower.last_position, dtype=np.float64),
                target_vel=np.asarray(lower.last_velocity, dtype=np.float64),
            )
            action = codec.rpm_to_normalized_action(rpm).astype(np.float32)
            observation, reward, terminated, truncated, info = env.step(action.reshape(1, -1))
            state = preprocess_state(observation.reshape(-1))
            position = np.asarray(env.pos[0], dtype=np.float64)
            path_length += float(np.linalg.norm(position - previous_position))
            previous_position = position.copy()
            clearance = float(info.get("min_tree_clearance", float("inf")))
            minimum_clearance = min(minimum_clearance, clearance)
            reward_sum += float(reward)
            step += 1

        return {
            "scenario": scenario,
            "controller": controller,
            "seed": int(seed),
            "steps": int(step),
            "success": bool(info.get("success", False)),
            "collision": bool(info.get("collision", False)),
            "done_reason": str(info.get("done_reason", "unknown")),
            "reward": float(reward_sum),
            "final_distance": float(np.linalg.norm(np.asarray(env.TARGET_POS) - env.pos[0])),
            "minimum_clearance": (
                None if not np.isfinite(minimum_clearance) else float(minimum_clearance)
            ),
            "path_length": float(path_length),
            "tracking_rmse": float(np.sqrt(np.mean(np.square(tracking_errors)))),
            "low_level": lower.metrics(),
            "scheduler": scheduler.metrics(),
        }
    finally:
        env.close()


def summarize(episodes):
    def mean(name):
        return float(np.mean([float(episode[name]) for episode in episodes]))

    clearances = [
        episode["minimum_clearance"]
        for episode in episodes
        if episode["minimum_clearance"] is not None
    ]
    return {
        "episodes": len(episodes),
        "success_rate": mean("success"),
        "collision_rate": mean("collision"),
        "mean_reward": mean("reward"),
        "mean_final_distance": mean("final_distance"),
        "mean_minimum_clearance": (
            None if not clearances else float(np.mean(clearances))
        ),
        "worst_minimum_clearance": (
            None if not clearances else float(np.min(clearances))
        ),
        "mean_path_length": mean("path_length"),
        "mean_tracking_rmse": mean("tracking_rmse"),
        "mean_packet_age_seconds": float(np.mean([
            episode["low_level"]["mean_packet_age_seconds"] for episode in episodes
        ])),
        "mean_stale_ratio": float(np.mean([
            episode["low_level"]["stale_ratio"] for episode in episodes
        ])),
        "max_reference_switching_error": float(max(
            episode["scheduler"]["max_reference_switching_error"] for episode in episodes
        )),
        "low_level_interruptions": int(sum(
            episode["low_level"]["low_level_interruptions"] for episode in episodes
        )),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/latent_only_affordance_dim16/model_final",
    )
    parser.add_argument(
        "--planner",
        default="checkpoints/reference_planner/latent_affordance_margin_v2/planner_final.pt",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--episode-len-sec", type=float, default=12.0)
    parser.add_argument("--delay-max-ms", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument(
        "--output",
        default="analysis_outputs/reference_planner/evaluation.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.episodes <= 0 or args.delay_max_ms < 0.0:
        raise ValueError("Episode count must be positive and delay must be non-negative.")
    device = torch.device("cpu")
    _, planner_metadata = load_planner(args.planner, device)
    results = {
        "latent_checkpoint": str(Path(args.checkpoint)),
        "planner_checkpoint": str(Path(args.planner)),
        "planner_training_epoch": int(planner_metadata.get("epoch", 0)),
        "high_level_frequency_hz": 15.0,
        "low_level_frequency_hz": 120.0,
        "delay_range_seconds": [0.0, float(args.delay_max_ms) / 1000.0],
        "scenarios": {},
    }
    for scenario_index, scenario in enumerate(SCENARIOS):
        results["scenarios"][scenario] = {}
        for controller in CONTROLLERS:
            episodes = [
                run_episode(
                    scenario=scenario,
                    controller=controller,
                    seed=int(args.seed) + scenario_index * 1000 + index,
                    checkpoint_prefix=args.checkpoint,
                    planner_checkpoint=args.planner,
                    episode_len_sec=args.episode_len_sec,
                    delay_max_seconds=float(args.delay_max_ms) / 1000.0,
                    device=device,
                )
                for index in range(args.episodes)
            ]
            summary = summarize(episodes)
            results["scenarios"][scenario][controller] = {
                "summary": summary,
                "episodes": episodes,
            }
            print(
                f"{scenario}/{controller}: success={summary['success_rate']:.1%}, "
                f"collision={summary['collision_rate']:.1%}, "
                f"final={summary['mean_final_distance']:.3f}m, "
                f"tracking={summary['mean_tracking_rmse']:.3f}m"
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
