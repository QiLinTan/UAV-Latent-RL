"""Evaluate the frozen latent-only checkpoint through a logical async PID stack.

This is deliberately an evaluation-only prototype: the environment, reward,
Encoder and Actor weights are not modified.  The old Actor is a direct-motor
policy, so a small documented adapter converts its motor differential into a
bounded velocity residual before the ReferenceGenerator produces a trajectory.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from dataclasses import asdict, dataclass
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
    HighLevelDecision,
    LowLevelReferenceSelector,
    ReferenceGenerator,
)
from utils.gym_pybullet_compat import ensure_gym_pybullet_envs_compat

ensure_gym_pybullet_envs_compat()

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

from algos.td3.td3_latent_affordance import TD3LatentAffordance
from envs.ForestAviary import CustomForestAviary
from envs.preprocess import preprocess_state
from models.motor_action_codec import ASYMMETRIC_RPM, MotorActionCodec, MotorPhysicalLimits


@dataclass(frozen=True)
class ExperimentProfile:
    name: str
    high_frequency_hz: float
    delay_min_seconds: float = 0.0
    delay_max_seconds: float = 0.0
    skip_probability: float = 0.0


def default_profiles() -> tuple[ExperimentProfile, ...]:
    return (
        # A uses a high-level start every lower period; B-D are independently
        # clocked at 15 Hz and never wait for an upper result.
        ExperimentProfile("A_sync_baseline", 120.0),
        ExperimentProfile("B_async_no_delay_15hz", 15.0),
        ExperimentProfile("C_async_delay_0_20ms", 15.0, 0.0, 0.020),
        ExperimentProfile("C_async_delay_0_50ms", 15.0, 0.0, 0.050),
        ExperimentProfile("C_async_delay_0_100ms", 15.0, 0.0, 0.100),
        ExperimentProfile("D_async_skip_10pct", 15.0, 0.0, 0.0, 0.10),
    )


def _quiet_env(**kwargs):
    # pybullet prints connection diagnostics that obscure a compact experiment
    # result, but we do not change environment construction or its reward.
    with contextlib.redirect_stdout(io.StringIO()):
        return CustomForestAviary(**kwargs)


class FrozenLatentPolicyReferenceAdapter:
    """Use the existing 16D policy without reinterpreting it as a trained waypoint head.

    The legacy Actor outputs four motor commands.  It therefore cannot be made
    a waypoint policy without training a different semantic output head.  This
    adapter retains its task latent in each packet and converts motor
    differentials into a deliberately small velocity residual around a
    goal-directed command.  It makes the asynchronous boundary testable while
    keeping the original network and checkpoint untouched.
    """

    def __init__(self, agent: TD3LatentAffordance, *, max_speed: float = 0.6):
        self.agent = agent
        self.max_speed = float(max_speed)

    def decision(self, state: np.ndarray, env) -> HighLevelDecision:
        tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.agent.device
        ).reshape(1, -1)
        with torch.no_grad():
            latent = self.agent.encoder(tensor)
            motor_action = self.agent.actor(self.agent._policy_input(latent))
        latent_value = latent.squeeze(0).cpu().numpy().astype(np.float32)
        motor = motor_action.squeeze(0).cpu().numpy().astype(np.float32)

        position = np.asarray(env.pos[0], dtype=np.float32)
        goal = np.asarray(env.TARGET_POS, dtype=np.float32)
        delta = goal - position
        planar_norm = float(np.linalg.norm(delta[:2]))
        nominal = np.zeros(3, dtype=np.float32)
        if planar_norm > 1e-6:
            nominal[:2] = delta[:2] / planar_norm * min(self.max_speed, planar_norm)
        nominal[2] = float(np.clip(delta[2], -0.25, 0.25))

        # Motor mixing is only a low-gain residual: direct motor values are not
        # falsely claimed to be a trained Cartesian-action policy.
        body_residual = np.array(
            [
                0.25 * (motor[0] + motor[1] - motor[2] - motor[3]),
                0.25 * (-motor[0] + motor[1] + motor[2] - motor[3]),
            ],
            dtype=np.float32,
        )
        yaw = float(env.rpy[0][2])
        rotation = np.array(
            [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]],
            dtype=np.float32,
        )
        residual_world = rotation @ body_residual
        residual = np.array(
            [0.08 * residual_world[0], 0.08 * residual_world[1], 0.025 * motor.mean()],
            dtype=np.float32,
        )
        desired_velocity = nominal + residual
        planar = float(np.linalg.norm(desired_velocity[:2]))
        if planar > self.max_speed:
            desired_velocity[:2] *= self.max_speed / planar
        desired_velocity[2] = np.clip(desired_velocity[2], -0.35, 0.35)
        return HighLevelDecision(
            target_position=goal,
            desired_velocity=desired_velocity,
            latent=latent_value,
            policy_action=motor,
        )


def load_frozen_agent(checkpoint_prefix: str, state_dim: int, device: torch.device):
    """Load only inference weights; no optimiser state or training is touched."""

    agent = TD3LatentAffordance(
        state_dim=state_dim,
        action_dim=4,
        max_action=1.0,
        latent_dim=16,
        target_pos=(3.5, 0.0, 1.0),
        device=device,
    )
    prefix = str(checkpoint_prefix)
    agent.encoder.load_state_dict(torch.load(prefix + "_encoder", map_location=device))
    agent.actor.load_state_dict(torch.load(prefix + "_actor", map_location=device))
    agent.encoder.eval()
    agent.actor.eval()
    return agent


def make_motor_codec(env) -> MotorActionCodec:
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


def run_episode(
    *,
    profile: ExperimentProfile,
    seed: int,
    checkpoint_prefix: str,
    episode_len_sec: float,
    num_trees: int,
    route_blocking_tree: bool,
    device: torch.device,
) -> dict:
    env = _quiet_env(
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        pyb_freq=240,
        ctrl_freq=120,
        gui=False,
        curriculum=False,
        num_trees=int(num_trees),
        route_blocking_tree=bool(route_blocking_tree),
        episode_len_sec=float(episode_len_sec),
    )
    try:
        observation, _ = env.reset(seed=int(seed))
        state = preprocess_state(observation.reshape(-1))
        agent = load_frozen_agent(
            checkpoint_prefix,
            int(state.size),
            device,
        )
        policy = FrozenLatentPolicyReferenceAdapter(agent)
        codec = make_motor_codec(env)
        teacher = DSLPIDControl(drone_model=env.DRONE_MODEL)
        teacher.reset()
        buffer = AsyncBuffer()
        generator = ReferenceGenerator(
            horizon_seconds=0.5,
            sequence_length=16,
            max_speed=0.6,
            max_acceleration=1.2,
            max_vertical_speed=0.35,
        )
        scheduler = AsyncScheduler(
            AsyncSchedulerConfig(
                high_frequency_hz=profile.high_frequency_hz,
                delay_min_seconds=profile.delay_min_seconds,
                delay_max_seconds=profile.delay_max_seconds,
                skip_probability=profile.skip_probability,
                seed=int(seed),
            ),
            generator,
            buffer,
        )
        lower = LowLevelReferenceSelector(env.pos[0])
        reward_sum = 0.0
        stale_flags: list[bool] = []
        packet_ages: list[float] = []
        reference_switch_errors: list[float] = []
        step_count = 0
        info = {}
        terminated = truncated = False

        while not (terminated or truncated):
            now = step_count * float(env.CTRL_TIMESTEP)
            state_snapshot = state.copy()
            scheduler.advance(
                now,
                lambda _t_generate: policy.decision(state_snapshot, env),
                fallback_position=lower.last_position,
                fallback_velocity=np.zeros(3, dtype=np.float32),
            )
            sample, fallback = lower.select(buffer, now)
            target_position = lower.last_position
            target_velocity = lower.last_velocity
            if sample is not None:
                packet_ages.append(sample.packet_age)
            stale_flags.append(bool(fallback and buffer.latest_packet is not None))
            if scheduler.switching_errors:
                reference_switch_errors.append(scheduler.switching_errors[-1])

            rpm, _, _ = teacher.computeControl(
                control_timestep=float(env.CTRL_TIMESTEP),
                cur_pos=np.asarray(env.pos[0], dtype=np.float64),
                cur_quat=np.asarray(p.getQuaternionFromEuler(env.rpy[0]), dtype=np.float64),
                cur_vel=np.asarray(env.vel[0], dtype=np.float64),
                cur_ang_vel=np.asarray(env.ang_v[0], dtype=np.float64),
                target_pos=np.asarray(target_position, dtype=np.float64),
                target_vel=np.asarray(target_velocity, dtype=np.float64),
            )
            action = codec.rpm_to_normalized_action(rpm).astype(np.float32)
            observation, reward, terminated, truncated, info = env.step(action.reshape(1, -1))
            state = preprocess_state(observation.reshape(-1))
            reward_sum += float(reward)
            step_count += 1

        lower_metrics = lower.metrics()
        scheduler_metrics = scheduler.metrics()
        final_distance = float(np.linalg.norm(np.asarray(env.TARGET_POS) - np.asarray(env.pos[0])))
        return {
            "seed": int(seed),
            "steps": int(step_count),
            "reward": float(reward_sum),
            "success": bool(info.get("success", False)),
            "collision": bool(info.get("collision", False)),
            "done_reason": str(info.get("done_reason", "unknown")),
            "final_distance": final_distance,
            "mean_packet_age_seconds": float(np.mean(packet_ages)) if packet_ages else 0.0,
            "max_packet_age_seconds": float(np.max(packet_ages)) if packet_ages else 0.0,
            "stale_ratio_from_trace": float(np.mean(stale_flags)) if stale_flags else 0.0,
            "reference_switching_error_from_trace": (
                float(np.max(reference_switch_errors)) if reference_switch_errors else 0.0
            ),
            "low_level": lower_metrics,
            "scheduler": scheduler_metrics,
            "buffer": {
                "accepted_packets": int(buffer.accepted_packets),
                "rejected_packets": int(buffer.rejected_packets),
                "rejected_versions": int(buffer.rejected_versions),
            },
        }
    finally:
        env.close()


def summarize(episodes: list[dict]) -> dict:
    if not episodes:
        return {"episode_count": 0}

    def mean(path: str) -> float:
        values = []
        for episode in episodes:
            value = episode
            for key in path.split("."):
                value = value[key]
            values.append(float(value))
        return float(np.mean(values))

    return {
        "episode_count": len(episodes),
        "success_rate": mean("success"),
        "collision_rate": mean("collision"),
        "mean_reward": mean("reward"),
        "mean_final_distance": mean("final_distance"),
        "mean_packet_age_seconds": mean("low_level.mean_packet_age_seconds"),
        "p95_packet_age_seconds": mean("low_level.p95_packet_age_seconds"),
        "mean_stale_ratio": mean("low_level.stale_ratio"),
        "max_reference_switching_error": float(max(
            episode["scheduler"]["max_reference_switching_error"]
            for episode in episodes
        )),
        "low_level_interruptions": int(sum(
            episode["low_level"]["low_level_interruptions"] for episode in episodes
        )),
        "total_stale_packet_cycles": int(sum(
            episode["low_level"]["stale_packet_cycles"] for episode in episodes
        )),
        "total_missing_packet_cycles": int(sum(
            episode["low_level"]["missing_packet_cycles"] for episode in episodes
        )),
        "total_high_level_skips": int(sum(
            episode["scheduler"]["high_level_skips"] for episode in episodes
        )),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/latent_only_affordance_dim16/model_final",
        help="Prefix of the frozen latent_only_affordance_dim16 checkpoint.",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-len-sec", type=float, default=12.0)
    parser.add_argument(
        "--num-trees",
        type=int,
        default=0,
        help="0 isolates the async/PID mechanism; use 24 for the original forest difficulty.",
    )
    parser.add_argument("--route-blocking-tree", action="store_true")
    parser.add_argument(
        "--output",
        default="analysis_outputs/latent_async_pid/async_results.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    prefix = Path(args.checkpoint)
    for suffix in ("_encoder", "_actor"):
        if not Path(str(prefix) + suffix).is_file():
            raise FileNotFoundError(f"Frozen checkpoint component is missing: {prefix}{suffix}")

    # Evaluation is intentionally CPU-only: no training occurs, and this keeps
    # the prototype reproducible on workstations without a compatible CUDA
    # driver.
    device = torch.device("cpu")
    result = {
        "checkpoint": str(prefix),
        "checkpoint_loaded_read_only": True,
        "latent_dim": 16,
        "controller": "DSLPIDControl",
        "low_frequency_hz": 120.0,
        "environment": {
            "num_trees": int(args.num_trees),
            "route_blocking_tree": bool(args.route_blocking_tree),
            "episode_len_sec": float(args.episode_len_sec),
        },
        "profiles": {},
    }
    for profile in default_profiles():
        episodes = [
            run_episode(
                profile=profile,
                seed=int(args.seed) + index,
                checkpoint_prefix=str(prefix),
                episode_len_sec=args.episode_len_sec,
                num_trees=args.num_trees,
                route_blocking_tree=args.route_blocking_tree,
                device=device,
            )
            for index in range(args.episodes)
        ]
        result["profiles"][profile.name] = {
            "config": asdict(profile),
            "summary": summarize(episodes),
            "episodes": episodes,
        }
        summary = result["profiles"][profile.name]["summary"]
        print(
            f"{profile.name}: success={summary['success_rate']:.2%}, "
            f"final_distance={summary['mean_final_distance']:.3f}, "
            f"age={summary['mean_packet_age_seconds'] * 1000.0:.1f}ms, "
            f"stale={summary['mean_stale_ratio']:.2%}, "
            f"interruptions={summary['low_level_interruptions']}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
