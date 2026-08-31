from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
import pybullet as p
import torch

from utils.gym_pybullet_compat import ensure_gym_pybullet_envs_compat

ensure_gym_pybullet_envs_compat()

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

from envs.preprocess import preprocess_state
from models.async_reference_channel import AsyncChannelProfile, AsyncReferenceChannel
from scripts.bc_scenarios import (
    BCScenario,
    apply_initial_disturbance,
    apply_runtime_impulse,
    impulse_step,
    make_env,
    make_reference_packet,
    make_reference_window,
    physical_state,
    recovery_state,
)
from scripts.evaluate_bc_actor import make_agent


CONTROLLERS = ("L0_teacher_t3", "L1_bc_mlp_t3")
UNSTABLE_REASONS = {"attitude_bound", "height_bound", "collision", "xy_bound"}


@dataclass(frozen=True)
class PlantRobustnessProfile:
    name: str
    thrust_coefficient_scale: float = 1.0
    mass_scale: float = 1.0
    observation_delay_steps: int = 0
    action_delay_steps: int = 0
    position_noise_std: float = 0.0
    attitude_noise_std: float = 0.0
    velocity_noise_std: float = 0.0
    angular_velocity_noise_std: float = 0.0
    motor_thrust_effectiveness: tuple[float, float, float, float] = (
        1.0,
        1.0,
        1.0,
        1.0,
    )
    engineering_gate_required: bool = True

    def __post_init__(self):
        if float(self.thrust_coefficient_scale) <= 0.0:
            raise ValueError("thrust_coefficient_scale must be positive.")
        if float(self.mass_scale) <= 0.0:
            raise ValueError("mass_scale must be positive.")
        if int(self.observation_delay_steps) < 0 or int(self.action_delay_steps) < 0:
            raise ValueError("Delay steps cannot be negative.")
        if any(float(value) <= 0.0 for value in self.motor_thrust_effectiveness):
            raise ValueError("Motor thrust effectiveness values must be positive.")


def ideal_channel_profile(*, seed: int = 0) -> AsyncChannelProfile:
    return AsyncChannelProfile(
        name="ideal_15hz",
        upper_frequency_hz=15.0,
        seed=int(seed),
    )


def _noisy_observation(
    value: np.ndarray,
    profile: PlantRobustnessProfile,
    rng: np.random.Generator,
) -> np.ndarray:
    observed = np.asarray(value, dtype=np.float64).copy()
    observed[0:3] += rng.normal(0.0, profile.position_noise_std, size=3)
    observed[3:6] += rng.normal(0.0, profile.attitude_noise_std, size=3)
    observed[6:9] += rng.normal(0.0, profile.velocity_noise_std, size=3)
    observed[9:12] += rng.normal(
        0.0,
        profile.angular_velocity_noise_std,
        size=3,
    )
    return observed


def _apply_motor_effectiveness(action, codec, effectiveness):
    effectiveness = np.asarray(effectiveness, dtype=np.float64).reshape(4)
    if np.allclose(effectiveness, 1.0):
        return np.asarray(action, dtype=np.float32).reshape(4)
    rpm = codec.normalized_action_to_rpm(action)
    effective_rpm = rpm * np.sqrt(effectiveness)
    return codec.rpm_to_normalized_action(effective_rpm).astype(np.float32)


def run_dual_lower_episode(
    *,
    controller: str,
    scenario: BCScenario,
    seed: int,
    duration: float,
    checkpoint: dict,
    device,
    plant_profile: PlantRobustnessProfile,
    channel_profile: AsyncChannelProfile,
    episode_uid: int,
) -> tuple[dict, dict]:
    if controller not in CONTROLLERS:
        raise ValueError(f"Unknown controller {controller!r}.")
    env = make_env(scenario, duration)
    obs, _ = env.reset(seed=seed)
    apply_initial_disturbance(env, scenario, seed)
    agent = make_agent(env, checkpoint, device)
    agent.reset_episode()
    codec = agent.configure_motor_action_interface(env)
    agent.set_high_level_enabled(False)
    teacher = DSLPIDControl(drone_model=env.DRONE_MODEL)
    teacher.reset()

    nominal_kf = float(env.KF)
    env.KF = nominal_kf * float(plant_profile.thrust_coefficient_scale)
    if abs(float(plant_profile.mass_scale) - 1.0) > 1e-12:
        p.changeDynamics(
            env.DRONE_IDS[0],
            -1,
            mass=float(env.M) * float(plant_profile.mass_scale),
            physicsClientId=env.CLIENT,
        )

    full_reference = make_reference_packet(scenario.reference_kind, duration)
    channel = AsyncReferenceChannel(channel_profile)
    generation_interval_steps = max(
        1,
        int(round(float(env.CTRL_FREQ) / channel_profile.upper_frequency_hz)),
    )
    reference_version = 0
    event_step = impulse_step(scenario, env.CTRL_FREQ)
    recovery_active = scenario.category == "initial_recovery"
    recovery_start = 0 if recovery_active else None
    stable_count = 0
    recovery_time = None
    observed_history: deque[np.ndarray] = deque(
        maxlen=max(2, int(plant_profile.observation_delay_steps) + 2)
    )
    action_queue: deque[np.ndarray] = deque(
        [
            np.zeros(4, dtype=np.float32)
            for _ in range(int(plant_profile.action_delay_steps))
        ]
    )
    rng = np.random.default_rng(
        int(seed) * 1009
        + sum(ord(char) for char in plant_profile.name)
    )

    steps = 0
    info = {}
    was_valid = False
    ever_valid = False
    validity_transitions = 0
    expiry_transitions = 0
    resumption_transitions = 0
    fallback_steps = 0
    fallback_contraction_violations = 0
    resumption_action_jumps = []
    max_action_jump = 0.0
    previous_computed_action = np.zeros(4, dtype=np.float32)
    position_errors = []
    attitude_maxima = []
    angular_velocity_norms = []
    applied_actions = []
    recovery_active_steps = 0

    step_log = {
        "episode_uid": [],
        "step_id": [],
        "time": [],
        "reference_valid": [],
        "reference_version": [],
        "reference_age_seconds": [],
        "computed_action": [],
        "applied_action": [],
        "position_error": [],
        "rpy": [],
        "angular_velocity": [],
    }

    while True:
        # Use the exact same floating-point clock as TD3ReferenceTracking.
        # Division and multiplication can differ by one ulp and falsely make a
        # just-generated packet appear to start in the future.
        now = agent.runtime_step * agent.control_dt
        if event_step is not None and steps == event_step:
            apply_runtime_impulse(env, scenario, seed)
            recovery_active = True
            recovery_start = steps
            stable_count = 0

        if steps % generation_interval_steps == 0:
            reference_version += 1
            packet = make_reference_window(
                full_reference,
                now=now,
                version=reference_version,
                horizon_seconds=agent.reference_horizon_seconds,
                sequence_length=agent.sequence_length,
            )
            channel.submit(packet, now)
        channel.deliver(now, agent.reference_buffer)

        current_physical = physical_state(env).astype(np.float64)
        observed_history.append(current_physical)
        delay = int(plant_profile.observation_delay_steps)
        if len(observed_history) <= delay:
            delayed_physical = observed_history[0]
        else:
            delayed_physical = observed_history[-1 - delay]
        observed_physical = _noisy_observation(
            delayed_physical,
            plant_profile,
            rng,
        )
        actor_state = preprocess_state(observed_physical)
        context = agent.prepare_runtime_context(actor_state)
        reference_sample = agent.last_reference_sample
        reference_valid = bool(
            reference_sample is not None and reference_sample["valid"]
        )
        if reference_valid != was_valid:
            validity_transitions += 1
            if was_valid and not reference_valid:
                expiry_transitions += 1
            if not was_valid and reference_valid and ever_valid:
                resumption_transitions += 1
        resumed_now = not was_valid and reference_valid and ever_valid
        was_valid = reference_valid
        ever_valid = ever_valid or reference_valid

        if not reference_valid:
            fallback_steps += 1
            computed_action = agent.fallback_controller(agent.previous_action)
            if np.linalg.norm(computed_action) > np.linalg.norm(agent.previous_action) + 1e-7:
                fallback_contraction_violations += 1
        elif controller == "L0_teacher_t3":
            rpm, _, _ = teacher.computeControl(
                control_timestep=env.CTRL_TIMESTEP,
                cur_pos=observed_physical[0:3],
                cur_quat=np.asarray(
                    p.getQuaternionFromEuler(observed_physical[3:6]),
                    dtype=np.float64,
                ),
                cur_vel=observed_physical[6:9],
                cur_ang_vel=observed_physical[9:12],
                target_pos=reference_sample["lookahead_position"],
                target_vel=reference_sample["lookahead_velocity"],
            )
            computed_action = codec.rpm_to_normalized_action(rpm).astype(np.float32)
        else:
            computed_action = agent.action_from_context(context)

        action_jump = float(
            np.max(np.abs(computed_action - previous_computed_action))
        )
        max_action_jump = max(max_action_jump, action_jump)
        if resumed_now:
            resumption_action_jumps.append(action_jump)
        previous_computed_action = computed_action.copy()

        if int(plant_profile.action_delay_steps) > 0:
            action_queue.append(computed_action.copy())
            delayed_action = action_queue.popleft()
        else:
            delayed_action = computed_action.copy()
        applied_action = _apply_motor_effectiveness(
            delayed_action,
            codec,
            plant_profile.motor_thrust_effectiveness,
        )

        ideal_reference = full_reference.sample(now, lookahead_seconds=0.0)
        current_error = float(
            np.linalg.norm(current_physical[0:3] - ideal_reference["current_position"])
        )
        if recovery_active:
            recovery_active_steps += 1
            stable_now = recovery_state(env, ideal_reference)
            stable_count = stable_count + 1 if stable_now else 0
            if stable_count >= 30:
                first_stable_step = steps - stable_count + 1
                recovery_time = (
                    first_stable_step - int(recovery_start)
                ) / float(env.CTRL_FREQ)
                recovery_active = False

        position_errors.append(current_error)
        attitude_maxima.append(float(np.max(np.abs(current_physical[3:6]))))
        angular_velocity_norms.append(float(np.linalg.norm(current_physical[9:12])))
        applied_actions.append(applied_action)
        step_log["episode_uid"].append(episode_uid)
        step_log["step_id"].append(steps)
        step_log["time"].append(now)
        step_log["reference_valid"].append(reference_valid)
        step_log["reference_version"].append(
            0 if reference_sample is None else reference_sample["version"]
        )
        step_log["reference_age_seconds"].append(
            0.0 if reference_sample is None else reference_sample["age_seconds"]
        )
        step_log["computed_action"].append(computed_action)
        step_log["applied_action"].append(applied_action)
        step_log["position_error"].append(current_error)
        step_log["rpy"].append(current_physical[3:6])
        step_log["angular_velocity"].append(current_physical[9:12])

        next_obs, _, terminated, truncated, info = env.step(
            applied_action.reshape(1, -1)
        )
        del next_obs
        agent.record_executed_action(applied_action)
        agent.advance_runtime_step()
        steps += 1
        if terminated or truncated:
            break

    reason = str(info.get("done_reason", "unknown"))
    position_errors_value = np.asarray(position_errors, dtype=np.float64)
    applied_actions_value = np.asarray(applied_actions, dtype=np.float64)
    recovery_required = scenario.category in {"initial_recovery", "impulse_recovery"}
    functional_success = bool(
        reason == "timeout"
        and reason not in UNSTABLE_REASONS
        and float(np.max(position_errors_value)) <= 0.60
        and float(np.max(attitude_maxima)) <= 0.70
        and (not recovery_required or recovery_time is not None)
    )
    result = {
        "episode_uid": int(episode_uid),
        "controller": controller,
        "scenario": scenario.name,
        "category": scenario.category,
        "reference_kind": scenario.reference_kind,
        "seed": int(seed),
        "duration": float(duration),
        "steps": int(steps),
        "done_reason": reason,
        "full_horizon": reason == "timeout",
        "unstable": reason in UNSTABLE_REASONS,
        "functional_success": functional_success,
        "recovery_required": recovery_required,
        "recovery_time": recovery_time,
        "max_position_error": float(np.max(position_errors_value)),
        "position_rmse": float(np.sqrt(np.mean(np.square(position_errors_value)))),
        "steady_state_position_error": float(
            np.mean(position_errors_value[-min(120, steps) :])
        ),
        "max_abs_attitude": float(np.max(attitude_maxima)),
        "max_angular_velocity": float(np.max(angular_velocity_norms)),
        "max_action_jump": float(max_action_jump),
        "max_resumption_action_jump": (
            0.0 if not resumption_action_jumps else float(max(resumption_action_jumps))
        ),
        "normalized_action_saturation_fraction": float(
            np.mean(np.abs(applied_actions_value) >= 1.0 - 1e-6)
        ),
        "reference_valid_fraction": float(1.0 - fallback_steps / max(1, steps)),
        "fallback_steps": int(fallback_steps),
        "fallback_contraction_violations": int(fallback_contraction_violations),
        "validity_transitions": int(validity_transitions),
        "expiry_transitions": int(expiry_transitions),
        "resumption_transitions": int(resumption_transitions),
        "recovery_active_steps": int(recovery_active_steps),
        "plant_profile": asdict(plant_profile),
        "channel_profile": asdict(channel_profile),
        "channel_metrics": channel.metrics(),
        "buffer_metrics": {
            "accepted_packets": int(agent.reference_buffer.accepted_packets),
            "rejected_packets": int(agent.reference_buffer.rejected_packets),
            "rejected_stale_version": int(
                agent.reference_buffer.rejected_stale_version
            ),
            "rejected_out_of_order": int(
                agent.reference_buffer.rejected_out_of_order
            ),
        },
        "teacher_takeover_used": False,
    }
    env.close()
    scalar_dtypes = {
        "episode_uid": np.int32,
        "step_id": np.int32,
        "time": np.float32,
        "reference_valid": np.bool_,
        "reference_version": np.int32,
        "reference_age_seconds": np.float32,
        "position_error": np.float32,
    }
    arrays = {
        name: np.asarray(values, dtype=scalar_dtypes.get(name, np.float32))
        for name, values in step_log.items()
    }
    return result, arrays


def merge_logs(logs: list[dict]) -> dict:
    if not logs:
        return {}
    return {
        name: np.concatenate([log[name] for log in logs], axis=0)
        for name in logs[0]
    }


def summarize_results(results: list[dict]) -> dict:
    if not results:
        return {"episode_count": 0}
    recoveries = [item for item in results if item["recovery_required"]]
    nominal = [item for item in results if not item["recovery_required"]]
    return {
        "episode_count": len(results),
        "functional_success_rate": float(
            np.mean([item["functional_success"] for item in results])
        ),
        "nominal_success_rate": (
            None
            if not nominal
            else float(np.mean([item["functional_success"] for item in nominal]))
        ),
        "disturbance_recovery_success_rate": (
            None
            if not recoveries
            else float(np.mean([item["functional_success"] for item in recoveries]))
        ),
        "instability_rate": float(np.mean([item["unstable"] for item in results])),
        "full_horizon_rate": float(
            np.mean([item["full_horizon"] for item in results])
        ),
        "mean_position_rmse": float(
            np.mean([item["position_rmse"] for item in results])
        ),
        "max_position_error": float(
            np.max([item["max_position_error"] for item in results])
        ),
        "max_abs_attitude": float(
            np.max([item["max_abs_attitude"] for item in results])
        ),
        "max_angular_velocity": float(
            np.max([item["max_angular_velocity"] for item in results])
        ),
        "mean_reference_valid_fraction": float(
            np.mean([item["reference_valid_fraction"] for item in results])
        ),
        "total_fallback_steps": int(
            np.sum([item["fallback_steps"] for item in results])
        ),
        "total_expiry_transitions": int(
            np.sum([item["expiry_transitions"] for item in results])
        ),
        "total_resumption_transitions": int(
            np.sum([item["resumption_transitions"] for item in results])
        ),
        "max_resumption_action_jump": float(
            np.max([item["max_resumption_action_jump"] for item in results])
        ),
        "fallback_contraction_violations": int(
            np.sum([item["fallback_contraction_violations"] for item in results])
        ),
    }
