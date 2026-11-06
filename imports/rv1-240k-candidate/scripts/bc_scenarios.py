from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass

import numpy as np
import pybullet as p

from utils.gym_pybullet_compat import ensure_gym_pybullet_envs_compat

ensure_gym_pybullet_envs_compat()

from gym_pybullet_drones.utils.enums import ActionType, ObservationType

from envs.ForestAviary import CustomForestAviary
from models.reference_packet import ReferencePacket


REFERENCE_ORIGIN = np.array([-3.5, 0.0, 1.0], dtype=np.float32)
GROUP_NAMES = {
    0: "nominal",
    1: "initial_recovery",
    2: "impulse_recovery",
}


@dataclass(frozen=True)
class BCScenario:
    name: str
    category: str
    reference_kind: str = "hover"
    initial_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_angular_velocity: tuple[float, float, float] | None = None
    random_initial_angular_velocity: bool = False
    height_offset: float = 0.0
    impulse_linear_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    impulse_angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    impulse_time: float | None = None
    holdout_condition: bool = False

    @property
    def group_id(self) -> int:
        if self.category == "nominal":
            return 0
        if self.category == "initial_recovery":
            return 1
        if self.category == "impulse_recovery":
            return 2
        raise ValueError(f"Unknown scenario category {self.category!r}.")


def build_scenarios() -> list[BCScenario]:
    scenarios = [
        BCScenario("nominal_hover", "nominal"),
        BCScenario("nominal_height_step_up", "nominal", reference_kind="height_step_up"),
        BCScenario("nominal_height_step_down", "nominal", reference_kind="height_step_down"),
        BCScenario("nominal_low_speed_line", "nominal", reference_kind="line"),
        BCScenario("nominal_gentle_curve", "nominal", reference_kind="curve"),
        BCScenario("nominal_circle", "nominal", reference_kind="circle"),
    ]

    for axis_index, axis in enumerate(("roll", "pitch", "yaw")):
        for amplitude in (-0.20, -0.10, -0.05, 0.05, 0.10, 0.20):
            rpy = [0.0, 0.0, 0.0]
            rpy[axis_index] = amplitude
            scenarios.append(
                BCScenario(
                    f"initial_{axis}_{amplitude:+.2f}",
                    "initial_recovery",
                    initial_rpy=tuple(rpy),
                )
            )
    for amplitude in (-0.20, -0.10, 0.10, 0.20):
        scenarios.append(
            BCScenario(
                f"initial_mixed_rpy_{amplitude:+.2f}",
                "initial_recovery",
                initial_rpy=(amplitude, -0.8 * amplitude, 0.6 * amplitude),
            )
        )
    scenarios.extend(
        [
            BCScenario(
                "initial_angular_velocity_random",
                "initial_recovery",
                random_initial_angular_velocity=True,
            ),
            BCScenario(
                "initial_velocity_x_pos",
                "initial_recovery",
                initial_velocity=(0.4, 0.0, 0.0),
            ),
            BCScenario(
                "initial_velocity_x_neg",
                "initial_recovery",
                initial_velocity=(-0.4, 0.0, 0.0),
            ),
            BCScenario(
                "initial_velocity_y_pos",
                "initial_recovery",
                initial_velocity=(0.0, 0.4, 0.0),
            ),
            BCScenario(
                "initial_velocity_y_neg",
                "initial_recovery",
                initial_velocity=(0.0, -0.4, 0.0),
            ),
            BCScenario("initial_height_-0.20", "initial_recovery", height_offset=-0.20),
            BCScenario("initial_height_-0.10", "initial_recovery", height_offset=-0.10),
            BCScenario("initial_height_+0.10", "initial_recovery", height_offset=0.10),
            BCScenario("initial_height_+0.20", "initial_recovery", height_offset=0.20),
        ]
    )

    impulse_specs = [
        ("impulse_linear_x", (0.60, 0.0, 0.10), (0.0, 0.0, 0.0)),
        ("impulse_linear_y", (0.0, 0.60, 0.10), (0.0, 0.0, 0.0)),
        ("impulse_linear_mixed", (0.50, -0.40, 0.20), (0.0, 0.0, 0.0)),
        ("impulse_angular_roll", (0.0, 0.0, 0.0), (0.80, 0.0, 0.0)),
        ("impulse_angular_pitch", (0.0, 0.0, 0.0), (0.0, 0.80, 0.0)),
        ("impulse_angular_yaw", (0.0, 0.0, 0.0), (0.0, 0.0, 0.65)),
        ("impulse_angular_mixed", (0.0, 0.0, 0.0), (0.70, -0.60, 0.40)),
        ("impulse_combined", (0.60, -0.50, 0.25), (0.80, -0.60, 0.40)),
    ]
    for name, linear, angular in impulse_specs:
        scenarios.append(
            BCScenario(
                name,
                "impulse_recovery",
                impulse_linear_velocity=linear,
                impulse_angular_velocity=angular,
                impulse_time=3.0,
            )
        )

    scenarios.extend(
        [
            BCScenario(
                "holdout_mixed_rpy",
                "initial_recovery",
                initial_rpy=(0.12, -0.07, 0.09),
                holdout_condition=True,
            ),
            BCScenario(
                "holdout_diagonal_velocity",
                "initial_recovery",
                initial_velocity=(0.32, -0.28, 0.0),
                holdout_condition=True,
            ),
            BCScenario(
                "holdout_attitude_and_angular_velocity",
                "initial_recovery",
                initial_rpy=(-0.08, 0.11, -0.06),
                initial_angular_velocity=(0.28, -0.22, 0.18),
                holdout_condition=True,
            ),
            BCScenario(
                "holdout_combined_impulse",
                "impulse_recovery",
                reference_kind="line",
                impulse_linear_velocity=(-0.45, 0.35, -0.15),
                impulse_angular_velocity=(-0.55, 0.45, -0.30),
                impulse_time=4.25,
                holdout_condition=True,
            ),
            BCScenario(
                "holdout_reverse_arc",
                "nominal",
                reference_kind="reverse_curve",
                holdout_condition=True,
            ),
        ]
    )
    return scenarios


def _minimum_jerk(timestamps: np.ndarray, duration: float):
    u = np.clip(timestamps / duration, 0.0, 1.0)
    position = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    velocity = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / duration
    return position, velocity


def make_reference_packet(kind: str, duration: float) -> ReferencePacket:
    timestamps = np.linspace(0.0, duration + 0.1, 181, dtype=np.float64)
    smooth, smooth_rate = _minimum_jerk(timestamps, duration)
    positions = np.zeros((timestamps.size, 3), dtype=np.float64)
    velocities = np.zeros_like(positions)

    if kind == "hover":
        pass
    elif kind in {"height_step_up", "height_step_down"}:
        sign = 1.0 if kind.endswith("up") else -1.0
        transition = np.clip((timestamps - 3.0) / 1.0, 0.0, 1.0)
        transition_position = 3.0 * transition**2 - 2.0 * transition**3
        transition_velocity = np.where(
            (timestamps >= 3.0) & (timestamps <= 4.0),
            6.0 * transition - 6.0 * transition**2,
            0.0,
        )
        positions[:, 2] = sign * 0.18 * transition_position
        velocities[:, 2] = sign * 0.18 * transition_velocity
    elif kind == "line":
        positions[:, 0] = 1.8 * smooth
        velocities[:, 0] = 1.8 * smooth_rate
    elif kind in {"curve", "reverse_curve"}:
        lateral_sign = -1.0 if kind == "reverse_curve" else 1.0
        positions[:, 0] = 1.5 * smooth
        positions[:, 1] = lateral_sign * 0.25 * np.sin(np.pi * smooth)
        velocities[:, 0] = 1.5 * smooth_rate
        velocities[:, 1] = (
            lateral_sign
            * 0.25
            * np.pi
            * np.cos(np.pi * smooth)
            * smooth_rate
        )
    elif kind == "circle":
        radius = 0.30
        angle = 2.0 * np.pi * smooth
        angle_rate = 2.0 * np.pi * smooth_rate
        positions[:, 0] = radius * np.sin(angle)
        positions[:, 1] = radius * (1.0 - np.cos(angle))
        velocities[:, 0] = radius * np.cos(angle) * angle_rate
        velocities[:, 1] = radius * np.sin(angle) * angle_rate
    else:
        raise ValueError(f"Unknown reference kind {kind!r}.")

    return ReferencePacket(
        positions=positions.astype(np.float32),
        velocities=velocities.astype(np.float32),
        relative_timestamps=timestamps,
        t_gen=0.0,
        t_start=0.0,
        t_receive=0.0,
        valid_duration=duration + 0.1,
        version=1,
        frame_id="local_navigation",
        origin_position=REFERENCE_ORIGIN,
        origin_attitude=np.zeros(3, dtype=np.float32),
    )


def make_reference_window(
    full_reference: ReferencePacket,
    *,
    now: float,
    version: int,
    horizon_seconds: float = 1.0,
    sequence_length: int = 15,
) -> ReferencePacket:
    relative_timestamps = np.linspace(
        0.0,
        float(horizon_seconds),
        int(sequence_length),
        dtype=np.float64,
    )
    positions = []
    velocities = []
    for relative_time in relative_timestamps:
        sample = full_reference.sample(
            float(now) + float(relative_time),
            lookahead_seconds=0.0,
        )
        positions.append(sample["current_position"])
        velocities.append(sample["current_velocity"])
    return ReferencePacket(
        positions=np.asarray(positions, dtype=np.float32),
        velocities=np.asarray(velocities, dtype=np.float32),
        relative_timestamps=relative_timestamps,
        t_gen=float(now),
        t_start=float(now),
        t_receive=float(now),
        valid_duration=float(horizon_seconds),
        version=int(version),
        frame_id="world",
        origin_position=np.zeros(3, dtype=np.float32),
        origin_attitude=np.zeros(3, dtype=np.float32),
    )


def make_env(scenario: BCScenario, episode_len_sec: float, gui: bool = False):
    initial_xyz = REFERENCE_ORIGIN.copy()
    initial_xyz[2] += scenario.height_offset
    with contextlib.redirect_stdout(io.StringIO()):
        return CustomForestAviary(
            obs=ObservationType.KIN,
            act=ActionType.RPM,
            initial_xyzs=np.array([initial_xyz], dtype=np.float32),
            initial_rpys=np.array([scenario.initial_rpy], dtype=np.float32),
            pyb_freq=240,
            ctrl_freq=120,
            gui=gui,
            curriculum=False,
            num_trees=0,
            route_blocking_tree=False,
            target_pos=[3.5, 0.0, 1.0],
            episode_len_sec=episode_len_sec,
        )


def signed_for_seed(values, seed: int):
    values = np.asarray(values, dtype=np.float64)
    return values if int(seed) % 2 == 0 else -values


def initial_angular_velocity(scenario: BCScenario, seed: int) -> np.ndarray:
    if scenario.random_initial_angular_velocity:
        return np.random.default_rng(seed + 1771).uniform(-0.45, 0.45, size=3)
    if scenario.initial_angular_velocity is None:
        return np.zeros(3, dtype=np.float64)
    return np.asarray(scenario.initial_angular_velocity, dtype=np.float64)


def apply_initial_disturbance(env, scenario: BCScenario, seed: int):
    angular = initial_angular_velocity(scenario, seed)
    linear = np.asarray(scenario.initial_velocity, dtype=np.float64)
    if np.max(np.abs(linear)) > 0.0 or np.max(np.abs(angular)) > 0.0:
        p.resetBaseVelocity(
            env.DRONE_IDS[0],
            linearVelocity=linear,
            angularVelocity=angular,
            physicsClientId=env.CLIENT,
        )
        env._updateAndStoreKinematicInformation()
    return linear, angular


def impulse_step(scenario: BCScenario, control_frequency: int) -> int | None:
    if scenario.impulse_time is None:
        return None
    return int(round(float(scenario.impulse_time) * int(control_frequency)))


def apply_runtime_impulse(env, scenario: BCScenario, seed: int):
    linear = signed_for_seed(scenario.impulse_linear_velocity, seed)
    angular = signed_for_seed(scenario.impulse_angular_velocity, seed)
    p.resetBaseVelocity(
        env.DRONE_IDS[0],
        linearVelocity=linear,
        angularVelocity=angular,
        physicsClientId=env.CLIENT,
    )
    env._updateAndStoreKinematicInformation()
    return linear, angular


def disturbance_vector(scenario: BCScenario, seed: int) -> np.ndarray:
    if scenario.impulse_time is not None:
        linear = signed_for_seed(scenario.impulse_linear_velocity, seed)
        angular = signed_for_seed(scenario.impulse_angular_velocity, seed)
        rpy = np.zeros(3)
        height = 0.0
    else:
        linear = np.asarray(scenario.initial_velocity, dtype=np.float64)
        angular = initial_angular_velocity(scenario, seed)
        rpy = np.asarray(scenario.initial_rpy, dtype=np.float64)
        height = float(scenario.height_offset)
    return np.concatenate([rpy, linear, angular, [height]]).astype(np.float32)


def physical_state(env) -> np.ndarray:
    state = env._getDroneStateVector(0)
    return np.concatenate(
        [state[0:3], state[7:10], state[10:13], state[13:16]]
    ).astype(np.float32)


def recovery_state(
    env,
    reference_sample,
    *,
    position_threshold: float = 0.05,
    attitude_threshold: float = 0.05,
    velocity_threshold: float = 0.15,
    angular_velocity_threshold: float = 0.15,
) -> bool:
    state = env._getDroneStateVector(0)
    return bool(
        np.linalg.norm(state[0:3] - reference_sample["current_position"])
        <= position_threshold
        and np.max(np.abs(state[7:10])) <= attitude_threshold
        and np.linalg.norm(state[10:13] - reference_sample["current_velocity"])
        <= velocity_threshold
        and np.linalg.norm(state[13:16]) <= angular_velocity_threshold
    )
