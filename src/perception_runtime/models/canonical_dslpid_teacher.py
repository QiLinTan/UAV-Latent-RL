"""Deterministic DSLPID label adapters for the frozen Stage2-v1 context.

The upstream DSLPID implementation contains episode-history state.  These
adapters canonicalize that hidden state before every label so identical
observable inputs produce identical motor RPMs.  They are teacher diagnostics,
not Stage 1 controller replacements.
"""

from __future__ import annotations

import numpy as np
import pybullet as p

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl


ZERO_RATE = "zero_rate"
WORLD_ANGULAR_VELOCITY = "world_angular_velocity"
SUPPORTED_RATE_MODES = (ZERO_RATE, WORLD_ANGULAR_VELOCITY)
CANONICAL_DSLPID_WORLD_RATE_ID = "stage2_v1_canonical_dslpid_world_rate"


def previous_rpy_from_world_angular_velocity(
    cur_quat,
    cur_ang_vel,
    control_timestep: float,
) -> np.ndarray:
    """Infer the prior Euler sample using observable world angular velocity."""

    quaternion = np.asarray(cur_quat, dtype=np.float64).reshape(4)
    angular_velocity = np.asarray(cur_ang_vel, dtype=np.float64).reshape(3)
    dt = float(control_timestep)
    if dt <= 0.0:
        raise ValueError("control_timestep must be positive.")
    current_rpy = np.asarray(p.getEulerFromQuaternion(quaternion), dtype=np.float64)
    speed = float(np.linalg.norm(angular_velocity))
    if speed <= 1e-12:
        return current_rpy
    delta_quaternion = p.getQuaternionFromAxisAngle(
        angular_velocity / speed,
        -speed * dt,
    )
    _, previous_quaternion = p.multiplyTransforms(
        [0.0, 0.0, 0.0],
        delta_quaternion,
        [0.0, 0.0, 0.0],
        quaternion,
    )
    wrapped_previous = np.asarray(
        p.getEulerFromQuaternion(previous_quaternion),
        dtype=np.float64,
    )
    shortest_delta = (
        wrapped_previous - current_rpy + np.pi
    ) % (2.0 * np.pi) - np.pi
    return current_rpy + shortest_delta


class CanonicalDSLPIDTeacher:
    """Make DSLPID a deterministic function of its explicit control inputs."""

    def __init__(self, drone_model, *, rate_mode: str = WORLD_ANGULAR_VELOCITY):
        if rate_mode not in SUPPORTED_RATE_MODES:
            raise ValueError(f"Unsupported canonical rate mode {rate_mode!r}.")
        self.rate_mode = str(rate_mode)
        self.controller = DSLPIDControl(drone_model=drone_model)
        self.calls = 0
        self.last_hidden_before = None
        self.last_hidden_after = None

    def reset(self) -> None:
        self.controller.reset()
        self.calls = 0
        self.last_hidden_before = None
        self.last_hidden_after = None

    def _hidden_state(self) -> dict[str, np.ndarray | int]:
        return {
            "integral_pos_e": np.asarray(
                self.controller.integral_pos_e,
                dtype=np.float32,
            ).copy(),
            "integral_rpy_e": np.asarray(
                self.controller.integral_rpy_e,
                dtype=np.float32,
            ).copy(),
            "last_rpy": np.asarray(
                self.controller.last_rpy,
                dtype=np.float32,
            ).copy(),
            "last_rpy_e": np.asarray(
                self.controller.last_rpy_e,
                dtype=np.float32,
            ).copy(),
            "control_counter": int(self.controller.control_counter),
        }

    def computeControl(
        self,
        *,
        control_timestep,
        cur_pos,
        cur_quat,
        cur_vel,
        cur_ang_vel,
        target_pos,
        target_rpy=np.zeros(3),
        target_vel=np.zeros(3),
        target_rpy_rates=np.zeros(3),
    ):
        self.controller.reset()
        current_rpy = np.asarray(
            p.getEulerFromQuaternion(cur_quat),
            dtype=np.float64,
        )
        if self.rate_mode == ZERO_RATE:
            previous_rpy = current_rpy
        else:
            previous_rpy = previous_rpy_from_world_angular_velocity(
                cur_quat,
                cur_ang_vel,
                control_timestep,
            )
        self.controller.integral_pos_e.fill(0.0)
        self.controller.integral_rpy_e.fill(0.0)
        self.controller.last_rpy = previous_rpy.copy()
        self.controller.last_rpy_e.fill(0.0)
        self.controller.control_counter = 0
        self.last_hidden_before = self._hidden_state()
        result = self.controller.computeControl(
            control_timestep=control_timestep,
            cur_pos=cur_pos,
            cur_quat=cur_quat,
            cur_vel=cur_vel,
            cur_ang_vel=cur_ang_vel,
            target_pos=target_pos,
            target_rpy=target_rpy,
            target_vel=target_vel,
            target_rpy_rates=target_rpy_rates,
        )
        self.last_hidden_after = self._hidden_state()
        self.calls += 1
        return result
