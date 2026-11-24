"""Isolated DSL-form attitude feedback and CF2X motor mixing.

The installed private function combines mutation with clipping and returns
RPM only. Its error law, gains and mixer are reproduced here, with explicit
state, diagnostics, initialized/wrapped Euler differences and conditional
integral anti-windup. No installed source or global monkey patch is used.
Unsaturated numerical equivalence to that exact function is tested.
"""
from __future__ import annotations

import hashlib
import inspect

import numpy as np
import pybullet as p
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel

from ..contracts import (DSLPID_SHA256, ControlState, ControlSpec, ModelParameters,
                         body_to_world, force_to_attitude, collective_force_to_pwm,
                         scalar, vector, wrap_pi, yaw_from_quaternion)


class DSLAttitudeAdapter:
    implementation = 'isolated_dsl_form_initialized_wrapped_conditional_aw_v1'

    def __init__(self, model=None, spec=None):
        self.model = model or ModelParameters.from_installed()
        self.spec = spec or ControlSpec()
        path = inspect.getfile(DSLPIDControl)
        with open(path, 'rb') as stream:
            actual_hash = hashlib.sha256(stream.read()).hexdigest()
        if actual_hash != DSLPID_SHA256:
            raise RuntimeError('installed DSLPID differs from audited version')
        # Read model-specific constants only; never use its position controller.
        original = DSLPIDControl(DroneModel.CF2X)
        self.kp = original.P_COEFF_TOR.copy()
        self.ki = original.I_COEFF_TOR.copy()
        self.kd = original.D_COEFF_TOR.copy()
        self.mixer = original.MIXER_MATRIX.copy()
        self.inverse_mixer = np.linalg.pinv(self.mixer)
        self.integral = np.zeros(3)
        self.last_rpy = np.zeros(3)
        self.yaw_target = 0.
        self.active = False

    def reset(self, state):
        if not isinstance(state, ControlState):
            raise ValueError('ControlState required')
        rpy = np.asarray(p.getEulerFromQuaternion(state.q_world_from_body_xyzw), dtype=float)
        yaw = yaw_from_quaternion(state.q_world_from_body_xyzw)
        self.integral = np.zeros(3)
        self.last_rpy = rpy
        self.yaw_target = yaw
        self.active = True

    def snapshot(self):
        return dict(integral=self.integral.tolist(), last_rpy=self.last_rpy.tolist(),
                    yaw_target=self.yaw_target, active=self.active)

    def restore(self, snapshot):
        self.integral = np.asarray(snapshot['integral'], dtype=float).copy()
        self.last_rpy = np.asarray(snapshot['last_rpy'], dtype=float).copy()
        self.yaw_target = float(snapshot['yaw_target'])
        self.active = bool(snapshot['active'])

    def finish_episode(self):
        self.integral = np.zeros(3)
        self.last_rpy = np.zeros(3)
        self.yaw_target = 0.
        self.active = False

    def _allocate(self, collective_pwm, rotation_error, measured_euler_rates, integral):
        torque_raw = -self.kp * rotation_error - self.kd * measured_euler_rates + self.ki * integral
        torque_limited = np.clip(torque_raw, -3200., 3200.)
        pwm_raw = collective_pwm + self.mixer @ torque_limited
        pwm_limited = np.clip(pwm_raw, self.model.min_pwm, self.model.max_pwm)
        rpm_before_physical = self.model.pwm_scale * pwm_limited + self.model.pwm_offset
        rpm = np.clip(rpm_before_physical, 0., self.model.max_rpm)
        pwm_actual = (rpm - self.model.pwm_offset) / self.model.pwm_scale
        torque_allocated = self.inverse_mixer @ (pwm_actual - collective_pwm)
        return rpm, dict(torque_raw=torque_raw, torque_limited=torque_limited,
                         torque_allocated=torque_allocated, pwm_raw=pwm_raw, pwm_limited=pwm_limited,
                         rpm_before_physical=rpm_before_physical,
                         torque_saturation_mask=abs(torque_raw - torque_limited) > 1e-9,
                         pwm_saturation_mask=abs(pwm_raw - pwm_limited) > 1e-9,
                         physical_rpm_saturation_mask=abs(rpm-rpm_before_physical) > 1e-9)

    def update(self, force_world_n, state, dt=1/120):
        if not self.active:
            raise RuntimeError('attitude reset required')
        if not isinstance(state, ControlState):
            raise ValueError('ControlState required')
        if abs(scalar(dt, 'dt', positive=True) - self.spec.dt) > 1e-12:
            raise ValueError('fixed 120Hz timestep required')
        force = vector(force_world_n, 3, 'force_world_n')
        target_rotation, target_euler = force_to_attitude(force, self.yaw_target)
        current_rotation = body_to_world(state.q_world_from_body_xyzw)
        rpy = np.asarray(p.getEulerFromQuaternion(state.q_world_from_body_xyzw), dtype=float)
        measured_rates = wrap_pi(rpy - self.last_rpy) / dt
        matrix_error = target_rotation.T @ current_rotation - current_rotation.T @ target_rotation
        rotation_error = np.array([matrix_error[2, 1], matrix_error[0, 2], matrix_error[1, 0]])
        integral_candidate = np.clip(self.integral - rotation_error * dt,
                                     -np.asarray(self.spec.attitude_integral_limit),
                                     self.spec.attitude_integral_limit)
        thrust_n = max(0., float(force @ current_rotation[:, 2]))
        collective_pwm = collective_force_to_pwm(thrust_n, self.model)
        rpm, allocation = self._allocate(collective_pwm, rotation_error, measured_rates, integral_candidate)
        increment_torque = self.ki * (integral_candidate - self.integral)
        unachieved = allocation['torque_allocated'] - allocation['torque_raw']
        blocked = increment_torque * unachieved < -1e-9
        if blocked.any():
            integral_candidate = np.where(blocked, self.integral, integral_candidate)
            rpm, allocation = self._allocate(collective_pwm, rotation_error, measured_rates, integral_candidate)
        saturation = any(np.any(allocation[k]) for k in (
            'torque_saturation_mask', 'pwm_saturation_mask', 'physical_rpm_saturation_mask'))
        diagnostic = dict(target_rotation=target_rotation, target_euler_intrinsic_XYZ=target_euler,
                          target_force_direction=target_rotation[:, 2], collective_requested_n=thrust_n,
                          collective_pwm=collective_pwm, measured_euler_rates=measured_rates,
                          attitude_rotation_error=rotation_error, attitude_integral_before=self.integral.copy(),
                          attitude_integral_after=integral_candidate.copy(), attitude_integrator_blocked=blocked,
                          allocation_saturated=bool(saturation), yaw_target=self.yaw_target,
                          **allocation)
        if not np.isfinite(rpm).all():
            raise ValueError('nonfinite mixer result')
        # State mutation only after every calculation and check succeeds.
        self.last_rpy = rpy.copy()
        self.integral = integral_candidate.copy()
        return rpm, diagnostic
