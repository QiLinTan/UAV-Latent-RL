"""World-velocity PI. Position is deliberately absent from the control law."""
from __future__ import annotations

import numpy as np

from ..contracts import ControlSpec, ModelParameters, vector


def constrain_force(force_world_n, model: ModelParameters, spec: ControlSpec):
    raw = vector(force_world_n, 3, 'force_world_n')
    maximum = model.max_force_n * spec.max_force_fraction
    minimum_z = model.weight_n * spec.min_vertical_weight_fraction
    if maximum <= minimum_z:
        raise ValueError('inconsistent minimum/maximum force')
    force = raw.copy()
    force[2] = np.clip(force[2], minimum_z, maximum)
    lateral = np.linalg.norm(force[:2])
    lateral_max = force[2] * np.tan(spec.max_tilt_rad)
    if lateral > lateral_max:
        force[:2] *= lateral_max / lateral
    norm = np.linalg.norm(force)
    if norm > maximum:
        force *= maximum / norm
    return force, dict(force_limited=bool(np.max(abs(force - raw)) > 1e-12),
                       tilt_limited=bool(lateral > lateral_max + 1e-12),
                       total_force_limited=bool(norm > maximum + 1e-12),
                       vertical_force_limited=bool(raw[2] < minimum_z or raw[2] > maximum))


class VelocityPI:
    """Two-phase propose/commit permits an atomic complete-controller update.

    eta is the integral contribution in m/s². Conditional integration blocks
    errors that push farther into constraints. Back-calculation unwinds an
    existing integral without creating an opposing bias from a saturated P
    step alone. Allocation saturation freezes error integration as well.
    """

    def __init__(self, model=None, spec=None):
        self.model = model or ModelParameters.from_installed()
        self.spec = spec or ControlSpec()
        self.integral = np.zeros(3)

    def reset(self):
        self.integral = np.zeros(3)

    def propose(self, command_world, actual_velocity_world, *, allocation_saturated=False):
        command = vector(command_world, 3, 'command_world')
        velocity = vector(actual_velocity_world, 3, 'actual_velocity_world')
        error = command - velocity
        acceleration = np.asarray(self.spec.kp) * error + self.integral
        raw_force = self.model.mass_kg * (acceleration + [0., 0., self.model.gravity_mps2])
        force, limits = constrain_force(raw_force, self.model, self.spec)
        residual_acceleration = (force - raw_force) / self.model.mass_kg
        blocked = error * residual_acceleration < -1e-12
        if allocation_saturated:
            blocked[:] = True
        error_increment = np.where(blocked, 0., np.asarray(self.spec.ki) * error * self.spec.dt)
        unwind = np.where(self.integral * residual_acceleration < 0,
                          np.asarray(self.spec.kaw) * residual_acceleration * self.spec.dt, 0.)
        unwind = np.sign(unwind) * np.minimum(abs(unwind), abs(self.integral))
        candidate = np.clip(self.integral + unwind + error_increment,
                            -np.asarray(self.spec.integral_limit_mps2), self.spec.integral_limit_mps2)
        diagnostics = dict(command_world=command, actual_velocity_world=velocity,
                           velocity_error=error, acceleration_raw=acceleration,
                           force_raw_n=raw_force, force_limited_n=force,
                           velocity_integral_before=self.integral.copy(),
                           velocity_integral_after=candidate.copy(),
                           integrator_blocked=blocked, allocation_feedback_saturated=bool(allocation_saturated),
                           **limits)
        return force, diagnostics, candidate

    def commit(self, candidate):
        self.integral = vector(candidate, 3, 'integral_candidate')


class VelocityTrackingController:
    """Atomic, synchronous 120Hz velocity control; output is four motor RPMs."""

    def __init__(self, model=None, spec=None, attitude_adapter=None):
        from .velocity_control_adapter import DSLAttitudeAdapter
        self.model = model or ModelParameters.from_installed()
        self.spec = spec or ControlSpec()
        self.pi = VelocityPI(self.model, self.spec)
        self.adapter = attitude_adapter or DSLAttitudeAdapter(self.model, self.spec)
        self.active = False
        self.tick = 0
        self.command = None
        self.last_saturation = False

    def reset(self, state, lower_tick=0):
        from ..contracts import ControlState, integer
        if not isinstance(state, ControlState):
            raise ValueError('ControlState required')
        tick = integer(lower_tick, 'lower_tick')
        self.adapter.reset(state)
        self.pi.reset()
        self.tick = tick
        self.command = None
        self.last_saturation = False
        self.active = True

    def snapshot(self):
        return dict(active=self.active, tick=self.tick, integral=self.pi.integral.tolist(),
                    command=None if self.command is None else dict(sequence_id=self.command.sequence_id,
                      accepted_at_lower_tick=self.command.accepted_at_lower_tick,
                      velocity_world=self.command.velocity_world.tolist()),
                    last_saturation=self.last_saturation, attitude=self.adapter.snapshot())

    def finish_episode(self):
        self.active = False
        self.command = None
        self.pi.reset()
        self.last_saturation = False
        self.adapter.finish_episode()

    def update(self, command, state, dt=1/120):
        from ..contracts import ControlState, VelocityCommand, project_world_velocity, scalar
        if not self.active:
            raise RuntimeError('reset required before update')
        if not isinstance(state, ControlState):
            raise ValueError('ControlState required')
        if abs(scalar(dt, 'dt', positive=True) - self.spec.dt) > 1e-12:
            raise ValueError('fixed 120Hz timestep required')
        accepted = self.command
        if command is not None:
            if not isinstance(command, VelocityCommand):
                raise ValueError('VelocityCommand required')
            if command.accepted_at_lower_tick > self.tick:
                raise ValueError('future command')
            if accepted is not None:
                if command.sequence_id < accepted.sequence_id:
                    raise ValueError('command sequence moved backwards')
                if command.sequence_id == accepted.sequence_id and (
                    command.accepted_at_lower_tick != accepted.accepted_at_lower_tick or
                    not np.array_equal(command.velocity_world, accepted.velocity_world)):
                    raise ValueError('same command id changed contents')
            if accepted is None or command.sequence_id > accepted.sequence_id:
                if command.accepted_at_lower_tick != self.tick:
                    raise ValueError('new command must be accepted at current tick')
                accepted = command
        applied = np.zeros(3) if accepted is None else project_world_velocity(
            accepted.velocity_world, self.spec.max_xy_mps, self.spec.max_z_mps)
        force, diagnostics, candidate = self.pi.propose(applied, state.velocity_world,
                                                       allocation_saturated=self.last_saturation)
        old_adapter = self.adapter.snapshot()
        try:
            rpm, inner = self.adapter.update(force, state, self.spec.dt)
            rpm = vector(rpm, 4, 'output_rpm')
            if np.any(rpm < 0) or np.any(rpm > self.model.max_rpm + 1e-9):
                raise ValueError('invalid physical RPM')
        except Exception:
            self.adapter.restore(old_adapter)
            raise
        self.pi.commit(candidate)
        self.command = accepted
        self.last_saturation = bool(inner['allocation_saturated'])
        diagnostics.update(inner)
        diagnostics.update(lower_tick=self.tick, timestamp_sim=self.tick*self.spec.dt,
                           command_id=-1 if accepted is None else accepted.sequence_id,
                           command_raw_world=np.zeros(3) if accepted is None else accepted.velocity_world.copy(),
                           command_applied_world=applied.copy())
        self.tick += 1
        return rpm, diagnostics
