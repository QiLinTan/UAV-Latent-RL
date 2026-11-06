"""Units and frames: world z-up, body-to-world xyzw quaternion, SI motion.

No navigation position or desired-position variable is a control input.
The installed DSL attitude mixer's collective input has PWM units.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

SCHEMA = 'direct_velocity_control_v1'
DSLPID_SHA256 = 'b1b9f40b3289610c3b3dfdd5f9990294e4ee11ec1c1b8d1f966e5db3ed0d6312'
MODEL_PACKAGE = Path(__file__).resolve().parents[2] / '.wsl2/drones-cpu/lib/python3.10/site-packages/gym_pybullet_drones'


def vector(value, n, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (n,) or not np.isfinite(result).all():
        raise ValueError(f'{name} must be finite with shape ({n},)')
    return result.copy()


def scalar(value, name, *, positive=False, nonnegative=False):
    if np.ndim(value) != 0 or isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be a real scalar')
    result = float(value)
    if not np.isfinite(result) or (positive and result <= 0) or (nonnegative and result < 0):
        raise ValueError(f'invalid {name}')
    return result


def integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return int(value)


def quaternion(value):
    q = vector(value, 4, 'q_world_from_body_xyzw')
    norm = float(np.linalg.norm(q))
    if abs(norm - 1) > 1e-3:
        raise ValueError('quaternion must be unit length within 1e-3')
    return q / norm


def body_to_world(q_xyzw):
    return Rotation.from_quat(quaternion(q_xyzw)).as_matrix()


def yaw_rotation(yaw):
    yaw = scalar(yaw, 'yaw')
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def yaw_from_quaternion(q_xyzw):
    return float(Rotation.from_quat(quaternion(q_xyzw)).as_euler('xyz')[2])


def wrap_pi(angles):
    return (np.asarray(angles) + np.pi) % (2 * np.pi) - np.pi


def local_to_world(velocity_local, yaw):
    return yaw_rotation(yaw) @ vector(velocity_local, 3, 'velocity_local')


def world_to_local(velocity_world, yaw):
    return yaw_rotation(yaw).T @ vector(velocity_world, 3, 'velocity_world')


def body_angular_velocity(angular_velocity_world, q_xyzw):
    return body_to_world(q_xyzw).T @ vector(angular_velocity_world, 3, 'angular_velocity_world')


def project_world_velocity(value, max_xy=.8, max_z=.3):
    v = vector(value, 3, 'velocity_world')
    max_xy = scalar(max_xy, 'max_xy', positive=True)
    max_z = scalar(max_z, 'max_z', positive=True)
    norm = np.linalg.norm(v[:2])
    if norm > max_xy:
        v[:2] *= max_xy / norm
    v[2] = np.clip(v[2], -max_z, max_z)
    return v


def normalized_action_to_world(action, yaw):
    a = np.clip(vector(action, 3, 'action'), -1, 1)
    a[:2] /= max(1., np.linalg.norm(a[:2]))
    return local_to_world(a * [.8, .8, .3], yaw)


@dataclass(frozen=True)
class ControlState:
    q_world_from_body_xyzw: np.ndarray
    velocity_world: np.ndarray
    angular_velocity_world: np.ndarray

    def __post_init__(self):
        for name, array in (
            ('q_world_from_body_xyzw', quaternion(self.q_world_from_body_xyzw)),
            ('velocity_world', vector(self.velocity_world, 3, 'velocity_world')),
            ('angular_velocity_world', vector(self.angular_velocity_world, 3, 'angular_velocity_world')),
        ):
            array.flags.writeable = False
            object.__setattr__(self, name, array)


@dataclass(frozen=True)
class VelocityCommand:
    sequence_id: int
    accepted_at_lower_tick: int
    velocity_world: np.ndarray

    def __post_init__(self):
        integer(self.sequence_id, 'sequence_id')
        integer(self.accepted_at_lower_tick, 'accepted_at_lower_tick')
        v = vector(self.velocity_world, 3, 'velocity_world')
        v.flags.writeable = False
        object.__setattr__(self, 'velocity_world', v)


@dataclass(frozen=True)
class ModelParameters:
    mass_kg: float = .027
    gravity_mps2: float = 9.8
    kf: float = 3.16e-10
    km: float = 7.94e-12
    arm_m: float = .0397
    inertia_kg_m2: tuple = (1.4e-5, 1.4e-5, 2.17e-5)
    thrust_weight_ratio: float = 2.25
    pwm_scale: float = .2685
    pwm_offset: float = 4070.3
    min_pwm: float = 20000.
    max_pwm: float = 65535.

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name != 'inertia_kg_m2':
                scalar(value, name, positive=True)
        if np.any(vector(self.inertia_kg_m2, 3, 'inertia') <= 0) or self.max_pwm <= self.min_pwm:
            raise ValueError('invalid model parameters')

    @property
    def weight_n(self):
        return self.mass_kg * self.gravity_mps2

    @property
    def hover_rpm(self):
        return np.sqrt(self.weight_n / (4 * self.kf))

    @property
    def max_rpm(self):
        return self.hover_rpm * np.sqrt(self.thrust_weight_ratio)

    @property
    def max_force_n(self):
        return self.weight_n * self.thrust_weight_ratio

    @classmethod
    def from_installed(cls):
        root = ET.parse(MODEL_PACKAGE / 'assets/cf2x.urdf').getroot()
        p = root.find('properties').attrib
        inertial = root.find('link/inertial')
        inertia = inertial.find('inertia').attrib
        return cls(mass_kg=float(inertial.find('mass').attrib['value']), kf=float(p['kf']),
                   km=float(p['km']), arm_m=float(p['arm']), thrust_weight_ratio=float(p['thrust2weight']),
                   inertia_kg_m2=tuple(float(inertia[k]) for k in ('ixx', 'iyy', 'izz')))


@dataclass(frozen=True)
class ControlSpec:
    control_hz: int = 120
    physics_hz: int = 240
    upper_interval: int = 8
    max_xy_mps: float = .8
    max_z_mps: float = .3
    kp: tuple = (3., 3., 4.)
    ki: tuple = (1., 1., 1.5)
    kaw: tuple = (4., 4., 4.)
    integral_limit_mps2: tuple = (.8, .8, 1.)
    max_tilt_rad: float = .45
    max_force_fraction: float = .9
    min_vertical_weight_fraction: float = .45
    attitude_integral_limit: tuple = (1., 1., 1.)

    def __post_init__(self):
        if (self.control_hz, self.physics_hz, self.upper_interval) != (120, 240, 8):
            raise ValueError('Step 1 requires 15/120/240 Hz')
        for name in ('max_xy_mps', 'max_z_mps', 'max_tilt_rad', 'max_force_fraction', 'min_vertical_weight_fraction'):
            scalar(getattr(self, name), name, positive=True)
        if not 0 < self.max_tilt_rad < .6 or not 0 < self.max_force_fraction <= 1:
            raise ValueError('invalid force or tilt constraint')
        for name in ('kp', 'ki', 'kaw', 'integral_limit_mps2', 'attitude_integral_limit'):
            if np.any(vector(getattr(self, name), 3, name) < 0):
                raise ValueError('negative gain or integral bound')

    @property
    def dt(self):
        return 1. / self.control_hz


def collective_force_to_pwm(force_n, model):
    t = scalar(force_n, 'collective_force_n', nonnegative=True)
    return (np.sqrt(t / (4 * model.kf)) - model.pwm_offset) / model.pwm_scale


def pwm_to_collective_force(pwm, model):
    pwm = scalar(pwm, 'collective_pwm')
    rpm = model.pwm_scale * pwm + model.pwm_offset
    if rpm < 0:
        raise ValueError('PWM implies negative RPM')
    return 4 * model.kf * rpm**2


def rpm_to_collective_force(rpm, model):
    rpm = vector(rpm, 4, 'rpm')
    if np.any(rpm < 0):
        raise ValueError('negative RPM')
    return float(model.kf * np.sum(rpm**2))


def force_to_attitude(force_world_n, yaw):
    force = vector(force_world_n, 3, 'force_world_n')
    if force[2] <= 0:
        raise ValueError('positive vertical support required')
    b3 = force / np.linalg.norm(force)
    heading = yaw_rotation(yaw)[:, 0]
    b2 = np.cross(b3, heading)
    if np.linalg.norm(b2) < 1e-9:
        raise ValueError('degenerate attitude geometry')
    b2 /= np.linalg.norm(b2)
    target = np.column_stack((np.cross(b2, b3), b2, b3))
    return target, Rotation.from_matrix(target).as_euler('XYZ')


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
