"""Control-only CustomForestAviary; unmodified Full rotor dynamics underneath."""
from __future__ import annotations

import numpy as np
import pybullet as p

from envs.forest.core import CustomForestAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ObservationType, ActionType
from models.motor_action_codec import MotorActionCodec, MotorPhysicalLimits, ASYMMETRIC_RPM
from ..contracts import ControlState


class FullControlFixture(CustomForestAviary):
    def __init__(self, *, rpy=(0, 0, 0), position=(0, 0, 1), duration=40.):
        self.physics_ticks = 0
        self.applied_disturbance_substeps = 0
        self.disturbances = []
        self.tick_observer = None
        super().__init__(drone_model=DroneModel.CF2X, physics=Physics.PYB,
                         initial_xyzs=np.array([position]), initial_rpys=np.array([rpy]),
                         pyb_freq=240, ctrl_freq=120, gui=False, record=False,
                         obs=ObservationType.KIN, act=ActionType.RPM,
                         target_pos=[19., 19., 1.], forest_half_extent=20.,
                         num_trees=0, route_blocking_tree=False, curriculum=False,
                         route_count_curriculum=False, curriculum_simple_mix_probability=0.,
                         episode_len_sec=duration, range_azimuth_count=32,
                         range_elevation_degrees=(-30., 0., 30.))
        self.set_motor_action_codec(MotorActionCodec(MotorPhysicalLimits(
            min_rpm=0., hover_rpm=self.HOVER_RPM, max_rpm=self.MAX_RPM, kf=self.KF), mode=ASYMMETRIC_RPM))

    def reset(self, *, seed=None, options=None):
        self.physics_ticks = 0
        self.applied_disturbance_substeps = 0
        return super().reset(seed=seed, options=options)

    def initialize_motion(self, velocity=(0, 0, 0), angular_velocity=(0, 0, 0)):
        if self.step_counter != 0:
            raise RuntimeError('initial motion injection is reset-only')
        p.resetBaseVelocity(int(self.DRONE_IDS[0]), linearVelocity=velocity,
                            angularVelocity=angular_velocity, physicsClientId=self.CLIENT)
        self._updateAndStoreKinematicInformation()

    def motion_state(self):
        return ControlState(self.quat[0], self.vel[0], self.ang_v[0])

    def _physics(self, rpm, nth_drone):
        for disturbance in self.disturbances:
            start = round(disturbance['start_s'] * self.PYB_FREQ)
            stop = round((disturbance['start_s'] + disturbance['duration_s']) * self.PYB_FREQ)
            if start <= self.physics_ticks < stop:
                position, _ = p.getBasePositionAndOrientation(int(self.DRONE_IDS[nth_drone]), physicsClientId=self.CLIENT)
                p.applyExternalForce(int(self.DRONE_IDS[nth_drone]), -1,
                                     disturbance['force_world_n'], position, p.WORLD_FRAME,
                                     physicsClientId=self.CLIENT)
                self.applied_disturbance_substeps += 1
        super()._physics(rpm, nth_drone)
        self.physics_ticks += 1

    def step(self, action):
        result = super().step(action)
        if self.tick_observer is not None:
            self.tick_observer(result)
        return result
