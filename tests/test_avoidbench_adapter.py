from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from envs.avoidbench.adapter import AvoidBenchBridgeAdapter, AvoidBenchImageError


class FakeState:
    def setStateEstimate(self, position, orientation, velocity, timestamp):
        self.position = np.array(position, copy=True)
        self.orientation = np.array(orientation, copy=True)
        self.velocity = np.array(velocity, copy=True)
        self.timestamp = timestamp


class FakeBindings:
    def create_state(self):
        return FakeState()

    def create_mission(self):
        return SimpleNamespace()


class FakeAvoidBridge:
    def __init__(self):
        self.update_count = 0
        self.spawn_requested = False
        self.mission = None

    def updateUnity(self, state):
        self.update_count += 1
        self.state = state
        return True

    def setParamFromMission(self, mission):
        self.mission = mission

    def spawnObstacles(self):
        return True

    def SpawnNewObs(self):
        self.spawn_requested = True

    def ifSceneChanged(self):
        return self.spawn_requested and self.update_count >= 2

    def getUnityDepthImages(self):
        left = np.full((4, 6, 3), 12, dtype=np.uint8)
        depth = np.full((4, 6, 1), 250.0, dtype=np.float32)
        return left, depth

    def getImages(self):
        left = np.full((4, 6, 3), 12, dtype=np.uint8)
        right = np.full((4, 6, 3), 13, dtype=np.uint8)
        depth = np.full((4, 6, 1), 1000, dtype=np.uint16)
        return left, right, depth

    def getQuadCollisionState(self):
        return False


def make_adapter(bridge=None):
    return AvoidBenchBridgeAdapter(
        "task_indoor.yaml",
        bridge=bridge or FakeAvoidBridge(),
        bindings=FakeBindings(),
    )


def test_state_mission_scene_and_unity_images():
    adapter = make_adapter()
    state = adapter.create_state(
        position=(1.0, 2.0, 3.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
        velocity=(0.1, 0.2, 0.3),
        timestamp=4.0,
    )
    assert adapter.update_unity(state)
    np.testing.assert_allclose(state.position, [1.0, 2.0, 3.0])

    mission = adapter.configure_mission(seed=9, radius=1.5)
    assert mission.m_seed == 9
    assert mission.m_radius == 1.5
    assert adapter.spawn_obstacles(state=state, sleep_seconds=0.0)

    left, depth = adapter.get_unity_depth_images()
    assert left.shape == (4, 6, 3)
    assert left.dtype == np.uint8
    assert depth.shape == (4, 6, 1)
    assert depth.dtype == np.float32
    assert not adapter.collision()


def test_stereo_images_are_supported_separately():
    adapter = make_adapter()
    left, right, depth = adapter.get_stereo_images()
    assert left.shape == right.shape == (4, 6, 3)
    assert depth.shape == (4, 6, 1)
    assert depth.dtype == np.uint16


def test_missing_unity_depth_binding_has_patch_guidance():
    class StereoOnlyBridge(FakeAvoidBridge):
        getUnityDepthImages = None

    adapter = make_adapter(StereoOnlyBridge())
    with pytest.raises(AvoidBenchImageError, match="avoidbench_unity_depth_pybind.patch"):
        adapter.get_unity_depth_images()


def test_empty_images_are_rejected():
    class EmptyBridge(FakeAvoidBridge):
        def getUnityDepthImages(self):
            return (
                np.empty((0, 0, 3), dtype=np.uint8),
                np.empty((0, 0, 1), dtype=np.float32),
            )

    adapter = make_adapter(EmptyBridge())
    with pytest.raises(AvoidBenchImageError, match="empty left RGB"):
        adapter.get_unity_depth_images()


def test_invalid_state_shape_is_rejected():
    adapter = make_adapter()
    with pytest.raises(ValueError, match="position"):
        adapter.create_state(position=(0.0, 1.0))
