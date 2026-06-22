from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .backend import AvoidBridgeBindings, create_avoidbridge_backend


class AvoidBenchImageError(RuntimeError):
    pass


class AvoidBenchBridgeAdapter:
    """Thin Python adapter for the ROS/catkin ``avoidbridge`` binding."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        bridge=None,
        bindings: AvoidBridgeBindings | None = None,
    ):
        self.config_path = Path(config_path).expanduser()
        if bridge is None:
            bridge, loaded_bindings = create_avoidbridge_backend(self.config_path)
            bindings = bindings or loaded_bindings
        if bindings is None:
            raise ValueError("bindings are required when injecting a bridge.")
        self.bridge = bridge
        self.bindings = bindings
        self.last_state = None

    def create_state(
        self,
        position=(0.0, 0.0, 2.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
        velocity=(0.0, 0.0, 0.0),
        timestamp: float = 0.0,
    ):
        position = np.asarray(position, dtype=np.float64)
        orientation = np.asarray(orientation, dtype=np.float64)
        velocity = np.asarray(velocity, dtype=np.float64)
        if position.shape != (3,):
            raise ValueError(f"position must have shape (3,), got {position.shape}.")
        if orientation.shape != (4,):
            raise ValueError(f"orientation must have shape (4,), got {orientation.shape}.")
        if velocity.shape != (3,):
            raise ValueError(f"velocity must have shape (3,), got {velocity.shape}.")
        if not (
            np.isfinite(position).all()
            and np.isfinite(orientation).all()
            and np.isfinite(velocity).all()
            and np.isfinite(timestamp)
        ):
            raise FloatingPointError("AvoidBench state contains NaN or Inf.")

        state = self.bindings.create_state()
        state.setStateEstimate(
            position,
            orientation,
            velocity,
            float(timestamp),
        )
        return state

    def update_unity(self, state=None, **state_kwargs) -> bool:
        if state is None:
            state = self.create_state(**state_kwargs)
        ready = bool(self.bridge.updateUnity(state))
        self.last_state = state
        return ready

    def configure_mission(
        self,
        *,
        start_point=(0.0, 0.0, 2.0, 0.0),
        end_point=(0.0, 15.0, 2.0),
        trials: int = 1,
        radius: float = 2.0,
        seed: int = 32,
        opacity: float = 0.5,
        pointcloud_file: str = "pointcloud-unity-test",
    ):
        mission = self.bindings.create_mission()
        mission.m_start_point = [float(value) for value in start_point]
        mission.m_end_point = [float(value) for value in end_point]
        mission.trials = int(trials)
        mission.m_radius = float(radius)
        mission.m_seed = int(seed)
        mission.m_opacity = float(opacity)
        mission.m_pc_file_name = str(pointcloud_file)
        self.bridge.setParamFromMission(mission)
        return mission

    def spawn_obstacles(
        self,
        *,
        state=None,
        max_updates: int = 20,
        sleep_seconds: float = 0.05,
    ) -> bool:
        if not self.bridge.spawnObstacles():
            return False
        state = state or self.last_state or self.create_state()
        for _ in range(max_updates):
            self.bridge.updateUnity(state)
            self.bridge.SpawnNewObs()
            if self.bridge.ifSceneChanged():
                return True
            if sleep_seconds > 0.0:
                time.sleep(sleep_seconds)
        return bool(self.bridge.ifSceneChanged())

    @staticmethod
    def _validate_image(name: str, image, expected_channels: int):
        array = np.asarray(image)
        if array.size == 0:
            raise AvoidBenchImageError(
                f"AvoidBench returned an empty {name} image with shape {array.shape}."
            )
        if array.ndim == 2 and expected_channels == 1:
            array = array[..., None]
        if array.ndim != 3 or array.shape[2] != expected_channels:
            raise AvoidBenchImageError(
                f"Unexpected {name} image shape {array.shape}; expected HxWx{expected_channels}."
            )
        if not np.isfinite(array).all():
            raise AvoidBenchImageError(f"{name} image contains NaN or Inf.")
        return np.array(array, copy=True)

    def get_unity_depth_images(self):
        method = getattr(self.bridge, "getUnityDepthImages", None)
        if method is None:
            raise AvoidBenchImageError(
                "This avoidbridge build does not expose getUnityDepthImages(). "
                "Apply patches/avoidbench_unity_depth_pybind.patch and rebuild avoidlib. "
                "Changing camera.perform_sgm=true is only a stereo-SGM workaround."
            )
        images = method()
        if len(images) != 2:
            raise AvoidBenchImageError(
                f"getUnityDepthImages() returned {len(images)} values; expected RGB and depth."
            )
        left = self._validate_image("left RGB", images[0], 3)
        depth = self._validate_image("Unity depth", images[1], 1)
        return left, depth

    def get_stereo_images(self):
        images = self.bridge.getImages()
        if len(images) != 3:
            raise AvoidBenchImageError(
                f"getImages() returned {len(images)} values; expected left, right and depth."
            )
        left = self._validate_image("left RGB", images[0], 3)
        right = self._validate_image("right RGB", images[1], 3)
        depth = self._validate_image("SGM depth", images[2], 1)
        return left, right, depth

    def get_images(self, mode: str = "unity"):
        if mode == "unity":
            return self.get_unity_depth_images()
        if mode == "stereo":
            return self.get_stereo_images()
        if mode == "auto":
            if hasattr(self.bridge, "getUnityDepthImages"):
                return self.get_unity_depth_images()
            return self.get_stereo_images()
        raise ValueError("image mode must be one of: unity, stereo, auto.")

    def collision(self) -> bool:
        return bool(self.bridge.getQuadCollisionState())
