from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


class AvoidBridgeUnavailableError(ImportError):
    pass


@dataclass(frozen=True)
class AvoidBridgeBindings:
    module: ModuleType

    def create_bridge(self, config_path: str | Path):
        return self.module.AvoidbenchBridge(str(Path(config_path)))

    def create_state(self):
        return self.module.quadStateEstimate()

    def create_mission(self):
        return self.module.mission_parameter()


def load_avoidbridge() -> AvoidBridgeBindings:
    try:
        module = importlib.import_module("avoidbridge")
    except ImportError as exc:
        raise AvoidBridgeUnavailableError(
            "Could not import avoidbridge. Enter the ROS Noetic AvoidBench container, "
            "then source /opt/ros/noetic/setup.bash and /AvoidBench/devel/setup.bash."
        ) from exc

    required = (
        "AvoidbenchBridge",
        "quadStateEstimate",
        "mission_parameter",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AvoidBridgeUnavailableError(
            f"avoidbridge is missing required bindings: {', '.join(missing)}"
        )
    return AvoidBridgeBindings(module)


def create_avoidbridge_backend(config_path: str | Path):
    bindings = load_avoidbridge()
    return bindings.create_bridge(config_path), bindings
