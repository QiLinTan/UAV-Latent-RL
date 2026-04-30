from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path


def ensure_gym_pybullet_envs_compat() -> None:
    """Load `gym_pybullet_drones.envs` submodules without relying on a stale package __init__.

    Some local installs still expose `gym_pybullet_drones.envs.__init__` that imports
    `ForestAviary`, even when that module is no longer shipped. Importing any env submodule
    then fails before the real target module is loaded.
    """
    try:
        importlib.import_module("gym_pybullet_drones.envs.BaseRLAviary")
        return
    except ModuleNotFoundError as exc:
        if exc.name != "gym_pybullet_drones.envs.ForestAviary":
            raise

    import gym_pybullet_drones

    envs_dir = Path(gym_pybullet_drones.__file__).resolve().parent / "envs"
    envs_pkg_name = "gym_pybullet_drones.envs"
    envs_pkg = sys.modules.get(envs_pkg_name)

    if envs_pkg is None:
        envs_pkg = types.ModuleType(envs_pkg_name)
        envs_pkg.__file__ = str(envs_dir / "__init__.py")
        envs_pkg.__path__ = [str(envs_dir)]
        sys.modules[envs_pkg_name] = envs_pkg
    elif not hasattr(envs_pkg, "__path__"):
        envs_pkg.__path__ = [str(envs_dir)]

    for module_name in ("BaseAviary", "BaseRLAviary", "HoverAviary"):
        _load_env_module(module_name, envs_dir, envs_pkg)


def _load_env_module(module_name: str, envs_dir: Path, envs_pkg: types.ModuleType) -> None:
    full_name = f"gym_pybullet_drones.envs.{module_name}"
    if full_name in sys.modules:
        return

    module_path = envs_dir / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(full_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {full_name} from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    setattr(envs_pkg, module_name, module)
    spec.loader.exec_module(module)
