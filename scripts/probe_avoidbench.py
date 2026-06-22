from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.avoidbench import AvoidBenchBridgeAdapter


DEFAULT_CONTAINER_CONFIG = pathlib.Path(
    "/AvoidBench/src/avoidbench/avoid_manage/params/task_indoor.yaml"
)


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def _describe(name, value):
    array = np.asarray(value)
    finite = np.isfinite(array)
    finite_values = array[finite]
    value_range = (
        f"[{finite_values.min():.4f}, {finite_values.max():.4f}]"
        if finite_values.size
        else "no finite values"
    )
    print(
        f"{name}: shape={array.shape}, dtype={array.dtype}, "
        f"size={array.size}, range={value_range}, finite={bool(finite.all())}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Probe AvoidBench's ROS/catkin avoidbridge image interface."
    )
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONTAINER_CONFIG)
    parser.add_argument("--image-mode", choices=("unity", "stereo", "auto"), default="unity")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--spawn-obstacles", type=str2bool, default=True)
    parser.add_argument("--position", type=float, nargs=3, default=(0.0, 0.0, 2.0))
    parser.add_argument(
        "--orientation",
        type=float,
        nargs=4,
        default=(0.0, 0.0, 0.0, 1.0),
        help="Quaternion x y z w.",
    )
    parser.add_argument("--velocity", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--mission-end", type=float, nargs=3, default=(0.0, 15.0, 2.0))
    parser.add_argument("--mission-radius", type=float, default=2.0)
    parser.add_argument("--mission-seed", type=int, default=32)
    args = parser.parse_args()

    if not args.config.is_file():
        raise FileNotFoundError(f"AvoidBench config does not exist: {args.config}")
    if args.steps <= 0:
        raise ValueError("steps must be positive.")

    adapter = AvoidBenchBridgeAdapter(args.config)
    state = adapter.create_state(
        position=args.position,
        orientation=args.orientation,
        velocity=args.velocity,
        timestamp=0.0,
    )
    ready = adapter.update_unity(state)
    print(f"Unity ready: {ready}")
    if not ready:
        raise RuntimeError("AvoidBench Unity bridge did not become ready.")

    if args.spawn_obstacles:
        adapter.configure_mission(
            start_point=(*args.position, 0.0),
            end_point=args.mission_end,
            trials=1,
            radius=args.mission_radius,
            seed=args.mission_seed,
            opacity=0.5,
            pointcloud_file="pointcloud-unity-probe",
        )
        changed = adapter.spawn_obstacles(state=state)
        print(f"Scene changed: {changed}")
        if not changed:
            raise RuntimeError("AvoidBench did not confirm the obstacle scene change.")

    start = time.perf_counter()
    images = None
    for step in range(args.steps):
        if not adapter.update_unity(state):
            raise RuntimeError(f"Unity update failed at step {step}.")
        images = adapter.get_images(args.image_mode)
        if args.sleep > 0.0:
            time.sleep(args.sleep)
    elapsed = time.perf_counter() - start

    for index, image in enumerate(images):
        _describe(f"image {index}", image)
    print(f"collision: {adapter.collision()}")
    print(f"throughput: {args.steps / elapsed:.2f} image updates/s")


if __name__ == "__main__":
    main()
