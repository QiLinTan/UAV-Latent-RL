from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from envs.avoidbench.rl_env import ACTION_PRESETS, AvoidBenchRLEnv


@dataclass
class EpisodeRecord:
    strategy: str
    frame: str
    episode_index: int
    initial_position: list[float]
    final_position: list[float]
    initial_distance: float
    final_distance: float
    distance_delta: float
    min_distance: float
    mean_progress: float
    final_height: float
    min_height: float
    max_height: float
    collision: bool
    done: bool
    done_reason: str
    action_mean: list[float]
    action_std: list[float]
    autopilot_state_start: str
    autopilot_state_end: str
    reset_retry_count: int
    wall_time: float
    error: str = ""


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, records: list[EpisodeRecord]) -> None:
    if not records:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def yaw_from_xyzw(quat: np.ndarray) -> float:
    x, y, z, w = [float(value) for value in quat]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def goal_direction_action(
    info: dict[str, Any],
    obs: np.ndarray,
    *,
    frame: str,
    speed: float,
    action_bounds: np.ndarray,
) -> np.ndarray:
    position = np.asarray(info["position"], dtype=np.float32)
    goal = np.asarray(info.get("goal_position", [5.0, 0.0, 1.2]), dtype=np.float32)
    delta = goal[:2] - position[:2]
    norm = float(np.linalg.norm(delta))
    action = np.zeros(4, dtype=np.float32)
    if norm <= 1e-6:
        return action
    world_xy = delta / norm * float(speed)
    if frame == "world":
        action[0] = world_xy[0]
        action[1] = world_xy[1]
    elif frame == "body":
        yaw = yaw_from_xyzw(np.asarray(obs[6:10], dtype=np.float32))
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        action[0] = cos_yaw * world_xy[0] + sin_yaw * world_xy[1]
        action[1] = -sin_yaw * world_xy[0] + cos_yaw * world_xy[1]
    else:
        raise ValueError(f"Unsupported frame {frame!r}.")
    return np.clip(action, -action_bounds, action_bounds).astype(np.float32)


def build_action(
    strategy: str,
    frame: str,
    env: AvoidBenchRLEnv,
    obs: np.ndarray,
    info: dict[str, Any],
    speed: float,
) -> np.ndarray:
    action = np.zeros(4, dtype=np.float32)
    if strategy == "zero":
        return action
    if strategy == "constant_forward":
        action[0] = min(float(speed), float(env.action_bounds[0]))
        return action
    if strategy == "goal_direction":
        return goal_direction_action(
            info,
            obs,
            frame=frame,
            speed=speed,
            action_bounds=env.action_bounds,
        )
    raise ValueError(f"Unknown strategy {strategy!r}.")


def run_episode(
    env: AvoidBenchRLEnv,
    *,
    strategy: str,
    frame: str,
    episode_index: int,
    steps: int,
    speed: float,
) -> EpisodeRecord:
    started = time.perf_counter()
    obs, info = env.reset()
    initial_position = list(info["position"])
    initial_distance = float(info["distance_to_goal"])
    autopilot_state_start = str(info.get("autopilot_state", "UNKNOWN"))
    distances = [initial_distance]
    progress_values: list[float] = []
    heights = [float(info["height"])]
    actions: list[np.ndarray] = []
    final_info = dict(info)
    done = False

    try:
        for _ in range(steps):
            action = build_action(strategy, frame, env, obs, final_info, speed)
            actions.append(action.copy())
            obs, _reward, done, final_info = env.step(action)
            distances.append(float(final_info["distance_to_goal"]))
            progress_values.append(float(final_info.get("progress", 0.0)))
            heights.append(float(final_info["height"]))
            if done:
                break
        action_array = np.stack(actions) if actions else np.zeros((1, env.action_dim), dtype=np.float32)
        final_position = list(final_info["position"])
        final_distance = float(final_info["distance_to_goal"])
        return EpisodeRecord(
            strategy=strategy,
            frame=frame,
            episode_index=episode_index,
            initial_position=initial_position,
            final_position=final_position,
            initial_distance=initial_distance,
            final_distance=final_distance,
            distance_delta=float(initial_distance - final_distance),
            min_distance=float(min(distances)),
            mean_progress=float(np.mean(progress_values)) if progress_values else 0.0,
            final_height=float(final_info["height"]),
            min_height=float(min(heights)),
            max_height=float(max(heights)),
            collision=bool(final_info.get("collision", False)),
            done=bool(done),
            done_reason=str(final_info.get("done_reason", "running")),
            action_mean=action_array.mean(axis=0).astype(float).tolist(),
            action_std=action_array.std(axis=0).astype(float).tolist(),
            autopilot_state_start=autopilot_state_start,
            autopilot_state_end=str(final_info.get("autopilot_state", "UNKNOWN")),
            reset_retry_count=int(final_info.get("reset_retry_count", 0)),
            wall_time=float(time.perf_counter() - started),
        )
    except Exception as exc:
        return EpisodeRecord(
            strategy=strategy,
            frame=frame,
            episode_index=episode_index,
            initial_position=initial_position,
            final_position=list(final_info.get("position", initial_position)),
            initial_distance=initial_distance,
            final_distance=float(final_info.get("distance_to_goal", initial_distance)),
            distance_delta=float(initial_distance - float(final_info.get("distance_to_goal", initial_distance))),
            min_distance=float(min(distances)),
            mean_progress=float(np.mean(progress_values)) if progress_values else 0.0,
            final_height=float(final_info.get("height", heights[-1])),
            min_height=float(min(heights)),
            max_height=float(max(heights)),
            collision=bool(final_info.get("collision", False)),
            done=bool(done),
            done_reason=str(final_info.get("done_reason", "exception")),
            action_mean=[0.0, 0.0, 0.0, 0.0],
            action_std=[0.0, 0.0, 0.0, 0.0],
            autopilot_state_start=autopilot_state_start,
            autopilot_state_end=str(final_info.get("autopilot_state", "UNKNOWN")),
            reset_retry_count=int(final_info.get("reset_retry_count", 0)),
            wall_time=float(time.perf_counter() - started),
            error=f"{type(exc).__name__}: {exc}",
        )


def summarize(records: list[EpisodeRecord]) -> dict[str, Any]:
    grouped: dict[str, list[EpisodeRecord]] = {}
    for record in records:
        key = f"{record.strategy}:{record.frame}"
        grouped.setdefault(key, []).append(record)
    result = {}
    for key, rows in grouped.items():
        deltas = np.asarray([row.distance_delta for row in rows], dtype=np.float64)
        final_lt_initial = np.asarray([row.final_distance < row.initial_distance for row in rows])
        result[key] = {
            "episode_count": len(rows),
            "mean_distance_delta": float(deltas.mean()) if deltas.size else 0.0,
            "min_distance_delta": float(deltas.min()) if deltas.size else 0.0,
            "max_distance_delta": float(deltas.max()) if deltas.size else 0.0,
            "positive_progress_count": int(final_lt_initial.sum()),
            "positive_progress_ratio": float(final_lt_initial.mean()) if final_lt_initial.size else 0.0,
            "collision_count": int(sum(row.collision for row in rows)),
            "errors": [row.error for row in rows if row.error],
            "done_reasons": dict(sorted({reason: sum(row.done_reason == reason for row in rows) for reason in {row.done_reason for row in rows}}.items())),
            "mean_final_height": float(np.mean([row.final_height for row in rows])) if rows else 0.0,
        }
    zero_delta = result.get("zero:world", result.get("zero:none", {})).get("mean_distance_delta", 0.0)
    best_goal_key = None
    best_goal_delta = None
    for key, payload in result.items():
        if not key.startswith("goal_direction:"):
            continue
        value = float(payload["mean_distance_delta"])
        if best_goal_delta is None or value > best_goal_delta:
            best_goal_key = key
            best_goal_delta = value
    return {
        "by_strategy": result,
        "best_goal_direction_key": best_goal_key,
        "best_goal_direction_mean_delta": best_goal_delta,
        "zero_mean_delta": zero_delta,
        "goal_direction_better_than_zero": (
            best_goal_delta is not None and best_goal_delta > zero_delta + 0.05
        ),
        "majority_goal_direction_positive": (
            best_goal_key is not None
            and result[best_goal_key]["positive_progress_ratio"] > 0.5
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe hand-coded goal-direction policies in AvoidBenchRLEnv.")
    parser.add_argument("--namespace", default="/hummingbird")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--speed", type=float, default=0.08)
    parser.add_argument("--action-preset", choices=tuple(ACTION_PRESETS), default="conservative")
    parser.add_argument("--frame", choices=("world", "body", "auto"), default="auto")
    parser.add_argument("--output-root", default="runs/avoidbench_goal_direction_probe")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive.")
    if args.steps <= 0:
        parser.error("--steps must be positive.")
    if args.speed <= 0.0:
        parser.error("--speed must be positive.")

    output_dir = Path(args.output_root) / timestamp_slug()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = ("world", "body") if args.frame == "auto" else (args.frame,)
    scenarios = [("zero", "world"), ("constant_forward", "world")]
    scenarios.extend(("goal_direction", frame) for frame in frames)

    env = AvoidBenchRLEnv(namespace=args.namespace, action_preset=args.action_preset)
    records: list[EpisodeRecord] = []
    try:
        for strategy, frame in scenarios:
            for episode_index in range(args.episodes):
                record = run_episode(
                    env,
                    strategy=strategy,
                    frame=frame,
                    episode_index=episode_index,
                    steps=args.steps,
                    speed=args.speed,
                )
                records.append(record)
                print(
                    f"{strategy}:{frame} ep={episode_index} "
                    f"delta={record.distance_delta:.4f} final={record.final_distance:.4f} "
                    f"done={record.done_reason} collision={record.collision} error={record.error}"
                )
                if record.error:
                    break
    finally:
        env.close()

    summary = {
        "namespace": args.namespace,
        "episodes": args.episodes,
        "steps": args.steps,
        "speed": args.speed,
        "action_preset": args.action_preset,
        "frame": args.frame,
        "output_dir": str(output_dir),
        "records": [asdict(record) for record in records],
        "summary": summarize(records),
    }
    write_csv(output_dir / "episodes.csv", records)
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
