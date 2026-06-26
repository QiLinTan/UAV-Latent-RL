from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass, is_dataclass
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
    initial_distance: float | None
    final_distance: float | None
    distance_delta: float | None
    min_distance: float | None
    mean_progress: float
    final_height: float | None
    min_height: float | None
    max_height: float | None
    collision: bool
    done: bool
    done_reason: str
    action_mean: list[float]
    action_std: list[float]
    autopilot_state_start: str
    autopilot_state_end: str
    reset_retry_count: int
    collision_step: int | None
    collision_position: list[float] | None
    collision_distance_to_goal: float | None
    collision_height: float | None
    collision_action: list[float] | None
    collision_autopilot_state: str
    collision_before_first_action: bool
    wall_time: float
    error: str = ""


@dataclass
class ActionMappingRecord:
    direction: str
    episode_index: int
    action: list[float]
    initial_position: list[float]
    final_position: list[float]
    delta_position: list[float]
    delta_x: float | None
    delta_y: float | None
    delta_z: float | None
    initial_distance: float | None
    final_distance: float | None
    distance_delta: float | None
    collision: bool
    done: bool
    done_reason: str
    collision_step: int | None
    collision_position: list[float] | None
    collision_distance_to_goal: float | None
    collision_height: float | None
    collision_action: list[float] | None
    collision_autopilot_state: str
    collision_before_first_action: bool
    autopilot_state_start: str
    autopilot_state_end: str
    reset_retry_count: int
    wall_time: float
    error: str = ""


@dataclass
class ResetSanityRecord:
    episode_index: int
    observe_seconds: float
    sample_count: int
    initial_position: list[float]
    final_position: list[float]
    initial_distance: float | None
    final_distance: float | None
    collision_before_observe: bool
    collision_observed: bool
    first_collision_time: float | None
    first_collision_position: list[float] | None
    first_collision_distance_to_goal: float | None
    first_collision_height: float | None
    first_collision_autopilot_state: str
    autopilot_state_start: str
    autopilot_state_end: str
    reset_retry_count: int
    wall_time: float
    error: str = ""


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return jsonable(asdict(value))
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


def row_dict(record: Any) -> dict[str, Any]:
    if is_dataclass(record) and not isinstance(record, type):
        return asdict(record)
    return dict(record)


def write_csv(path: Path, records: list[Any]) -> None:
    rows = [row_dict(record) for record in records]
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def info_float(info: dict[str, Any], key: str) -> float | None:
    return optional_float(info.get(key))


def info_list(info: dict[str, Any], key: str) -> list[float] | None:
    value = info.get(key)
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32).astype(float).tolist()


def action_list(action: np.ndarray | list[float] | tuple[float, ...] | None) -> list[float] | None:
    if action is None:
        return None
    return np.asarray(action, dtype=np.float32).reshape(-1).astype(float).tolist()


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


def collision_fields(
    info: dict[str, Any],
    *,
    step_index: int,
    action: np.ndarray | None,
) -> dict[str, Any]:
    return {
        "collision_step": int(step_index),
        "collision_position": info_list(info, "position"),
        "collision_distance_to_goal": info_float(info, "distance_to_goal"),
        "collision_height": info_float(info, "height"),
        "collision_action": action_list(action),
        "collision_autopilot_state": str(info.get("autopilot_state", "UNKNOWN")),
    }


def empty_collision_fields() -> dict[str, Any]:
    return {
        "collision_step": None,
        "collision_position": None,
        "collision_distance_to_goal": None,
        "collision_height": None,
        "collision_action": None,
        "collision_autopilot_state": "",
    }


def append_step_trace(
    traces: list[dict[str, Any]],
    *,
    mode: str,
    scenario: str,
    episode_index: int,
    step_index: int,
    phase: str,
    info: dict[str, Any],
    action: np.ndarray | None,
    reward: float | None = None,
    done: bool = False,
    error: str = "",
) -> None:
    traces.append(
        {
            "mode": mode,
            "scenario": scenario,
            "episode_index": int(episode_index),
            "step_index": int(step_index),
            "phase": phase,
            "position": info_list(info, "position"),
            "velocity": info_list(info, "velocity"),
            "distance_to_goal": info_float(info, "distance_to_goal"),
            "height": info_float(info, "height"),
            "collision": bool(info.get("collision", False)),
            "done": bool(done),
            "done_reason": str(info.get("done_reason", "running")),
            "autopilot_state": str(info.get("autopilot_state", "UNKNOWN")),
            "action": action_list(action),
            "reward": optional_float(reward),
            "progress": info_float(info, "progress"),
            "step_position_delta": info_list(info, "step_position_delta"),
            "error": error,
        }
    )


def snapshot_info(env: AvoidBenchRLEnv, snapshot: Any, *, done_reason: str = "running") -> dict[str, Any]:
    distance = float(env._distance_to_goal(snapshot.position))
    collision = bool(env._latest_collision)
    return {
        "position": snapshot.position.astype(np.float32).tolist(),
        "velocity": snapshot.velocity.astype(np.float32).tolist(),
        "distance_to_goal": distance,
        "collision": collision,
        "height": float(snapshot.position[2]),
        "done_reason": done_reason,
        "autopilot_state": env._autopilot_state_name(),
    }


def action_stats(actions: list[np.ndarray], action_dim: int) -> tuple[list[float], list[float]]:
    action_array = (
        np.stack(actions)
        if actions
        else np.zeros((1, action_dim), dtype=np.float32)
    )
    return (
        action_array.mean(axis=0).astype(float).tolist(),
        action_array.std(axis=0).astype(float).tolist(),
    )


def failed_episode_record(
    *,
    strategy: str,
    frame: str,
    episode_index: int,
    started: float,
    error: Exception,
) -> EpisodeRecord:
    return EpisodeRecord(
        strategy=strategy,
        frame=frame,
        episode_index=episode_index,
        initial_position=[],
        final_position=[],
        initial_distance=None,
        final_distance=None,
        distance_delta=None,
        min_distance=None,
        mean_progress=0.0,
        final_height=None,
        min_height=None,
        max_height=None,
        collision=False,
        done=True,
        done_reason="reset_exception",
        action_mean=[0.0, 0.0, 0.0, 0.0],
        action_std=[0.0, 0.0, 0.0, 0.0],
        autopilot_state_start="UNKNOWN",
        autopilot_state_end="UNKNOWN",
        reset_retry_count=0,
        **empty_collision_fields(),
        collision_before_first_action=False,
        wall_time=float(time.perf_counter() - started),
        error=f"{type(error).__name__}: {error}",
    )


def run_episode(
    env: AvoidBenchRLEnv,
    *,
    strategy: str,
    frame: str,
    episode_index: int,
    steps: int,
    speed: float,
    traces: list[dict[str, Any]],
) -> EpisodeRecord:
    started = time.perf_counter()
    try:
        obs, info = env.reset()
    except Exception as exc:
        return failed_episode_record(
            strategy=strategy,
            frame=frame,
            episode_index=episode_index,
            started=started,
            error=exc,
        )

    initial_position = list(info["position"])
    initial_distance = float(info["distance_to_goal"])
    autopilot_state_start = str(info.get("autopilot_state", "UNKNOWN"))
    distances = [initial_distance]
    progress_values: list[float] = []
    heights = [float(info["height"])]
    actions: list[np.ndarray] = []
    final_info = dict(info)
    done = False
    collision_before_first_action = bool(info.get("collision", False))
    collision = empty_collision_fields()

    scenario = f"{strategy}:{frame}"
    append_step_trace(
        traces,
        mode="goal_policy",
        scenario=scenario,
        episode_index=episode_index,
        step_index=0,
        phase="reset",
        info=info,
        action=None,
        done=collision_before_first_action,
    )

    if collision_before_first_action:
        collision = collision_fields(info, step_index=0, action=np.zeros(env.action_dim, dtype=np.float32))
        action_mean, action_std = action_stats(actions, env.action_dim)
        return EpisodeRecord(
            strategy=strategy,
            frame=frame,
            episode_index=episode_index,
            initial_position=initial_position,
            final_position=initial_position,
            initial_distance=initial_distance,
            final_distance=initial_distance,
            distance_delta=0.0,
            min_distance=initial_distance,
            mean_progress=0.0,
            final_height=float(info["height"]),
            min_height=float(info["height"]),
            max_height=float(info["height"]),
            collision=True,
            done=True,
            done_reason="collision_before_first_action",
            action_mean=action_mean,
            action_std=action_std,
            autopilot_state_start=autopilot_state_start,
            autopilot_state_end=autopilot_state_start,
            reset_retry_count=int(info.get("reset_retry_count", 0)),
            **collision,
            collision_before_first_action=True,
            wall_time=float(time.perf_counter() - started),
        )

    error = ""
    try:
        for step_index in range(1, steps + 1):
            action = build_action(strategy, frame, env, obs, final_info, speed)
            actions.append(action.copy())
            obs, reward, done, final_info = env.step(action)
            distances.append(float(final_info["distance_to_goal"]))
            progress_values.append(float(final_info.get("progress", 0.0)))
            heights.append(float(final_info["height"]))
            append_step_trace(
                traces,
                mode="goal_policy",
                scenario=scenario,
                episode_index=episode_index,
                step_index=step_index,
                phase="step",
                info=final_info,
                action=action,
                reward=reward,
                done=done,
            )
            if bool(final_info.get("collision", False)) and collision["collision_step"] is None:
                collision = collision_fields(final_info, step_index=step_index, action=action)
            if done:
                break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    action_mean, action_std = action_stats(actions, env.action_dim)
    final_position = list(final_info.get("position", initial_position))
    final_distance = float(final_info.get("distance_to_goal", initial_distance))
    final_height = float(final_info.get("height", heights[-1]))
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
        final_height=final_height,
        min_height=float(min(heights)),
        max_height=float(max(heights)),
        collision=bool(final_info.get("collision", False)) or collision["collision_step"] is not None,
        done=bool(done) or bool(error),
        done_reason=str(final_info.get("done_reason", "exception" if error else "running")),
        action_mean=action_mean,
        action_std=action_std,
        autopilot_state_start=autopilot_state_start,
        autopilot_state_end=str(final_info.get("autopilot_state", "UNKNOWN")),
        reset_retry_count=int(final_info.get("reset_retry_count", 0)),
        **collision,
        collision_before_first_action=False,
        wall_time=float(time.perf_counter() - started),
        error=error,
    )


def direction_actions(speed: float, action_bounds: np.ndarray) -> dict[str, np.ndarray]:
    limited = min(float(speed), float(np.min(action_bounds[:2])))
    return {
        "+x": np.asarray([limited, 0.0, 0.0, 0.0], dtype=np.float32),
        "-x": np.asarray([-limited, 0.0, 0.0, 0.0], dtype=np.float32),
        "+y": np.asarray([0.0, limited, 0.0, 0.0], dtype=np.float32),
        "-y": np.asarray([0.0, -limited, 0.0, 0.0], dtype=np.float32),
    }


def failed_mapping_record(
    *,
    direction: str,
    episode_index: int,
    action: np.ndarray,
    started: float,
    error: Exception,
) -> ActionMappingRecord:
    return ActionMappingRecord(
        direction=direction,
        episode_index=episode_index,
        action=action_list(action) or [0.0, 0.0, 0.0, 0.0],
        initial_position=[],
        final_position=[],
        delta_position=[],
        delta_x=None,
        delta_y=None,
        delta_z=None,
        initial_distance=None,
        final_distance=None,
        distance_delta=None,
        collision=False,
        done=True,
        done_reason="reset_exception",
        **empty_collision_fields(),
        collision_before_first_action=False,
        autopilot_state_start="UNKNOWN",
        autopilot_state_end="UNKNOWN",
        reset_retry_count=0,
        wall_time=float(time.perf_counter() - started),
        error=f"{type(error).__name__}: {error}",
    )


def run_action_mapping_episode(
    env: AvoidBenchRLEnv,
    *,
    direction: str,
    action: np.ndarray,
    episode_index: int,
    steps: int,
    traces: list[dict[str, Any]],
) -> ActionMappingRecord:
    started = time.perf_counter()
    try:
        _obs, info = env.reset()
    except Exception as exc:
        return failed_mapping_record(
            direction=direction,
            episode_index=episode_index,
            action=action,
            started=started,
            error=exc,
        )

    initial_position = np.asarray(info["position"], dtype=np.float32)
    initial_distance = float(info["distance_to_goal"])
    autopilot_state_start = str(info.get("autopilot_state", "UNKNOWN"))
    final_info = dict(info)
    done = False
    collision = empty_collision_fields()
    collision_before_first_action = bool(info.get("collision", False))

    append_step_trace(
        traces,
        mode="action_map",
        scenario=direction,
        episode_index=episode_index,
        step_index=0,
        phase="reset",
        info=info,
        action=None,
        done=collision_before_first_action,
    )

    if collision_before_first_action:
        collision = collision_fields(info, step_index=0, action=np.zeros(env.action_dim, dtype=np.float32))
    else:
        try:
            for step_index in range(1, steps + 1):
                _obs, reward, done, final_info = env.step(action)
                append_step_trace(
                    traces,
                    mode="action_map",
                    scenario=direction,
                    episode_index=episode_index,
                    step_index=step_index,
                    phase="step",
                    info=final_info,
                    action=action,
                    reward=reward,
                    done=done,
                )
                if bool(final_info.get("collision", False)) and collision["collision_step"] is None:
                    collision = collision_fields(final_info, step_index=step_index, action=action)
                if done:
                    break
        except Exception as exc:
            final_position = np.asarray(final_info.get("position", initial_position), dtype=np.float32)
            delta = final_position - initial_position
            return ActionMappingRecord(
                direction=direction,
                episode_index=episode_index,
                action=action_list(action) or [0.0, 0.0, 0.0, 0.0],
                initial_position=initial_position.astype(float).tolist(),
                final_position=final_position.astype(float).tolist(),
                delta_position=delta.astype(float).tolist(),
                delta_x=float(delta[0]),
                delta_y=float(delta[1]),
                delta_z=float(delta[2]),
                initial_distance=initial_distance,
                final_distance=info_float(final_info, "distance_to_goal"),
                distance_delta=(
                    initial_distance - float(final_info["distance_to_goal"])
                    if "distance_to_goal" in final_info
                    else None
                ),
                collision=bool(final_info.get("collision", False)) or collision["collision_step"] is not None,
                done=True,
                done_reason=str(final_info.get("done_reason", "exception")),
                **collision,
                collision_before_first_action=False,
                autopilot_state_start=autopilot_state_start,
                autopilot_state_end=str(final_info.get("autopilot_state", "UNKNOWN")),
                reset_retry_count=int(final_info.get("reset_retry_count", 0)),
                wall_time=float(time.perf_counter() - started),
                error=f"{type(exc).__name__}: {exc}",
            )

    final_position = np.asarray(final_info.get("position", initial_position), dtype=np.float32)
    delta = final_position - initial_position
    final_distance = info_float(final_info, "distance_to_goal")
    return ActionMappingRecord(
        direction=direction,
        episode_index=episode_index,
        action=action_list(action) or [0.0, 0.0, 0.0, 0.0],
        initial_position=initial_position.astype(float).tolist(),
        final_position=final_position.astype(float).tolist(),
        delta_position=delta.astype(float).tolist(),
        delta_x=float(delta[0]),
        delta_y=float(delta[1]),
        delta_z=float(delta[2]),
        initial_distance=initial_distance,
        final_distance=final_distance,
        distance_delta=initial_distance - final_distance if final_distance is not None else None,
        collision=bool(final_info.get("collision", False)) or collision["collision_step"] is not None,
        done=bool(done) or collision_before_first_action,
        done_reason=(
            "collision_before_first_action"
            if collision_before_first_action
            else str(final_info.get("done_reason", "running"))
        ),
        **collision,
        collision_before_first_action=collision_before_first_action,
        autopilot_state_start=autopilot_state_start,
        autopilot_state_end=str(final_info.get("autopilot_state", autopilot_state_start)),
        reset_retry_count=int(final_info.get("reset_retry_count", 0)),
        wall_time=float(time.perf_counter() - started),
    )


def failed_reset_record(
    *,
    episode_index: int,
    observe_seconds: float,
    started: float,
    error: Exception,
) -> ResetSanityRecord:
    return ResetSanityRecord(
        episode_index=episode_index,
        observe_seconds=observe_seconds,
        sample_count=0,
        initial_position=[],
        final_position=[],
        initial_distance=None,
        final_distance=None,
        collision_before_observe=False,
        collision_observed=False,
        first_collision_time=None,
        first_collision_position=None,
        first_collision_distance_to_goal=None,
        first_collision_height=None,
        first_collision_autopilot_state="",
        autopilot_state_start="UNKNOWN",
        autopilot_state_end="UNKNOWN",
        reset_retry_count=0,
        wall_time=float(time.perf_counter() - started),
        error=f"{type(error).__name__}: {error}",
    )


def run_reset_sanity(
    env: AvoidBenchRLEnv,
    *,
    episode_index: int,
    observe_seconds: float,
    sample_period: float,
    traces: list[dict[str, Any]],
) -> ResetSanityRecord:
    started = time.perf_counter()
    try:
        _obs, info = env.reset()
    except Exception as exc:
        return failed_reset_record(
            episode_index=episode_index,
            observe_seconds=observe_seconds,
            started=started,
            error=exc,
        )

    initial_position = list(info["position"])
    initial_distance = float(info["distance_to_goal"])
    autopilot_state_start = str(info.get("autopilot_state", "UNKNOWN"))
    reset_retry_count = int(info.get("reset_retry_count", 0))
    collision_before_observe = bool(info.get("collision", False))
    first_collision_time: float | None = None
    first_collision_position: list[float] | None = None
    first_collision_distance_to_goal: float | None = None
    first_collision_height: float | None = None
    first_collision_autopilot_state = ""
    if collision_before_observe:
        first_collision_time = 0.0
        first_collision_position = info_list(info, "position")
        first_collision_distance_to_goal = info_float(info, "distance_to_goal")
        first_collision_height = info_float(info, "height")
        first_collision_autopilot_state = str(info.get("autopilot_state", "UNKNOWN"))

    append_step_trace(
        traces,
        mode="reset_sanity",
        scenario="reset_only",
        episode_index=episode_index,
        step_index=0,
        phase="reset",
        info=info,
        action=None,
        done=collision_before_observe,
    )

    final_info = dict(info)
    sample_count = 0
    collision_observed = collision_before_observe
    deadline = time.monotonic() + observe_seconds
    latest_snapshot = env._latest_snapshot
    last_stamp = latest_snapshot.stamp if latest_snapshot is not None else 0.0

    while time.monotonic() < deadline and not rospy_shutdown():
        remaining = max(0.0, deadline - time.monotonic())
        try:
            snapshot = env._wait_for_fresh_snapshot(
                after_stamp=last_stamp,
                timeout=min(sample_period, remaining),
            )
            last_stamp = snapshot.stamp
        except TimeoutError:
            snapshot = env._wait_for_snapshot(timeout=0.1)
        final_info = snapshot_info(
            env,
            snapshot,
            done_reason="collision" if bool(env._latest_collision) else "running",
        )
        sample_count += 1
        append_step_trace(
            traces,
            mode="reset_sanity",
            scenario="reset_only",
            episode_index=episode_index,
            step_index=sample_count,
            phase="observe",
            info=final_info,
            action=None,
            done=bool(final_info.get("collision", False)),
        )
        if bool(final_info.get("collision", False)) and first_collision_time is None:
            collision_observed = True
            first_collision_time = float(time.perf_counter() - started)
            first_collision_position = info_list(final_info, "position")
            first_collision_distance_to_goal = info_float(final_info, "distance_to_goal")
            first_collision_height = info_float(final_info, "height")
            first_collision_autopilot_state = str(final_info.get("autopilot_state", "UNKNOWN"))
        else:
            collision_observed = collision_observed or bool(final_info.get("collision", False))
        if sample_period > 0.0:
            time.sleep(min(sample_period, max(0.0, deadline - time.monotonic())))

    return ResetSanityRecord(
        episode_index=episode_index,
        observe_seconds=float(observe_seconds),
        sample_count=sample_count,
        initial_position=initial_position,
        final_position=list(final_info.get("position", initial_position)),
        initial_distance=initial_distance,
        final_distance=info_float(final_info, "distance_to_goal"),
        collision_before_observe=collision_before_observe,
        collision_observed=collision_observed,
        first_collision_time=first_collision_time,
        first_collision_position=first_collision_position,
        first_collision_distance_to_goal=first_collision_distance_to_goal,
        first_collision_height=first_collision_height,
        first_collision_autopilot_state=first_collision_autopilot_state,
        autopilot_state_start=autopilot_state_start,
        autopilot_state_end=str(final_info.get("autopilot_state", autopilot_state_start)),
        reset_retry_count=reset_retry_count,
        wall_time=float(time.perf_counter() - started),
    )


def rospy_shutdown() -> bool:
    try:
        import rospy

        return bool(rospy.is_shutdown())
    except Exception:
        return False


def numeric_stats(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if finite.size == 0:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def summarize_policy(records: list[EpisodeRecord]) -> dict[str, Any]:
    grouped: dict[str, list[EpisodeRecord]] = {}
    for record in records:
        key = f"{record.strategy}:{record.frame}"
        grouped.setdefault(key, []).append(record)
    result = {}
    for key, rows in grouped.items():
        deltas = [row.distance_delta for row in rows if row.distance_delta is not None]
        final_lt_initial = [
            row.final_distance < row.initial_distance
            for row in rows
            if row.final_distance is not None and row.initial_distance is not None
        ]
        collision_steps = [row.collision_step for row in rows if row.collision_step is not None]
        result[key] = {
            "episode_count": len(rows),
            "mean_distance_delta": float(np.mean(deltas)) if deltas else None,
            "min_distance_delta": float(np.min(deltas)) if deltas else None,
            "max_distance_delta": float(np.max(deltas)) if deltas else None,
            "positive_progress_count": int(sum(final_lt_initial)),
            "positive_progress_ratio": float(np.mean(final_lt_initial)) if final_lt_initial else None,
            "collision_count": int(sum(row.collision for row in rows)),
            "collision_before_first_action_count": int(
                sum(row.collision_before_first_action for row in rows)
            ),
            "collision_step": numeric_stats([float(step) for step in collision_steps]),
            "errors": [row.error for row in rows if row.error],
            "done_reasons": dict(
                sorted(
                    {
                        reason: sum(row.done_reason == reason for row in rows)
                        for reason in {row.done_reason for row in rows}
                    }.items()
                )
            ),
            "mean_final_height": float(np.mean([row.final_height for row in rows if row.final_height is not None]))
            if any(row.final_height is not None for row in rows)
            else None,
        }
    zero_delta = result.get("zero:world", result.get("zero:none", {})).get("mean_distance_delta")
    best_goal_key = None
    best_goal_delta = None
    for key, payload in result.items():
        if not key.startswith("goal_direction:"):
            continue
        value = payload["mean_distance_delta"]
        if value is None:
            continue
        if best_goal_delta is None or value > best_goal_delta:
            best_goal_key = key
            best_goal_delta = value
    return {
        "by_strategy": result,
        "best_goal_direction_key": best_goal_key,
        "best_goal_direction_mean_delta": best_goal_delta,
        "zero_mean_delta": zero_delta,
        "goal_direction_better_than_zero": (
            best_goal_delta is not None
            and zero_delta is not None
            and best_goal_delta > zero_delta + 0.05
        ),
        "majority_goal_direction_positive": (
            best_goal_key is not None
            and result[best_goal_key]["positive_progress_ratio"] is not None
            and result[best_goal_key]["positive_progress_ratio"] > 0.5
        ),
    }


def summarize_action_map(records: list[ActionMappingRecord]) -> dict[str, Any]:
    grouped: dict[str, list[ActionMappingRecord]] = {}
    for record in records:
        grouped.setdefault(record.direction, []).append(record)
    by_direction: dict[str, Any] = {}
    for direction, rows in sorted(grouped.items()):
        deltas = [
            row.delta_position
            for row in rows
            if row.delta_position and len(row.delta_position) >= 3
        ]
        mean_delta = (
            np.mean(np.asarray(deltas, dtype=np.float64), axis=0)
            if deltas
            else np.zeros(3, dtype=np.float64)
        )
        horizontal_delta = mean_delta[:2]
        dominant_index = int(np.argmax(np.abs(horizontal_delta))) if deltas else None
        dominant_axis = ("x", "y")[dominant_index] if dominant_index is not None else None
        dominant_sign = None
        if dominant_index is not None:
            dominant_sign = "+" if horizontal_delta[dominant_index] >= 0.0 else "-"
        expected_axis = direction[1]
        expected_sign = direction[0]
        by_direction[direction] = {
            "episode_count": len(rows),
            "mean_delta_position": mean_delta.astype(float).tolist(),
            "horizontal_delta_norm": float(np.linalg.norm(horizontal_delta)),
            "mean_z_drift": float(mean_delta[2]) if deltas else None,
            "dominant_axis": dominant_axis,
            "dominant_sign": dominant_sign,
            "expected_axis": expected_axis,
            "expected_sign": expected_sign,
            "axis_sign_matches_command": (
                dominant_axis == expected_axis and dominant_sign == expected_sign
            )
            if dominant_axis is not None
            else None,
            "collision_count": int(sum(row.collision for row in rows)),
            "collision_before_first_action_count": int(
                sum(row.collision_before_first_action for row in rows)
            ),
            "done_reasons": dict(
                sorted(
                    {
                        reason: sum(row.done_reason == reason for row in rows)
                        for reason in {row.done_reason for row in rows}
                    }.items()
                )
            ),
            "errors": [row.error for row in rows if row.error],
        }
    return {
        "by_direction": by_direction,
        "directions_checked": sorted(grouped),
        "all_axis_signs_match": (
            all(
                payload["axis_sign_matches_command"] is True
                for payload in by_direction.values()
            )
            if by_direction
            else None
        ),
        "any_collision_before_first_action": (
            any(row.collision_before_first_action for row in records) if records else None
        ),
        "collision_count": int(sum(row.collision for row in records)),
    }


def summarize_reset_sanity(records: list[ResetSanityRecord]) -> dict[str, Any]:
    return {
        "episode_count": len(records),
        "collision_before_observe_count": int(sum(row.collision_before_observe for row in records)),
        "collision_observed_count": int(sum(row.collision_observed for row in records)),
        "first_collision_time": numeric_stats(
            [
                float(row.first_collision_time)
                for row in records
                if row.first_collision_time is not None
            ]
        ),
        "errors": [row.error for row in records if row.error],
        "reset_collision_sanity_passed": (
            all(not row.collision_observed and not row.error for row in records)
            if records
            else None
        ),
    }


def build_stage1_gate_summary(
    policy_records: list[EpisodeRecord],
    action_map_records: list[ActionMappingRecord],
    reset_records: list[ResetSanityRecord],
) -> dict[str, Any]:
    reasons: list[str] = []
    reset_pass = (
        all(not row.collision_observed and not row.error for row in reset_records)
        if reset_records
        else None
    )
    action_map_summary = summarize_action_map(action_map_records)
    action_map_pass = action_map_summary["all_axis_signs_match"]
    if action_map_pass is not None:
        action_map_pass = bool(action_map_pass) and action_map_summary["collision_count"] == 0
    goal_rows = [
        row
        for row in policy_records
        if row.strategy == "goal_direction" and not row.error
    ]
    goal_pass = (
        any(
            row.distance_delta is not None
            and row.distance_delta > 0.05
            and not row.collision
            for row in goal_rows
        )
        if goal_rows
        else None
    )

    if reset_pass is not True:
        reasons.append("reset-only collision sanity has not passed")
    if action_map_pass is not True:
        reasons.append("four-direction action mapping has not passed without collision")
    if goal_pass is not True:
        reasons.append("goal-direction policy has not proven stable non-collision progress")
    return {
        "reset_collision_sanity_passed": reset_pass,
        "action_mapping_passed": action_map_pass,
        "goal_direction_policy_passed": goal_pass,
        "stage1_short_navigation_allowed": (
            reset_pass is True and action_map_pass is True and goal_pass is True
        ),
        "reasons": reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe AvoidBench collision timing, action mapping, and hand-coded goal-direction policies."
    )
    parser.add_argument("--namespace", default="/hummingbird")
    parser.add_argument(
        "--mode",
        choices=("goal-policy", "action-map", "reset-sanity", "all"),
        default="goal-policy",
        help="Which triage phase to run. The default preserves the original goal-policy probe.",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--speed", type=float, default=0.08)
    parser.add_argument("--mapping-steps", type=int, default=30)
    parser.add_argument("--mapping-speed", type=float, default=None)
    parser.add_argument("--reset-checks", type=int, default=None)
    parser.add_argument("--reset-observe-seconds", type=float, default=2.0)
    parser.add_argument("--reset-sample-period", type=float, default=0.10)
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
    if args.mapping_steps <= 0:
        parser.error("--mapping-steps must be positive.")
    if args.mapping_speed is not None and args.mapping_speed <= 0.0:
        parser.error("--mapping-speed must be positive.")
    if args.reset_observe_seconds <= 0.0:
        parser.error("--reset-observe-seconds must be positive.")
    if args.reset_sample_period <= 0.0:
        parser.error("--reset-sample-period must be positive.")
    if args.reset_checks is not None and args.reset_checks <= 0:
        parser.error("--reset-checks must be positive.")

    output_dir = Path(args.output_root) / timestamp_slug()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = ("world", "body") if args.frame == "auto" else (args.frame,)
    scenarios = [("zero", "world"), ("constant_forward", "world")]
    scenarios.extend(("goal_direction", frame) for frame in frames)

    env = AvoidBenchRLEnv(namespace=args.namespace, action_preset=args.action_preset)
    policy_records: list[EpisodeRecord] = []
    action_map_records: list[ActionMappingRecord] = []
    reset_records: list[ResetSanityRecord] = []
    traces: list[dict[str, Any]] = []
    try:
        if args.mode in {"reset-sanity", "all"}:
            reset_checks = args.reset_checks or args.episodes
            for episode_index in range(reset_checks):
                record = run_reset_sanity(
                    env,
                    episode_index=episode_index,
                    observe_seconds=args.reset_observe_seconds,
                    sample_period=args.reset_sample_period,
                    traces=traces,
                )
                reset_records.append(record)
                print(
                    "reset_sanity "
                    f"ep={episode_index} collision_before={record.collision_before_observe} "
                    f"collision_observed={record.collision_observed} "
                    f"first_collision_time={record.first_collision_time} error={record.error}"
                )

        if args.mode in {"action-map", "all"}:
            mapping_speed = float(args.mapping_speed if args.mapping_speed is not None else args.speed)
            for direction, action in direction_actions(mapping_speed, env.action_bounds).items():
                for episode_index in range(args.episodes):
                    record = run_action_mapping_episode(
                        env,
                        direction=direction,
                        action=action,
                        episode_index=episode_index,
                        steps=args.mapping_steps,
                        traces=traces,
                    )
                    action_map_records.append(record)
                    print(
                        f"action_map {direction} ep={episode_index} "
                        f"delta=({record.delta_x}, {record.delta_y}, {record.delta_z}) "
                        f"done={record.done_reason} collision={record.collision} "
                        f"collision_step={record.collision_step} "
                        f"before_first={record.collision_before_first_action} error={record.error}"
                    )
                    if record.error:
                        break

        if args.mode in {"goal-policy", "all"}:
            for strategy, frame in scenarios:
                for episode_index in range(args.episodes):
                    record = run_episode(
                        env,
                        strategy=strategy,
                        frame=frame,
                        episode_index=episode_index,
                        steps=args.steps,
                        speed=args.speed,
                        traces=traces,
                    )
                    policy_records.append(record)
                    print(
                        f"{strategy}:{frame} ep={episode_index} "
                        f"delta={record.distance_delta} final={record.final_distance} "
                        f"done={record.done_reason} collision={record.collision} "
                        f"collision_step={record.collision_step} "
                        f"before_first={record.collision_before_first_action} error={record.error}"
                    )
                    if record.error:
                        break
    finally:
        env.close()

    summary = {
        "namespace": args.namespace,
        "mode": args.mode,
        "episodes": args.episodes,
        "steps": args.steps,
        "speed": args.speed,
        "mapping_steps": args.mapping_steps,
        "mapping_speed": args.mapping_speed if args.mapping_speed is not None else args.speed,
        "reset_checks": args.reset_checks or args.episodes,
        "reset_observe_seconds": args.reset_observe_seconds,
        "reset_sample_period": args.reset_sample_period,
        "action_preset": args.action_preset,
        "frame": args.frame,
        "output_dir": str(output_dir),
        "records": {
            "policy": policy_records,
            "action_map": action_map_records,
            "reset_sanity": reset_records,
        },
        "summary": {
            "policy": summarize_policy(policy_records),
            "action_map": summarize_action_map(action_map_records),
            "reset_sanity": summarize_reset_sanity(reset_records),
            "stage1_gate": build_stage1_gate_summary(
                policy_records,
                action_map_records,
                reset_records,
            ),
        },
    }
    write_csv(output_dir / "episodes.csv", policy_records)
    write_csv(output_dir / "action_map.csv", action_map_records)
    write_csv(output_dir / "reset_sanity.csv", reset_records)
    write_csv(output_dir / "step_trace.csv", traces)
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
