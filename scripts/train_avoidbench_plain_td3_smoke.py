from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, deque
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from algos.td3.td3_plain import TD3Plain
from data.replay_buffer import ReplayBuffer
from envs.avoidbench.rl_env import (
    ACTION_NAMES,
    ACTION_PRESETS,
    AvoidBenchRLEnv,
    reward_done_config_for_task,
)


SATURATION_THRESHOLD = 0.95


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


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(jsonable(payload), sort_keys=True) + "\n")


def write_episodes(path: Path, episodes: list[dict[str, Any]]) -> None:
    if not episodes:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episodes[0].keys()))
        writer.writeheader()
        writer.writerows(episodes)


def dimension_stats(values: list[np.ndarray]) -> dict[str, dict[str, float]]:
    if not values:
        return {
            name: {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "saturation_percentage": 0.0,
            }
            for name in ACTION_NAMES
        }
    array = np.stack(values)
    result = {}
    for index, name in enumerate(ACTION_NAMES):
        column = array[:, index]
        result[name] = {
            "mean": float(column.mean()),
            "std": float(column.std()),
            "min": float(column.min()),
            "max": float(column.max()),
            "saturation_percentage": float(
                np.mean(np.abs(column) >= SATURATION_THRESHOLD) * 100.0
            ),
        }
    return result


def build_episode_record(
    *,
    episode_index: int,
    end_step: int,
    episode_return: float,
    episode_steps: list[dict[str, Any]],
    final_info: dict[str, Any],
    initial_distance: float,
) -> dict[str, Any]:
    heights = [float(row["height"]) for row in episode_steps]
    height_errors = [float(row["height_error"]) for row in episode_steps]
    vertical_velocities = [float(row["vertical_velocity"]) for row in episode_steps]
    progress = [float(row["progress"]) for row in episode_steps]
    raw_actions = [
        np.asarray(row["raw_actor_action"], dtype=np.float32)
        for row in episode_steps
        if row["raw_actor_action"] is not None
    ]
    saturation = (
        float(
            np.mean(
                np.abs(np.stack(raw_actions)) >= SATURATION_THRESHOLD,
            )
            * 100.0
        )
        if raw_actions
        else 0.0
    )
    return {
        "episode_index": episode_index,
        "end_step": end_step,
        "episode_return": float(episode_return),
        "episode_length": len(episode_steps),
        "done_reason": str(final_info["done_reason"]),
        "collision": bool(final_info["collision"]),
        "initial_distance": float(initial_distance),
        "final_distance": float(final_info["distance_to_goal"]),
        "distance_improvement": float(initial_distance - final_info["distance_to_goal"]),
        "progress_sum": float(sum(progress)),
        "min_height": float(min(heights)),
        "max_height": float(max(heights)),
        "mean_height": float(np.mean(heights)),
        "mean_height_error": float(np.mean(height_errors)),
        "max_height_error": float(max(height_errors)),
        "mean_vertical_velocity": float(np.mean(vertical_velocities)),
        "max_abs_vertical_velocity": float(np.max(np.abs(vertical_velocities))),
        "raw_actor_saturation_percentage": saturation,
        "reset_retry_count": int(final_info["reset_retry_count"]),
    }


def hover_gate(
    episodes: list[dict[str, Any]],
    raw_actor_actions: list[np.ndarray],
    failure_reason: str,
) -> dict[str, Any]:
    completed = len(episodes)
    height_terminations = sum(
        episode["done_reason"] in {"height_too_low", "height_too_high"}
        for episode in episodes
    )
    mean_length = (
        float(np.mean([episode["episode_length"] for episode in episodes]))
        if episodes
        else 0.0
    )
    saturation = (
        float(
            np.mean(np.abs(np.stack(raw_actor_actions)) >= SATURATION_THRESHOLD) * 100.0
        )
        if raw_actor_actions
        else 100.0
    )
    height_percentage = 100.0 * height_terminations / completed if completed else 100.0
    checks = {
        "no_runtime_failure": not failure_reason,
        "completed_episodes": completed > 0,
        "height_termination_percentage_le_30": height_percentage <= 30.0,
        "mean_episode_length_ge_100": mean_length >= 100.0,
        "raw_actor_saturation_percentage_le_25": saturation <= 25.0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "completed_episodes": completed,
        "height_termination_count": height_terminations,
        "height_termination_percentage": height_percentage,
        "mean_episode_length": mean_length,
        "raw_actor_saturation_percentage": saturation,
        "comparison_baseline": {
            "old_height_termination_percentage": 95.34883720930233,
            "old_mean_episode_length": 45.30232558139535,
            "old_actor_action_std_mean": 0.9028084356533853,
            "old_per_dimension_saturation": "unavailable",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Plain TD3 smoke training on AvoidBenchRLEnv.")
    parser.add_argument("--namespace", default="/hummingbird")
    parser.add_argument(
        "--task-mode",
        choices=("hover_smoke", "navigation_smoke"),
        default="hover_smoke",
    )
    parser.add_argument(
        "--action-preset",
        choices=tuple(ACTION_PRESETS),
        default="conservative",
    )
    parser.add_argument("--total-steps", type=int, default=5000)
    parser.add_argument("--episode-steps", type=int, default=200)
    parser.add_argument("--start-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--exploration-noise", type=float, default=0.10)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--output-root",
        default="runs/avoidbench_plain_td3_smoke",
    )
    args = parser.parse_args()
    if not 1 <= args.total_steps <= 10000:
        parser.error("--total-steps must be within [1, 10000].")
    if not 1 <= args.episode_steps <= 1000:
        parser.error("--episode-steps must be within [1, 1000].")
    if args.start_steps < 0:
        parser.error("--start-steps must be non-negative.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    output_dir = Path(args.output_root) / f"{timestamp_slug()}-{args.task_mode}"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    steps_path = output_dir / "steps.jsonl"
    episodes_path = output_dir / "episodes.csv"
    config_path = output_dir / "config.json"
    summary_path = output_dir / "summary.json"

    reward_config = reward_done_config_for_task(
        args.task_mode,
        target_height=1.2,
        max_episode_steps=args.episode_steps,
    )
    config = {
        "namespace": args.namespace,
        "task_mode": args.task_mode,
        "action_preset": args.action_preset,
        "action_names": list(ACTION_NAMES),
        "action_bounds": list(ACTION_PRESETS[args.action_preset]),
        "total_steps": args.total_steps,
        "episode_steps": args.episode_steps,
        "start_steps": args.start_steps,
        "batch_size": args.batch_size,
        "exploration_noise": args.exploration_noise,
        "log_interval": args.log_interval,
        "saturation_threshold": SATURATION_THRESHOLD,
        "seed": args.seed,
        "observation_mode": "lowdim_only",
        "action_interface": "/hummingbird/autopilot/velocity_command",
        "reward_done_config": asdict(reward_config),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    write_json(config_path, config)

    env = AvoidBenchRLEnv(
        namespace=args.namespace,
        action_preset=args.action_preset,
        reward_done_config=reward_config,
    )
    replay_buffer = ReplayBuffer(
        state_dim=env.observation_dim,
        action_dim=env.action_dim,
        max_size=max(10000, args.total_steps),
    )
    policy = TD3Plain(
        state_dim=env.observation_dim,
        action_dim=env.action_dim,
        max_action=1.0,
    )

    episodes: list[dict[str, Any]] = []
    all_raw_actor_actions: list[np.ndarray] = []
    all_scaled_actions: list[np.ndarray] = []
    recent_raw_actor_actions: deque[np.ndarray] = deque(maxlen=args.log_interval)
    recent_scaled_actions: deque[np.ndarray] = deque(maxlen=args.log_interval)
    recent_step_times: deque[float] = deque(maxlen=args.log_interval)
    recent_height_errors: deque[float] = deque(maxlen=args.log_interval)
    recent_vertical_velocities: deque[float] = deque(maxlen=args.log_interval)
    latest_train_metrics: dict[str, Any] = {}
    done_reason_counts: Counter[str] = Counter()
    collision_count = 0
    episode_return = 0.0
    episode_index = 0
    episode_steps: list[dict[str, Any]] = []
    training_started = time.perf_counter()
    failure_reason = ""

    try:
        state, reset_info = env.reset()
        initial_distance = float(reset_info["distance_to_goal"])
        for total_step in range(1, args.total_steps + 1):
            raw_actor_action: np.ndarray | None = None
            exploration_noise = np.zeros(env.action_dim, dtype=np.float32)
            if total_step <= args.start_steps:
                normalized_pre_clip = rng.uniform(
                    -1.0,
                    1.0,
                    size=env.action_dim,
                ).astype(np.float32)
                action_source = "random"
            else:
                raw_actor_action = policy.select_action(state).astype(np.float32)
                exploration_noise = rng.normal(
                    0.0,
                    args.exploration_noise,
                    size=env.action_dim,
                ).astype(np.float32)
                normalized_pre_clip = raw_actor_action + exploration_noise
                all_raw_actor_actions.append(raw_actor_action.copy())
                recent_raw_actor_actions.append(raw_actor_action.copy())
                action_source = "actor"

            normalized_action = np.clip(normalized_pre_clip, -1.0, 1.0)
            scaled_action = normalized_action * env.action_bounds
            all_scaled_actions.append(scaled_action.copy())
            recent_scaled_actions.append(scaled_action.copy())

            step_started = time.perf_counter()
            next_state, reward, done, info = env.step(scaled_action)
            wall_step_time = time.perf_counter() - step_started
            if wall_step_time > 3.0:
                raise RuntimeError(f"Abnormal environment step time: {wall_step_time:.3f}s.")

            replay_buffer.push(
                state=state,
                action=normalized_action,
                reward=reward,
                next_state=next_state,
                done=done,
            )
            if replay_buffer.size >= args.batch_size and total_step > args.start_steps:
                latest_train_metrics = policy.train(
                    replay_buffer,
                    batch_size=args.batch_size,
                )

            step_record = {
                "step": total_step,
                "episode_index": episode_index,
                "episode_step": len(episode_steps) + 1,
                "action_source": action_source,
                "state": state,
                "next_state": next_state,
                "raw_actor_action": raw_actor_action,
                "exploration_noise": exploration_noise,
                "normalized_action_pre_clip": normalized_pre_clip,
                "normalized_action": normalized_action,
                "normalized_action_clipped": bool(
                    np.any(np.abs(normalized_pre_clip - normalized_action) > 1e-6)
                ),
                "scaled_velocity_command": scaled_action,
                "action_norm": info["action_norm"],
                "z_action_abs": abs(float(scaled_action[2])),
                "reward": float(reward),
                "done": bool(done),
                "done_reason": info["done_reason"],
                "position": info["position"],
                "velocity": info["velocity"],
                "height": info["height"],
                "target_height": info["target_height"],
                "height_error": info["height_error"],
                "vertical_velocity": info["vertical_velocity"],
                "distance_to_goal": info["distance_to_goal"],
                "progress": info["progress"],
                "collision": info["collision"],
                "height_penalty": info["height_penalty"],
                "vertical_velocity_penalty": info["vertical_velocity_penalty"],
                "z_action_penalty": info["z_action_penalty"],
                "step_time": wall_step_time,
                "reset_retry_count": info["reset_retry_count"],
                "autopilot_state": info["autopilot_state"],
                "critic_loss": latest_train_metrics.get("critic_loss"),
                "actor_loss": latest_train_metrics.get("actor_loss"),
                "actor_sat_pct_batch": latest_train_metrics.get("actor_sat_pct"),
            }
            append_jsonl(steps_path, step_record)
            episode_steps.append(jsonable(step_record))

            state = next_state
            episode_return += float(reward)
            recent_step_times.append(wall_step_time)
            recent_height_errors.append(float(info["height_error"]))
            recent_vertical_velocities.append(float(info["vertical_velocity"]))
            collision_count += int(bool(info["collision"]))

            if done:
                episodes.append(
                    build_episode_record(
                        episode_index=episode_index,
                        end_step=total_step,
                        episode_return=episode_return,
                        episode_steps=episode_steps,
                        final_info=info,
                        initial_distance=initial_distance,
                    )
                )
                done_reason_counts[info["done_reason"]] += 1
                episode_index += 1
                state, reset_info = env.reset()
                initial_distance = float(reset_info["distance_to_goal"])
                episode_return = 0.0
                episode_steps = []

            if total_step % args.log_interval == 0 or total_step == args.total_steps:
                raw_stats = dimension_stats(list(recent_raw_actor_actions))
                scaled_stats = dimension_stats(
                    [
                        action / env.action_bounds
                        for action in recent_scaled_actions
                    ]
                )
                metrics = {
                    "step": total_step,
                    "replay_buffer_size": replay_buffer.size,
                    "episode_index": episode_index,
                    "episode_return": episode_return,
                    "episode_length": len(episode_steps),
                    "collision_count": collision_count,
                    "done_reason_counts": dict(done_reason_counts),
                    "raw_actor_action": raw_stats,
                    "scaled_action_normalized": scaled_stats,
                    "mean_height_error": float(np.mean(recent_height_errors)),
                    "mean_abs_vertical_velocity": float(
                        np.mean(np.abs(recent_vertical_velocities))
                    ),
                    "critic_loss": latest_train_metrics.get("critic_loss"),
                    "actor_loss": latest_train_metrics.get("actor_loss"),
                    "actor_sat_pct_batch": latest_train_metrics.get("actor_sat_pct"),
                    "mean_step_time": float(np.mean(recent_step_times)),
                    "action_source": action_source,
                }
                append_jsonl(metrics_path, metrics)
                print(
                    f"step={total_step} replay={replay_buffer.size} "
                    f"episode={episode_index} ep_return={episode_return:.4f} "
                    f"ep_len={len(episode_steps)} collisions={collision_count} "
                    f"height_error={metrics['mean_height_error']:.4f} "
                    f"abs_vz={metrics['mean_abs_vertical_velocity']:.4f} "
                    f"vz_sat={raw_stats['vz']['saturation_percentage']:.2f}% "
                    f"critic_loss={metrics['critic_loss']} "
                    f"step_time={metrics['mean_step_time']:.4f}",
                    flush=True,
                )
    except Exception as exc:  # pragma: no cover - live ROS smoke gate
        failure_reason = f"{type(exc).__name__}: {exc}"
    finally:
        env.close()

    write_episodes(episodes_path, episodes)
    checkpoint_prefix = output_dir / "td3_smoke"
    if not failure_reason:
        policy.save(str(checkpoint_prefix))

    action_statistics = {
        "raw_actor_action": dimension_stats(all_raw_actor_actions),
        "scaled_action_as_fraction_of_bounds": dimension_stats(
            [action / env.action_bounds for action in all_scaled_actions]
        ),
    }
    gate = (
        hover_gate(episodes, all_raw_actor_actions, failure_reason)
        if args.task_mode == "hover_smoke"
        else None
    )
    summary = {
        "status": "FAILED_GATE" if failure_reason else "OK",
        "failure_reason": failure_reason,
        "task_mode": args.task_mode,
        "action_preset": args.action_preset,
        "output_dir": str(output_dir),
        "completed_steps": replay_buffer.size,
        "replay_buffer_size": replay_buffer.size,
        "episodes_completed": len(episodes),
        "collision_count": collision_count,
        "done_reason_counts": dict(done_reason_counts),
        "td3_updates": policy.total_it,
        "action_statistics": action_statistics,
        "hover_gate": gate,
        "checkpoint_prefix": str(checkpoint_prefix) if not failure_reason else None,
        "elapsed_seconds": time.perf_counter() - training_started,
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if failure_reason:
        return 1
    if args.task_mode == "hover_smoke" and gate is not None and not gate["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
