from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from envs.avoidbench.rl_env import AvoidBenchRLEnv


@dataclass
class EpisodeRecord:
    scenario: str
    episode_index: int
    episode_return: float
    episode_length: int
    initial_distance: float
    final_distance: float
    distance_delta: float
    collision: bool
    done: bool
    done_reason: str
    min_height: float
    max_height: float
    mean_height: float
    mean_speed: float
    max_speed: float
    mean_action_norm: float
    mean_step_time: float
    max_step_time: float
    reset_time: float
    autopilot_state_start: str
    autopilot_state_end: str
    wall_time: float
    error: str = ""


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def ensure_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): ensure_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [ensure_jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(ensure_jsonable(payload), indent=2, sort_keys=True) + "\n")


def write_episode_csv(path: Path, records: list[EpisodeRecord]) -> None:
    if not records:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def infer_done_reason(done: bool, info: dict[str, Any], episode_length: int, step_budget: int) -> str:
    info_reason = str(info.get("done_reason", "running"))
    if info_reason != "running":
        return info_reason
    if done and info.get("collision"):
        return "collision"
    if done and info.get("success"):
        return "goal_reached"
    if done and episode_length >= step_budget:
        return "timeout"
    if episode_length >= step_budget:
        return "budget_exhausted"
    return "running"


def build_action_factory(name: str, rng: np.random.Generator):
    if name == "zero_action":
        return lambda env: np.zeros(env.action_dim, dtype=np.float32)
    if name == "constant_forward_action":
        return lambda env: np.array(
            [0.75 * float(env.action_bounds[0]), 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
    if name == "random_action":
        return lambda env: env.sample_random_action(rng)
    raise ValueError(f"Unknown action scenario {name}.")


def run_reset_only(env: AvoidBenchRLEnv, count: int) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for reset_idx in range(count):
        t0 = time.perf_counter()
        try:
            obs, info = env.reset()
            elapsed = time.perf_counter() - t0
            position = info.get("position", obs[0:3].tolist())
            velocity = info.get("velocity", obs[3:6].tolist())
            records.append(
                {
                    "reset_index": reset_idx,
                    "success": True,
                    "reset_time": elapsed,
                    "obs_shape": list(obs.shape),
                    "autopilot_state": info.get("autopilot_state", "UNKNOWN"),
                    "final_z": float(info.get("height", position[2])),
                    "final_position": position,
                    "final_velocity": velocity,
                    "retry_count": int(info.get("reset_retry_count", 0)),
                    "failure_reason": "",
                    "distance_to_goal": info.get("distance_to_goal", info.get("distance")),
                }
            )
        except Exception as exc:  # pragma: no cover - exercised in live runtime
            elapsed = time.perf_counter() - t0
            failures.append(f"reset_only failure at index {reset_idx}: {exc}")
            records.append(
                {
                    "reset_index": reset_idx,
                    "success": False,
                    "reset_time": elapsed,
                    "final_z": None,
                    "final_position": None,
                    "final_velocity": None,
                    "autopilot_state": "UNKNOWN",
                    "retry_count": int(getattr(env, "_last_reset_retry_count", 0)),
                    "failure_reason": repr(exc),
                }
            )
    return records, failures


def run_episode(
    env: AvoidBenchRLEnv,
    scenario: str,
    episode_index: int,
    step_budget: int,
    action_factory,
) -> tuple[EpisodeRecord, dict[str, Any]]:
    reset_t0 = time.perf_counter()
    obs, info = env.reset()
    reset_time = time.perf_counter() - reset_t0

    episode_return = 0.0
    step_times: list[float] = []
    speeds: list[float] = []
    action_norms: list[float] = []
    heights: list[float] = [float(info.get("height", info.get("position", [0.0, 0.0, 0.0])[2]))]
    done = False
    final_info = dict(info)
    initial_distance = float(info.get("distance_to_goal", info.get("distance", np.nan)))
    autopilot_state_start = str(info.get("autopilot_state", "UNKNOWN"))

    for step_idx in range(step_budget):
        action = action_factory(env)
        action_norms.append(float(np.linalg.norm(action)))
        step_t0 = time.perf_counter()
        obs, reward, done, info = env.step(action)
        step_times.append(time.perf_counter() - step_t0)
        episode_return += float(reward)
        final_info = dict(info)
        velocity = np.asarray(info.get("velocity", obs[3:6]), dtype=np.float32)
        position = np.asarray(info.get("position", obs[0:3]), dtype=np.float32)
        speeds.append(float(np.linalg.norm(velocity)))
        heights.append(float(info.get("height", position[2])))
        if done:
            break

    record = EpisodeRecord(
        scenario=scenario,
        episode_index=episode_index,
        episode_return=float(episode_return),
        episode_length=(step_idx + 1) if step_budget > 0 else 0,
        initial_distance=initial_distance,
        final_distance=float(final_info.get("distance_to_goal", final_info.get("distance", np.nan))),
        distance_delta=float(
            initial_distance
            - float(final_info.get("distance_to_goal", final_info.get("distance", np.nan)))
        ),
        collision=bool(final_info.get("collision", False)),
        done=bool(done),
        done_reason=infer_done_reason(done, final_info, step_idx + 1, step_budget),
        min_height=float(min(heights)) if heights else float("nan"),
        max_height=float(max(heights)) if heights else float("nan"),
        mean_height=float(np.mean(heights)) if heights else float("nan"),
        mean_speed=float(np.mean(speeds)) if speeds else 0.0,
        max_speed=float(np.max(speeds)) if speeds else 0.0,
        mean_action_norm=float(np.mean(action_norms)) if action_norms else 0.0,
        mean_step_time=float(np.mean(step_times)) if step_times else 0.0,
        max_step_time=float(np.max(step_times)) if step_times else 0.0,
        reset_time=float(reset_time),
        autopilot_state_start=autopilot_state_start,
        autopilot_state_end=str(final_info.get("autopilot_state", "UNKNOWN")),
        wall_time=float(sum(step_times) + reset_time),
    )
    return record, final_info


def main() -> int:
    parser = argparse.ArgumentParser(description="Stress test the minimal AvoidBenchRLEnv.")
    parser.add_argument("--namespace", default="/hummingbird", help="ROS namespace to use.")
    parser.add_argument(
        "--action-preset",
        choices=("legacy", "conservative"),
        default="conservative",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed for random-action episodes.")
    parser.add_argument(
        "--output-root",
        default="runs/avoidbench_env_stress",
        help="Directory under which timestamped stress results are written.",
    )
    parser.add_argument(
        "--consecutive-collision-threshold",
        type=int,
        default=3,
        help="Abort if this many episode-ending collisions happen consecutively.",
    )
    parser.add_argument(
        "--mode",
        choices=("full", "reset-only"),
        default="full",
        help="Run only repeated resets or the complete environment stress suite.",
    )
    parser.add_argument(
        "--num-resets",
        type=int,
        default=20,
        help="Number of reset attempts in the reset-only gate.",
    )
    args = parser.parse_args()
    if args.num_resets <= 0:
        parser.error("--num-resets must be positive.")

    output_dir = Path(args.output_root) / timestamp_slug()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    env = AvoidBenchRLEnv(namespace=args.namespace, action_preset=args.action_preset)
    scenarios = [
        ("zero_action", 3, 100),
        ("constant_forward_action", 3, 100),
        ("random_action", 5, 200),
    ]

    episode_records: list[EpisodeRecord] = []
    failures: list[str] = []
    reset_records, reset_failures = run_reset_only(env, count=args.num_resets)
    failures.extend(reset_failures)

    write_json(
        output_dir / "reset_only.json",
        {
            "namespace": args.namespace,
            "seed": args.seed,
            "records": reset_records,
            "failures": reset_failures,
        },
    )

    successful_resets = sum(bool(record.get("success")) for record in reset_records)
    max_consecutive_successes = 0
    current_consecutive_successes = 0
    for record in reset_records:
        if record.get("success"):
            current_consecutive_successes += 1
            max_consecutive_successes = max(
                max_consecutive_successes,
                current_consecutive_successes,
            )
        else:
            current_consecutive_successes = 0

    reset_gate_passed = (
        successful_resets == args.num_resets
        and max_consecutive_successes >= min(10, args.num_resets)
        and all(
            record.get("final_z") is not None and record["final_z"] >= env.takeoff_height
            for record in reset_records
            if record.get("success")
        )
    )
    if not reset_gate_passed and not failures:
        failures.append(
            "Reset-only gate failed: "
            f"successes={successful_resets}/{args.num_resets}, "
            f"max_consecutive={max_consecutive_successes}."
        )

    if args.mode == "reset-only" or failures:
        write_episode_csv(output_dir / "episodes.csv", episode_records)
        summary = {
            "status": "OK" if reset_gate_passed and not failures else "FAILED_GATE",
            "mode": args.mode,
            "failures": failures,
            "reset_only_count": len(reset_records),
            "successful_resets": successful_resets,
            "max_consecutive_successes": max_consecutive_successes,
            "episode_count": len(episode_records),
            "output_dir": str(output_dir),
        }
        write_json(output_dir / "summary.json", summary)
        env.close()
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["status"] == "OK" else 1

    consecutive_collision_episodes = 0
    for scenario_name, episode_count, step_budget in scenarios:
        action_factory = build_action_factory(scenario_name, rng)
        for episode_index in range(episode_count):
            try:
                record, final_info = run_episode(
                    env=env,
                    scenario=scenario_name,
                    episode_index=episode_index,
                    step_budget=step_budget,
                    action_factory=action_factory,
                )
                episode_records.append(record)
            except Exception as exc:  # pragma: no cover - exercised in live runtime
                failures.append(
                    f"{scenario_name} episode {episode_index} failed with {type(exc).__name__}: {exc}"
                )
                break

            print(
                f"{scenario_name} ep={episode_index} "
                f"len={record.episode_length} return={record.episode_return:.4f} "
                f"dist={record.final_distance:.4f} collision={record.collision} "
                f"done_reason={record.done_reason} autopilot={record.autopilot_state_end}"
            )

            if record.collision and record.done_reason == "collision":
                consecutive_collision_episodes += 1
            else:
                consecutive_collision_episodes = 0

            if consecutive_collision_episodes >= args.consecutive_collision_threshold:
                failures.append(
                    f"Aborted after {consecutive_collision_episodes} consecutive collision-terminated episodes."
                )
                break

            if record.autopilot_state_end == "OFF":
                failures.append(
                    f"Autopilot returned to OFF during {scenario_name} episode {episode_index}."
                )
                break

            if record.max_step_time > 2.0:
                failures.append(
                    f"Abnormal step time {record.max_step_time:.3f}s in "
                    f"{scenario_name} episode {episode_index}."
                )
                break

        if failures:
            break

    write_episode_csv(output_dir / "episodes.csv", episode_records)
    scenario_summary: dict[str, dict[str, Any]] = {}
    for scenario_name, _, _ in scenarios:
        records = [record for record in episode_records if record.scenario == scenario_name]
        if not records:
            continue
        scenario_summary[scenario_name] = {
            "episode_count": len(records),
            "mean_return": float(np.mean([record.episode_return for record in records])),
            "mean_length": float(np.mean([record.episode_length for record in records])),
            "collision_episodes": int(sum(record.collision for record in records)),
            "mean_final_distance": float(np.mean([record.final_distance for record in records])),
            "mean_distance_delta": float(np.mean([record.distance_delta for record in records])),
            "mean_height": float(np.mean([record.mean_height for record in records])),
            "max_speed": float(np.max([record.max_speed for record in records])),
            "mean_action_norm": float(np.mean([record.mean_action_norm for record in records])),
            "mean_reset_time": float(np.mean([record.reset_time for record in records])),
            "mean_step_time": float(np.mean([record.mean_step_time for record in records])),
            "max_step_time": float(np.max([record.max_step_time for record in records])),
        }

    summary = {
        "status": "FAILED_GATE" if failures else "OK",
        "namespace": args.namespace,
        "seed": args.seed,
        "output_dir": str(output_dir),
        "reset_only_count": len(reset_records),
        "episode_count": len(episode_records),
        "failures": failures,
        "scenario_summary": scenario_summary,
    }
    write_json(output_dir / "summary.json", summary)
    env.close()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
