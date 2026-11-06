"""Evaluate runtime asynchronous high-level inference plus low-level DSLPID.

The main process is the low-level loop.  It never calls Encoder.forward(),
Actor.forward(), a learned Planner, or a decision_factory.  A spawned high-level
worker owns frozen-network inference and publishes latest-only ReferencePacket
messages back to the low-level loop.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from async_control import AsyncBuffer, LowLevelReferenceSelector, RuntimeHighLevelWorker, RuntimeWorkerConfig
from async_control.runtime_timing import (
    LOW_LEVEL_PERIOD_SECONDS,
    RuntimeTimingStats,
    count_low_cycles_during_high_compute,
    summarize_high_level_events,
    summarize_series,
)
from envs.ForestAviary import CustomForestAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import ActionType, ObservationType
from models.motor_action_codec import ASYMMETRIC_RPM, MotorActionCodec, MotorPhysicalLimits
from scripts.evaluate_native_latent_actor import clean_json, terminal_reason, write_json


PROFILES = (
    ("A_runtime_no_extra_delay", 0.0),
    ("B_runtime_worker_sleep_20ms", 0.020),
    ("C_runtime_worker_sleep_50ms", 0.050),
    ("D_runtime_worker_sleep_100ms", 0.100),
)


def quiet_env(**kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return CustomForestAviary(**kwargs)


def make_motor_codec(env) -> MotorActionCodec:
    codec = MotorActionCodec(
        MotorPhysicalLimits(
            min_rpm=0.0,
            hover_rpm=float(env.HOVER_RPM),
            max_rpm=float(env.MAX_RPM),
            kf=float(env.KF),
        ),
        mode=ASYMMETRIC_RPM,
    )
    env.set_motor_action_codec(codec)
    return codec


def make_snapshot(env, observation, *, sim_time: float, wall_time: float, buffer: AsyncBuffer, lower: LowLevelReferenceSelector) -> dict:
    return {
        "sim_time": float(sim_time),
        "wall_time": float(wall_time),
        "observation": np.asarray(observation, dtype=np.float32).reshape(-1).copy(),
        "position": np.asarray(env.pos[0], dtype=np.float32).copy(),
        "velocity": np.asarray(env.vel[0], dtype=np.float32).copy(),
        "yaw": float(env.rpy[0][2]),
        "target_position": np.asarray(env.TARGET_POS, dtype=np.float32).copy(),
        "fallback_position": lower.last_position.copy(),
        "fallback_velocity": lower.last_velocity.copy(),
        "previous_packet": buffer.latest_packet,
    }


def profile_summary(episodes: list[dict]) -> dict:
    if not episodes:
        return {"episode_count": 0}

    def mean(key):
        return float(np.mean([episode[key] for episode in episodes]))

    return {
        "episode_count": len(episodes),
        "success_rate": mean("success"),
        "collision_rate": mean("collision"),
        "mean_final_distance": mean("final_distance"),
        "mean_reward": mean("reward"),
        "total_low_level_cycles": int(sum(ep["low_level_selector"]["low_level_cycles"] for ep in episodes)),
        "total_fallback_cycles": int(sum(ep["fallback_cycles"] for ep in episodes)),
        "total_stale_packet_cycles": int(sum(ep["low_level_selector"]["stale_packet_cycles"] for ep in episodes)),
        "total_missing_packet_cycles": int(sum(ep["low_level_selector"]["missing_packet_cycles"] for ep in episodes)),
        "accepted_packets": int(sum(ep["buffer"]["accepted_packets"] for ep in episodes)),
        "rejected_packets": int(sum(ep["buffer"]["rejected_packets"] for ep in episodes)),
        "packet_update_interval_wall_seconds": summarize_series(
            [
                interval
                for episode in episodes
                for interval in episode["packet_update_intervals_wall_seconds"]
            ]
        ),
        "high_level": summarize_high_level_events(
            [event for episode in episodes for event in episode["high_level_events"]]
        ),
        "low_level_cycle_duration": summarize_series(
            [
                value
                for episode in episodes
                for value in episode["low_level_timing_raw"]["cycle_duration"]
            ],
            deadline_seconds=LOW_LEVEL_PERIOD_SECONDS,
        ),
        "low_level_environment_step": summarize_series(
            [
                value
                for episode in episodes
                for value in episode["low_level_timing_raw"]["environment_step_time"]
            ],
            deadline_seconds=LOW_LEVEL_PERIOD_SECONDS,
        ),
        "low_level_cycles_during_high_compute": {
            "max": int(max(ep["low_cycles_during_high_compute"]["max"] for ep in episodes)),
            "mean": float(np.mean([ep["low_cycles_during_high_compute"]["mean"] for ep in episodes])),
        },
    }


def run_episode(*, profile_name: str, artificial_delay: float, seed: int, args) -> dict:
    env = quiet_env(
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        pyb_freq=240,
        ctrl_freq=120,
        gui=False,
        curriculum=False,
        num_trees=int(args.num_trees),
        route_blocking_tree=bool(args.route_blocking_tree),
        episode_len_sec=float(args.episode_len_sec),
    )
    worker = RuntimeHighLevelWorker(
        RuntimeWorkerConfig(
            checkpoint_prefix=args.checkpoint,
            high_frequency_hz=args.high_frequency_hz,
            artificial_delay_seconds=float(artificial_delay),
            policy_mode="latent_actor_adapter",
        )
    )
    worker.start()
    high_events = []
    startup_deadline = time.perf_counter() + float(args.worker_startup_timeout)
    try:
        while time.perf_counter() < startup_deadline:
            high_events.extend(worker.get_all_status())
            if any(event.get("type") == "started" for event in high_events):
                break
            if any(event.get("type") == "error" for event in high_events):
                raise RuntimeError(high_events[-1]["traceback"])
            time.sleep(0.005)
        if not any(event.get("type") == "started" for event in high_events):
            raise TimeoutError("High-level worker did not report startup before timeout.")

        observation, _ = env.reset(seed=int(seed))
        codec = make_motor_codec(env)
        controller = DSLPIDControl(drone_model=env.DRONE_MODEL)
        controller.reset()
        buffer = AsyncBuffer()
        lower = LowLevelReferenceSelector(env.pos[0])
        timing = RuntimeTimingStats()
        previous_cycle_start = None
        packet_commit_wall_times = []
        packet_versions = []
        fallback_cycles = 0
        reward_sum = 0.0
        step_count = 0
        info = {}
        terminated = truncated = False

        while not (terminated or truncated):
            cycle_start = time.perf_counter()
            sim_time = step_count * float(env.CTRL_TIMESTEP)

            t0 = time.perf_counter()
            worker.publish_observation(
                make_snapshot(
                    env,
                    observation,
                    sim_time=sim_time,
                    wall_time=cycle_start,
                    buffer=buffer,
                    lower=lower,
                )
            )
            t1 = time.perf_counter()

            packet_message = worker.get_packet()
            read_wall = time.perf_counter()
            if packet_message is not None:
                packet = packet_message["packet"]
                if buffer.commit(packet):
                    packet_commit_wall_times.append(read_wall)
                    packet_versions.append(int(packet.version))
                    timing.record_packet_receive(
                        receive_wall_time=read_wall,
                        packet_generate_wall_time=float(packet.t_generate),
                        packet_publish_wall_time=float(packet_message["worker_timing"]["publish_wall_time"]),
                    )
            high_events.extend(worker.get_all_status())
            sample, fallback = lower.select(buffer, read_wall)
            fallback_cycles += int(fallback)
            t2 = time.perf_counter()

            target_position = lower.last_position
            target_velocity = lower.last_velocity
            rpm, _, _ = controller.computeControl(
                control_timestep=float(env.CTRL_TIMESTEP),
                cur_pos=np.asarray(env.pos[0], dtype=np.float64),
                cur_quat=np.asarray(p.getQuaternionFromEuler(env.rpy[0]), dtype=np.float64),
                cur_vel=np.asarray(env.vel[0], dtype=np.float64),
                cur_ang_vel=np.asarray(env.ang_v[0], dtype=np.float64),
                target_pos=np.asarray(target_position, dtype=np.float64),
                target_vel=np.asarray(target_velocity, dtype=np.float64),
            )
            t3 = time.perf_counter()
            action = codec.rpm_to_normalized_action(rpm).astype(np.float32)
            observation, reward, terminated, truncated, info = env.step(action.reshape(1, -1))
            t4 = time.perf_counter()

            reward_sum += float(reward)
            timing.record_cycle(
                start=cycle_start,
                end=t4,
                previous_start=previous_cycle_start,
                observation_publish_time=t1 - t0,
                packet_read_select_time=t2 - t1,
                pid_compute_time=t3 - t2,
                environment_step_time=t4 - t3,
            )
            previous_cycle_start = cycle_start
            step_count += 1

        stop_info = worker.stop(timeout=max(5.0, artificial_delay + 2.0))
        high_events.extend(worker.get_all_status())
        packet_intervals = np.diff(np.asarray(packet_commit_wall_times, dtype=np.float64)).tolist()
        packet_events = [event for event in high_events if event.get("type") == "packet"]
        low_high_overlap = count_low_cycles_during_high_compute(timing.cycle_start_time, packet_events)
        final_distance = float(np.linalg.norm(np.asarray(env.TARGET_POS) - np.asarray(env.pos[0])))
        versions_monotonic = all(a < b for a, b in zip(packet_versions, packet_versions[1:]))
        return {
            "profile": profile_name,
            "seed": int(seed),
            "steps": int(step_count),
            "reward": float(reward_sum),
            "success": bool(info.get("success", False)),
            "collision": bool(info.get("collision", False)),
            "done_reason": terminal_reason(env, info, terminated, truncated),
            "final_distance": final_distance,
            "fallback_cycles": int(fallback_cycles),
            "packet_update_intervals_wall_seconds": packet_intervals,
            "packet_versions": packet_versions,
            "packet_versions_strictly_increasing": bool(versions_monotonic),
            "buffer": {
                "accepted_packets": int(buffer.accepted_packets),
                "rejected_packets": int(buffer.rejected_packets),
                "rejected_versions": int(buffer.rejected_versions),
            },
            "low_level_selector": lower.metrics(),
            "low_level_timing": timing.summary(),
            "low_level_timing_raw": {
                "cycle_start_time": timing.cycle_start_time,
                "cycle_duration": timing.cycle_duration,
                "environment_step_time": timing.environment_step_time,
            },
            "high_level_events": packet_events,
            "worker_status_events": [event for event in high_events if event.get("type") != "packet"],
            "worker_stop": stop_info,
            "low_cycles_during_high_compute": low_high_overlap,
        }
    finally:
        if worker.process.is_alive():
            worker.stop()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/latent_only_affordance_dim16/model_final")
    parser.add_argument("--output", default="analysis_outputs/runtime_async_pid/runtime_async_results.json")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-len-sec", type=float, default=4.0)
    parser.add_argument("--num-trees", type=int, default=0)
    parser.add_argument("--route-blocking-tree", action="store_true")
    parser.add_argument("--high-frequency-hz", type=float, default=15.0)
    parser.add_argument("--worker-startup-timeout", type=float, default=20.0)
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    for suffix in ("_encoder", "_actor"):
        if not Path(str(args.checkpoint) + suffix).is_file():
            raise FileNotFoundError(str(args.checkpoint) + suffix)

    result = {
        "checkpoint": args.checkpoint,
        "controller": "DSLPIDControl",
        "runtime_async": True,
        "logical_scheduler_used": False,
        "low_level_calls_encoder_or_planner": False,
        "ipc": {
            "process_context": "multiprocessing.get_context('spawn')",
            "observation_channel": "shared-memory latest slot with monotonic version",
            "packet_channel": "multiprocessing.Queue(maxsize=1), latest-only overwrite",
            "status_channel": "unbounded telemetry queue, not used for control",
        },
        "timing_note": (
            "The low-level loop is not wall-clock paced; wall timing measures compute "
            "latency and overlap, not enforced real-time scheduling."
        ),
        "low_level_deadline_seconds": LOW_LEVEL_PERIOD_SECONDS,
        "environment": {
            "num_trees": int(args.num_trees),
            "route_blocking_tree": bool(args.route_blocking_tree),
            "episode_len_sec": float(args.episode_len_sec),
        },
        "profiles": {},
    }
    for profile_name, artificial_delay in PROFILES:
        episodes = [
            run_episode(
                profile_name=profile_name,
                artificial_delay=artificial_delay,
                seed=int(args.seed) + idx,
                args=args,
            )
            for idx in range(args.episodes)
        ]
        result["profiles"][profile_name] = {
            "artificial_high_level_delay_seconds": float(artificial_delay),
            "summary": profile_summary(episodes),
            "episodes": episodes,
        }
        summary = result["profiles"][profile_name]["summary"]
        cycle = summary["low_level_cycle_duration"]
        high = summary["high_level"]
        print(
            f"{profile_name}: cycles={summary['total_low_level_cycles']}, "
            f"packets={summary['accepted_packets']}, "
            f"high_decisions={high['decision_count']}, "
            f"cycle_p99={cycle['p99_seconds'] * 1000:.3f}ms, "
            f"over_deadline={cycle['count_above_deadline']}, "
            f"overlap_max={summary['low_level_cycles_during_high_compute']['max']}",
            flush=True,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, clean_json(result))
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
