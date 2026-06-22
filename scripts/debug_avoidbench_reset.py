from __future__ import annotations

import argparse
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import rospy
from nav_msgs.msg import Odometry
from rpg_quadrotor_msgs.msg import AutopilotFeedback

from envs.avoidbench.rl_env import AvoidBenchRLEnv


AUTOPILOT_STATE_NAMES = {
    AutopilotFeedback.OFF: "OFF",
    AutopilotFeedback.START: "START",
    AutopilotFeedback.HOVER: "HOVER",
    AutopilotFeedback.LAND: "LAND",
    AutopilotFeedback.EMERGENCY_LAND: "EMERGENCY_LAND",
    AutopilotFeedback.BREAKING: "BREAKING",
    AutopilotFeedback.GO_TO_POSE: "GO_TO_POSE",
    AutopilotFeedback.VELOCITY_CONTROL: "VELOCITY_CONTROL",
    AutopilotFeedback.REFERENCE_CONTROL: "REFERENCE_CONTROL",
    AutopilotFeedback.TRAJECTORY_CONTROL: "TRAJECTORY_CONTROL",
    AutopilotFeedback.COMMAND_FEEDTHROUGH: "COMMAND_FEEDTHROUGH",
    AutopilotFeedback.RC_MANUAL: "RC_MANUAL",
}


def state_name(state: int | None) -> str:
    if state is None:
        return "UNKNOWN"
    return AUTOPILOT_STATE_NAMES.get(int(state), f"UNKNOWN({state})")


class TimelineRecorder:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start_time = time.monotonic()
        self._odometry: list[dict[str, Any]] = []
        self._autopilot: list[dict[str, Any]] = []
        self._latest_position: list[float] | None = None
        self._latest_velocity: list[float] | None = None
        self._latest_state: int | None = None

    def set_start(self, start_time: float) -> None:
        with self._lock:
            self._start_time = start_time
            self._odometry = []
            self._autopilot = []

    def odom_callback(self, msg: Odometry) -> None:
        now = time.monotonic()
        position = [
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y),
            float(msg.pose.pose.position.z),
        ]
        velocity = [
            float(msg.twist.twist.linear.x),
            float(msg.twist.twist.linear.y),
            float(msg.twist.twist.linear.z),
        ]
        with self._lock:
            self._latest_position = position
            self._latest_velocity = velocity
            self._odometry.append(
                {
                    "elapsed": now - self._start_time,
                    "position": position,
                    "velocity": velocity,
                    "z": position[2],
                    "vertical_velocity": velocity[2],
                }
            )

    def feedback_callback(self, msg: AutopilotFeedback) -> None:
        now = time.monotonic()
        state = int(msg.autopilot_state)
        with self._lock:
            self._latest_state = state
            if not self._autopilot or self._autopilot[-1]["state"] != state:
                self._autopilot.append(
                    {
                        "elapsed": now - self._start_time,
                        "state": state,
                        "state_name": state_name(state),
                    }
                )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "position": list(self._latest_position) if self._latest_position is not None else None,
                "velocity": list(self._latest_velocity) if self._latest_velocity is not None else None,
                "autopilot_state": state_name(self._latest_state),
            }

    def export(self) -> dict[str, Any]:
        with self._lock:
            return {
                "odometry_samples": list(self._odometry),
                "autopilot_state_changes": list(self._autopilot),
                "odometry_received": bool(self._odometry),
            }


class ResetInstrumentation:
    def __init__(self, env: AvoidBenchRLEnv) -> None:
        self.env = env
        self.events: list[dict[str, Any]] = []
        self.set_model_state_called = False
        self.set_model_state_target: list[float] | None = None
        self.set_model_state_success = False
        self._install()

    def clear(self) -> None:
        self.events = []
        self.set_model_state_called = False
        self.set_model_state_target = None
        self.set_model_state_success = False

    def _install(self) -> None:
        original_call_reset_pose = self.env._call_reset_pose
        original_publish_for_connections = self.env._publish_for_connections

        def wrapped_call_reset_pose(env_self, position, yaw):
            self.set_model_state_called = True
            self.set_model_state_target = np.asarray(position, dtype=np.float32).tolist()
            started = time.monotonic()
            event = {
                "kind": "set_model_state",
                "started": started,
                "target_position": self.set_model_state_target,
                "target_yaw": float(yaw),
                "success": False,
            }
            self.events.append(event)
            result = original_call_reset_pose(position, yaw)
            event["finished"] = time.monotonic()
            event["duration"] = event["finished"] - started
            event["success"] = True
            self.set_model_state_success = True
            return result

        def wrapped_publish_for_connections(
            env_self,
            publisher,
            msg,
            repeat,
            rate_hz=None,
            stamp_each_time=False,
        ):
            topic = getattr(publisher, "resolved_name", getattr(publisher, "name", "UNKNOWN"))
            started = time.monotonic()
            event = {
                "kind": "publish",
                "topic": topic,
                "started": started,
                "subscriber_count_before": int(publisher.get_num_connections()),
                "requested_repeat": int(repeat),
                "rate_hz": None if rate_hz is None else float(rate_hz),
                "stamp_each_time": bool(stamp_each_time),
            }
            self.events.append(event)
            result = original_publish_for_connections(
                publisher,
                msg,
                repeat,
                rate_hz=rate_hz,
                stamp_each_time=stamp_each_time,
            )
            event["finished"] = time.monotonic()
            event["duration"] = event["finished"] - started
            event["subscriber_count_after"] = int(publisher.get_num_connections())
            event["actual_publish_count"] = int(repeat)
            return result

        self.env._call_reset_pose = MethodType(wrapped_call_reset_pose, self.env)
        self.env._publish_for_connections = MethodType(
            wrapped_publish_for_connections,
            self.env,
        )


def summarize_publishes(events: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for event in events:
        if event.get("kind") != "publish":
            continue
        topic = str(event["topic"])
        entry = summary.setdefault(
            topic,
            {
                "called": False,
                "publish_count": 0,
                "subscriber_counts_before": [],
                "subscriber_counts_after": [],
            },
        )
        entry["called"] = True
        entry["publish_count"] += int(event.get("actual_publish_count", 0))
        entry["subscriber_counts_before"].append(int(event.get("subscriber_count_before", 0)))
        entry["subscriber_counts_after"].append(int(event.get("subscriber_count_after", 0)))
    return summary


def downsample_odometry(samples: list[dict[str, Any]], interval: float = 0.05) -> list[dict[str, Any]]:
    if not samples:
        return []
    selected: list[dict[str, Any]] = []
    next_elapsed = 0.0
    for sample in samples:
        if sample["elapsed"] >= next_elapsed:
            selected.append(sample)
            next_elapsed = sample["elapsed"] + interval
    if selected[-1] is not samples[-1]:
        selected.append(samples[-1])
    return selected


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Instrument the current AvoidBench reset sequence.")
    parser.add_argument("--namespace", default="/hummingbird")
    parser.add_argument("--num-resets", type=int, default=1)
    parser.add_argument("--takeoff-height", type=float, default=1.10)
    parser.add_argument("--takeoff-timeout", type=float, default=8.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.num_resets <= 0:
        parser.error("--num-resets must be positive.")

    env = AvoidBenchRLEnv(
        namespace=args.namespace,
        takeoff_height=args.takeoff_height,
        takeoff_timeout=args.takeoff_timeout,
    )
    recorder = TimelineRecorder()
    rospy.Subscriber(
        f"{args.namespace.rstrip('/')}/ground_truth/odometry",
        Odometry,
        recorder.odom_callback,
        queue_size=500,
    )
    rospy.Subscriber(
        f"{args.namespace.rstrip('/')}/autopilot/feedback",
        AutopilotFeedback,
        recorder.feedback_callback,
        queue_size=100,
    )
    instrumentation = ResetInstrumentation(env)
    time.sleep(0.5)

    results: list[dict[str, Any]] = []
    for reset_index in range(args.num_resets):
        instrumentation.clear()
        reset_started = time.monotonic()
        recorder.set_start(reset_started)
        before = recorder.snapshot()
        success = False
        failure_reason = ""
        reset_info: dict[str, Any] | None = None
        try:
            _, reset_info = env.reset()
            success = True
        except Exception as exc:  # pragma: no cover - live ROS diagnostic
            failure_reason = f"{type(exc).__name__}: {exc}"
        reset_finished = time.monotonic()
        after = recorder.snapshot()
        timeline = recorder.export()
        odometry_samples = downsample_odometry(timeline["odometry_samples"])
        max_height = max((sample["z"] for sample in odometry_samples), default=float("-inf"))
        reached_takeoff_height = max_height >= args.takeoff_height
        publishes = summarize_publishes(instrumentation.events)

        result = {
            "reset_index": reset_index,
            "reset_started_iso": datetime.now().isoformat(timespec="milliseconds"),
            "reset_duration": reset_finished - reset_started,
            "success": success,
            "failure_reason": failure_reason,
            "takeoff_height": args.takeoff_height,
            "takeoff_timeout": args.takeoff_timeout,
            "reached_takeoff_height": reached_takeoff_height,
            "max_height": max_height if np.isfinite(max_height) else None,
            "before": before,
            "after": after,
            "set_model_state_called": instrumentation.set_model_state_called,
            "set_model_state_success": instrumentation.set_model_state_success,
            "set_model_state_target": instrumentation.set_model_state_target,
            "odometry_received": timeline["odometry_received"],
            "odometry_samples": odometry_samples,
            "autopilot_state_changes": timeline["autopilot_state_changes"],
            "publisher_summary": publishes,
            "events": instrumentation.events,
            "reset_info": reset_info,
        }
        results.append(result)

        print(
            f"reset={reset_index} success={success} "
            f"reached_takeoff={reached_takeoff_height} "
            f"max_z={result['max_height']} "
            f"state={after['autopilot_state']} "
            f"failure={failure_reason or 'none'}"
        )
        if args.verbose:
            print(f"  before position={before['position']} velocity={before['velocity']}")
            print(f"  after  position={after['position']} velocity={after['velocity']}")
            print(f"  autopilot transitions={timeline['autopilot_state_changes']}")
            for topic, publish_info in publishes.items():
                print(f"  publish {topic}: {publish_info}")

    payload = {
        "namespace": args.namespace,
        "num_resets": args.num_resets,
        "takeoff_height": args.takeoff_height,
        "takeoff_timeout": args.takeoff_timeout,
        "successful_resets": sum(result["success"] for result in results),
        "resets_reaching_takeoff_height": sum(
            result["reached_takeoff_height"] for result in results
        ),
        "results": results,
    }
    if args.json_out is not None:
        write_json(args.json_out, payload)
        print(f"json_out={args.json_out}")

    env.close()
    return 0 if all(result["success"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
