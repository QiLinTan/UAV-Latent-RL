from __future__ import annotations

import argparse
import csv
import json
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CONTAINER_CONFIG = Path(
    "/AvoidBench/src/avoidbench/avoid_manage/params/task_indoor.yaml"
)


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def run_cli(command: list[str], timeout: float = 5.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def rostopic_list() -> list[str]:
    result = run_cli(["rostopic", "list"])
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def rostopic_type(topic: str) -> str:
    result = run_cli(["rostopic", "type", topic])
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def discover_contact_topics() -> list[dict[str, str]]:
    topics = rostopic_list()
    discovered: list[dict[str, str]] = []
    for topic in topics:
        lower = topic.lower()
        msg_type = rostopic_type(topic)
        if msg_type == "gazebo_msgs/ContactsState" or "contact" in lower or "bumper" in lower:
            discovered.append({"topic": topic, "type": msg_type})
    return discovered


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def vec3(values: Any) -> list[float] | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size < 3:
        return None
    return array[:3].astype(float).tolist()


def ros_time_to_float(stamp: Any) -> float | None:
    if stamp is None:
        return None
    try:
        return float(stamp.to_sec())
    except Exception:
        return None


class LatestValue:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.value: Any = None
        self.wall_time: float | None = None
        self.count = 0

    def set(self, value: Any) -> None:
        with self._lock:
            self.value = value
            self.wall_time = time.monotonic()
            self.count += 1

    def snapshot(self) -> tuple[Any, float | None, int]:
        with self._lock:
            return self.value, self.wall_time, self.count


class TopicTextMonitor:
    def __init__(self, topic: str) -> None:
        import rospy
        import rostopic

        self.topic = topic
        self.msg_type = ""
        self.latest = LatestValue()
        msg_class = None
        try:
            resolved = rostopic.get_topic_class(topic, blocking=False)
            if resolved is not None:
                msg_class, real_topic, _msg_eval = resolved
                self.topic = real_topic
                self.msg_type = getattr(msg_class, "_type", "") or rostopic_type(real_topic)
        except Exception:
            msg_class = None
        if msg_class is None:
            from rospy.msg import AnyMsg

            msg_class = AnyMsg
            self.msg_type = rostopic_type(topic)

        def callback(msg) -> None:
            payload: dict[str, Any] = {
                "type": self.msg_type,
                "text": str(msg).strip(),
            }
            if hasattr(msg, "data"):
                payload["data"] = getattr(msg, "data")
            header = getattr(msg, "header", None)
            if header is not None:
                payload["stamp"] = ros_time_to_float(getattr(header, "stamp", None))
                payload["frame_id"] = getattr(header, "frame_id", "")
            self.latest.set(payload)

        self._subscriber = rospy.Subscriber(self.topic, msg_class, callback, queue_size=50)

    def sample(self) -> dict[str, Any]:
        value, wall_time, count = self.latest.snapshot()
        return {
            "topic": self.topic,
            "type": self.msg_type,
            "count": count,
            "age_s": None if wall_time is None else time.monotonic() - wall_time,
            "latest": value,
        }


class RosRuntimeMonitors:
    def __init__(self, namespace: str, *, contact_topics: list[dict[str, str]]) -> None:
        import rospy
        from gazebo_msgs.msg import ContactsState
        from nav_msgs.msg import Odometry
        from rpg_quadrotor_msgs.msg import AutopilotFeedback
        from std_msgs.msg import Bool

        self.namespace = namespace.rstrip("/")
        self.odom = LatestValue()
        self.collision = LatestValue()
        self.autopilot = LatestValue()
        self.contacts = LatestValue()
        self.contact_topics = contact_topics
        self.task_state = TopicTextMonitor(f"{self.namespace}/task_state")
        self.metrics = TopicTextMonitor(f"{self.namespace}/metrics")
        self._contact_latest_by_topic: dict[str, dict[str, Any]] = {}

        def odom_callback(msg: Odometry) -> None:
            pos = msg.pose.pose.position
            vel = msg.twist.twist.linear
            self.odom.set(
                {
                    "position": [float(pos.x), float(pos.y), float(pos.z)],
                    "velocity": [float(vel.x), float(vel.y), float(vel.z)],
                    "stamp": ros_time_to_float(msg.header.stamp),
                    "frame_id": msg.header.frame_id,
                }
            )

        def collision_callback(msg: Bool) -> None:
            self.collision.set({"data": bool(msg.data)})

        def autopilot_callback(msg: AutopilotFeedback) -> None:
            mapping = {
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
            state = int(msg.autopilot_state)
            self.autopilot.set({"state": state, "state_name": mapping.get(state, f"UNKNOWN({state})")})

        def make_contact_callback(topic: str):
            def callback(msg: ContactsState) -> None:
                names = [
                    [str(state.collision1_name), str(state.collision2_name)]
                    for state in msg.states[:10]
                ]
                payload = {
                    "topic": topic,
                    "state_count": len(msg.states),
                    "pairs": names,
                    "stamp": ros_time_to_float(msg.header.stamp),
                }
                self._contact_latest_by_topic[topic] = payload
                self.contacts.set(dict(self._contact_latest_by_topic))

            return callback

        rospy.Subscriber(f"{self.namespace}/ground_truth/odometry", Odometry, odom_callback, queue_size=200)
        rospy.Subscriber(f"{self.namespace}/collision", Bool, collision_callback, queue_size=200)
        rospy.Subscriber(
            f"{self.namespace}/autopilot/feedback",
            AutopilotFeedback,
            autopilot_callback,
            queue_size=100,
        )
        for topic in contact_topics:
            if topic.get("type") == "gazebo_msgs/ContactsState":
                rospy.Subscriber(topic["topic"], ContactsState, make_contact_callback(topic["topic"]), queue_size=100)

    def sample(self) -> dict[str, Any]:
        odom, odom_time, odom_count = self.odom.snapshot()
        collision, collision_time, collision_count = self.collision.snapshot()
        autopilot, autopilot_time, autopilot_count = self.autopilot.snapshot()
        contacts, contacts_time, contacts_count = self.contacts.snapshot()
        contact_payloads = list((contacts or {}).values()) if isinstance(contacts, dict) else []
        active_contact_count = int(sum(int(item.get("state_count", 0)) for item in contact_payloads))
        return {
            "odom": odom,
            "odom_count": odom_count,
            "odom_age_s": None if odom_time is None else time.monotonic() - odom_time,
            "ros_collision": None if collision is None else bool(collision.get("data", False)),
            "ros_collision_count": collision_count,
            "ros_collision_age_s": None
            if collision_time is None
            else time.monotonic() - collision_time,
            "autopilot": autopilot,
            "autopilot_count": autopilot_count,
            "autopilot_age_s": None
            if autopilot_time is None
            else time.monotonic() - autopilot_time,
            "contact_topics": self.contact_topics,
            "gazebo_contact_topic_count": len(self.contact_topics),
            "gazebo_contact_message_count": contacts_count,
            "gazebo_contact_active_count": active_contact_count,
            "gazebo_contact_payloads": contact_payloads,
            "task_state": self.task_state.sample(),
            "metrics": self.metrics.sample(),
        }


class GazeboModelStateProbe:
    def __init__(self, model_name: str) -> None:
        import rospy
        from gazebo_msgs.srv import GetModelState

        self.model_name = model_name
        self.error = ""
        self._service = None
        try:
            rospy.wait_for_service("/gazebo/get_model_state", timeout=3.0)
            self._service = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def sample(self) -> dict[str, Any]:
        if self._service is None:
            return {"available": False, "success": False, "error": self.error}
        try:
            response = self._service(self.model_name, "world")
        except Exception as exc:
            return {"available": True, "success": False, "error": f"{type(exc).__name__}: {exc}"}
        pose = response.pose
        twist = response.twist
        return {
            "available": True,
            "success": bool(response.success),
            "status_message": str(response.status_message),
            "position": [
                float(pose.position.x),
                float(pose.position.y),
                float(pose.position.z),
            ],
            "velocity": [
                float(twist.linear.x),
                float(twist.linear.y),
                float(twist.linear.z),
            ],
        }


class DirectBridgeProbe:
    def __init__(
        self,
        config_path: Path,
        *,
        spawn_obstacles: bool,
        mission_end: tuple[float, float, float],
        mission_radius: float,
        mission_seed: int,
    ) -> None:
        from envs.avoidbench.adapter import AvoidBenchBridgeAdapter

        self.adapter = AvoidBenchBridgeAdapter(config_path)
        self.spawn_obstacles = bool(spawn_obstacles)
        self.mission_end = tuple(float(value) for value in mission_end)
        self.mission_radius = float(mission_radius)
        self.mission_seed = int(mission_seed)
        self.scene_changed: bool | None = None
        self.initialized = False

    def initialize(self, position: list[float], velocity: list[float]) -> dict[str, Any]:
        state = self.adapter.create_state(
            position=position,
            orientation=(0.0, 0.0, 0.0, 1.0),
            velocity=velocity,
            timestamp=0.0,
        )
        ready = bool(self.adapter.update_unity(state))
        if ready and self.spawn_obstacles:
            self.adapter.configure_mission(
                start_point=(*position, 0.0),
                end_point=self.mission_end,
                trials=1,
                radius=self.mission_radius,
                seed=self.mission_seed,
                opacity=0.5,
                pointcloud_file="pointcloud-collision-ownership",
            )
            self.scene_changed = bool(self.adapter.spawn_obstacles(state=state))
        self.initialized = True
        return {
            "unity_ready": ready,
            "scene_changed": self.scene_changed,
            "collision": bool(self.adapter.collision()) if ready else None,
        }

    def sample(
        self,
        *,
        position: list[float],
        velocity: list[float],
        timestamp: float,
    ) -> dict[str, Any]:
        state = self.adapter.create_state(
            position=position,
            orientation=(0.0, 0.0, 0.0, 1.0),
            velocity=velocity,
            timestamp=timestamp,
        )
        ready = bool(self.adapter.update_unity(state))
        scene_changed = None
        try:
            scene_changed = bool(self.adapter.bridge.ifSceneChanged())
        except Exception:
            scene_changed = self.scene_changed
        return {
            "unity_ready": ready,
            "scene_changed": scene_changed,
            "collision": bool(self.adapter.collision()) if ready else None,
        }


def first_true_time(samples: list[dict[str, Any]], key: str) -> float | None:
    for sample in samples:
        if sample.get(key) is True:
            return safe_float(sample.get("elapsed_s"))
    return None


def bool_count(samples: list[dict[str, Any]], key: str) -> int:
    return int(sum(sample.get(key) is True for sample in samples))


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    heights = [
        float(sample["position"][2])
        for sample in samples
        if isinstance(sample.get("position"), list) and len(sample["position"]) >= 3
    ]
    speeds = [
        float(np.linalg.norm(np.asarray(sample["velocity"], dtype=np.float32)))
        for sample in samples
        if isinstance(sample.get("velocity"), list) and len(sample["velocity"]) >= 3
    ]
    return {
        "sample_count": len(samples),
        "ros_collision_true_count": bool_count(samples, "ros_collision"),
        "ros_collision_first_true_elapsed_s": first_true_time(samples, "ros_collision"),
        "direct_bridge_collision_true_count": bool_count(samples, "direct_bridge_collision"),
        "direct_bridge_collision_first_true_elapsed_s": first_true_time(
            samples,
            "direct_bridge_collision",
        ),
        "gazebo_contact_active_count_max": max(
            [int(sample.get("gazebo_contact_active_count") or 0) for sample in samples] or [0]
        ),
        "height_min": min(heights) if heights else None,
        "height_max": max(heights) if heights else None,
        "speed_max": max(speeds) if speeds else None,
        "latest_task_state": samples[-1].get("task_state_latest") if samples else None,
        "latest_metrics": samples[-1].get("metrics_latest") if samples else None,
    }


def summarize_bridge_sweep(samples: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for sample in samples:
        index = int(sample.get("sweep_index", -1))
        grouped.setdefault(index, []).append(sample)

    by_position: list[dict[str, Any]] = []
    for index in sorted(grouped):
        rows = grouped[index]
        first = rows[0] if rows else {}
        collisions = [row.get("direct_bridge_collision") for row in rows]
        ready = [row.get("direct_bridge_unity_ready") for row in rows]
        by_position.append(
            {
                "sweep_index": index,
                "position": first.get("position"),
                "sample_count": len(rows),
                "unity_ready_all": all(value is True for value in ready) if ready else None,
                "collision_true_count": int(sum(value is True for value in collisions)),
                "collision_false_count": int(sum(value is False for value in collisions)),
                "collision_first_true_elapsed_s": first_true_time(rows, "direct_bridge_collision"),
                "scene_changed_values": sorted(
                    {str(row.get("direct_bridge_scene_changed")) for row in rows}
                ),
            }
        )
    return {
        "position_count": len(by_position),
        "by_position": by_position,
        "any_position_collision_false": any(
            item["collision_false_count"] > 0 for item in by_position
        ),
        "all_positions_collision_true": (
            all(
                item["sample_count"] > 0
                and item["collision_true_count"] == item["sample_count"]
                for item in by_position
            )
            if by_position
            else None
        ),
    }


def flatten_official_sample(
    *,
    elapsed_s: float,
    monitor_sample: dict[str, Any],
    gazebo_model_state: dict[str, Any],
    direct_bridge: dict[str, Any] | None,
) -> dict[str, Any]:
    odom = monitor_sample.get("odom") or {}
    autopilot = monitor_sample.get("autopilot") or {}
    task_state = monitor_sample.get("task_state") or {}
    metrics = monitor_sample.get("metrics") or {}
    position = odom.get("position") or gazebo_model_state.get("position")
    velocity = odom.get("velocity") or gazebo_model_state.get("velocity")
    return {
        "elapsed_s": float(elapsed_s),
        "position": position,
        "velocity": velocity,
        "height": position[2] if isinstance(position, list) and len(position) >= 3 else None,
        "ros_collision": monitor_sample.get("ros_collision"),
        "autopilot_state": (autopilot or {}).get("state_name"),
        "gazebo_model_success": gazebo_model_state.get("success"),
        "gazebo_model_position": gazebo_model_state.get("position"),
        "gazebo_model_velocity": gazebo_model_state.get("velocity"),
        "gazebo_model_status": gazebo_model_state.get("status_message"),
        "gazebo_contact_topic_count": monitor_sample.get("gazebo_contact_topic_count"),
        "gazebo_contact_message_count": monitor_sample.get("gazebo_contact_message_count"),
        "gazebo_contact_active_count": monitor_sample.get("gazebo_contact_active_count"),
        "gazebo_contact_payloads": monitor_sample.get("gazebo_contact_payloads"),
        "task_state_topic_type": task_state.get("type"),
        "task_state_count": task_state.get("count"),
        "task_state_latest": (task_state.get("latest") or {}).get("text")
        if isinstance(task_state.get("latest"), dict)
        else task_state.get("latest"),
        "metrics_topic_type": metrics.get("type"),
        "metrics_count": metrics.get("count"),
        "metrics_latest": (metrics.get("latest") or {}).get("text")
        if isinstance(metrics.get("latest"), dict)
        else metrics.get("latest"),
        "direct_bridge_unity_ready": None if direct_bridge is None else direct_bridge.get("unity_ready"),
        "direct_bridge_scene_changed": None if direct_bridge is None else direct_bridge.get("scene_changed"),
        "direct_bridge_collision": None if direct_bridge is None else direct_bridge.get("collision"),
        "direct_bridge_error": None if direct_bridge is None else direct_bridge.get("error"),
    }


def run_official_reset(args) -> dict[str, Any]:
    import rospy

    from envs.avoidbench.rl_env import AvoidBenchRLEnv

    if not rospy.core.is_initialized():
        rospy.init_node("avoidbench_collision_ownership_probe", anonymous=True, disable_signals=True)

    contact_topics = discover_contact_topics()
    monitors = RosRuntimeMonitors(args.namespace, contact_topics=contact_topics)
    gazebo_probe = GazeboModelStateProbe(args.model_name)
    time.sleep(0.5)

    env = AvoidBenchRLEnv(
        namespace=args.namespace,
        model_name=args.model_name,
        reset_position=tuple(args.reset_position),
        reset_yaw=args.reset_yaw,
        action_preset=args.action_preset,
    )

    reset_started = time.monotonic()
    reset_info: dict[str, Any] | None = None
    reset_error = ""
    try:
        _obs, reset_info = env.reset()
    except Exception as exc:
        reset_error = f"{type(exc).__name__}: {exc}"
    reset_finished = time.monotonic()

    direct_bridge_probe: DirectBridgeProbe | None = None
    direct_bridge_error = ""
    if args.direct_bridge:
        try:
            initial_position = (
                vec3(reset_info.get("position")) if reset_info is not None else list(args.reset_position)
            )
            initial_velocity = (
                vec3(reset_info.get("velocity")) if reset_info is not None else [0.0, 0.0, 0.0]
            )
            direct_bridge_probe = DirectBridgeProbe(
                args.config,
                spawn_obstacles=args.spawn_obstacles,
                mission_end=tuple(args.mission_end),
                mission_radius=args.mission_radius,
                mission_seed=args.mission_seed,
            )
            direct_bridge_probe.initialize(initial_position or list(args.reset_position), initial_velocity or [0.0, 0.0, 0.0])
        except Exception as exc:
            direct_bridge_error = f"{type(exc).__name__}: {exc}"
            direct_bridge_probe = None

    samples: list[dict[str, Any]] = []
    observe_started = time.monotonic()
    next_sample = observe_started
    while time.monotonic() - observe_started <= args.observe_seconds and not rospy.is_shutdown():
        now = time.monotonic()
        if now < next_sample:
            time.sleep(min(0.01, max(0.0, next_sample - now)))
            continue
        monitor_sample = monitors.sample()
        gazebo_model_state = gazebo_probe.sample()
        direct_bridge: dict[str, Any] | None = None
        if direct_bridge_probe is not None:
            odom = monitor_sample.get("odom") or {}
            position = vec3(odom.get("position")) or list(args.reset_position)
            velocity = vec3(odom.get("velocity")) or [0.0, 0.0, 0.0]
            try:
                direct_bridge = direct_bridge_probe.sample(
                    position=position,
                    velocity=velocity,
                    timestamp=time.monotonic() - observe_started,
                )
            except Exception as exc:
                direct_bridge = {
                    "unity_ready": None,
                    "scene_changed": None,
                    "collision": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        elif direct_bridge_error:
            direct_bridge = {
                "unity_ready": None,
                "scene_changed": None,
                "collision": None,
                "error": direct_bridge_error,
            }
        samples.append(
            flatten_official_sample(
                elapsed_s=time.monotonic() - observe_started,
                monitor_sample=monitor_sample,
                gazebo_model_state=gazebo_model_state,
                direct_bridge=direct_bridge,
            )
        )
        next_sample += args.sample_period

    try:
        env.close()
    except Exception:
        pass

    summary = summarize_samples(samples)
    summary.update(
        {
            "mode": "official-reset",
            "reset_error": reset_error,
            "reset_wall_time_s": reset_finished - reset_started,
            "reset_info": reset_info,
            "contact_topics": contact_topics,
            "gazebo_contact_signal_available": any(
                topic.get("type") == "gazebo_msgs/ContactsState" for topic in contact_topics
            ),
            "direct_bridge_enabled": bool(args.direct_bridge),
            "direct_bridge_setup_error": direct_bridge_error,
        }
    )
    return {"summary": summary, "samples": samples}


def run_bridge_static(args) -> dict[str, Any]:
    bridge = DirectBridgeProbe(
        args.config,
        spawn_obstacles=args.spawn_obstacles,
        mission_end=tuple(args.mission_end),
        mission_radius=args.mission_radius,
        mission_seed=args.mission_seed,
    )
    initial = bridge.initialize(list(args.reset_position), [0.0, 0.0, 0.0])
    samples: list[dict[str, Any]] = []
    started = time.monotonic()
    next_sample = started
    while time.monotonic() - started <= args.observe_seconds:
        now = time.monotonic()
        if now < next_sample:
            time.sleep(min(0.01, max(0.0, next_sample - now)))
            continue
        elapsed = time.monotonic() - started
        direct = bridge.sample(
            position=list(args.reset_position),
            velocity=[0.0, 0.0, 0.0],
            timestamp=elapsed,
        )
        samples.append(
            {
                "elapsed_s": elapsed,
                "position": list(args.reset_position),
                "velocity": [0.0, 0.0, 0.0],
                "height": float(args.reset_position[2]),
                "direct_bridge_unity_ready": direct.get("unity_ready"),
                "direct_bridge_scene_changed": direct.get("scene_changed"),
                "direct_bridge_collision": direct.get("collision"),
            }
        )
        next_sample += args.sample_period
    summary = summarize_samples(samples)
    summary.update(
        {
            "mode": "bridge-static",
            "initial_direct_bridge": initial,
            "spawn_obstacles": bool(args.spawn_obstacles),
        }
    )
    return {"summary": summary, "samples": samples}


def default_sweep_positions() -> list[list[float]]:
    return [
        [0.0, 0.0, 1.2],
        [0.0, 0.0, 2.0],
        [0.0, 0.0, 5.0],
        [5.0, 0.0, 2.0],
        [0.0, 5.0, 2.0],
        [-5.0, 0.0, 2.0],
    ]


def run_bridge_sweep(args) -> dict[str, Any]:
    positions = args.sweep_position or default_sweep_positions()
    bridge = DirectBridgeProbe(
        args.config,
        spawn_obstacles=args.spawn_obstacles,
        mission_end=tuple(args.mission_end),
        mission_radius=args.mission_radius,
        mission_seed=args.mission_seed,
    )
    initial = bridge.initialize(list(positions[0]), [0.0, 0.0, 0.0])
    samples: list[dict[str, Any]] = []
    for sweep_index, position in enumerate(positions):
        position_started = time.monotonic()
        next_sample = position_started
        while time.monotonic() - position_started <= args.observe_seconds:
            now = time.monotonic()
            if now < next_sample:
                time.sleep(min(0.01, max(0.0, next_sample - now)))
                continue
            elapsed = time.monotonic() - position_started
            direct = bridge.sample(
                position=list(position),
                velocity=[0.0, 0.0, 0.0],
                timestamp=float(sweep_index) + elapsed,
            )
            samples.append(
                {
                    "sweep_index": int(sweep_index),
                    "elapsed_s": elapsed,
                    "position": list(position),
                    "velocity": [0.0, 0.0, 0.0],
                    "height": float(position[2]),
                    "direct_bridge_unity_ready": direct.get("unity_ready"),
                    "direct_bridge_scene_changed": direct.get("scene_changed"),
                    "direct_bridge_collision": direct.get("collision"),
                }
            )
            next_sample += args.sample_period
    summary = summarize_samples(samples)
    summary.update(
        {
            "mode": "bridge-sweep",
            "initial_direct_bridge": initial,
            "spawn_obstacles": bool(args.spawn_obstacles),
            "sweep": summarize_bridge_sweep(samples),
        }
    )
    return {"summary": summary, "samples": samples}


def make_csv_rows(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        row = dict(sample)
        for key in ("position", "velocity", "gazebo_model_position", "gazebo_model_velocity"):
            if isinstance(row.get(key), list):
                values = row[key]
                for index, axis in enumerate(("x", "y", "z")):
                    row[f"{key}_{axis}"] = values[index] if len(values) > index else None
        for key in ("position", "velocity", "gazebo_model_position", "gazebo_model_velocity", "gazebo_contact_payloads"):
            if key in row:
                row[key] = json.dumps(jsonable(row[key]), sort_keys=True)
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose AvoidBench collision ownership across ROS, Gazebo, and avoidbridge."
    )
    parser.add_argument(
        "--mode",
        choices=("official-reset", "bridge-static", "bridge-sweep"),
        default="official-reset",
    )
    parser.add_argument("--namespace", default="/hummingbird")
    parser.add_argument("--model-name", default="hummingbird")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONTAINER_CONFIG)
    parser.add_argument("--observe-seconds", type=float, default=5.0)
    parser.add_argument("--sample-period", type=float, default=0.05)
    parser.add_argument("--reset-position", type=float, nargs=3, default=(0.0, 0.0, 1.2))
    parser.add_argument("--reset-yaw", type=float, default=0.0)
    parser.add_argument("--action-preset", choices=("legacy", "conservative"), default="conservative")
    parser.add_argument(
        "--direct-bridge",
        action="store_true",
        help=(
            "Also create a Python avoidbridge client during official-reset. "
            "Use carefully: the official avoid_manage_node already owns Unity."
        ),
    )
    parser.add_argument("--spawn-obstacles", action="store_true")
    parser.add_argument("--mission-end", type=float, nargs=3, default=(0.0, 15.0, 2.0))
    parser.add_argument("--mission-radius", type=float, default=2.0)
    parser.add_argument("--mission-seed", type=int, default=32)
    parser.add_argument(
        "--sweep-position",
        type=float,
        nargs=3,
        action="append",
        metavar=("X", "Y", "Z"),
        help=(
            "Position for bridge-sweep mode. Repeat for multiple positions. "
            "If omitted, a small default position/height set is used."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=Path("runs/avoidbench_collision_ownership"))
    args = parser.parse_args()

    if args.observe_seconds <= 0.0:
        parser.error("--observe-seconds must be positive.")
    if args.sample_period <= 0.0:
        parser.error("--sample-period must be positive.")
    if args.mode in {"bridge-static", "bridge-sweep"} and not args.config.is_file():
        parser.error(f"--config does not exist: {args.config}")

    output_dir = args.output_root / timestamp_slug()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "official-reset":
        result = run_official_reset(args)
    elif args.mode == "bridge-static":
        result = run_bridge_static(args)
    else:
        result = run_bridge_sweep(args)

    summary = {
        **result["summary"],
        "output_dir": str(output_dir),
        "observe_seconds": float(args.observe_seconds),
        "sample_period": float(args.sample_period),
        "namespace": args.namespace,
        "model_name": args.model_name,
        "reset_position": list(args.reset_position),
    }
    payload = {
        "args": vars(args),
        "summary": summary,
        "samples": result["samples"],
    }
    write_json(output_dir / "summary.json", payload)
    write_jsonl(output_dir / "samples.jsonl", result["samples"])
    write_csv(output_dir / "samples.csv", make_csv_rows(result["samples"]))
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
