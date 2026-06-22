from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry, Path
from rpg_quadrotor_msgs.msg import AutopilotFeedback, TrajectoryPoint
from std_msgs.msg import Bool, Empty


ACTION_NAMES = ("vx", "vy", "vz", "yaw_rate")
ACTION_PRESETS = {
    "legacy": (0.20, 0.20, 0.15, 0.30),
    "conservative": (0.12, 0.12, 0.04, 0.12),
}


@dataclass
class QuadSnapshot:
    position: np.ndarray
    velocity: np.ndarray
    orientation_xyzw: np.ndarray
    angular_velocity: np.ndarray
    stamp: float


@dataclass(frozen=True)
class AvoidBenchRewardDoneConfig:
    progress_scale: float = 1.0
    collision_penalty: float = 5.0
    height_error_penalty_scale: float = 0.50
    vertical_velocity_penalty_scale: float = 0.20
    z_action_penalty_scale: float = 0.10
    speed_penalty_scale: float = 0.0
    action_penalty_scale: float = 0.02
    goal_bonus: float = 5.0
    timeout_penalty: float = 0.5
    target_height: float = 1.2
    min_height: float = 0.40
    max_height: float = 2.50
    out_of_bounds_xy: float = 10.0
    goal_tolerance: float = 0.40
    max_episode_steps: int = 200
    odometry_timeout: float = 2.0


def reward_done_config_for_task(
    task_mode: str,
    *,
    target_height: float,
    goal_tolerance: float = 0.40,
    max_episode_steps: int = 200,
) -> AvoidBenchRewardDoneConfig:
    common = {
        "target_height": float(target_height),
        "goal_tolerance": float(goal_tolerance),
        "max_episode_steps": int(max_episode_steps),
    }
    if task_mode == "navigation_smoke":
        return AvoidBenchRewardDoneConfig(**common)
    if task_mode == "hover_smoke":
        return AvoidBenchRewardDoneConfig(
            progress_scale=0.0,
            height_error_penalty_scale=1.0,
            vertical_velocity_penalty_scale=0.50,
            z_action_penalty_scale=0.25,
            speed_penalty_scale=0.10,
            action_penalty_scale=0.05,
            goal_bonus=0.0,
            timeout_penalty=0.0,
            **common,
        )
    raise ValueError(f"Unknown AvoidBench task mode {task_mode!r}.")


class AvoidBenchRLEnv:
    """Minimal ROS-backed AvoidBench reset/step skeleton.

    This class does not launch ROS, Gazebo, or Unity. It assumes the official
    `roslaunch avoid_manage rotors_gazebo.launch` stack is already running.
    """

    def __init__(
        self,
        namespace: str = "/hummingbird",
        model_name: str = "hummingbird",
        reset_position: tuple[float, float, float] = (0.0, 0.0, 1.2),
        reset_yaw: float = 0.0,
        goal_position: tuple[float, float, float] | None = None,
        action_preset: str = "legacy",
        action_bounds: tuple[float, float, float, float] | None = None,
        action_duration: float = 0.30,
        action_publish_rate: float = 20.0,
        goal_tolerance: float = 0.40,
        max_episode_steps: int = 200,
        takeoff_height: float = 1.10,
        takeoff_timeout: float = 10.0,
        reset_retry: int = 2,
        publish_repeat_duration: float = 0.75,
        publish_interval: float = 0.075,
        odom_settle_timeout: float = 3.0,
        hover_state_timeout: float = 10.0,
        odom_settle_position_tolerance: float = 0.08,
        odom_settle_velocity_tolerance: float = 0.30,
        odom_settle_frames: int = 5,
        reward_done_config: AvoidBenchRewardDoneConfig | None = None,
    ) -> None:
        self.namespace = namespace.rstrip("/")
        self.model_name = model_name
        self.reset_position = np.asarray(reset_position, dtype=np.float32)
        self.reset_yaw = float(reset_yaw)
        self.default_goal_position = (
            np.asarray(goal_position, dtype=np.float32)
            if goal_position is not None
            else np.asarray((5.0, 0.0, 1.2), dtype=np.float32)
        )
        if action_preset not in ACTION_PRESETS:
            raise ValueError(
                f"Unknown action preset {action_preset!r}; choose from {sorted(ACTION_PRESETS)}."
            )
        selected_bounds = ACTION_PRESETS[action_preset] if action_bounds is None else action_bounds
        self.action_preset = action_preset
        self.action_names = ACTION_NAMES
        self.action_bounds = np.asarray(selected_bounds, dtype=np.float32)
        if self.action_bounds.shape != (4,) or np.any(self.action_bounds <= 0.0):
            raise ValueError("action_bounds must contain four positive values.")
        self.action_duration = float(action_duration)
        self.action_publish_rate = float(action_publish_rate)
        self.reward_done_config = reward_done_config or AvoidBenchRewardDoneConfig(
            target_height=float(self.reset_position[2]),
            goal_tolerance=float(goal_tolerance),
            max_episode_steps=int(max_episode_steps),
        )
        self.goal_tolerance = float(self.reward_done_config.goal_tolerance)
        self.max_episode_steps = int(self.reward_done_config.max_episode_steps)
        self.takeoff_height = float(takeoff_height)
        self.takeoff_timeout = float(takeoff_timeout)
        self.reset_retry = int(reset_retry)
        self.publish_repeat_duration = float(publish_repeat_duration)
        self.publish_interval = float(publish_interval)
        self.odom_settle_timeout = float(odom_settle_timeout)
        self.hover_state_timeout = float(hover_state_timeout)
        self.odom_settle_position_tolerance = float(odom_settle_position_tolerance)
        self.odom_settle_velocity_tolerance = float(odom_settle_velocity_tolerance)
        self.odom_settle_frames = int(odom_settle_frames)
        if self.reset_retry < 0:
            raise ValueError("reset_retry must be non-negative.")
        if self.publish_repeat_duration <= 0.0 or self.publish_interval <= 0.0:
            raise ValueError("Reset publish duration and interval must be positive.")
        if self.odom_settle_frames <= 0:
            raise ValueError("odom_settle_frames must be positive.")

        self.action_dim = 4
        self.observation_dim = 17

        self._latest_snapshot: QuadSnapshot | None = None
        self._latest_collision = False
        self._latest_goal = self.default_goal_position.copy()
        self._latest_feedback: AutopilotFeedback | None = None
        self._episode_step = 0
        self._prev_distance: float | None = None
        self._state_condition = threading.Condition()
        self._last_reset_retry_count = 0
        self._last_reset_failure_reason = ""
        self._last_reset_publish_counts: dict[str, int] = {}
        self._last_done_reason = "running"

        if not rospy.core.is_initialized():
            rospy.init_node("avoidbench_rl_env", anonymous=True, disable_signals=True)

        self._odom_topic = f"{self.namespace}/ground_truth/odometry"
        self._collision_topic = f"{self.namespace}/collision"
        self._goal_topic = f"{self.namespace}/goal_point"
        self._feedback_topic = f"{self.namespace}/autopilot/feedback"
        self._velocity_topic = f"{self.namespace}/autopilot/velocity_command"
        self._reset_reference_topic = f"{self.namespace}/autopilot/reset_reference_state"
        self._start_topic = f"{self.namespace}/autopilot/start"
        self._force_hover_topic = f"{self.namespace}/autopilot/force_hover"
        self._arm_topic = f"{self.namespace}/bridge/arm"

        rospy.Subscriber(self._odom_topic, Odometry, self._odom_callback, queue_size=200)
        rospy.Subscriber(self._collision_topic, Bool, self._collision_callback, queue_size=50)
        rospy.Subscriber(self._goal_topic, Path, self._goal_callback, queue_size=10)
        rospy.Subscriber(self._feedback_topic, AutopilotFeedback, self._feedback_callback, queue_size=50)

        self._velocity_pub = rospy.Publisher(self._velocity_topic, TwistStamped, queue_size=1)
        self._reset_reference_pub = rospy.Publisher(self._reset_reference_topic, TrajectoryPoint, queue_size=1)
        self._start_pub = rospy.Publisher(self._start_topic, Empty, queue_size=1)
        self._force_hover_pub = rospy.Publisher(self._force_hover_topic, Empty, queue_size=1)
        self._arm_pub = rospy.Publisher(self._arm_topic, Bool, queue_size=1)
        self._set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

    def reset(self) -> tuple[np.ndarray, dict]:
        self._wait_for_snapshot(timeout=3.0)
        self._wait_for_service("/gazebo/set_model_state", timeout=5.0)
        self._last_reset_failure_reason = ""
        self._last_reset_publish_counts = {}
        self._latest_collision = False

        last_error: Exception | None = None
        snapshot: QuadSnapshot | None = None
        for attempt in range(self.reset_retry + 1):
            self._last_reset_retry_count = attempt
            try:
                snapshot = self._reset_once()
                last_error = None
                break
            except (RuntimeError, TimeoutError, rospy.ROSException) as exc:
                last_error = exc
                self._last_reset_failure_reason = f"{type(exc).__name__}: {exc}"
                if attempt < self.reset_retry:
                    time.sleep(0.25)

        if snapshot is None:
            raise RuntimeError(
                "AvoidBench reset failed after "
                f"{self.reset_retry + 1} attempts: {self._last_reset_failure_reason}"
            ) from last_error

        self._episode_step = 0
        self._prev_distance = self._distance_to_goal(snapshot.position)
        self._last_done_reason = "running"
        obs = self._build_observation(snapshot)
        info = self._build_info(
            snapshot,
            previous_distance=self._prev_distance,
            progress=0.0,
            collision=self._latest_collision,
            done_reason="running",
            action_norm=0.0,
            step_time=0.0,
        )
        info["reset_retry_count"] = self._last_reset_retry_count
        info["reset_failure_reason"] = self._last_reset_failure_reason
        info["reset_publish_counts"] = dict(self._last_reset_publish_counts)
        return obs, info

    def _reset_once(self) -> QuadSnapshot:
        previous_stamp = self._latest_snapshot.stamp if self._latest_snapshot is not None else 0.0
        self._call_reset_pose(self.reset_position, self.reset_yaw)
        settled = self._wait_for_odom_settle(
            target_position=self.reset_position,
            after_stamp=previous_stamp,
            timeout=self.odom_settle_timeout,
        )

        self._publish_reset_reference(self.reset_position, self.reset_yaw)
        self._publish_reset_message(self._arm_pub, Bool(data=True))

        if self._autopilot_state() == AutopilotFeedback.OFF:
            self._publish_reset_message(self._start_pub, Empty())
            self._wait_for_autopilot_state(
                {AutopilotFeedback.START, AutopilotFeedback.HOVER},
                timeout=self.hover_state_timeout,
            )

        self._publish_reset_message(self._force_hover_pub, Empty())
        self._wait_for_autopilot_state(
            {AutopilotFeedback.HOVER},
            timeout=self.hover_state_timeout,
        )
        return self._wait_for_stable_takeoff(
            target_z=self.takeoff_height,
            after_stamp=settled.stamp,
            timeout=self.takeoff_timeout,
        )

    def step(self, action: np.ndarray | list[float] | tuple[float, ...]) -> tuple[np.ndarray, float, bool, dict]:
        step_started = time.perf_counter()
        raw_action = np.asarray(action, dtype=np.float32)
        if raw_action.shape != (self.action_dim,):
            raise ValueError(f"Expected action shape {(self.action_dim,)}, got {raw_action.shape}.")
        clipped = np.clip(raw_action, -self.action_bounds, self.action_bounds)
        normalized_action = clipped / self.action_bounds

        if self._autopilot_state() == AutopilotFeedback.OFF:
            raise RuntimeError("Autopilot is OFF. Call reset() before step().")

        before = self._wait_for_snapshot(timeout=2.0)
        previous_distance = (
            self._distance_to_goal(before.position)
            if self._prev_distance is None
            else float(self._prev_distance)
        )
        self._publish_velocity(clipped)
        odometry_timeout = False
        try:
            after = self._wait_for_fresh_snapshot(
                after_stamp=before.stamp,
                timeout=self.reward_done_config.odometry_timeout,
            )
        except TimeoutError:
            after = before
            odometry_timeout = True

        self._episode_step += 1
        distance = self._distance_to_goal(after.position)
        progress = float(previous_distance - distance)
        collision = bool(self._latest_collision)
        action_norm = float(np.linalg.norm(clipped))
        normalized_action_norm = float(np.linalg.norm(normalized_action))
        vertical_velocity = float(after.velocity[2])
        height_error = abs(float(after.position[2]) - self.reward_done_config.target_height)
        done_reason = self._compute_done_reason(
            after,
            distance=distance,
            collision=collision,
            odometry_timeout=odometry_timeout,
        )
        done = done_reason != "running"
        reward = self._compute_reward(
            progress=progress,
            collision=collision,
            height=float(after.position[2]),
            speed=float(np.linalg.norm(after.velocity)),
            vertical_velocity=vertical_velocity,
            normalized_action_norm=normalized_action_norm,
            normalized_z_action=float(normalized_action[2]),
            done_reason=done_reason,
        )
        self._prev_distance = distance
        self._last_done_reason = done_reason
        step_time = time.perf_counter() - step_started

        obs = self._build_observation(after)
        info = self._build_info(
            after,
            previous_distance=previous_distance,
            progress=progress,
            collision=collision,
            done_reason=done_reason,
            action_norm=action_norm,
            step_time=step_time,
            height_error=height_error,
            vertical_velocity=vertical_velocity,
            z_action=float(clipped[2]),
            normalized_z_action=float(normalized_action[2]),
        )
        info["raw_action"] = raw_action.tolist()
        info["action"] = clipped.tolist()
        info["normalized_action"] = normalized_action.astype(np.float32).tolist()
        info["normalized_action_norm"] = normalized_action_norm
        info["action_clipped"] = bool(np.any(np.abs(raw_action - clipped) > 1e-6))
        info["step_position_delta"] = (after.position - before.position).astype(np.float32).tolist()
        return obs, reward, done, info

    def close(self) -> None:
        try:
            self._publish_zero_velocity()
        except Exception:
            pass

    def sample_random_action(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(-self.action_bounds, self.action_bounds).astype(np.float32)

    def _odom_callback(self, msg: Odometry) -> None:
        snapshot = QuadSnapshot(
            position=np.array(
                [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z],
                dtype=np.float32,
            ),
            velocity=np.array(
                [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z],
                dtype=np.float32,
            ),
            orientation_xyzw=np.array(
                [
                    msg.pose.pose.orientation.x,
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z,
                    msg.pose.pose.orientation.w,
                ],
                dtype=np.float32,
            ),
            angular_velocity=np.array(
                [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z],
                dtype=np.float32,
            ),
            stamp=time.monotonic(),
        )
        with self._state_condition:
            self._latest_snapshot = snapshot
            self._state_condition.notify_all()

    def _collision_callback(self, msg: Bool) -> None:
        self._latest_collision = bool(msg.data)

    def _goal_callback(self, msg: Path) -> None:
        if msg.poses:
            pose = msg.poses[-1].pose.position
            self._latest_goal = np.array([pose.x, pose.y, pose.z], dtype=np.float32)

    def _feedback_callback(self, msg: AutopilotFeedback) -> None:
        with self._state_condition:
            self._latest_feedback = msg
            self._state_condition.notify_all()

    def _wait_for_snapshot(self, timeout: float) -> QuadSnapshot:
        deadline = time.monotonic() + timeout
        with self._state_condition:
            while time.monotonic() < deadline and not rospy.is_shutdown():
                if self._latest_snapshot is not None:
                    return self._latest_snapshot
                self._state_condition.wait(timeout=min(0.05, max(0.0, deadline - time.monotonic())))
        if self._latest_snapshot is None:
            raise TimeoutError(f"No odometry received on {self._odom_topic}.")
        return self._latest_snapshot

    def _wait_for_fresh_snapshot(self, after_stamp: float, timeout: float) -> QuadSnapshot:
        deadline = time.monotonic() + timeout
        with self._state_condition:
            while time.monotonic() < deadline and not rospy.is_shutdown():
                if (
                    self._latest_snapshot is not None
                    and self._latest_snapshot.stamp > after_stamp
                ):
                    return self._latest_snapshot
                self._state_condition.wait(timeout=min(0.05, max(0.0, deadline - time.monotonic())))
        raise TimeoutError(f"No fresh odometry received on {self._odom_topic}.")

    def _wait_for_odom_settle(
        self,
        target_position: np.ndarray,
        after_stamp: float,
        timeout: float,
    ) -> QuadSnapshot:
        deadline = time.monotonic() + timeout
        stable_frames = 0
        last_stamp = after_stamp
        latest: QuadSnapshot | None = None
        while time.monotonic() < deadline and not rospy.is_shutdown():
            with self._state_condition:
                snapshot = self._latest_snapshot
                if snapshot is None or snapshot.stamp <= last_stamp:
                    self._state_condition.wait(
                        timeout=min(0.05, max(0.0, deadline - time.monotonic()))
                    )
                    continue
            latest = snapshot
            last_stamp = snapshot.stamp
            position_error = float(np.linalg.norm(snapshot.position - target_position))
            speed = float(np.linalg.norm(snapshot.velocity))
            if (
                position_error <= self.odom_settle_position_tolerance
                and speed <= self.odom_settle_velocity_tolerance
            ):
                stable_frames += 1
                if stable_frames >= self.odom_settle_frames:
                    return snapshot
            else:
                stable_frames = 0
        if latest is None:
            raise TimeoutError("Odometry did not update after /gazebo/set_model_state.")
        raise TimeoutError(
            "Odometry did not settle near reset target: "
            f"position={latest.position.tolist()} velocity={latest.velocity.tolist()}."
        )

    def _wait_for_stable_takeoff(
        self,
        target_z: float,
        after_stamp: float,
        timeout: float,
    ) -> QuadSnapshot:
        deadline = time.monotonic() + timeout
        stable_frames = 0
        last_stamp = after_stamp
        latest: QuadSnapshot | None = None
        while time.monotonic() < deadline and not rospy.is_shutdown():
            with self._state_condition:
                snapshot = self._latest_snapshot
                if snapshot is None or snapshot.stamp <= last_stamp:
                    self._state_condition.wait(
                        timeout=min(0.05, max(0.0, deadline - time.monotonic()))
                    )
                    continue
            latest = snapshot
            last_stamp = snapshot.stamp
            if (
                snapshot.position[2] >= target_z
                and abs(float(snapshot.velocity[2])) <= self.odom_settle_velocity_tolerance
            ):
                stable_frames += 1
                if stable_frames >= self.odom_settle_frames:
                    return snapshot
            else:
                stable_frames = 0
        if latest is None:
            raise TimeoutError("Odometry did not update while waiting for takeoff height.")
        raise TimeoutError(
            f"Vehicle did not stabilize above {target_z:.2f} m: "
            f"z={latest.position[2]:.3f}, vz={latest.velocity[2]:.3f}."
        )

    def _wait_for_autopilot_state(self, allowed_states: set[int], timeout: float) -> None:
        deadline = time.monotonic() + timeout
        with self._state_condition:
            while time.monotonic() < deadline and not rospy.is_shutdown():
                if (
                    self._latest_feedback is not None
                    and self._latest_feedback.autopilot_state in allowed_states
                ):
                    return
                self._state_condition.wait(timeout=min(0.05, max(0.0, deadline - time.monotonic())))
        current = None if self._latest_feedback is None else self._latest_feedback.autopilot_state
        raise TimeoutError(f"Autopilot did not reach allowed states {allowed_states}; current={current}.")

    def _wait_for_service(self, service_name: str, timeout: float) -> None:
        rospy.wait_for_service(service_name, timeout=timeout)

    def _call_reset_pose(self, position: np.ndarray, yaw: float) -> None:
        state = ModelState()
        state.model_name = self.model_name
        state.pose.position.x = float(position[0])
        state.pose.position.y = float(position[1])
        state.pose.position.z = float(position[2])
        qw, qx, qy, qz = self._yaw_to_quat(yaw)
        state.pose.orientation.w = qw
        state.pose.orientation.x = qx
        state.pose.orientation.y = qy
        state.pose.orientation.z = qz
        state.twist.linear.x = 0.0
        state.twist.linear.y = 0.0
        state.twist.linear.z = 0.0
        state.twist.angular.x = 0.0
        state.twist.angular.y = 0.0
        state.twist.angular.z = 0.0
        state.reference_frame = "world"
        response = self._set_model_state(state)
        if not response.success:
            raise RuntimeError(f"/gazebo/set_model_state failed: {response.status_message}")

    def _publish_reset_reference(self, position: np.ndarray, yaw: float) -> None:
        msg = TrajectoryPoint()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        msg.heading = float(yaw)
        self._publish_reset_message(self._reset_reference_pub, msg, stamp_each_time=True)

    def _publish_reset_message(self, publisher, msg, stamp_each_time: bool = False) -> int:
        repeat = max(1, int(np.ceil(self.publish_repeat_duration / self.publish_interval)))
        published = self._publish_for_connections(
            publisher,
            msg,
            repeat=repeat,
            rate_hz=1.0 / self.publish_interval,
            stamp_each_time=stamp_each_time,
        )
        topic = getattr(publisher, "resolved_name", getattr(publisher, "name", "UNKNOWN"))
        self._last_reset_publish_counts[topic] = (
            self._last_reset_publish_counts.get(topic, 0) + published
        )
        return published

    def _publish_velocity(self, action: np.ndarray) -> None:
        msg = TwistStamped()
        msg.twist.linear.x = float(action[0])
        msg.twist.linear.y = float(action[1])
        msg.twist.linear.z = float(action[2])
        msg.twist.angular.z = float(action[3])
        self._publish_for_connections(
            self._velocity_pub,
            msg,
            repeat=max(1, int(self.action_duration * self.action_publish_rate)),
            rate_hz=self.action_publish_rate,
            stamp_each_time=True,
        )
        self._publish_zero_velocity()

    def _publish_zero_velocity(self) -> None:
        stop = TwistStamped()
        stop.header.stamp = rospy.Time.now()
        self._publish_for_connections(self._velocity_pub, stop, repeat=3, rate_hz=20.0, stamp_each_time=True)

    def _publish_for_connections(
        self,
        publisher,
        msg,
        repeat: int,
        rate_hz: float | None = None,
        stamp_each_time: bool = False,
    ) -> int:
        deadline = time.monotonic() + 2.0
        while publisher.get_num_connections() == 0 and time.monotonic() < deadline and not rospy.is_shutdown():
            time.sleep(0.05)
        interval = 0.0 if rate_hz is None or rate_hz <= 0.0 else 1.0 / rate_hz
        for _ in range(repeat):
            if stamp_each_time and hasattr(msg, "header"):
                msg.header.stamp = rospy.Time.now()
            publisher.publish(msg)
            if interval > 0.0:
                time.sleep(interval)
        return repeat

    def _build_observation(self, snapshot: QuadSnapshot) -> np.ndarray:
        goal_delta = (self._latest_goal - snapshot.position).astype(np.float32)
        obs = np.concatenate(
            [
                snapshot.position,
                snapshot.velocity,
                snapshot.orientation_xyzw,
                snapshot.angular_velocity,
                goal_delta,
                np.array([1.0 if self._latest_collision else 0.0], dtype=np.float32),
            ]
        ).astype(np.float32)
        if obs.shape != (self.observation_dim,):
            raise RuntimeError(f"Unexpected observation shape {obs.shape}.")
        return obs

    def _distance_to_goal(self, position: np.ndarray) -> float:
        return float(np.linalg.norm(self._latest_goal - position))

    def _compute_done_reason(
        self,
        snapshot: QuadSnapshot,
        distance: float,
        collision: bool,
        odometry_timeout: bool,
    ) -> str:
        config = self.reward_done_config
        if odometry_timeout:
            return "odometry_timeout"
        if collision:
            return "collision"
        if distance <= config.goal_tolerance:
            return "goal_reached"
        if snapshot.position[2] < config.min_height:
            return "height_too_low"
        if snapshot.position[2] > config.max_height:
            return "height_too_high"
        if np.max(np.abs(snapshot.position[:2])) > config.out_of_bounds_xy:
            return "out_of_bounds"
        if self._episode_step >= config.max_episode_steps:
            return "timeout"
        return "running"

    def _compute_reward(
        self,
        progress: float,
        collision: bool,
        height: float,
        speed: float,
        vertical_velocity: float,
        normalized_action_norm: float,
        normalized_z_action: float,
        done_reason: str,
    ) -> float:
        config = self.reward_done_config
        height_error = abs(height - config.target_height)
        reward = config.progress_scale * progress
        reward -= config.height_error_penalty_scale * height_error
        reward -= config.vertical_velocity_penalty_scale * abs(vertical_velocity)
        reward -= config.z_action_penalty_scale * abs(normalized_z_action)
        reward -= config.speed_penalty_scale * speed
        reward -= config.action_penalty_scale * (normalized_action_norm**2)
        if collision:
            reward -= config.collision_penalty
        if done_reason == "goal_reached":
            reward += config.goal_bonus
        if done_reason in {"timeout", "odometry_timeout"}:
            reward -= config.timeout_penalty
        return float(reward)

    def _build_info(
        self,
        snapshot: QuadSnapshot,
        previous_distance: float,
        progress: float,
        collision: bool,
        done_reason: str,
        action_norm: float,
        step_time: float,
        height_error: float = 0.0,
        vertical_velocity: float = 0.0,
        z_action: float = 0.0,
        normalized_z_action: float = 0.0,
    ) -> dict:
        distance = self._distance_to_goal(snapshot.position)
        config = self.reward_done_config
        return {
            "position": snapshot.position.astype(np.float32).tolist(),
            "velocity": snapshot.velocity.astype(np.float32).tolist(),
            "distance_to_goal": distance,
            "previous_distance_to_goal": float(previous_distance),
            "progress": float(progress),
            "collision": bool(collision),
            "height": float(snapshot.position[2]),
            "target_height": float(config.target_height),
            "height_error": float(height_error),
            "vertical_velocity": float(vertical_velocity),
            "z_action": float(z_action),
            "normalized_z_action": float(normalized_z_action),
            "height_penalty": float(config.height_error_penalty_scale * height_error),
            "vertical_velocity_penalty": float(
                config.vertical_velocity_penalty_scale * abs(vertical_velocity)
            ),
            "z_action_penalty": float(
                config.z_action_penalty_scale * abs(normalized_z_action)
            ),
            "done_reason": str(done_reason),
            "autopilot_state": self._autopilot_state_name(),
            "action_norm": float(action_norm),
            "step_time": float(step_time),
            "reset_retry_count": int(self._last_reset_retry_count),
            "goal_position": self._latest_goal.astype(np.float32).tolist(),
            "success": done_reason == "goal_reached",
            "distance": distance,
        }

    def _autopilot_state(self) -> int:
        if self._latest_feedback is None:
            return AutopilotFeedback.OFF
        return int(self._latest_feedback.autopilot_state)

    def _autopilot_state_name(self) -> str:
        state = self._autopilot_state()
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
        return mapping.get(state, f"UNKNOWN({state})")

    @staticmethod
    def _yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        return float(cy), 0.0, 0.0, float(sy)
