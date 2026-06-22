from __future__ import annotations

import argparse
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass

import rospy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Empty

try:
    from rpg_quadrotor_msgs.msg import AutopilotFeedback
except ImportError:  # pragma: no cover - available in the sourced ROS runtime
    AutopilotFeedback = None


CONTROL_KEYWORDS = ("velocity", "trajectory", "command", "autopilot", "reference")
HELPER_SUFFIXES = (
    "autopilot/start",
    "autopilot/force_hover",
    "autopilot/reset_reference_state",
    "bridge/arm",
)
PREFERRED_ACTION_TOPICS = (
    "autopilot/velocity_command",
    "autopilot/reference_state",
    "autopilot/pose_command",
    "autopilot/trajectory",
    "autopilot/control_command_input",
)


@dataclass
class OdomSample:
    wall_time: float
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]


class OdomRecorder:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._samples: deque[OdomSample] = deque()
        self._latest: OdomSample | None = None

    def callback(self, msg: Odometry) -> None:
        sample = OdomSample(
            wall_time=time.monotonic(),
            position=(
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ),
            velocity=(
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z,
            ),
        )
        with self._lock:
            self._latest = sample
            self._samples.append(sample)
            while len(self._samples) > 10000:
                self._samples.popleft()

    def samples_since(self, start_time: float) -> list[OdomSample]:
        with self._lock:
            return [sample for sample in self._samples if sample.wall_time >= start_time]

    @property
    def latest(self) -> OdomSample | None:
        with self._lock:
            return self._latest


def run_cli(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, text=True, capture_output=True)


def rostopic_list() -> list[str]:
    result = run_cli(["rostopic", "list"])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "rostopic list failed")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def rostopic_info(topic: str) -> str:
    result = run_cli(["rostopic", "info", topic])
    if result.returncode != 0:
        return f"  info_error: {result.stderr.strip() or 'unknown error'}"
    return "\n".join(f"  {line}" for line in result.stdout.strip().splitlines())


def topic_matches(topic: str, namespace: str) -> bool:
    if not topic.startswith(namespace.rstrip("/") + "/") and topic not in {
        namespace.rstrip("/") + "/" + suffix for suffix in HELPER_SUFFIXES
    }:
        return False
    lowered = topic.lower()
    return any(keyword in lowered for keyword in CONTROL_KEYWORDS) or any(
        topic.endswith(suffix) for suffix in HELPER_SUFFIXES
    )


def classify_topic(topic: str) -> str:
    if topic.endswith(
        ("control_command_input", "command/motor_speed", "gazebo/command/motor_speed", "control_command")
    ):
        return "low_level_or_output"
    if topic.endswith(HELPER_SUFFIXES):
        return "mode_or_reset_helper"
    if topic.endswith(PREFERRED_ACTION_TOPICS):
        return "high_level_action"
    return "other_candidate"


def print_candidates(namespace: str) -> list[str]:
    topics = rostopic_list()
    namespace = namespace.rstrip("/")
    candidates = sorted(topic for topic in topics if topic_matches(topic, namespace))
    print("=== avoidbench action probe ===")
    print(f"namespace: {namespace}")
    print(f"candidate_topic_count: {len(candidates)}")
    if not candidates:
        print("status: NO_CANDIDATE_TOPICS")
        return []
    for topic in candidates:
        print(f"\n[{classify_topic(topic)}] {topic}")
        print(rostopic_info(topic))
    return candidates


def wait_for_odometry(recorder: OdomRecorder, timeout: float) -> OdomSample | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not rospy.is_shutdown():
        latest = recorder.latest
        if latest is not None:
            return latest
        time.sleep(0.02)
    return recorder.latest


def publish_velocity_test_action(
    topic: str,
    axis: str,
    magnitude: float,
    duration: float,
    publish_rate: float,
) -> tuple[float, float]:
    publisher = rospy.Publisher(topic, TwistStamped, queue_size=1)
    time.sleep(0.2)
    msg = TwistStamped()
    value_map = {"x": (magnitude, 0.0, 0.0, 0.0), "y": (0.0, magnitude, 0.0, 0.0), "z": (0.0, 0.0, magnitude, 0.0)}
    if axis == "yaw":
        linear_x, linear_y, linear_z, angular_z = (0.0, 0.0, 0.0, magnitude)
    else:
        linear_x, linear_y, linear_z, angular_z = value_map[axis]
    msg.twist.linear.x = linear_x
    msg.twist.linear.y = linear_y
    msg.twist.linear.z = linear_z
    msg.twist.angular.z = angular_z

    start_time = time.monotonic()
    rate = rospy.Rate(max(publish_rate, 1.0))
    while time.monotonic() - start_time < duration and not rospy.is_shutdown():
        msg.header.stamp = rospy.Time.now()
        publisher.publish(msg)
        rate.sleep()

    stop_msg = TwistStamped()
    stop_msg.header.stamp = rospy.Time.now()
    publisher.publish(stop_msg)
    return start_time, time.monotonic()


def publish_bool_once(topic: str, value: bool) -> None:
    publisher = rospy.Publisher(topic, Bool, queue_size=1)
    deadline = time.monotonic() + 2.0
    while publisher.get_num_connections() == 0 and time.monotonic() < deadline and not rospy.is_shutdown():
        time.sleep(0.05)
    for _ in range(5):
        publisher.publish(Bool(data=value))
        time.sleep(0.05)


def publish_empty_once(topic: str) -> None:
    publisher = rospy.Publisher(topic, Empty, queue_size=1)
    deadline = time.monotonic() + 2.0
    while publisher.get_num_connections() == 0 and time.monotonic() < deadline and not rospy.is_shutdown():
        time.sleep(0.05)
    for _ in range(5):
        publisher.publish(Empty())
        time.sleep(0.05)


def wait_for_autopilot_feedback(topic: str, timeout: float) -> AutopilotFeedback | None:
    if AutopilotFeedback is None:
        return None
    try:
        return rospy.wait_for_message(topic, AutopilotFeedback, timeout=timeout)
    except rospy.ROSException:
        return None


def autopilot_state_name(feedback: AutopilotFeedback | None) -> str:
    if feedback is None or AutopilotFeedback is None:
        return "UNKNOWN"
    state_map = {
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
    return state_map.get(feedback.autopilot_state, f"UNKNOWN({feedback.autopilot_state})")


def wait_for_autopilot_state(topic: str, allowed_states: set[int], timeout: float) -> AutopilotFeedback | None:
    deadline = time.monotonic() + timeout
    latest: AutopilotFeedback | None = None
    while time.monotonic() < deadline and not rospy.is_shutdown():
        feedback = wait_for_autopilot_feedback(topic, timeout=0.5)
        if feedback is None:
            continue
        latest = feedback
        if feedback.autopilot_state in allowed_states:
            return feedback
    return latest


def evaluate_response(
    baseline: OdomSample,
    post_samples: list[OdomSample],
    axis: str,
) -> tuple[bool, float, float]:
    if not post_samples:
        return False, 0.0, 0.0
    axis_index = {"x": 0, "y": 1, "z": 2}.get(axis, 2)
    final_sample = post_samples[-1]
    delta_position = final_sample.position[axis_index] - baseline.position[axis_index]
    max_velocity_delta = max(
        abs(sample.velocity[axis_index] - baseline.velocity[axis_index]) for sample in post_samples
    )
    if axis == "yaw":
        return False, delta_position, max_velocity_delta
    responded = abs(delta_position) > 0.005 or max_velocity_delta > 0.02
    return responded, delta_position, max_velocity_delta


def main() -> int:
    parser = argparse.ArgumentParser(
        description="List AvoidBench action topics and optionally send a tiny guarded velocity test."
    )
    parser.add_argument("--namespace", default="/hummingbird", help="ROS namespace to inspect.")
    parser.add_argument(
        "--odom-topic",
        default="/hummingbird/ground_truth/odometry",
        help="Odometry topic used to confirm action response.",
    )
    parser.add_argument(
        "--send-test-action",
        action="store_true",
        help="Publish a tiny velocity command after confirming odometry is readable.",
    )
    parser.add_argument(
        "--action-topic",
        default="/hummingbird/autopilot/velocity_command",
        help="Topic used for the guarded test action.",
    )
    parser.add_argument(
        "--axis",
        choices=("x", "y", "z", "yaw"),
        default="x",
        help="Axis used by the guarded test action.",
    )
    parser.add_argument(
        "--magnitude",
        type=float,
        default=0.05,
        help="Linear speed in m/s or yaw rate in rad/s for the guarded test action.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.20,
        help="How long to publish the guarded test action in seconds.",
    )
    parser.add_argument(
        "--publish-rate",
        type=float,
        default=20.0,
        help="Publish rate for the guarded test action.",
    )
    parser.add_argument(
        "--response-window",
        type=float,
        default=1.0,
        help="How long to observe odometry after publishing the test action.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if candidate topics are missing or if the test action shows no response.",
    )
    parser.add_argument(
        "--feedback-topic",
        default="/hummingbird/autopilot/feedback",
        help="Autopilot feedback topic used to report controller mode.",
    )
    parser.add_argument(
        "--publish-start-before-test",
        action="store_true",
        help="Publish one /autopilot/start message before the guarded test action.",
    )
    parser.add_argument(
        "--publish-arm-before-test",
        action="store_true",
        help="Publish one /bridge/arm Bool(true) message before the guarded test action.",
    )
    parser.add_argument(
        "--prepare-timeout",
        type=float,
        default=8.0,
        help="How long to wait for HOVER or VELOCITY_CONTROL after the optional prepare step.",
    )
    parser.add_argument(
        "--settle-time",
        type=float,
        default=0.5,
        help="Extra wait after the optional prepare step before capturing the action baseline.",
    )
    args = parser.parse_args()

    candidates = print_candidates(args.namespace)
    if not candidates:
        return 1 if args.strict else 0

    if not args.send_test_action:
        print("\nstatus: LIST_ONLY")
        print("No action was published. Re-run with --send-test-action to attempt a tiny guarded command.")
        return 0

    rospy.init_node("avoidbench_action_probe", anonymous=True, disable_signals=True)
    recorder = OdomRecorder()
    subscriber = rospy.Subscriber(args.odom_topic, Odometry, recorder.callback, queue_size=200)
    del subscriber

    initial_odometry = wait_for_odometry(recorder, timeout=3.0)
    if initial_odometry is None:
        print("\nstatus: ODOMETRY_UNAVAILABLE")
        print("Refusing to publish because odometry is not readable.")
        return 1

    feedback_before = wait_for_autopilot_feedback(args.feedback_topic, timeout=1.0)
    print("\n=== autopilot feedback ===")
    print(f"feedback_topic: {args.feedback_topic}")
    print(f"autopilot_state_before: {autopilot_state_name(feedback_before)}")

    if args.publish_arm_before_test:
        arm_topic = args.namespace.rstrip("/") + "/bridge/arm"
        print(f"publishing_arm_before_test: {arm_topic} -> True")
        publish_bool_once(arm_topic, True)
        time.sleep(0.3)

    if args.publish_start_before_test:
        start_topic = args.namespace.rstrip("/") + "/autopilot/start"
        print(f"publishing_start_before_test: {start_topic}")
        publish_empty_once(start_topic)
    feedback_after_prepare = wait_for_autopilot_feedback(args.feedback_topic, timeout=1.0)
    print(f"autopilot_state_after_prepare: {autopilot_state_name(feedback_after_prepare)}")

    if args.publish_start_before_test:
        hover_feedback = wait_for_autopilot_state(
            topic=args.feedback_topic,
            allowed_states={
                AutopilotFeedback.HOVER,
                AutopilotFeedback.VELOCITY_CONTROL,
            }
            if AutopilotFeedback is not None
            else set(),
            timeout=args.prepare_timeout,
        )
        print(f"autopilot_state_after_wait: {autopilot_state_name(hover_feedback)}")
        if args.settle_time > 0.0:
            time.sleep(args.settle_time)

    baseline = recorder.latest
    if baseline is None:
        print("status: ODOMETRY_LOST_AFTER_PREPARE")
        return 1

    print("\n=== guarded action test ===")
    print(f"action_topic: {args.action_topic}")
    print(f"axis: {args.axis}")
    print(f"magnitude: {args.magnitude:.4f}")
    print(f"duration_s: {args.duration:.3f}")
    print(f"publish_rate_hz: {args.publish_rate:.1f}")
    print(f"baseline_position: {baseline.position}")
    print(f"baseline_velocity: {baseline.velocity}")

    command_start, command_end = publish_velocity_test_action(
        topic=args.action_topic,
        axis=args.axis,
        magnitude=args.magnitude,
        duration=args.duration,
        publish_rate=args.publish_rate,
    )
    observe_until = command_end + args.response_window
    while time.monotonic() < observe_until and not rospy.is_shutdown():
        time.sleep(0.02)

    post_samples = recorder.samples_since(command_start)
    responded, delta_position, max_velocity_delta = evaluate_response(
        baseline=baseline, post_samples=post_samples, axis=args.axis
    )
    final_sample = recorder.latest
    print(f"post_sample_count: {len(post_samples)}")
    print(f"delta_position_axis: {delta_position:+.5f}")
    print(f"max_velocity_delta_axis: {max_velocity_delta:+.5f}")
    if final_sample is not None:
        print(f"final_position: {final_sample.position}")
        print(f"final_velocity: {final_sample.velocity}")

    if responded:
        print("status: ACTION_RESPONSE_DETECTED")
        return 0

    print("status: ACTION_RESPONSE_NOT_DETECTED")
    print(
        "Possible causes: autopilot still OFF, vehicle still grounded, "
        "avoid_manage overrides commands, or the chosen topic is not authoritative."
    )
    return 1 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
