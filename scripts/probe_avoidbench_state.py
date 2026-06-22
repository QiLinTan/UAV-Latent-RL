from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import rospy
from nav_msgs.msg import Odometry


@dataclass
class OdomStats:
    count: int = 0
    first_wall_time: float | None = None
    last_wall_time: float | None = None
    last_message: Odometry | None = None

    def update(self, msg: Odometry) -> None:
        now = time.monotonic()
        if self.first_wall_time is None:
            self.first_wall_time = now
        self.last_wall_time = now
        self.last_message = msg
        self.count += 1

    @property
    def frequency_hz(self) -> float:
        if self.count <= 1 or self.first_wall_time is None or self.last_wall_time is None:
            return 0.0
        elapsed = self.last_wall_time - self.first_wall_time
        if elapsed <= 0.0:
            return 0.0
        return (self.count - 1) / elapsed


def format_vec3(x: float, y: float, z: float) -> str:
    return f"({x:+.4f}, {y:+.4f}, {z:+.4f})"


def format_quat(x: float, y: float, z: float, w: float) -> str:
    return f"({x:+.5f}, {y:+.5f}, {z:+.5f}, {w:+.5f})"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only odometry probe for the AvoidBench ROS/Gazebo/autopilot runtime."
        )
    )
    parser.add_argument(
        "--odom-topic",
        default="/hummingbird/ground_truth/odometry",
        help="Odometry topic to subscribe to.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="How long to observe odometry in seconds.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return exit code 1 if no odometry message is received.",
    )
    args = parser.parse_args()
    if args.duration <= 0.0:
        parser.error("--duration must be positive.")

    rospy.init_node("avoidbench_state_probe", anonymous=True, disable_signals=True)
    stats = OdomStats()

    def callback(msg: Odometry) -> None:
        stats.update(msg)

    subscriber = rospy.Subscriber(args.odom_topic, Odometry, callback, queue_size=50)
    del subscriber

    deadline = time.monotonic() + args.duration
    print("=== avoidbench state probe ===")
    print(f"topic: {args.odom_topic}")
    print(f"duration_s: {args.duration:.2f}")
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        time.sleep(0.05)

    print("\n=== summary ===")
    print(f"messages_received: {stats.count}")
    print(f"estimated_frequency_hz: {stats.frequency_hz:.2f}")

    if stats.last_message is None:
        print("status: NO_ODOMETRY")
        print("This probe did not publish messages.")
        return 1 if args.strict else 0

    msg = stats.last_message
    pos = msg.pose.pose.position
    ori = msg.pose.pose.orientation
    lin = msg.twist.twist.linear
    ang = msg.twist.twist.angular
    linear_speed = math.sqrt(lin.x**2 + lin.y**2 + lin.z**2)
    angular_speed = math.sqrt(ang.x**2 + ang.y**2 + ang.z**2)

    print("status: ODOMETRY_OK")
    print(f"frame_id: {msg.header.frame_id}")
    print(f"child_frame_id: {msg.child_frame_id}")
    print(f"position: {format_vec3(pos.x, pos.y, pos.z)}")
    print(f"linear_velocity: {format_vec3(lin.x, lin.y, lin.z)}")
    print(f"orientation_xyzw: {format_quat(ori.x, ori.y, ori.z, ori.w)}")
    print(f"angular_velocity: {format_vec3(ang.x, ang.y, ang.z)}")
    print(f"linear_speed_mps: {linear_speed:.4f}")
    print(f"angular_speed_radps: {angular_speed:.4f}")
    if msg.header.stamp != rospy.Time():
        age = (rospy.Time.now() - msg.header.stamp).to_sec()
        print(f"message_age_s: {age:.4f}")
    print("This probe did not publish messages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
