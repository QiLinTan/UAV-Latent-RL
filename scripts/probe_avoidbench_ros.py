from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from dataclasses import dataclass
from enum import Enum


EXPECTED_TOPIC_SUFFIXES = (
    "ground_truth/odometry",
    "autopilot/reset_reference_state",
    "autopilot/start",
    "autopilot/force_hover",
    "bridge/arm",
    "goal_point",
    "autopilot/velocity_command",
    "autopilot/reference_state",
    "autopilot/pose_command",
)

EXPECTED_SERVICES = (
    "/gazebo/set_model_state",
)


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class GraphSnapshot:
    topics: set[str]
    services: set[str]
    topic_result: CommandResult
    service_result: CommandResult


class ProbeState(Enum):
    ROS_CLI_UNAVAILABLE = "ROS_CLI_UNAVAILABLE"
    MASTER_UNAVAILABLE = "ROS_MASTER_UNAVAILABLE"
    INTERFACES_MISSING = "ROS_MASTER_AVAILABLE_INTERFACES_MISSING"
    READY = "ROS_INTERFACES_READY"


def run_command(command: list[str], timeout: float) -> CommandResult:
    executable = command[0]
    if shutil.which(executable) is None:
        return CommandResult(
            tuple(command),
            127,
            "",
            f"{executable} is not available. Source the ROS Noetic environment.",
        )

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return CommandResult(
            tuple(command),
            124,
            "",
            f"command timed out after {timeout:.1f}s",
        )

    return CommandResult(
        tuple(command),
        completed.returncode,
        completed.stdout.strip(),
        completed.stderr.strip(),
    )


def print_result(result: CommandResult, *, indent: str = "") -> None:
    command = " ".join(result.command)
    print(f"{indent}$ {command}")
    if result.stdout:
        for line in result.stdout.splitlines():
            print(f"{indent}{line}")
    if result.stderr:
        for line in result.stderr.splitlines():
            print(f"{indent}stderr: {line}")
    if result.returncode != 0:
        print(f"{indent}return code: {result.returncode}")


def list_names(command: list[str], timeout: float) -> tuple[set[str], CommandResult]:
    result = run_command(command, timeout)
    names = {
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip().startswith("/")
    }
    return names, result


def resolve_topic(namespace: str, suffix: str) -> str:
    namespace = namespace.strip("/")
    suffix = suffix.strip("/")
    if namespace:
        return f"/{namespace}/{suffix}"
    return f"/{suffix}"


def collect_graph(timeout: float) -> GraphSnapshot:
    topics, topic_result = list_names(["rostopic", "list"], timeout)
    services, service_result = list_names(["rosservice", "list"], timeout)
    return GraphSnapshot(
        topics=topics,
        services=services,
        topic_result=topic_result,
        service_result=service_result,
    )


def classify_graph(
    snapshot: GraphSnapshot,
    expected_topics: tuple[str, ...],
) -> ProbeState:
    results = (snapshot.topic_result, snapshot.service_result)
    if any(result.returncode == 127 for result in results):
        return ProbeState.ROS_CLI_UNAVAILABLE
    if any(result.returncode != 0 for result in results):
        return ProbeState.MASTER_UNAVAILABLE

    topics_ready = all(name in snapshot.topics for name in expected_topics)
    services_ready = all(name in snapshot.services for name in EXPECTED_SERVICES)
    if topics_ready and services_ready:
        return ProbeState.READY
    return ProbeState.INTERFACES_MISSING


def wait_for_graph(
    wait_timeout: float,
    command_timeout: float,
    expected_topics: tuple[str, ...],
) -> tuple[GraphSnapshot, ProbeState]:
    deadline = time.monotonic() + wait_timeout
    attempt = 0

    while True:
        attempt += 1
        snapshot = collect_graph(command_timeout)
        state = classify_graph(snapshot, expected_topics)
        if state is ProbeState.READY or time.monotonic() >= deadline:
            return snapshot, state

        remaining = max(0.0, deadline - time.monotonic())
        print(
            f"waiting for ROS graph: attempt={attempt}, "
            f"state={state.value}, remaining={remaining:.1f}s"
        )
        time.sleep(min(1.0, remaining))


def inspect_endpoint(
    kind: str,
    name: str,
    available: set[str],
    timeout: float,
) -> bool:
    present = name in available
    print(f"\n[{kind}] {name}: {'FOUND' if present else 'MISSING'}")
    if not present:
        return False

    command = "rostopic" if kind == "topic" else "rosservice"
    for operation in ("type", "info"):
        print_result(
            run_command([command, operation, name], timeout),
            indent="  ",
        )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only probe for ROS topics and services needed by a future "
            "AvoidBenchRLEnv reset()/step() adapter."
        )
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Timeout in seconds for each ROS CLI command.",
    )
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=0.0,
        help=(
            "Maximum seconds to wait for the ROS master and expected interfaces. "
            "The default performs one immediate read-only check."
        ),
    )
    parser.add_argument(
        "--namespace",
        default="/hummingbird",
        help=(
            "Quadrotor namespace used by the official AvoidBench launch "
            "(default: /hummingbird)."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return exit code 1 unless the ROS master and all expected endpoints are ready.",
    )
    args = parser.parse_args()
    if args.timeout <= 0.0:
        parser.error("--timeout must be positive.")
    if args.wait_timeout < 0.0:
        parser.error("--wait-timeout must be non-negative.")

    expected_topics = tuple(
        resolve_topic(args.namespace, suffix) for suffix in EXPECTED_TOPIC_SUFFIXES
    )
    snapshot, state = wait_for_graph(
        args.wait_timeout,
        args.timeout,
        expected_topics,
    )

    print("\n=== ROS graph state ===")
    print(f"state: {state.value}")
    if state is ProbeState.ROS_CLI_UNAVAILABLE:
        print("ROS CLI tools are unavailable; source ROS Noetic and AvoidBench first.")
    elif state is ProbeState.MASTER_UNAVAILABLE:
        print("ROS CLI tools are available, but the ROS master cannot be reached.")
    elif state is ProbeState.INTERFACES_MISSING:
        print("ROS master is reachable, but one or more required interfaces are missing.")
    else:
        print("ROS master is reachable and all required interfaces are present.")

    print("\n=== rostopic list ===")
    print_result(snapshot.topic_result)
    print("\n=== rosservice list ===")
    print_result(snapshot.service_result)

    print("\n=== expected RL-adapter interfaces ===")
    found = []
    print(
        "official remap: flight_pilot/state_estimate -> "
        f"{resolve_topic(args.namespace, 'ground_truth/odometry')}"
    )
    for topic in expected_topics:
        found.append(inspect_endpoint("topic", topic, snapshot.topics, args.timeout))
    for service in EXPECTED_SERVICES:
        found.append(
            inspect_endpoint("service", service, snapshot.services, args.timeout)
        )

    total = len(found)
    found_count = sum(found)
    print("\n=== summary ===")
    print(f"topics discovered: {len(snapshot.topics)}")
    print(f"services discovered: {len(snapshot.services)}")
    print(f"expected endpoints found: {found_count}/{total}")
    print("This probe did not publish messages or call services.")

    if args.strict and state is not ProbeState.READY:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
