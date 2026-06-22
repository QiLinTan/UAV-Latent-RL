#!/usr/bin/env bash

# Source this file inside the AvoidBench container.
set -e

PROJECT_DIR="${UAV_AVOIDBENCH_RL_DIR:-/workspace/UAV-AvoidBench-RL}"

if [[ ! -f /opt/ros/noetic/setup.bash ]]; then
  echo "ROS Noetic setup not found: /opt/ros/noetic/setup.bash" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ ! -f /AvoidBench/devel/setup.bash ]]; then
  echo "AvoidBench setup not found: /AvoidBench/devel/setup.bash" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ ! -d "${PROJECT_DIR}/scripts" ]]; then
  echo "Project mount not found: ${PROJECT_DIR}" >&2
  echo "Create the container with tools/avoidbench_container.sh." >&2
  return 1 2>/dev/null || exit 1
fi

source /opt/ros/noetic/setup.bash
source /AvoidBench/devel/setup.bash

export UAV_AVOIDBENCH_RL_DIR="${PROJECT_DIR}"
case ":${PYTHONPATH:-}:" in
  *":${PROJECT_DIR}:"*) ;;
  *) export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

cd "${PROJECT_DIR}"
