#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CONTAINER_NAME="${AVOIDBENCH_CONTAINER_NAME:-noetic_ab_workspace}"
IMAGE="${AVOIDBENCH_IMAGE:-noetic_avoidbench_unitydepth_fixed:local}"
CONTAINER_PROJECT_DIR="/workspace/UAV-AvoidBench-RL"

usage() {
  cat <<EOF
Usage: $0 <create|start|enter|check|recreate|status>

Environment overrides:
  AVOIDBENCH_CONTAINER_NAME  Container name (default: ${CONTAINER_NAME})
  AVOIDBENCH_IMAGE           Docker image (default: ${IMAGE})

Typical workflow:
  $0 create
  $0 enter

Inside the container, the repository is mounted at:
  ${CONTAINER_PROJECT_DIR}
EOF
}

container_exists() {
  docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1
}

container_running() {
  [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null || true)" == "true" ]]
}

create_container() {
  if container_exists; then
    echo "Container already exists: ${CONTAINER_NAME}" >&2
    echo "Use '$0 start', '$0 enter', or '$0 recreate'." >&2
    return 1
  fi

  docker run -d \
    --name "${CONTAINER_NAME}" \
    --network host \
    --ipc host \
    --gpus all \
    --device /dev/dri:/dev/dri \
    -e DISPLAY="${DISPLAY:-:0}" \
    -e QT_X11_NO_MITSHM=1 \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e UAV_AVOIDBENCH_RL_DIR="${CONTAINER_PROJECT_DIR}" \
    -e PYTHONPATH="${CONTAINER_PROJECT_DIR}" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "${PROJECT_DIR}:${CONTAINER_PROJECT_DIR}" \
    --workdir "${CONTAINER_PROJECT_DIR}" \
    "${IMAGE}" \
    sleep infinity

  echo "Created ${CONTAINER_NAME} from ${IMAGE}."
}

start_container() {
  if ! container_exists; then
    create_container
    return
  fi
  if ! container_running; then
    docker start "${CONTAINER_NAME}" >/dev/null
  fi
  echo "${CONTAINER_NAME} is running."
}

enter_container() {
  start_container
  exec docker exec -it \
    --workdir "${CONTAINER_PROJECT_DIR}" \
    "${CONTAINER_NAME}" \
    bash -lc \
    "source ${CONTAINER_PROJECT_DIR}/tools/setup_avoidbench_env.sh && exec bash -i"
}

check_container() {
  start_container
  docker exec \
    --workdir "${CONTAINER_PROJECT_DIR}" \
    "${CONTAINER_NAME}" \
    bash -lc "
      source ${CONTAINER_PROJECT_DIR}/tools/setup_avoidbench_env.sh
      test -f scripts/stress_avoidbench_rl_env.py
      python3 -c 'import scripts.stress_avoidbench_rl_env; import rospy; print(\"AvoidBench workspace import OK\")'
      python3 -m scripts.stress_avoidbench_rl_env --help >/dev/null
    "
  echo "Container mount, ROS environment, and Python module imports are healthy."
}

recreate_container() {
  if container_exists; then
    backup="${CONTAINER_NAME}_backup_$(date +%Y%m%d_%H%M%S)"
    if container_running; then
      docker stop "${CONTAINER_NAME}" >/dev/null
    fi
    docker rename "${CONTAINER_NAME}" "${backup}"
    echo "Preserved old container as ${backup}."
  fi
  create_container
  check_container
}

show_status() {
  if ! container_exists; then
    echo "${CONTAINER_NAME}: missing"
    return
  fi
  docker inspect "${CONTAINER_NAME}" --format \
    'name={{.Name}} running={{.State.Running}} image={{.Config.Image}} workdir={{.Config.WorkingDir}}'
  docker inspect "${CONTAINER_NAME}" --format '{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}'
}

case "${1:-}" in
  create) create_container ;;
  start) start_container ;;
  enter) enter_container ;;
  check) check_container ;;
  recreate) recreate_container ;;
  status) show_status ;;
  *) usage; exit 2 ;;
esac
