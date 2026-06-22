#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# The current NVIDIA/GLX path fails inside noetic_ab_workspace. Force Mesa's
# llvmpipe renderer until the host/container NVIDIA stack is repaired.
unset __VK_LAYER_NV_optimus
unset __NV_PRIME_RENDER_OFFLOAD
export __GLX_VENDOR_LIBRARY_NAME=mesa
export LIBGL_ALWAYS_SOFTWARE=1
export LIBGL_ALWAYS_INDIRECT=0

echo "[AvoidBench] DISPLAY=${DISPLAY:-unset}"
echo "[AvoidBench] __GLX_VENDOR_LIBRARY_NAME=${__GLX_VENDOR_LIBRARY_NAME}"
echo "[AvoidBench] LIBGL_ALWAYS_SOFTWARE=${LIBGL_ALWAYS_SOFTWARE}"
echo "[AvoidBench] LIBGL_ALWAYS_INDIRECT=${LIBGL_ALWAYS_INDIRECT}"

set +u
source "${PROJECT_DIR}/tools/setup_avoidbench_env.sh"
set -u

exec roslaunch avoid_manage rotors_gazebo.launch "$@"
