# AvoidBench `avoidbridge` Integration

This AvoidBench build exposes the ROS/catkin module `avoidbridge`, not
`flightgym.AvoidVisionEnv_v1`.

## Runtime setup

Inside the `noetic_ab_glx` container:

```bash
source /opt/ros/noetic/setup.bash
source /AvoidBench/devel/setup.bash

python3 -c \
  "from avoidbridge import AvoidbenchBridge, quadStateEstimate, mission_parameter; print('avoidbridge OK')"
```

Start Unity separately:

```bash
/AvoidBench/src/avoidbench/unity_scene/AvoidBench/AvoidBench.x86_64
```

## Image API mismatch

The stock pybind11 method `getImages()` always calls the four-argument stereo
SGM overload. With `camera.perform_sgm: false`, that C++ method returns early
and produces empty arrays.

Recommended fix: apply the additive wrapper patch:

```bash
cd /AvoidBench
git apply --check /path/to/UAV-AvoidBench-RL/patches/avoidbench_unity_depth_pybind.patch
git apply /path/to/UAV-AvoidBench-RL/patches/avoidbench_unity_depth_pybind.patch

catkin clean -y avoidlib
catkin build avoidlib
source /AvoidBench/devel/setup.bash
```

The clean rebuild is required because the patch changes Eigen ABI compile
definitions for every avoidlib translation unit.

The patch preserves `getImages()` for stereo SGM and adds:

```python
left_bgr, unity_depth = bridge.getUnityDepthImages()
```

It also avoids constructing the CUDA SGM implementation when
`camera.perform_sgm: false`. This keeps the Unity-depth path independent of
stereo processing and its GPU allocations.

## Clean-exit fix

The shutdown crash was not caused by Unity, ZMQ, CUDA, `cv::Mat`, or NumPy.
It can be reproduced by constructing `quadStateEstimate()` without creating an
`AvoidbenchBridge`.

The root cause is a cross-package Eigen ABI mismatch:

- `quadrotor_common` is compiled as C++11 without AVX;
- avoidlib is compiled as C++17 with `-march=native`;
- `QuadStateEstimate` therefore uses different Eigen alignment and aligned
  allocation/deallocation rules across the two shared libraries.

The patch fixes this by:

1. fixing avoidlib's Eigen ABI alignment at 16 bytes to match
   `quadrotor_common`;
2. exposing a plain Python state proxy instead of heap-allocating the native
   C++11 `QuadStateEstimate`;
3. creating the native state on the stack only for the duration of
   `updateUnity()`;
4. returning NumPy-owned copies of OpenCV images.

Verify the state lifecycle independently:

```bash
python3 -c \
  "from avoidbridge import quadStateEstimate; quadStateEstimate(); print('clean exit')"
echo $?
```

The exit code must be `0`. No `os._exit(0)` workaround is required.

The returned arrays are:

- `left_bgr`: `uint8`, shape `(height, width, 3)`
- `unity_depth`: `float32`, shape `(height, width, 1)`

Unity depth uses the raw units produced by this AvoidBench branch. The source
currently scales Unity's float depth by `100` in `unity_bridge.cpp`; do not
silently reinterpret it as millimetres.

## Probe

After rebuilding:

```bash
cd /path/to/UAV-AvoidBench-RL
python3 -m scripts.probe_avoidbench \
  --config /AvoidBench/src/avoidbench/avoid_manage/params/task_indoor.yaml \
  --image-mode unity \
  --steps 20
```

For the original stereo path, set `camera.perform_sgm: true`, rebuild/restart
the bridge process if needed, then use:

```bash
python3 -m scripts.probe_avoidbench --image-mode stereo
```

Do not expect the unchanged `bridge.getImages()` method to return Unity depth:
it remains the three-output stereo API. With `perform_sgm: false`, call
`bridge.getUnityDepthImages()` or the project adapter's
`adapter.get_images("unity")`.

## Training boundary

`AvoidbenchBridge` only supplies rendering, obstacle management, point clouds,
and collision checks. It does not provide an RL `step(action)` transition,
reward, termination, or reset interface.

Before TD3 training, a separate control environment must:

1. send commands through RotorS/ROS or another dynamics backend;
2. read odometry and build `quadStateEstimate`;
3. call `updateUnity(state)` and acquire images;
4. define reward, success, collision, timeout, and reset;
5. expose a Gymnasium-style environment.
