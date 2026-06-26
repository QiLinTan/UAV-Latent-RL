# AvoidBench Collision Source Trace

Date: 2026-06-26

## Scope

This report traces where `collision=True` comes from in AvoidBench. It does not
authorize training, Stage 1, latent, or depth-encoder work.

Evidence run:

```text
runs/avoidbench_collision_ownership/20260626-121259/
```

Key lifecycle result at `(0, 0, 5)`:

- `before_update_collision=false`
- `after_first_update_collision=true`
- `after_10_updates_collision=true`
- `spawn_return=true`
- `scene_changed=false`
- `after_scene_changed_collision=true`

This means the direct bridge collision value is not a Python/C++ default true.
It becomes true after the first `updateUnity()` round trip.

Important lifecycle caveat: native `UnityBridge` is a process-local singleton:

```text
/AvoidBench/src/avoidbench/avoidlib/include/avoidlib/bridges/unity_bridge.hpp:69-73
```

Multiple Python `AvoidbenchBridge` objects in one process can reuse collision
state. Clean lifecycle tests must use one Python process per case.

## Answers

### 1. Is `getQuadCollisionState()` local C++ or Unity state?

It is Unity state.

The pybind wrapper exposes `AvoidbenchBridge::getQuadCollisionState()` directly:

```text
/AvoidBench/src/avoidbench/avoidlib/src/wrapper/avoidbench_wrapper.cpp:169-177
```

`AvoidbenchBridge::getQuadCollisionState()` returns
`unity_bridge_ptr_->collisions[0]` if a Unity collision vector exists, otherwise
false:

```text
/AvoidBench/src/avoidbench/avoidlib/src/bridges/avoidbench_bridge.cpp:270-277
```

`UnityBridge::handleOutput()` clears and repopulates that vector from the Unity
JSON field `pub_vehicles[*].collision`:

```text
/AvoidBench/src/avoidbench/avoidlib/src/bridges/unity_bridge.cpp:222-248
/AvoidBench/src/avoidbench/avoidlib/include/avoidlib/bridges/unity_message_types.hpp:139-149
/AvoidBench/src/avoidbench/avoidlib/include/avoidlib/bridges/unity_message_types.hpp:263-275
```

So the value is not computed from Gazebo contacts, and not computed by local
Python.

### 2. Is collision sticky?

The bridge stores the latest Unity output. `UnityBridge::handleOutput()` clears
`collisions` and pushes the latest `sub_msg.sub_vehicles[idx].collision` each
render frame:

```text
/AvoidBench/src/avoidbench/avoidlib/src/bridges/unity_bridge.cpp:240-248
```

There is no separate "clear collision" method in the exposed pybind API.
Collision can therefore behave sticky from Python's point of view if Unity keeps
returning true, but the C++ bridge itself is not initialized to true.

Supporting evidence:

- `Quadrotor` initializes `collision_(false)`:
  `/AvoidBench/src/avoidbench/avoidlib/src/objects/quadrotor.cpp:5-32`
- before the first update, direct bridge lifecycle reports collision false:
  `runs/avoidbench_collision_ownership/20260626-121259/summary.json`
- after the first `updateUnity()`, collision becomes true and remains true.

### 3. Is there a clear/reset collision interface?

No clear/reset collision interface was found in the exposed pybind methods:

```text
/AvoidBench/src/avoidbench/avoidlib/src/wrapper/avoidbench_wrapper.cpp:169-181
```

Available relevant methods are:

- `updateUnity`
- `getQuadCollisionState`
- `setParamFromMission`
- `spawnObstacles`
- `ifSceneChanged`
- `SpawnNewObs`
- `checkCollisionState`

None explicitly clears vehicle collision state.

### 4. Does `updateUnity()` update collision?

Yes.

`AvoidbenchBridge::updateUnity()` sets the quad state, sends a render request,
waits for `UnityBridge::handleOutput()`, and then returns true:

```text
/AvoidBench/src/avoidbench/avoidlib/src/bridges/avoidbench_bridge.cpp:150-170
```

`handleOutput()` is the point where Unity's returned collision flag is copied
into the bridge:

```text
/AvoidBench/src/avoidbench/avoidlib/src/bridges/unity_bridge.cpp:222-248
```

The lifecycle probe confirms the first `updateUnity()` changes collision from
false to true.

### 5. How do spawn/scene change relate to collision?

`spawnObstacles()` only populates object/tree parameters in the bridge. It does
not itself wait for the Unity scene to report a changed state.

Official C++ manager sequence:

```text
/AvoidBench/src/avoidbench/avoid_manage/src/avoid_manage.cpp:236-246
```

The sequence is:

1. `setParamFromMission(p_m)`
2. `spawnObstacles()`
3. repeat `SpawnNewObs()` until `ifSceneChanged()`
4. check start/end points with `checkCollisionState()`

Current lifecycle result:

- `spawn_return=true`
- `scene_changed=false`
- collision remains true

So `spawn_return=true` is not sufficient. A valid scene-change handshake is
still missing in the direct bridge runtime.

### 6. What is official `avoid_manage.py` order?

The Python manager follows the same order:

```text
/AvoidBench/src/avoidbench/avoid_manage/scripts/avoid_manage.py:196-205
```

It calls:

1. `getMissionParam`
2. `setParamFromMission`
3. `spawnObstacles`
4. loop `SpawnNewObs()` until `ifSceneChanged()`
5. `CheckCollision(start/end)`

It also initializes `collision_state=True`, but that does not explain the
direct bridge result because direct mode bypasses `avoid_manage.py`.

### 7. Is the current direct bridge probe consistent with official order?

The original direct bridge probe was sufficient to test the first
`updateUnity()` collision source, but it did not record every initialization
phase. The new mode does:

```bash
python3 -m scripts.probe_avoidbench_collision_ownership \
  --mode bridge-init-sequence \
  --config /AvoidBench/src/avoidbench/avoid_manage/params/task_indoor.yaml \
  --reset-position 0.0 0.0 5.0 \
  --sample-period 0.1 \
  --scene-change-timeout 3.0 \
  --spawn-obstacles
```

This proves the collision becomes true only after Unity output is received.

However, the direct bridge still fails to reach `scene_changed=True`, so it is
not equivalent to a fully valid official mission lifecycle.

### 8. Where does `asphalt_tile` come from?

`asphalt_tile` was not found in AvoidBench C++ or Python source. It was found
inside Unity build data:

```text
/AvoidBench/src/avoidbench/unity_scene/AvoidBench/AvoidBench_Data/level2
```

Runtime Unity log repeatedly reports:

```text
collision.collider.name: asphalt_tile
hitColliders[i].gameObject.name: asphalt_tile
```

Log path:

```text
/tmp/avoidbench_direct_bridge_collision.log
```

This strongly points to Unity scene/collider/layer behavior around the ground
tile, not Gazebo contact and not RL action.

## Current Diagnosis

The direct evidence supports this chain:

```text
Python bridge creates AvoidbenchBridge
  -> C++ bridge collision is initially false
  -> first updateUnity() sends pose to Unity
  -> Unity returns pub_vehicles[0].collision=true
  -> C++ bridge copies that value into unity_bridge_ptr_->collisions[0]
  -> getQuadCollisionState() returns true
```

The primary unresolved issue is Unity-side: the returned collision is with
`asphalt_tile`, including at ROS `(0,0,5)`, which maps to Unity `(0,5,0)` via
`positionRos2Unity(x,y,z) -> (x,z,y)`.

Next work should inspect or replace the Unity project/source, or add a narrow
C++/Python diagnostic workaround that refuses to accept a reset until Unity
returns collision false. Do not ignore this collision and proceed to training.
