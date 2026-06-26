# AvoidBench Unity Scene Assets Report

Date: 2026-06-26

## Question

Can the current container directly fix Unity collision layers, colliders, or
`asphalt_tile` scene objects?

## Findings

The current container has a built Unity player under:

```text
/AvoidBench/src/avoidbench/unity_scene/AvoidBench/
```

Representative files:

```text
/AvoidBench/src/avoidbench/unity_scene/AvoidBench/AvoidBench.x86_64
/AvoidBench/src/avoidbench/unity_scene/AvoidBench/UnityPlayer.so
/AvoidBench/src/avoidbench/unity_scene/AvoidBench/AvoidBench_Data/level0
/AvoidBench/src/avoidbench/unity_scene/AvoidBench/AvoidBench_Data/level1
/AvoidBench/src/avoidbench/unity_scene/AvoidBench/AvoidBench_Data/level2
/AvoidBench/src/avoidbench/unity_scene/AvoidBench/AvoidBench_Data/level3
/AvoidBench/src/avoidbench/unity_scene/AvoidBench/AvoidBench_Data/sharedassets*.assets
/AvoidBench/src/avoidbench/unity_scene/AvoidBench/AvoidBench_Data/Managed/Assembly-CSharp.dll
```

No editable Unity source assets were found by:

```bash
find /AvoidBench/src/avoidbench/unity_scene -type f \
  \( -name "*.cs" -o -name "*.unity" -o -name "*.prefab" -o -name "*.asset" \)
```

No Unity log files were stored under the Unity scene tree. The active runtime
log is the manually selected path:

```text
/tmp/avoidbench_direct_bridge_collision.log
```

## Answers

### 1. Does the current container include a Unity source project?

No. It includes a built Unity player and managed DLLs/assets, not an editable
Unity project.

### 2. Is it only `AvoidBench.x86_64` and data files?

Effectively yes. There is also `Assembly-CSharp.dll`, but no `.cs` source,
`.unity` scene, `.prefab`, or editable project asset files were found.

### 3. Can we directly edit collision layer/collider/asphalt_tile?

Not safely from the current files. `asphalt_tile` exists in Unity build data
(`/AvoidBench/src/avoidbench/unity_scene/AvoidBench/AvoidBench_Data/level2`),
but the scene source is absent.

Editing binary Unity level/assets files in-place would not be a controlled
engineering fix.

### 4. Is upstream Unity source needed?

Yes, if the real fix is to change:

- ground/asphalt collision layer;
- drone collider shape, size, or offset;
- the collision mask used for `pub_vehicles[0].collision`;
- scene-change feedback behavior;
- debug logging around `collision.collider.name`.

Those are Unity-side changes and require the Unity project source or a rebuilt
Unity player from upstream.

### 5. What fixes are currently feasible?

Current feasible work without Unity source:

- Python/C++ diagnostics around bridge lifecycle;
- reset validity guards that refuse to train on `collision=True`;
- reporting Unity-returned collision state with timing and scene-change status;
- narrow bridge-side experiments such as disabling collision checks for
  diagnostic comparison only.

Current infeasible work without Unity source:

- correctly changing `asphalt_tile` layer membership;
- changing the drone collider shape/offset;
- changing Unity's vehicle collision mask;
- fixing `scene_changed` generation inside Unity.

## Current Recommendation

Do not train. Do not suppress collision as a production workaround.

The next productive branch is either:

1. obtain the Unity source project and inspect `asphalt_tile`, vehicle collider,
   and collision mask logic; or
2. add a clearly marked C++/Python diagnostic switch that bypasses vehicle
   collision only to prove the rest of the RL reset/action stack can move,
   while keeping Stage 1 blocked until real Unity collision is fixed.
