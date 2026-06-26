# AvoidBench Bridge Lifecycle Report

Date: 2026-06-26

## Purpose

Determine whether direct bridge `collision=True` is:

- a default/local true value;
- set by the first Unity `updateUnity()` output;
- sticky state in a process-local bridge singleton;
- affected by spawn/scene-change.

No training was run.

## Probe Scripts

Primary staged lifecycle mode:

```text
scripts/probe_avoidbench_collision_ownership.py --mode bridge-init-sequence
```

Standalone lifecycle script:

```text
scripts/probe_avoidbench_bridge_lifecycle.py
```

The standalone script is intentionally guarded: clean lifecycle cases require
one Python process per case because native `UnityBridge::getInstance()` is a
process-local singleton.

Source evidence:

```text
/AvoidBench/src/avoidbench/avoidlib/include/avoidlib/bridges/unity_bridge.hpp:69-73
```

## Clean Lifecycle Result

Run:

```text
runs/avoidbench_collision_ownership/20260626-121259/
```

Command:

```bash
python3 -m scripts.probe_avoidbench_collision_ownership \
  --mode bridge-init-sequence \
  --config /AvoidBench/src/avoidbench/avoid_manage/params/task_indoor.yaml \
  --reset-position 0.0 0.0 5.0 \
  --sample-period 0.1 \
  --scene-change-timeout 3.0 \
  --spawn-obstacles
```

Result:

- `before_update_collision=false`
- `after_first_update_collision=true`
- `after_10_updates_collision=true`
- `spawn_return=true`
- `scene_changed=false`
- `after_scene_spawn_collision=true`
- `after_scene_changed_collision=true`

Interpretation:

`collision=True` is not a Python/C++ default before any Unity output. It appears
after the first successful `updateUnity()` round trip. That points to Unity
returning `pub_vehicles[0].collision=true`.

## Singleton Contamination Result

Run:

```text
runs/avoidbench_bridge_lifecycle/20260626-121906/
```

This run intentionally executed two lifecycle cases in one Python process before
the script was guarded.

Observed:

- first case `no-spawn`: before update false, after first update true;
- second case `spawn`: before update already true;
- second case then hit `RuntimeError: attempting to request a message part outside the valid range`.

Interpretation:

The second case was contaminated by the process-local `UnityBridge::getInstance`
state. Multi-case lifecycle data from one Python process should not be treated
as clean evidence.

## Dirty Unity Connection Result

Run:

```text
runs/avoidbench_bridge_lifecycle/20260626-122025/
```

This later single-case run did not reconnect cleanly to the already-used
standalone Unity instance:

- Unity connection timed out;
- `update_success_count=0`;
- collision stayed false because no Unity output was received.

Interpretation:

This does not disprove the previous Unity collision result. It shows the current
standalone Unity/ZMQ state was no longer suitable for fresh clean-client tests
after repeated bridge connections. Future lifecycle tests should start from one
fresh Unity process and one Python bridge process.

## Current Conclusion

The best clean evidence remains:

```text
before update: false
after first successful updateUnity(): true
Unity log collider: asphalt_tile
```

Therefore the collision source is Unity output, not Gazebo contact, not TD3,
and not a local Python default. Stage 1 remains blocked.

## Next Clean Test Protocol

1. Stop all old Python bridge probes.
2. Start exactly one fresh Unity process.
3. Run exactly one lifecycle case per Python process.
4. Prefer high-altitude discriminator first:

```bash
python3 -m scripts.probe_avoidbench_collision_ownership \
  --mode bridge-init-sequence \
  --config /AvoidBench/src/avoidbench/avoid_manage/params/task_indoor.yaml \
  --reset-position 0.0 0.0 5.0 \
  --sample-period 0.1 \
  --scene-change-timeout 3.0 \
  --spawn-obstacles
```

Passing condition for reset sanity remains strict: collision must stay false
after a successful Unity update and static observation.
