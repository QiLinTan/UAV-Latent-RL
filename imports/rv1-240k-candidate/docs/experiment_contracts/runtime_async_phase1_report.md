# Runtime Async Phase 1 Report

Date: 2026-09-07

## Scope

This phase upgrades the existing logical-asynchronous reference-control
prototype into a runtime-asynchronous prototype.  It does not modify the
Reference Planner, train networks, run DAgger, change sensor inputs, or address
24-tree navigation performance.

## Scheme Corrections

1. The runtime worker reproduces the validated logical prototype's upper stack:
   frozen Encoder + legacy Actor adapter + ReferenceGenerator.  It does not
   require a learned Reference Planner checkpoint.  This keeps the phase focused
   on runtime concurrency rather than planner quality.

2. The low-to-high observation channel uses a shared-memory latest slot instead
   of `multiprocessing.Queue(maxsize=1)`.  Initial experiments showed queue
   publish latency spikes when the worker slept.  A high-rate low-level loop
   should not pay Queue pickle/backpressure cost for stale observations.

3. The high-to-low packet channel remains a latest-only
   `multiprocessing.Queue(maxsize=1)` because packets are low-rate, small, and
   must be transferred as immutable `ReferencePacket` objects.

4. Status/timing telemetry is an unbounded non-control queue.  Control channels
   are latest-only, but timing samples must not be dropped or the report would
   undercount high-level latency.

5. Packet timestamps use wall-clock time for runtime validity/age.  Simulation
   time is recorded separately.  The environment loop is not wall-clock paced.

## Files Added or Modified

Added:

- `async_control/latest_channel.py`
- `async_control/runtime_timing.py`
- `async_control/runtime_high_level_worker.py`
- `scripts/evaluate_runtime_async_pid.py`
- `tests/test_runtime_async_control.py`
- `docs/experiment_contracts/runtime_async_phase1_report.md`

Modified:

- `async_control/__init__.py`

## Runtime Architecture

Low-level main process:

1. Owns `CustomForestAviary`, DSLPID, motor codec, `AsyncBuffer`, and
   `LowLevelReferenceSelector`.
2. Publishes the latest observation snapshot to shared memory.
3. Non-blockingly reads the latest packet message.
4. Commits strictly newer packets to `AsyncBuffer`.
5. Selects a reference or hover-hold fallback.
6. Computes DSLPID and calls `env.step()`.

High-level spawned process:

1. Starts via `multiprocessing.get_context("spawn")`.
2. Loads frozen Encoder and Actor internally.
3. Reads the newest observation snapshot from shared memory at about 15 Hz.
4. Runs preprocessing, Encoder inference, Actor inference/adapter, and
   ReferenceGenerator.
5. Publishes the newest `ReferencePacket`.

The low-level process does not call Encoder, Actor, Planner, or
`decision_factory`.

## IPC

- Low -> High: shared-memory latest observation slot with monotonic version.
- High -> Low: latest-only `multiprocessing.Queue(maxsize=1)` packet channel.
- Worker -> Low telemetry: unbounded status queue, not used by control.

## Experiment

Command:

```bash
/home/tequila/miniconda3/envs/drones/bin/python scripts/evaluate_runtime_async_pid.py --episodes 1 --episode-len-sec 4.0 --output analysis_outputs/runtime_async_pid/runtime_async_results_v2.json
```

Environment:

- Trees: 0
- Route-blocking tree: false
- Low-level nominal rate: 120 Hz
- High-level nominal rate: 15 Hz
- Checkpoint: `checkpoints/latent_only_affordance_dim16/model_final`
- Artificial delay location: inside the spawned high-level process

## Profile A-D Timing Summary

| Profile | Worker delay | Low cycles | Packets | High decisions | Low p99 | Low max | >8.33 ms | Env step p99 | Max low cycles during one high compute |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_runtime_no_extra_delay | 0 ms | 482 | 7 | 7 | 1.138 ms | 1.248 ms | 0 | 0.693 ms | 2 |
| B_runtime_worker_sleep_20ms | 20 ms | 482 | 7 | 7 | 1.362 ms | 1.602 ms | 0 | 0.894 ms | 27 |
| C_runtime_worker_sleep_50ms | 50 ms | 482 | 6 | 7 | 1.241 ms | 1.495 ms | 0 | 0.786 ms | 63 |
| D_runtime_worker_sleep_100ms | 100 ms | 482 | 4 | 5 | 1.240 ms | 1.507 ms | 0 | 0.721 ms | 120 |

High-level latency:

| Profile | High total mean | High total max | Encoder mean | Actor mean | Packet gen mean | Obs age at start mean |
|---|---:|---:|---:|---:|---:|---:|
| A_runtime_no_extra_delay | 0.831 ms | 1.323 ms | 0.213 ms | 0.132 ms | 0.245 ms | 0.470 ms |
| B_runtime_worker_sleep_20ms | 20.999 ms | 21.838 ms | 0.217 ms | 0.121 ms | 0.347 ms | 0.578 ms |
| C_runtime_worker_sleep_50ms | 51.198 ms | 51.803 ms | 0.278 ms | 0.155 ms | 0.403 ms | 0.542 ms |
| D_runtime_worker_sleep_100ms | 101.214 ms | 101.654 ms | 0.382 ms | 0.128 ms | 0.345 ms | 0.467 ms |

Packet/fallback:

| Profile | Fallback cycles | Missing cycles | Stale cycles | Packet interval mean |
|---|---:|---:|---:|---:|
| A_runtime_no_extra_delay | 3 | 3 | 0 | 66.564 ms |
| B_runtime_worker_sleep_20ms | 28 | 28 | 0 | 66.539 ms |
| C_runtime_worker_sleep_50ms | 64 | 64 | 0 | 66.562 ms |
| D_runtime_worker_sleep_100ms | 120 | 120 | 0 | 100.979 ms |

Navigation metrics were recorded but are not a success criterion here:

| Profile | Success | Collision | Final distance |
|---|---:|---:|---:|
| A_runtime_no_extra_delay | 0.0 | 0.0 | 6.8042 |
| B_runtime_worker_sleep_20ms | 0.0 | 0.0 | 6.8610 |
| C_runtime_worker_sleep_50ms | 0.0 | 0.0 | 6.9488 |
| D_runtime_worker_sleep_100ms | 0.0 | 0.0 | 6.9063 |

## Concurrency Result

The 100 ms artificial high-level delay did not stop the low-level loop.  In
Profile D, one high-level compute window overlapped with up to 120 low-level
cycles, while low-level cycle p99 stayed at 1.240 ms and no cycle exceeded the
8.33 ms nominal 120 Hz budget.

This directly addresses the phase question: high-level inference and artificial
delay now occur in a separate Python execution context and do not synchronously
block DSLPID/env stepping.

## Realtime Interpretation

The simulation was not paced to wall-clock 120 Hz.  The result means the current
headless environment/PID/control compute path is below the 8.33 ms per-cycle
compute budget in this run.  It is not a real-time OS scheduling certificate.

`env.step()` itself had p99 below 1 ms in all profiles, so the measured run does
not show PyBullet step cost preventing 120 Hz compute.  GUI mode, heavier maps,
or OS contention still need separate checks.

## Remaining Runtime Risks

- The worker currently runs the legacy Actor adapter, not a learned Reference
  Planner.
- Observation latest-only semantics intentionally drop intermediate snapshots.
- The first runtime packet may arrive after several fallback cycles.
- Packet continuity is anchored to the snapshot fallback state sent to the
  worker, not to a low-level local packet regeneration step.
- No process priority, CPU affinity, watchdog restart, or hard real-time
  scheduling is implemented.
- The stale/fallback pressure case was not exhausted because the default
  reference horizon and profile durations produced mostly missing startup
  fallback rather than long stale operation.

## Tests

Commands:

```bash
/home/tequila/miniconda3/envs/drones/bin/python -m unittest tests.test_runtime_async_control tests.test_async_control -v
```

Result:

- Runtime async tests: 7 passed
- Existing logical async tests: 4 passed

Additional targeted tests verify worker lifecycle, worker exception handling,
latest-only observation/packet channels, packet version monotonicity, 100 ms
delay non-blocking behavior, clean shutdown, and preservation of logical async
tests.

Full regression:

```bash
/home/tequila/miniconda3/envs/drones/bin/python -m unittest discover -s tests -v
```

Result:

- 67 tests passed

## Next Recommendation

Run a separate Phase 1.5 stale/fallback pressure test after this report is
reviewed.  Do not start Temporal Planner, DAgger, sensor redesign, or 24-tree
planner work from this phase.
