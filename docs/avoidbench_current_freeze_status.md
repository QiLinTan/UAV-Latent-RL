# AvoidBench Current Freeze Status

Date: 2026-06-22

## Workspace

- repository: `/home/tequial/projects/UAV-AvoidBench-RL`
- container path: `/workspace/UAV-AvoidBench-RL`
- branch: `avoidbench-rl-env-smoke`
- managed container: `noetic_ab_workspace`

## Purpose

Freeze the current AvoidBench engineering baseline before further navigation
diagnosis and Stage 1 work.

## Key files confirmed

- `envs/avoidbench/rl_env.py`
- `scripts/stress_avoidbench_rl_env.py`
- `scripts/train_avoidbench_plain_td3_smoke.py`
- `docs/avoidbench_plain_td3_baseline_report.md`
- `docs/avoidbench_next_status.md`
- `scripts/launch_avoidbench_ros_mesa.sh`
- `docs/avoidbench_workspace_container.md`

## Pre-freeze status summary

The working tree contains the AvoidBench ROS/Unity adapter, reset hardening,
stress probes, Plain TD3 smoke tooling, baseline reports, and the Mesa launcher
for `noetic_ab_workspace`.

Tracked files modified before freeze included:

- `.gitignore`
- `algos/td3/td3_plain.py`
- `scripts/run_change_td3.py`
- `skills/uav-rl-research/IMPROVEMENT_LOG.md`

AvoidBench-related untracked directories included:

- `docs/`
- `envs/avoidbench/`
- `patches/`
- `scripts/`
- `tests/`
- `tools/`
- `trainers/`

Experiment outputs remain outside git through `.gitignore`, especially:

- `runs/avoidbench_plain_td3_smoke/`
- `runs/avoidbench_env_stress/`
- checkpoints and TensorBoard event files

## Checks

- `git diff --check`: passed
- `bash -n scripts/launch_avoidbench_ros_mesa.sh`: passed
- `python3 -m py_compile` for AvoidBench environment, probes, stress,
  analysis, smoke trainer, and vector trainer files: passed
- `pytest -q tests`: unavailable on host (`pytest: command not found`)

## Freeze commit

Freeze commit:

```text
a2601b31ac5a11c4ae650086bb08cc2ffe8f2f54
```

Commit message:

```text
stabilize AvoidBench RL env and plain TD3 smoke baseline
```

Submitted files:

- `.gitignore`
- `algos/td3/td3_plain.py`
- `data/__init__.py`
- `data/replay_buffer.py`
- `docs/avoidbench_action_interface_notes.md`
- `docs/avoidbench_current_freeze_status.md`
- `docs/avoidbench_current_status.md`
- `docs/avoidbench_hover_smoke_analysis.md`
- `docs/avoidbench_integration.md`
- `docs/avoidbench_navigation_smoke_analysis.md`
- `docs/avoidbench_next_status.md`
- `docs/avoidbench_plain_td3_baseline_report.md`
- `docs/avoidbench_plain_td3_smoke_analysis.md`
- `docs/avoidbench_rl_adapter_design.md`
- `docs/avoidbench_ros_runtime_result.md`
- `docs/avoidbench_ros_startup_plan.md`
- `docs/avoidbench_workspace_container.md`
- `docs/reference_projects_for_avoidbench_rl.md`
- `envs/avoidbench/__init__.py`
- `envs/avoidbench/adapter.py`
- `envs/avoidbench/backend.py`
- `envs/avoidbench/configs/task_indoor.yaml`
- `envs/avoidbench/observation.py`
- `envs/avoidbench/rl_env.py`
- `patches/avoidbench_unity_depth_pybind.patch`
- `scripts/analyze_avoidbench_smoke_run.py`
- `scripts/debug_avoidbench_reset.py`
- `scripts/launch_avoidbench_ros_mesa.sh`
- `scripts/probe_avoidbench.py`
- `scripts/probe_avoidbench_action.py`
- `scripts/probe_avoidbench_rl_env.py`
- `scripts/probe_avoidbench_ros.py`
- `scripts/probe_avoidbench_state.py`
- `scripts/stress_avoidbench_rl_env.py`
- `scripts/train_avoidbench_plain_td3_smoke.py`
- `scripts/train_avoidbench_td3.py`
- `skills/uav-rl-research/IMPROVEMENT_LOG.md`
- `tests/test_avoidbench_adapter.py`
- `tests/test_replay_buffer_batch.py`
- `tests/test_vector_td3_trainer.py`
- `tools/avoidbench_container.sh`
- `tools/setup_avoidbench_env.sh`
- `trainers/vector_td3_trainer.py`

Remaining uncommitted files after the freeze commit:

- modified `scripts/run_change_td3.py`
- untracked `models/`

## Evidence runs not committed

- `runs/avoidbench_env_stress/20260610-085245/`
- `runs/avoidbench_env_stress/20260610-085341/`
- `runs/avoidbench_env_stress/20260610-085713/`
- `runs/avoidbench_plain_td3_smoke/20260609-051809/`
- `runs/avoidbench_plain_td3_smoke/20260609-133209-hover_smoke/`
- `runs/avoidbench_plain_td3_smoke/20260609-141223-navigation_smoke/`
