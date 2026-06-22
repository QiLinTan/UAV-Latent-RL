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

Pending local commit.

## Evidence runs not committed

- `runs/avoidbench_env_stress/20260610-085245/`
- `runs/avoidbench_env_stress/20260610-085341/`
- `runs/avoidbench_env_stress/20260610-085713/`
- `runs/avoidbench_plain_td3_smoke/20260609-051809/`
- `runs/avoidbench_plain_td3_smoke/20260609-133209-hover_smoke/`
- `runs/avoidbench_plain_td3_smoke/20260609-141223-navigation_smoke/`
