from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

from scripts.bc_scenarios import build_scenarios
from scripts.dual_lower_evaluator import (
    CONTROLLERS,
    PlantRobustnessProfile,
    ideal_channel_profile,
    merge_logs,
    run_dual_lower_episode,
    summarize_results,
)


def robustness_profiles():
    return [
        PlantRobustnessProfile("nominal"),
        PlantRobustnessProfile(
            "thrust_coefficient_minus_10pct",
            thrust_coefficient_scale=0.90,
        ),
        PlantRobustnessProfile("mass_plus_10pct", mass_scale=1.10),
        PlantRobustnessProfile("observation_delay_2_steps", observation_delay_steps=2),
        PlantRobustnessProfile(
            "moderate_sensor_noise",
            position_noise_std=0.005,
            attitude_noise_std=0.005,
            velocity_noise_std=0.02,
            angular_velocity_noise_std=0.03,
        ),
        PlantRobustnessProfile("action_delay_2_steps", action_delay_steps=2),
        PlantRobustnessProfile(
            "motor_0_thrust_effectiveness_90pct",
            motor_thrust_effectiveness=(0.90, 1.0, 1.0, 1.0),
        ),
        PlantRobustnessProfile(
            "combined_mild_uncertainty",
            thrust_coefficient_scale=0.95,
            mass_scale=1.05,
            observation_delay_steps=1,
            action_delay_steps=1,
            position_noise_std=0.0025,
            attitude_noise_std=0.0025,
            velocity_noise_std=0.01,
            angular_velocity_noise_std=0.015,
            motor_thrust_effectiveness=(0.95, 1.0, 1.0, 1.0),
        ),
    ]


def selected_scenarios():
    lookup = {scenario.name: scenario for scenario in build_scenarios()}
    return [
        lookup["nominal_hover"],
        lookup["nominal_low_speed_line"],
        lookup["nominal_gentle_curve"],
        lookup["initial_mixed_rpy_+0.10"],
        lookup["initial_velocity_x_pos"],
        lookup["impulse_combined"],
    ]


def profile_gate(summary):
    nominal = summary["nominal_success_rate"]
    recovery = summary["disturbance_recovery_success_rate"]
    checks = {
        "nominal_success_rate_ge_95pct": nominal is not None and nominal >= 0.95,
        "disturbance_recovery_success_rate_ge_90pct": (
            recovery is not None and recovery >= 0.90
        ),
        "instability_rate_le_2pct": summary["instability_rate"] <= 0.02,
        "fallback_not_used_in_ideal_channel": summary["total_fallback_steps"] == 0,
    }
    return {"checks": checks, "passed": bool(all(checks.values()))}


def main():
    parser = argparse.ArgumentParser(
        description="Compare frozen DSLPID+T3 and BC-MLP+T3 robustness envelopes."
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/behavior_cloning/"
            "asymmetric_rpm_v2_plain_mlp_b4096/actor_best.pt"
        ),
    )
    parser.add_argument(
        "--dataset-metadata",
        default="data/behavior_cloning/asymmetric_rpm_v2/metadata.json",
    )
    parser.add_argument(
        "--freeze-manifest",
        default="configs/lower_baseline_freeze_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/dual_lower/robustness_v1",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[60, 61])
    parser.add_argument("--duration", type=float, default=12.0)
    args = parser.parse_args()

    output = pathlib.Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "robustness_results.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite {result_path}.")
    metadata = json.loads(
        pathlib.Path(args.dataset_metadata).read_text(encoding="utf-8")
    )
    freeze = json.loads(
        pathlib.Path(args.freeze_manifest).read_text(encoding="utf-8")
    )
    checkpoint_path = pathlib.Path(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint["dataset_samples_sha256"] != metadata["samples_sha256"]:
        raise ValueError("Actor checkpoint and dataset hashes do not match.")
    if freeze["L1_bc_mlp_t3"]["dataset_samples_sha256"] != metadata["samples_sha256"]:
        raise ValueError("Freeze manifest and dataset hashes do not match.")

    results = []
    logs = []
    profiles = robustness_profiles()
    scenarios = selected_scenarios()
    total = len(CONTROLLERS) * len(profiles) * len(scenarios) * len(args.seeds)
    episode_uid = 0
    for profile in profiles:
        for scenario in scenarios:
            for seed in args.seeds:
                for controller in CONTROLLERS:
                    result, step_log = run_dual_lower_episode(
                        controller=controller,
                        scenario=scenario,
                        seed=seed,
                        duration=float(args.duration),
                        checkpoint=checkpoint,
                        device=device,
                        plant_profile=profile,
                        channel_profile=ideal_channel_profile(seed=seed),
                        episode_uid=episode_uid,
                    )
                    results.append(result)
                    logs.append(step_log)
                    episode_uid += 1
                    if episode_uid % 10 == 0 or episode_uid == total:
                        print(
                            f"[dual-robustness] completed {episode_uid}/{total}",
                            file=sys.stderr,
                            flush=True,
                        )

    summaries = {}
    gates = {}
    for profile in profiles:
        summaries[profile.name] = {}
        gates[profile.name] = {}
        for controller in CONTROLLERS:
            selected = [
                item
                for item in results
                if item["plant_profile"]["name"] == profile.name
                and item["controller"] == controller
            ]
            summary = summarize_results(selected)
            summaries[profile.name][controller] = summary
            gates[profile.name][controller] = profile_gate(summary)

    required_profiles = [
        profile.name for profile in profiles if profile.engineering_gate_required
    ]
    overall_passed = all(
        gates[profile][controller]["passed"]
        for profile in required_profiles
        for controller in CONTROLLERS
    )
    both_failed = []
    teacher_only_failed = []
    actor_only_failed = []
    for profile in required_profiles:
        teacher_pass = gates[profile]["L0_teacher_t3"]["passed"]
        actor_pass = gates[profile]["L1_bc_mlp_t3"]["passed"]
        if not teacher_pass and not actor_pass:
            both_failed.append(profile)
        elif not teacher_pass:
            teacher_only_failed.append(profile)
        elif not actor_pass:
            actor_only_failed.append(profile)

    report = {
        "freeze_version": freeze["freeze_version"],
        "checkpoint": args.checkpoint,
        "dataset_version": metadata["dataset_version"],
        "motor_action_codec": metadata["motor_action_codec"],
        "episode_count": len(results),
        "seeds": args.seeds,
        "scenario_names": [scenario.name for scenario in scenarios],
        "profile_names": [profile.name for profile in profiles],
        "summaries": summaries,
        "gates": gates,
        "failure_attribution": {
            "both_failed_upper_or_shared_interface_suspects": both_failed,
            "teacher_only_failed": teacher_only_failed,
            "actor_only_failed_lower_actor_suspects": actor_only_failed,
        },
        "dual_lower_robustness_gate_passed": bool(overall_passed),
        "results": results,
    }
    result_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(output / "robustness_step_logs.npz", **merge_logs(logs))
    print(
        json.dumps(
            {
                "dual_lower_robustness_gate_passed": bool(overall_passed),
                "episode_count": len(results),
                "failure_attribution": report["failure_attribution"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
