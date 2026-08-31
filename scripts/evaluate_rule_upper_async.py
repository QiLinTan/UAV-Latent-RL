from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

from models.async_reference_channel import AsyncChannelProfile
from scripts.bc_scenarios import build_scenarios
from scripts.dual_lower_evaluator import (
    CONTROLLERS,
    PlantRobustnessProfile,
    merge_logs,
    run_dual_lower_episode,
    summarize_results,
)


def async_profiles():
    return [
        AsyncChannelProfile("ideal_15hz", 15.0),
        AsyncChannelProfile("frequency_10hz", 10.0),
        AsyncChannelProfile("frequency_5hz", 5.0),
        AsyncChannelProfile("frequency_2hz", 2.0),
        AsyncChannelProfile("latency_100ms", 10.0, fixed_latency_seconds=0.10),
        AsyncChannelProfile("latency_200ms", 10.0, fixed_latency_seconds=0.20),
        AsyncChannelProfile("random_drop_10pct", 10.0, drop_probability=0.10),
        AsyncChannelProfile("random_drop_30pct", 10.0, drop_probability=0.30),
        AsyncChannelProfile(
            "burst_drop_500ms",
            10.0,
            burst_drop_windows=((4.0, 4.5),),
        ),
        AsyncChannelProfile(
            "burst_drop_1250ms",
            10.0,
            burst_drop_windows=((4.0, 5.25),),
            expected_to_expire=True,
        ),
        AsyncChannelProfile(
            "duplicate_and_reorder",
            10.0,
            duplicate_every_n=4,
            reorder_every_n=5,
            reorder_extra_delay_seconds=0.35,
        ),
        AsyncChannelProfile(
            "mixed_delay_drop_burst",
            10.0,
            fixed_latency_seconds=0.10,
            latency_jitter_seconds=0.04,
            drop_probability=0.20,
            burst_drop_windows=((4.0, 5.25),),
            duplicate_every_n=7,
            reorder_every_n=6,
            reorder_extra_delay_seconds=0.30,
            expected_to_expire=True,
            extreme=True,
        ),
    ]


def selected_scenarios():
    lookup = {scenario.name: scenario for scenario in build_scenarios()}
    return [
        lookup["nominal_low_speed_line"],
        lookup["nominal_gentle_curve"],
        lookup["initial_mixed_rpy_+0.10"],
        lookup["impulse_combined"],
    ]


def async_profile_gate(summary, profile):
    checks = {
        "nominal_success_rate_ge_95pct": (
            summary["nominal_success_rate"] is not None
            and summary["nominal_success_rate"] >= 0.95
        ),
        "disturbance_recovery_success_rate_ge_90pct": (
            summary["disturbance_recovery_success_rate"] is not None
            and summary["disturbance_recovery_success_rate"] >= 0.90
        ),
        "instability_rate_le_2pct": summary["instability_rate"] <= 0.02,
    }
    if profile.expected_to_expire:
        checks.update(
            {
                "packet_expiry_observed": summary["total_expiry_transitions"] > 0,
                "fallback_contracted_toward_hover": (
                    summary["fallback_contraction_violations"] == 0
                ),
                "communication_resumption_observed": (
                    summary["total_resumption_transitions"] > 0
                ),
                "resumption_action_jump_le_0_5": (
                    summary["max_resumption_action_jump"] <= 0.50
                ),
            }
        )
    return {"checks": checks, "passed": bool(all(checks.values()))}


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate rule-upper ReferencePacket execution under asynchronous "
            "frequency, latency, drop, duplicate, reorder, expiry, and resumption."
        )
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
        "--robustness-results",
        default="runs/dual_lower/robustness_v1/robustness_results.json",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/dual_lower/rule_upper_async_v1",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[70, 71])
    parser.add_argument("--duration", type=float, default=12.0)
    args = parser.parse_args()

    output = pathlib.Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "async_results.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite {result_path}.")
    metadata = json.loads(
        pathlib.Path(args.dataset_metadata).read_text(encoding="utf-8")
    )
    freeze = json.loads(
        pathlib.Path(args.freeze_manifest).read_text(encoding="utf-8")
    )
    robustness = json.loads(
        pathlib.Path(args.robustness_results).read_text(encoding="utf-8")
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if checkpoint["dataset_samples_sha256"] != metadata["samples_sha256"]:
        raise ValueError("Actor checkpoint and dataset hashes do not match.")
    if freeze["L1_bc_mlp_t3"]["dataset_samples_sha256"] != metadata["samples_sha256"]:
        raise ValueError("Freeze manifest and dataset hashes do not match.")

    profiles = async_profiles()
    scenarios = selected_scenarios()
    nominal_plant = PlantRobustnessProfile("nominal")
    results = []
    logs = []
    total = len(profiles) * len(scenarios) * len(args.seeds) * len(CONTROLLERS)
    episode_uid = 0
    for profile_index, profile in enumerate(profiles):
        for scenario in scenarios:
            for seed in args.seeds:
                for controller in CONTROLLERS:
                    seeded_profile = AsyncChannelProfile(
                        **{
                            **profile.__dict__,
                            "seed": int(seed) * 101 + profile_index,
                        }
                    )
                    result, step_log = run_dual_lower_episode(
                        controller=controller,
                        scenario=scenario,
                        seed=seed,
                        duration=float(args.duration),
                        checkpoint=checkpoint,
                        device=device,
                        plant_profile=nominal_plant,
                        channel_profile=seeded_profile,
                        episode_uid=episode_uid,
                    )
                    results.append(result)
                    logs.append(step_log)
                    episode_uid += 1
                    if episode_uid % 10 == 0 or episode_uid == total:
                        print(
                            f"[rule-upper-async] completed {episode_uid}/{total}",
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
                if item["channel_profile"]["name"] == profile.name
                and item["controller"] == controller
            ]
            summary = summarize_results(selected)
            summaries[profile.name][controller] = summary
            gates[profile.name][controller] = async_profile_gate(summary, profile)

    non_extreme = [profile for profile in profiles if not profile.extreme]
    expiry_profiles = [
        profile
        for profile in profiles
        if profile.expected_to_expire and not profile.extreme
    ]
    extended_expiry_profiles = [
        profile for profile in profiles if profile.expected_to_expire
    ]
    async_reference_gate = all(
        gates[profile.name][controller]["passed"]
        for profile in non_extreme
        for controller in CONTROLLERS
    )
    expiry_gate = all(
        gates[profile.name][controller]["checks"].get(
            "packet_expiry_observed",
            False,
        )
        and gates[profile.name][controller]["checks"].get(
            "fallback_contracted_toward_hover",
            False,
        )
        and summaries[profile.name][controller]["instability_rate"] <= 0.02
        for profile in expiry_profiles
        for controller in CONTROLLERS
    )
    recovery_gate = all(
        gates[profile.name][controller]["checks"].get(
            "communication_resumption_observed",
            False,
        )
        and gates[profile.name][controller]["checks"].get(
            "resumption_action_jump_le_0_5",
            False,
        )
        and summaries[profile.name][controller]["functional_success_rate"] >= 0.90
        for profile in expiry_profiles
        for controller in CONTROLLERS
    )
    extended_expiry_gate = all(
        gates[profile.name][controller]["checks"].get(
            "packet_expiry_observed",
            False,
        )
        and gates[profile.name][controller]["checks"].get(
            "fallback_contracted_toward_hover",
            False,
        )
        and summaries[profile.name][controller]["instability_rate"] <= 0.02
        for profile in extended_expiry_profiles
        for controller in CONTROLLERS
    )
    extended_recovery_gate = all(
        gates[profile.name][controller]["checks"].get(
            "communication_resumption_observed",
            False,
        )
        and gates[profile.name][controller]["checks"].get(
            "resumption_action_jump_le_0_5",
            False,
        )
        and summaries[profile.name][controller]["functional_success_rate"] >= 0.90
        for profile in extended_expiry_profiles
        for controller in CONTROLLERS
    )
    learning_upper_allowed = bool(
        robustness["dual_lower_robustness_gate_passed"]
        and async_reference_gate
        and expiry_gate
        and recovery_gate
    )
    report = {
        "freeze_version": freeze["freeze_version"],
        "dataset_version": metadata["dataset_version"],
        "motor_action_codec": metadata["motor_action_codec"],
        "episode_count": len(results),
        "seeds": args.seeds,
        "scenario_names": [scenario.name for scenario in scenarios],
        "profile_names": [profile.name for profile in profiles],
        "summaries": summaries,
        "gates": gates,
        "dual_lower_robustness_gate_passed": robustness[
            "dual_lower_robustness_gate_passed"
        ],
        "async_reference_gate_passed": bool(async_reference_gate),
        "packet_expiry_fallback_gate_passed": bool(expiry_gate),
        "communication_recovery_gate_passed": bool(recovery_gate),
        "extended_extreme_expiry_fallback_gate_passed": bool(
            extended_expiry_gate
        ),
        "extended_extreme_communication_recovery_gate_passed": bool(
            extended_recovery_gate
        ),
        "learning_upper_integration_allowed": learning_upper_allowed,
        "results": results,
    }
    result_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(output / "async_step_logs.npz", **merge_logs(logs))
    print(
        json.dumps(
            {
                "async_reference_gate_passed": bool(async_reference_gate),
                "packet_expiry_fallback_gate_passed": bool(expiry_gate),
                "communication_recovery_gate_passed": bool(recovery_gate),
                "extended_extreme_expiry_fallback_gate_passed": bool(
                    extended_expiry_gate
                ),
                "extended_extreme_communication_recovery_gate_passed": bool(
                    extended_recovery_gate
                ),
                "learning_upper_integration_allowed": learning_upper_allowed,
                "episode_count": len(results),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
