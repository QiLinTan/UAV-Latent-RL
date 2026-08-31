from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

from scripts.bc_scenarios import build_scenarios
from scripts.dual_lower_evaluator import (
    CONTROLLERS,
    PlantRobustnessProfile,
    ideal_channel_profile,
    run_dual_lower_episode,
    summarize_results,
)


def boundary_profiles():
    profiles = []
    for scale in (1.005, 1.01, 1.015, 1.02, 1.05, 1.08, 1.10):
        percent_label = f"{(scale - 1.0) * 100:.1f}".rstrip("0").rstrip(".")
        profiles.append(
            (
                "mass_scale",
                scale,
                PlantRobustnessProfile(
                    f"mass_plus_{percent_label.replace('.', 'p')}pct",
                    mass_scale=scale,
                ),
            )
        )
    for effectiveness in (0.99, 0.98, 0.97, 0.96, 0.95, 0.92, 0.90):
        profiles.append(
            (
                "motor_0_thrust_effectiveness",
                effectiveness,
                PlantRobustnessProfile(
                    f"motor_0_effectiveness_{int(round(effectiveness * 100))}pct",
                    motor_thrust_effectiveness=(effectiveness, 1.0, 1.0, 1.0),
                ),
            )
        )
    return profiles


def selected_scenarios():
    lookup = {scenario.name: scenario for scenario in build_scenarios()}
    return [
        lookup["nominal_hover"],
        lookup["initial_mixed_rpy_+0.10"],
        lookup["impulse_combined"],
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Resolve mass and single-motor-effectiveness capability boundaries."
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
        "--output-dir",
        default="runs/dual_lower/capability_boundary_v2",
    )
    parser.add_argument("--seed", type=int, default=80)
    parser.add_argument("--duration", type=float, default=12.0)
    args = parser.parse_args()
    output = pathlib.Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "capability_boundary_results.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite {result_path}.")
    metadata = json.loads(
        pathlib.Path(args.dataset_metadata).read_text(encoding="utf-8")
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if checkpoint["dataset_samples_sha256"] != metadata["samples_sha256"]:
        raise ValueError("Actor checkpoint and dataset hashes do not match.")

    profiles = boundary_profiles()
    scenarios = selected_scenarios()
    results = []
    total = len(profiles) * len(scenarios) * len(CONTROLLERS)
    episode_uid = 0
    for family, value, profile in profiles:
        for scenario in scenarios:
            for controller in CONTROLLERS:
                result, _ = run_dual_lower_episode(
                    controller=controller,
                    scenario=scenario,
                    seed=int(args.seed),
                    duration=float(args.duration),
                    checkpoint=checkpoint,
                    device=device,
                    plant_profile=profile,
                    channel_profile=ideal_channel_profile(seed=args.seed),
                    episode_uid=episode_uid,
                )
                result["boundary_family"] = family
                result["boundary_value"] = float(value)
                results.append(result)
                episode_uid += 1
                if episode_uid % 8 == 0 or episode_uid == total:
                    print(
                        f"[lower-boundary] completed {episode_uid}/{total}",
                        file=sys.stderr,
                        flush=True,
                    )

    summaries = {}
    all_pass = {}
    for family, value, profile in profiles:
        summaries[profile.name] = {}
        all_pass[profile.name] = {}
        for controller in CONTROLLERS:
            selected = [
                item
                for item in results
                if item["plant_profile"]["name"] == profile.name
                and item["controller"] == controller
            ]
            summaries[profile.name][controller] = summarize_results(selected)
            all_pass[profile.name][controller] = bool(
                all(item["functional_success"] for item in selected)
            )

    boundaries = {}
    for controller in CONTROLLERS:
        mass_passes = [
            value
            for family, value, profile in profiles
            if family == "mass_scale" and all_pass[profile.name][controller]
        ]
        motor_passes = [
            value
            for family, value, profile in profiles
            if family == "motor_0_thrust_effectiveness"
            and all_pass[profile.name][controller]
        ]
        boundaries[controller] = {
            "maximum_tested_passing_mass_scale": (
                None if not mass_passes else max(mass_passes)
            ),
            "minimum_tested_passing_motor_0_thrust_effectiveness": (
                None if not motor_passes else min(motor_passes)
            ),
        }

    report = {
        "dataset_version": metadata["dataset_version"],
        "motor_action_codec": metadata["motor_action_codec"],
        "episode_count": len(results),
        "seed": int(args.seed),
        "scenario_names": [scenario.name for scenario in scenarios],
        "summaries": summaries,
        "all_scenarios_passed": all_pass,
        "resolved_tested_boundaries": boundaries,
        "results": results,
    }
    result_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "episode_count": len(results),
                "resolved_tested_boundaries": boundaries,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
