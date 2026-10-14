"""Held-out Stage 0--3 / Full evaluations for the frozen 42--45 curricula.

Resumable: an existing evaluation_metrics.jsonl is parsed, not overwritten.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.run_change_td3 import build_mode_cmd


BASE = "upper_dslpid_recurrent_multitree_terminal_v2_noroute_h8_k1"
MAP = "procedural_forest_multitree_validation_random_v1"
ROOT = Path("analysis_outputs/startobscurr_multiseed_holdout_20261010")
NEW_ROOT = Path("checkpoints/startobscurr_v1_selected_20261010")
OLD_ROOT = Path("checkpoints/startcurr_v1_frozen_20261009")


def replace(command, option, value):
    command[command.index(option) + 1] = str(value)


def evaluate(seed, variant, stage, model):
    label = f"seed{seed}_{variant}_{model}_stage{stage}"
    root = ROOT / label
    output = root / "evaluation_metrics.jsonl"
    checkpoint_root = NEW_ROOT if variant == "new" else OLD_ROOT
    checkpoint = checkpoint_root / f"seed{seed}" / "model_final"
    manifest = json.loads((checkpoint_root / "manifest.json").read_text())
    actor = Path(str(checkpoint) + "_actor")
    digest = hashlib.sha256(actor.read_bytes()).hexdigest()
    assert digest == manifest["seeds"][str(seed)]["files"]["model_final_actor"]["sha256"]
    if not output.exists():
        root.mkdir(parents=True, exist_ok=True)
        protocol = {
            "training_seed": seed,
            "variant": variant,
            "checkpoint": str(checkpoint),
            "actor_sha256": digest,
            "model": model,
            "stage": stage,
            "map_id": MAP,
            "environment_seeds": list(range(20, 40)),
            "expected_route_blocking_tree_count": (1, 2, 3, 5)[stage],
            "original_start": [-3.5, 0.0, 1.0],
            "reward_and_network_preset": BASE,
        }
        (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
        command = build_mode_cmd(BASE)
        for option, value in (
            ("--seed", seed),
            ("--eval_map_id", MAP),
            ("--eval_episodes", 20),
            ("--eval_interval", 1),
            ("--log_dir", root),
            ("--trajectory_output_dir", root / "trajectory"),
            ("--collision_diagnostic_output_dir", root / "collision"),
            ("--timeout_diagnostic_output_dir", root / "timeout"),
            ("--eval_terminal_coverage_output_dir", root / "coverage"),
            ("--upper_execution_model", model),
        ):
            replace(command, option, value)
        command += [
            "--evaluate_only", "true",
            "--curriculum_stage_override", str(stage),
            "--eval_seed_offset", "20",
            "--load_checkpoint", str(checkpoint),
        ]
        with (root / "evaluation.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    protocol = json.loads((root / "protocol.json").read_text())
    if protocol["actor_sha256"] != digest or protocol["model"] != model or protocol["stage"] != stage:
        raise AssertionError(f"Protocol differs from requested evaluation: {root}")
    records = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
    if len(records) != 1 or len(records[0]["episodes"]) != 20:
        raise AssertionError(f"Unexpected evaluation records in {output}")
    record = records[0]
    episodes = record["episodes"]
    if [row["environment_seed"] for row in episodes] != list(range(20, 40)):
        raise AssertionError(f"Incorrect environment seeds: {root}")
    if int(record["summary"]["eval/curriculum_stage"]) != stage:
        raise AssertionError(f"Incorrect curriculum stage: {root}")
    collisions = json.loads((root / "collision" / "collision_windows_latest.json").read_text())
    timeouts = json.loads((root / "timeout" / "timeout_diagnostics_latest.json").read_text())
    result = {
        "label": label,
        "seed": seed,
        "variant": variant,
        "stage": stage,
        "model": model,
        "checkpoint_actor_sha256": digest,
        "success_count": sum(row["success"] for row in episodes),
        "collision_count": sum(row["collision"] for row in episodes),
        "timeout_count": sum(row["done_reason"] == "timeout" for row in episodes),
        "guidance_entry_fraction": record["summary"].get("eval/terminal_coverage/guidance_entered_episode_fraction"),
        "mean_final_goal_distance": record["summary"]["eval/final_goal_distance"],
        "failure_seeds": [row["environment_seed"] for row in episodes if not row["success"]],
        "collision_classifications": dict(Counter(row["classification"]["code"] for row in collisions["collisions"])),
        "timeout_classifications": dict(Counter(row["classification"]["code"] for row in timeouts["timeouts"])),
        "directory": str(root),
    }
    if result["success_count"] + result["collision_count"] + result["timeout_count"] != 20:
        raise AssertionError(f"Unexpected termination totals: {root}")
    return result


def tasks_for(phase):
    seeds = (42, 44, 45)
    if phase == "primary":
        return [(seed, "new", 3, "k1") for seed in seeds]
    if phase == "secondary":
        return [(seed, "new", 3, "full") for seed in seeds] + [
            (seed, "new", 0, "full") for seed in seeds
        ] + [(seed, "new", stage, "k1") for seed in seeds for stage in (0, 1, 2)]
    if phase == "baseline":
        return [(seed, "old", 3, model) for seed in seeds for model in ("k1", "full")]
    raise ValueError(phase)


def write_summary():
    rows = []
    for protocol_path in sorted(ROOT.glob("*/protocol.json")):
        protocol = json.loads(protocol_path.read_text())
        if (protocol_path.parent / "evaluation_metrics.jsonl").exists():
            rows.append(evaluate(protocol["training_seed"], protocol["variant"], protocol["stage"], protocol["model"]))
    (ROOT / "summary.json").write_text(json.dumps({"schema_version": 1, "rows": rows}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("primary", "secondary", "baseline"), required=True)
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    for seed, variant, stage, model in tasks_for(args.phase):
        print(f"Evaluating seed {seed}: {variant} {model} Stage {stage}", flush=True)
        row = evaluate(seed, variant, stage, model)
        print(f"{row['success_count']}/20 success, {row['collision_count']} collision, {row['timeout_count']} timeout", flush=True)
        write_summary()


if __name__ == "__main__":
    main()
