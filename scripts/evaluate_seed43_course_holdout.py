"""Evaluate selected seed-43 K1 policies on untouched map seeds 20--39."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from scripts.run_change_td3 import build_mode_cmd


BASE = "upper_dslpid_recurrent_multitree_terminal_v2_noroute_h8_k1"
MAP = "procedural_forest_multitree_validation_random_v1"
ROOT = Path("analysis_outputs/seed43_course_holdout_20261009")
CHECKPOINTS = {
    "frozen": Path("checkpoints/startcurr_v1_frozen_20261009/seed43/model_final"),
    "stage01_ft": Path("checkpoints/upper_dslpid_recurrent_multitree_terminal_v2_noroute_stage01_ft_v1_h8_k1_seed43/model_final"),
    "startobscurr": Path("checkpoints/upper_dslpid_recurrent_multitree_terminal_v2_noroute_startobscurr_v1_h8_k1_seed43/model_final"),
}
TASKS = [
    ("frozen", 1), ("frozen", 3),
    ("stage01_ft", 0), ("stage01_ft", 1),
    ("startobscurr", 0), ("startobscurr", 1),
    ("startobscurr", 2), ("startobscurr", 3),
]
FULL_TASKS = [("startobscurr", 0), ("startobscurr", 3)]
HIGHER_TASKS = [("stage01_ft", 2), ("stage01_ft", 3)]


def replace(command, option, value):
    command[command.index(option) + 1] = str(value)


def evaluate(variant, stage, model="k1"):
    label = f"{variant}_stage{stage}" if model == "k1" else f"{variant}_{model}_stage{stage}"
    root = ROOT / label
    output = root / "evaluation_metrics.jsonl"
    if output.exists():
        raise FileExistsError(output)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = CHECKPOINTS[variant]
    actor_path = Path(str(checkpoint) + "_actor")
    actor_hash = hashlib.sha256(actor_path.read_bytes()).hexdigest()
    protocol = {
        "variant": variant,
        "checkpoint": str(checkpoint),
        "actor_sha256": actor_hash,
        "model": model,
        "stage": stage,
        "map_id": MAP,
        "evaluation_environment_seeds": list(range(20, 40)),
        "expected_route_blocking_tree_count": (1, 2, 3, 5)[stage],
        "original_start": [-3.5, 0.0, 1.0],
        "reward_and_network_preset": BASE,
    }
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    command = build_mode_cmd(BASE)
    for option, value in (
        ("--seed", 43),
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
    records = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
    if len(records) != 1 or len(records[0]["episodes"]) != 20:
        raise AssertionError(f"Unexpected episode count in {output}")
    episodes = records[0]["episodes"]
    if [row["environment_seed"] for row in episodes] != list(range(20, 40)):
        raise AssertionError("Wrong held-out environment seeds")
    summary = records[0]["summary"]
    if int(summary["eval/curriculum_stage"]) != stage:
        raise AssertionError("Wrong curriculum stage")
    return {
        "variant": variant, "stage": stage,
        "successes": sum(row["success"] for row in episodes),
        "collisions": sum(row["collision"] for row in episodes),
        "timeouts": sum(row["done_reason"] == "timeout" for row in episodes),
        "mean_final_goal_distance": summary["eval/final_goal_distance"],
        "coverage": summary.get("eval/terminal_coverage/guidance_entered_episode_fraction"),
        "output": str(root),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Evaluate selected policy in Full Stage 0 and 3")
    parser.add_argument("--higher", action="store_true", help="Probe Stage 1-only finetune on unseen Stage 2 and 3")
    args = parser.parse_args()
    if args.full and args.higher:
        parser.error("Choose one additional evaluation set")
    rows = []
    tasks = FULL_TASKS if args.full else HIGHER_TASKS if args.higher else TASKS
    for variant, stage in tasks:
        model = "full" if args.full else "k1"
        print(f"Evaluating {variant} {model} Stage {stage} on seeds 20--39", flush=True)
        row = evaluate(variant, stage, model=model)
        rows.append(row)
        print(row, flush=True)
    name = "full_summary.json" if args.full else "higher_summary.json" if args.higher else "summary.json"
    (ROOT / name).write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
