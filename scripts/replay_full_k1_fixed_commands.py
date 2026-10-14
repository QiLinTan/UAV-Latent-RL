"""A2: execute identical K1 3D velocity tapes in K1 and Full, without Actor feedback."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from scripts.run_change_td3 import build_mode_cmd


BASE = "upper_dslpid_recurrent_multitree_terminal_v2_noroute_h8_k1"
MAP = "procedural_forest_multitree_validation_random_v1"
SOURCE = Path("analysis_outputs/full_k1_paired_trajectories_20261010/all")
ROOT = Path("analysis_outputs/full_k1_fixed_command_a2_20261010")
CHECKPOINTS = Path("checkpoints/startobscurr_v1_selected_20261010")
CASES = ((42, 34), (42, 32), (45, 20))
STEPS = (15, 45, None)


def replace(command, option, value):
    command[command.index(option) + 1] = str(value)


def run_one(seed, map_seed, steps, mode):
    label = "full_tape" if steps is None else f"steps{steps:03d}"
    directory = ROOT / f"seed{seed}_map{map_seed}" / label / mode
    tape_dir = SOURCE / f"seed{seed}_k1" / "traces"
    tape_path = tape_dir / f"trace_seed{map_seed}.npz"
    checkpoint = CHECKPOINTS / f"seed{seed}" / "model_final"
    manifest = json.loads((CHECKPOINTS / "manifest.json").read_text())
    actor_hash = hashlib.sha256(Path(str(checkpoint) + "_actor").read_bytes()).hexdigest()
    assert actor_hash == manifest["seeds"][str(seed)]["files"]["model_final_actor"]["sha256"]
    with np.load(tape_path) as tape:
        length = len(tape["upper_action"])
    assert steps is None or steps <= length
    protocol = {
        "schema_version": 1,
        "training_seed": seed,
        "environment_seed": map_seed,
        "mode": mode,
        "stage": 3,
        "map_id": MAP,
        "checkpoint": str(checkpoint),
        "actor_sha256": actor_hash,
        "source_k1_tape": str(tape_path),
        "source_k1_tape_sha256": hashlib.sha256(tape_path.read_bytes()).hexdigest(),
        "command_source": "K1 closed-loop executed 3D velocity command",
        "intervention_steps": length if steps is None else steps,
        "intervention_seconds_requested": (length if steps is None else steps) / 15.0,
    }
    directory.mkdir(parents=True, exist_ok=True)
    protocol_path = directory / "protocol.json"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    output = directory / "evaluation_metrics.jsonl"
    if not output.exists():
        command = build_mode_cmd(BASE)
        for option, value in (
            ("--seed", seed), ("--eval_map_id", MAP),
            ("--eval_episodes", 1), ("--eval_interval", 1),
            ("--log_dir", directory), ("--upper_execution_model", mode),
        ):
            replace(command, option, value)
        command += [
            "--evaluate_only", "true",
            "--curriculum_stage_override", "3",
            "--load_checkpoint", str(checkpoint),
            "--eval_episode_seeds", str(map_seed),
            "--eval_fixed_command_tape_dir", str(tape_dir),
            "--eval_paired_trace_output_dir", str(directory / "traces"),
        ]
        if steps is not None:
            command.extend(["--eval_fixed_command_steps", str(steps)])
        with (directory / "evaluation.log").open("w") as handle:
            subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)
    row = json.loads(output.read_text().splitlines()[0])["episodes"][0]
    assert row["environment_seed"] == map_seed
    return {
        "training_seed": seed,
        "map_seed": map_seed,
        "steps": steps,
        "mode": mode,
        "done_reason": row["done_reason"],
        "final_goal_distance": row["final_goal_distance"],
        "directory": str(directory),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "all", "collision_all"), required=True)
    args = parser.parse_args()
    if args.phase == "collision_all":
        summary = json.loads((SOURCE / "paired_summary.json").read_text())
        cases = tuple((row["training_seed"], row["map_seed"]) for row in summary["pairs"] if row["full_done_reason"] == "collision")
        assert len(cases) == 12
        steps_list = (None,)
        modes = ("full",)
    else:
        cases = CASES[:1] if args.phase == "pilot" else CASES
        steps_list = STEPS[:1] if args.phase == "pilot" else STEPS
        modes = ("k1", "full")
    rows = []
    for seed, map_seed in cases:
        for steps in steps_list:
            for mode in modes:
                print(seed, map_seed, steps, mode, flush=True)
                row = run_one(seed, map_seed, steps, mode)
                rows.append(row)
                ROOT.mkdir(parents=True, exist_ok=True)
                (ROOT / f"runs_{args.phase}.json").write_text(json.dumps(rows, indent=2) + "\n")
                print("  ", row["done_reason"], f"d={row['final_goal_distance']:.3f}", flush=True)


if __name__ == "__main__":
    main()
