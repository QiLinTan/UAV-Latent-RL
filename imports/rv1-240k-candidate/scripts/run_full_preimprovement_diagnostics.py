"""Frozen-policy runs for P0 branch tests and B terminal-time diagnostics."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.run_change_td3 import build_mode_cmd


BASE = "upper_dslpid_recurrent_multitree_terminal_v2_noroute_h8_k1"
CHECKPOINTS = Path("checkpoints/startobscurr_v1_selected_20261010")
ROOT = Path("analysis_outputs/full_preimprovement_diagnostics_20261010")
MAP_STAGE3 = "procedural_forest_multitree_validation_random_v1"
P0_CASES = ((42, 34), (45, 20), (42, 32))
B_CASES = ((45, 25), (45, 30))


def _replace(command, option, value):
    command[command.index(option) + 1] = str(value)


def run_one(seed, map_seed, *, mode, stage, tag, episode_seconds=12.0,
            intervention_start=None, intervention_steps=None):
    directory = ROOT / tag / f"seed{seed}_map{map_seed}_{mode}"
    checkpoint = CHECKPOINTS / f"seed{seed}" / "model_final"
    actor_file = Path(str(checkpoint) + "_actor")
    manifest = json.loads((CHECKPOINTS / "manifest.json").read_text())
    actor_sha = hashlib.sha256(actor_file.read_bytes()).hexdigest()
    assert actor_sha == manifest["seeds"][str(seed)]["files"]["model_final_actor"]["sha256"]
    tape_dir = Path("analysis_outputs/full_k1_paired_trajectories_20261010/all") / f"seed{seed}_k1" / "traces"
    protocol = {
        "training_seed": seed, "map_seed": map_seed, "mode": mode, "stage": stage,
        "tag": tag, "episode_seconds": episode_seconds, "checkpoint": str(checkpoint),
        "actor_sha256": actor_sha, "intervention_start_step": intervention_start,
        "intervention_steps": intervention_steps,
        "intervention_source": str(tape_dir / f"trace_seed{map_seed}.npz") if intervention_start is not None else None,
    }
    directory.mkdir(parents=True, exist_ok=True)
    protocol_path = directory / "protocol.json"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    output = directory / "evaluation_metrics.jsonl"
    if not output.exists():
        cmd = build_mode_cmd(BASE)
        for option, value in (
            ("--seed", seed), ("--eval_map_id", MAP_STAGE3),
            ("--eval_episodes", 1), ("--eval_interval", 1),
            ("--log_dir", directory), ("--upper_execution_model", mode),
        ):
            _replace(cmd, option, value)
        cmd += [
            "--evaluate_only", "true", "--curriculum_stage_override", str(stage),
            "--load_checkpoint", str(checkpoint), "--eval_episode_seeds", str(map_seed),
            "--eval_paired_trace_output_dir", str(directory / "traces"),
            "--eval_episode_len_sec", str(episode_seconds),
        ]
        if intervention_start is not None:
            assert mode == "full" and stage == 3 and intervention_steps is not None
            cmd += [
                "--eval_fixed_command_tape_dir", str(tape_dir),
                "--eval_command_intervention_start_step", str(intervention_start),
                "--eval_command_intervention_steps", str(intervention_steps),
            ]
        with (directory / "evaluation.log").open("w") as handle:
            subprocess.run(cmd, check=True, stdout=handle, stderr=subprocess.STDOUT)
    row = json.loads(output.read_text().splitlines()[0])["episodes"][0]
    assert row["environment_seed"] == map_seed
    trace = directory / "traces" / f"trace_seed{map_seed}.npz"
    assert trace.is_file()
    result = {**protocol, "done_reason": row["done_reason"],
              "final_goal_distance": row["final_goal_distance"], "trace": str(trace)}
    print(seed, map_seed, mode, tag, row["done_reason"], f"d={row['final_goal_distance']:.3f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("p0_baseline", "p0_intervention", "b"), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--map-seed", type=int)
    parser.add_argument("--intervention-start", type=int)
    parser.add_argument("--intervention-steps", type=int, default=8)
    args = parser.parse_args()
    if args.phase == "p0_baseline":
        rows = [run_one(seed, map_seed, mode=mode, stage=3, tag="p0_baseline")
                for seed, map_seed in P0_CASES for mode in ("k1", "full")]
    elif args.phase == "p0_intervention":
        if args.seed is None or args.map_seed is None or args.intervention_start is None:
            parser.error("P0 intervention requires seed, map-seed and intervention-start")
        assert (args.seed, args.map_seed) in P0_CASES
        tag = f"p0_intervention_step{args.intervention_start:03d}_len{args.intervention_steps:02d}"
        rows = [run_one(args.seed, args.map_seed, mode="full", stage=3, tag=tag,
                        intervention_start=args.intervention_start,
                        intervention_steps=args.intervention_steps)]
    else:
        rows = [run_one(seed, map_seed, mode="full", stage=0,
                        tag=f"b_{int(seconds)}s", episode_seconds=seconds)
                for seed, map_seed in B_CASES for seconds in (12.0, 14.0, 16.0)]
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / f"runs_{args.phase}.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
