"""Paired frozen-source and Full-FT evaluation on reserved map seeds 40--79."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.run_change_td3 import build_mode_cmd

BASE = "upper_dslpid_recurrent_multitree_terminal_v2_noroute_h8_k1"
MAP = "procedural_forest_multitree_validation_random_v1"
SOURCE_ROOT = Path("checkpoints/startobscurr_v1_selected_20261010")
ROOT = Path("analysis_outputs/full_ft_v1_heldout_20261010")
ENV_SEEDS = tuple(range(40, 80))
TASKS = (("full", 3), ("full", 0), ("k1", 3))


def _set(command, option, value):
    command[command.index(option) + 1] = str(value)


def _digest(prefix):
    actor = Path(str(prefix) + "_actor")
    if not actor.is_file():
        raise FileNotFoundError(actor)
    return hashlib.sha256(actor.read_bytes()).hexdigest()


def evaluate(seed, variant, checkpoint, mode, stage):
    folder = ROOT / f"seed{seed}" / f"{variant}_{mode}_stage{stage}"
    digest = _digest(checkpoint)
    protocol = {
        "schema_version": 1, "training_seed": seed, "variant": variant,
        "checkpoint": str(checkpoint), "checkpoint_actor_sha256": digest,
        "mode": mode, "stage": stage, "map_id": MAP,
        "environment_seeds": list(ENV_SEEDS), "upper_episode_seconds": 12,
        "success_radius_m": 0.2,
    }
    folder.mkdir(parents=True, exist_ok=True)
    protocol_path = folder / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise RuntimeError(f"Held-out protocol changed: {protocol_path}")
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    output = folder / "evaluation_metrics.jsonl"
    if not output.exists():
        command = build_mode_cmd(BASE)
        for option, value in (
            ("--seed", seed), ("--eval_map_id", MAP),
            ("--eval_episodes", len(ENV_SEEDS)), ("--eval_interval", 1),
            ("--log_dir", folder), ("--upper_execution_model", mode),
        ):
            _set(command, option, value)
        command += [
            "--evaluate_only", "true", "--curriculum_stage_override", str(stage),
            "--load_checkpoint", str(checkpoint), "--eval_episode_seeds",
            *[str(value) for value in ENV_SEEDS],
        ]
        if mode == "full" and stage == 3:
            command += ["--eval_paired_trace_output_dir", str(folder / "traces")]
        with (folder / "evaluation.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    records = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
    if len(records) != 1:
        raise RuntimeError(f"Expected one evaluation pass: {output}")
    rows = records[0]["episodes"]
    if [int(row["environment_seed"]) for row in rows] != list(ENV_SEEDS):
        raise RuntimeError(f"Held-out seeds differ from protocol: {output}")
    if int(records[0]["summary"]["eval/curriculum_stage"]) != stage:
        raise RuntimeError(f"Wrong evaluation stage: {output}")
    by_seed = {int(row["environment_seed"]): row for row in rows}
    result = {
        "variant": variant, "mode": mode, "stage": stage,
        "checkpoint": str(checkpoint), "actor_sha256": digest,
        "success_count": sum(bool(row["success"]) for row in rows),
        "collision_count": sum(bool(row["collision"]) for row in rows),
        "timeout_count": sum(row["done_reason"] == "timeout" for row in rows),
        "mean_final_goal_distance_m": records[0]["summary"]["eval/final_goal_distance"],
        "outcomes_by_environment_seed": {
            str(s): {"done_reason": by_seed[s]["done_reason"],
                     "success": bool(by_seed[s]["success"]),
                     "collision": bool(by_seed[s]["collision"]),
                     "final_goal_distance_m": by_seed[s]["final_goal_distance"]}
            for s in ENV_SEEDS
        },
        "directory": str(folder),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=(42, 43, 44, 45))
    parser.add_argument("--phase", choices=("source", "finetuned", "both"), default="both")
    parser.add_argument("--finetuned-checkpoint", type=Path,
                        help="Checkpoint prefix, e.g. checkpoints/full_ft_v1_seed43/model_final")
    args = parser.parse_args()
    if args.phase != "source" and args.finetuned_checkpoint is None:
        parser.error("Finetuned phase needs --finetuned-checkpoint")
    source = SOURCE_ROOT / f"seed{args.seed}" / "model_final"
    manifest = json.loads((SOURCE_ROOT / "manifest.json").read_text())
    if _digest(source) != manifest["seeds"][str(args.seed)]["files"]["model_final_actor"]["sha256"]:
        raise RuntimeError("Frozen source Actor checksum mismatch")
    variants = {}
    if args.phase in ("source", "both"):
        variants["source"] = source
    if args.phase in ("finetuned", "both"):
        variants["finetuned"] = args.finetuned_checkpoint
    root = ROOT / f"seed{args.seed}"
    root.mkdir(parents=True, exist_ok=True)
    for variant, checkpoint in variants.items():
        for mode, stage in TASKS:
            row = evaluate(args.seed, variant, checkpoint, mode, stage)
            print(f"seed {args.seed} {variant} {mode} Stage {stage}: "
                  f"{row['success_count']}/40 success, {row['collision_count']} collision", flush=True)
    summary = {"training_seed": args.seed, "environment_seeds": list(ENV_SEEDS),
               "map_id": MAP, "tasks": {}}
    for mode, stage in TASKS:
        key = f"{mode}_stage{stage}"
        summary["tasks"][key] = {}
        for variant in ("source", "finetuned"):
            folder = root / f"{variant}_{mode}_stage{stage}"
            if (folder / "evaluation_metrics.jsonl").exists():
                protocol = json.loads((folder / "protocol.json").read_text())
                summary["tasks"][key][variant] = evaluate(
                    args.seed, variant, Path(protocol["checkpoint"]), mode, stage,
                )
        if len(summary["tasks"][key]) == 2:
            source_rows = summary["tasks"][key]["source"]["outcomes_by_environment_seed"]
            ft_rows = summary["tasks"][key]["finetuned"]["outcomes_by_environment_seed"]
            summary["tasks"][key]["paired_changes"] = {
                "source_fail_to_ft_success": [s for s in source_rows if not source_rows[s]["success"] and ft_rows[s]["success"]],
                "source_success_to_ft_fail": [s for s in source_rows if source_rows[s]["success"] and not ft_rows[s]["success"]],
                "source_noncollision_to_ft_collision": [s for s in source_rows if not source_rows[s]["collision"] and ft_rows[s]["collision"]],
                "source_collision_to_ft_noncollision": [s for s in source_rows if source_rows[s]["collision"] and not ft_rows[s]["collision"]],
            }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
