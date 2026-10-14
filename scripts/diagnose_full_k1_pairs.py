"""Rerun frozen, paired Stage-3 K1/Full failures with 15/120 Hz traces."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.run_change_td3 import build_mode_cmd


BASE = "upper_dslpid_recurrent_multitree_terminal_v2_noroute_h8_k1"
MAP = "procedural_forest_multitree_validation_random_v1"
CHECKPOINT_ROOT = Path("checkpoints/startobscurr_v1_selected_20261010")
OLD_ROOT = Path("analysis_outputs/startobscurr_multiseed_holdout_20261010")
ROOT = Path("analysis_outputs/full_k1_paired_trajectories_20261010")
PILOT = {42: (20,), 43: (21,), 45: (20,)}


def _old_rows(seed, mode):
    if seed == 43:
        path = Path("analysis_outputs/seed43_course_holdout_20261009") / (
            "startobscurr_stage3" if mode == "k1" else "startobscurr_full_stage3"
        )
    else:
        path = OLD_ROOT / f"seed{seed}_new_{mode}_stage3"
    records = json.loads((path / "evaluation_metrics.jsonl").read_text().splitlines()[0])
    return {int(row["environment_seed"]): row for row in records["episodes"]}


def _pairs():
    result = {}
    for seed in (42, 43, 44, 45):
        k1 = _old_rows(seed, "k1")
        full = _old_rows(seed, "full")
        result[seed] = tuple(
            map_seed for map_seed in range(20, 40)
            if k1[map_seed]["success"] and not full[map_seed]["success"]
        )
    assert sum(map(len, result.values())) == 15, result
    return result


def _set(command, option, value):
    command[command.index(option) + 1] = str(value)


def evaluate(seed, mode, map_seeds, phase):
    directory = ROOT / phase / f"seed{seed}_{mode}"
    checkpoint = CHECKPOINT_ROOT / f"seed{seed}" / "model_final"
    manifest = json.loads((CHECKPOINT_ROOT / "manifest.json").read_text())
    actor_sha = hashlib.sha256(Path(str(checkpoint) + "_actor").read_bytes()).hexdigest()
    assert actor_sha == manifest["seeds"][str(seed)]["files"]["model_final_actor"]["sha256"]
    protocol = {
        "schema_version": 1,
        "training_seed": seed,
        "mode": mode,
        "stage": 3,
        "map_id": MAP,
        "environment_seeds": list(map_seeds),
        "checkpoint": str(checkpoint),
        "actor_sha256": actor_sha,
        "baseline_map_seeds": list(range(20, 40)),
        "upper_rate_hz": 15,
        "full_lower_rate_hz": 120,
    }
    protocol_path = directory / "protocol.json"
    output_path = directory / "evaluation_metrics.jsonl"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol
    else:
        directory.mkdir(parents=True, exist_ok=True)
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    if not output_path.exists():
        command = build_mode_cmd(BASE)
        for option, value in (
            ("--seed", seed), ("--eval_map_id", MAP),
            ("--eval_episodes", len(map_seeds)), ("--eval_interval", 1),
            ("--log_dir", directory), ("--upper_execution_model", mode),
        ):
            _set(command, option, value)
        command += [
            "--evaluate_only", "true",
            "--curriculum_stage_override", "3",
            "--load_checkpoint", str(checkpoint),
            "--eval_episode_seeds", *[str(value) for value in map_seeds],
            "--eval_paired_trace_output_dir", str(directory / "traces"),
        ]
        with (directory / "evaluation.log").open("w") as handle:
            subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)
    records = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
    assert len(records) == 1
    rows = records[0]["episodes"]
    assert [row["environment_seed"] for row in rows] == list(map_seeds)
    old = _old_rows(seed, mode)
    matches = []
    for row in rows:
        map_seed = row["environment_seed"]
        assert (directory / "traces" / f"trace_seed{map_seed}.npz").exists()
        matches.append({
            "map_seed": map_seed,
            "original_done_reason": old[map_seed]["done_reason"],
            "rerun_done_reason": row["done_reason"],
            "original_final_distance": old[map_seed]["final_goal_distance"],
            "rerun_final_distance": row["final_goal_distance"],
            "reason_matches": row["done_reason"] == old[map_seed]["done_reason"],
            "distance_delta": row["final_goal_distance"] - old[map_seed]["final_goal_distance"],
        })
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "all"), required=True)
    args = parser.parse_args()
    pairs = _pairs()
    selected = PILOT if args.phase == "pilot" else pairs
    ROOT.mkdir(parents=True, exist_ok=True)
    summary = []
    for seed, map_seeds in selected.items():
        for mode in ("k1", "full"):
            print(f"Tracing seed {seed} {mode} maps {map_seeds}", flush=True)
            matches = evaluate(seed, mode, map_seeds, args.phase)
            summary.extend({"training_seed": seed, "mode": mode, **row} for row in matches)
            (ROOT / args.phase / "reproduction.json").write_text(
                json.dumps({"phase": args.phase, "rows": summary}, indent=2) + "\n"
            )
            print(f"  reproduced {sum(row['reason_matches'] for row in matches)}/{len(matches)} reasons", flush=True)


if __name__ == "__main__":
    main()
