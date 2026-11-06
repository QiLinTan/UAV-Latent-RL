"""Check four frozen K1 Actors on matched K1/Full reset states before Full FT."""

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from algos.td3.td3_upper_recurrent import TD3UpperRecurrent
from scripts.run_change_td3 import build_mode_cmd

BASE = "upper_dslpid_recurrent_multitree_terminal_v2_noroute_h8_k1"
SOURCE = Path("checkpoints/startobscurr_v1_selected_20261010")
ROOT = Path("analysis_outputs/full_ft_v1_source_audit_20261010")
MAP = "procedural_forest_multitree_validation_random_v1"
MAP_SEED = 80


def _set(command, option, value):
    command[command.index(option) + 1] = str(value)


def _run(seed, mode, actor_hash):
    directory = ROOT / f"seed{seed}_{mode}"
    checkpoint = SOURCE / f"seed{seed}" / "model_final"
    protocol = {
        "training_seed": seed, "mode": mode, "environment_seed": MAP_SEED,
        "stage": 3, "map_id": MAP, "source_actor_sha256": actor_hash,
        "checkpoint": str(checkpoint),
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
            ("--seed", seed), ("--eval_map_id", MAP), ("--eval_episodes", 1),
            ("--eval_interval", 1), ("--log_dir", directory),
            ("--upper_execution_model", mode),
        ):
            _set(command, option, value)
        command += [
            "--evaluate_only", "true", "--curriculum_stage_override", "3",
            "--load_checkpoint", str(checkpoint), "--eval_episode_seeds", str(MAP_SEED),
            "--eval_paired_trace_output_dir", str(directory / "traces"),
        ]
        with (directory / "evaluation.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    records = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
    assert len(records) == 1 and len(records[0]["episodes"]) == 1
    episode = records[0]["episodes"][0]
    assert episode["environment_seed"] == MAP_SEED
    trace = directory / "traces" / f"trace_seed{MAP_SEED}.npz"
    with np.load(trace) as data:
        result = {
            "initial_pre_observation": data["upper_pre_observation"][0].tolist(),
            "first_policy_action": data["upper_policy_action"][0].tolist(),
            "first_executed_action": data["upper_action"][0].tolist(),
            "first_command_velocity": data["upper_command_velocity"][0].tolist(),
            "observation_dimension": int(data["upper_pre_observation"].shape[1]),
            "action_dimension": int(data["upper_action"].shape[1]),
            "upper_steps": int(data["upper_action"].shape[0]),
            "lower_steps": int(data["lower_velocity"].shape[0]) if mode == "full" else 0,
        }
    result.update({"seed": seed, "mode": mode, "done_reason": episode["done_reason"]})
    return result


def main():
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    rows = []
    for seed in (42, 43, 44, 45):
        actor_file = SOURCE / f"seed{seed}" / "model_final_actor"
        digest = hashlib.sha256(actor_file.read_bytes()).hexdigest()
        assert digest == manifest["seeds"][str(seed)]["files"]["model_final_actor"]["sha256"]
        k1 = _run(seed, "k1", digest)
        full = _run(seed, "full", digest)
        assert k1["observation_dimension"] == full["observation_dimension"] == 917
        assert k1["action_dimension"] == full["action_dimension"] == 3
        assert full["lower_steps"] > 0 and k1["lower_steps"] == 0
        obs_diff = float(np.max(np.abs(np.asarray(k1["initial_pre_observation"]) - full["initial_pre_observation"])))
        action_diff = float(np.max(np.abs(np.asarray(k1["first_policy_action"]) - full["first_policy_action"])))
        command_diff = float(np.max(np.abs(np.asarray(k1["first_command_velocity"]) - full["first_command_velocity"])))
        assert obs_diff <= 1e-5 and action_diff <= 1e-5 and command_diff <= 1e-5
        # Verify the actor-only initialization path used by full_ft_v1,
        # not just the legacy full-checkpoint inference path.
        fresh = TD3UpperRecurrent(
            state_dim=917, action_dim=3, max_action=1.0,
            base_state_dim=117, history_length=8, history_frame_dim=100,
            hidden_dim=64, project_planar_action=True,
            device=torch.device("cpu"),
        )
        fresh.load_actor_only(str(SOURCE / f"seed{seed}" / "model_final"))
        actor_only_action = fresh.select_action(np.asarray(k1["initial_pre_observation"], dtype=np.float32))
        actor_only_diff = float(np.max(np.abs(actor_only_action - k1["first_policy_action"])))
        assert actor_only_diff <= 1e-6
        assert hashlib.sha256(actor_file.read_bytes()).hexdigest() == digest
        row = {
            "training_seed": seed, "source_actor_sha256": digest,
            "initial_observation_max_abs_diff": obs_diff,
            "first_policy_action_max_abs_diff": action_diff,
            "first_command_velocity_max_abs_diff": command_diff,
            "actor_only_vs_frozen_first_action_max_abs_diff": actor_only_diff,
            "k1_done_reason": k1["done_reason"], "full_done_reason": full["done_reason"],
            "full_120hz_steps": full["lower_steps"],
            "contract_pass": True,
        }
        rows.append(row)
        print(row, flush=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "summary.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
