"""Zero-shot K1 or Full obstacle-stage evaluation of frozen actors.

Uses the original no_route preset and selected weights. Evaluation resets at
exactly the original 7 m start and disables the training map's 20% easier-stage
mix so every episode has the requested route-blocking-tree count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

from scripts import change_td3
from scripts.run_change_td3 import build_mode_cmd
from trainers.callbacks.eval_callback import EvalCallback as ProductionEvalCallback


PRESET = "upper_dslpid_recurrent_multitree_terminal_v2_noroute_h8_k1"
TRAIN_MAP = "procedural_forest_multitree_train_v2"
VALIDATION_MAP = "procedural_forest_validation_v1"
FREEZE = Path("checkpoints/startcurr_v1_frozen_20261009")
DEFAULT_OUTPUT = Path("analysis_outputs/startcurr_zero_shot_20261009")
ROUTE_COUNTS = (1, 2, 3, 5)


def evaluate(cli):
    checkpoint = FREEZE / f"seed{cli.train_seed}" / "model_final"
    manifest = json.loads((FREEZE / "manifest.json").read_text())
    actor_hash = hashlib.sha256(Path(str(checkpoint) + "_actor").read_bytes()).hexdigest()
    if actor_hash != manifest["seeds"][str(cli.train_seed)]["files"]["model_final_actor"]["sha256"]:
        raise RuntimeError("Frozen actor hash mismatch")
    label = f"{cli.model}_stage{cli.stage}_seed{cli.train_seed}"
    if cli.eval_map_id == VALIDATION_MAP:
        label += "_fixed_validation"
    root = cli.output_root / label
    if (root / "evaluation_metrics.jsonl").exists():
        raise FileExistsError(root / "evaluation_metrics.jsonl")
    root.mkdir(parents=True, exist_ok=True)
    protocol = {
        "training_seed": cli.train_seed,
        "model": cli.model,
        "forced_stage": cli.stage,
        "expected_route_blocking_tree_count": (
            5 if cli.eval_map_id == VALIDATION_MAP else ROUTE_COUNTS[cli.stage]
        ),
        "map_id": cli.eval_map_id,
        "evaluation_environment_seeds": list(range(cli.episodes)),
        "original_start": [-3.5, 0.0, 1.0],
        "goal": [3.5, 0.0, 1.0],
        "curriculum_simple_mix_probability": 0.0,
        "checkpoint": str(checkpoint),
        "actor_sha256": actor_hash,
        "zero_shot": True,
        "reward_and_network_preset": PRESET,
    }
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    class CheckedEvalCallback(ProductionEvalCallback):
        def __init__(self, env_cls, env_kwargs, *args, **kwargs):
            def checked_factory(**factory_kwargs):
                env = env_cls(**factory_kwargs)
                base = getattr(env, "base_env", env)
                # The map atlas owns this parameter, so a CLI override alone
                # would be ignored. Set it after construction, before reset.
                base.CURRICULUM_SIMPLE_MIX_PROBABILITY = 0.0
                original_reset = env.reset

                def checked_reset(*, seed=None, options=None):
                    observation, info = original_reset(seed=seed, options=options)
                    if cli.eval_map_id == TRAIN_MAP and base.get_curriculum_stage() != cli.stage:
                        raise AssertionError("Actual obstacle stage differs from requested stage")
                    if int(info["route_blocking_tree_count"]) != protocol["expected_route_blocking_tree_count"]:
                        raise AssertionError("Evaluation sampled a mixed easier obstacle stage")
                    if info["map_id"] != cli.eval_map_id:
                        raise AssertionError("Evaluation sampled the wrong map family")
                    if np.linalg.norm(base.pos[0] - base.START_POS) > 1e-5:
                        raise AssertionError("Evaluation did not reset at original start")
                    return observation, info

                env.reset = checked_reset
                return env

            super().__init__(checked_factory, env_kwargs, *args, **kwargs)

    command = build_mode_cmd(PRESET)[3:]

    def replace(option, value):
        command[command.index(option) + 1] = str(value)

    replace("--seed", cli.train_seed)
    replace("--eval_map_id", cli.eval_map_id)
    replace("--eval_interval", 1)
    replace("--eval_episodes", cli.episodes)
    replace("--log_dir", root)
    replace("--trajectory_output_dir", root / "trajectory")
    replace("--collision_diagnostic_output_dir", root / "collision")
    replace("--timeout_diagnostic_output_dir", root / "timeout")
    replace("--eval_terminal_coverage_output_dir", root / "coverage")
    command += [
        "--upper_execution_model", cli.model,
        "--curriculum_stage_override", str(cli.stage),
        "--evaluate_only", "true",
        "--load_checkpoint", str(checkpoint),
        "--eval_stepsleep", "false",
        "--eval_quiet", "true",
    ]
    with patch.object(change_td3, "EvalCallback", CheckedEvalCallback), \
            patch.object(sys, "argv", [__file__] + command):
        change_td3.main()
    rows = [json.loads(line) for line in (root / "evaluation_metrics.jsonl").read_text().splitlines()
            if line.strip()]
    if len(rows) != 1 or len(rows[0]["episodes"]) != cli.episodes:
        raise AssertionError("Unexpected evaluation episode count")
    print(json.dumps({"label": label, "summary": rows[0]["summary"]}, allow_nan=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-seed", type=int, choices=(42, 43, 44, 45), required=True)
    parser.add_argument("--model", choices=("k1", "full"), required=True)
    parser.add_argument("--stage", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--eval-map-id", choices=(TRAIN_MAP, VALIDATION_MAP),
                        default=TRAIN_MAP)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    cli = parser.parse_args()
    if cli.episodes <= 0:
        parser.error("--episodes must be positive")
    evaluate(cli)


if __name__ == "__main__":
    main()
