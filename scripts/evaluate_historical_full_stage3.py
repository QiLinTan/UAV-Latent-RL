"""Re-evaluate the selected v4.1 Full policy on both five-tree map families.

Read-only, no-training comparison with the current Stage-3 zero-shot protocol.
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


PRESET = "upper_dslpid_recurrent_multitree_v4_1"
CHECKPOINT = Path("checkpoints") / PRESET / "model_final"
MAPS = {
    "random_train": "procedural_forest_multitree_train_v2",
    "fixed_validation": "procedural_forest_validation_v1",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-family", choices=MAPS, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output-root", type=Path,
                        default=Path("analysis_outputs/historical_v4_1_full_stage3_20261009"))
    cli = parser.parse_args()
    if cli.episodes <= 0:
        parser.error("--episodes must be positive")
    root = cli.output_root / cli.map_family
    if (root / "evaluation_metrics.jsonl").exists():
        raise FileExistsError(root / "evaluation_metrics.jsonl")
    root.mkdir(parents=True, exist_ok=True)
    model_hash = hashlib.sha256(Path(str(CHECKPOINT) + "_actor").read_bytes()).hexdigest()
    protocol = {
        "preset": PRESET,
        "checkpoint": str(CHECKPOINT),
        "selected_checkpoint_step": 88000,
        "actor_sha256": model_hash,
        "upper_execution_model": "full",
        "forced_stage": 3,
        "map_id": MAPS[cli.map_family],
        "expected_route_blocking_tree_count": 5,
        "environment_seeds": list(range(cli.episodes)),
        "original_start": [-3.5, 0.0, 1.0],
        "simple_obstacle_mix_probability": 0.0,
        "zero_shot": True,
    }
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    class CheckedEvalCallback(ProductionEvalCallback):
        def __init__(self, env_cls, env_kwargs, *args, **kwargs):
            def checked_factory(**factory_kwargs):
                env = env_cls(**factory_kwargs)
                base = getattr(env, "base_env", env)
                base.CURRICULUM_SIMPLE_MIX_PROBABILITY = 0.0
                original_reset = env.reset

                def checked_reset(*, seed=None, options=None):
                    observation, info = original_reset(seed=seed, options=options)
                    if int(info["route_blocking_tree_count"]) != 5:
                        raise AssertionError("Expected five route-blocking trees")
                    if np.linalg.norm(base.pos[0] - base.START_POS) > 1e-5:
                        raise AssertionError("Expected original start")
                    if info["map_id"] != MAPS[cli.map_family]:
                        raise AssertionError("Wrong map family")
                    return observation, info

                env.reset = checked_reset
                return env

            super().__init__(checked_factory, env_kwargs, *args, **kwargs)

    command = build_mode_cmd(PRESET)[3:]

    def replace(option, value):
        command[command.index(option) + 1] = str(value)

    replace("--eval_map_id", MAPS[cli.map_family])
    replace("--eval_interval", 1)
    replace("--eval_episodes", cli.episodes)
    replace("--log_dir", root)
    replace("--trajectory_output_dir", root / "trajectory")
    command += [
        "--curriculum_stage_override", "3",
        "--evaluate_only", "true",
        "--load_checkpoint", str(CHECKPOINT),
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
    print(json.dumps({"map_family": cli.map_family,
                      "success_rate": rows[0]["summary"]["eval/success_rate"],
                      "collision_rate": rows[0]["summary"]["eval/collision_rate"],
                      "timeout_rate": rows[0]["summary"]["eval/done_reason_timeout_rate"]}))


if __name__ == "__main__":
    main()
