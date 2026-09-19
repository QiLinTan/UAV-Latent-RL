"""Evaluate a frozen latent Actor on sealed benchmark train/validation maps.

The script never reads the benchmark test split unless explicitly requested for
final evaluation.  Successful rollouts can be exported as a nonprivileged
reference-sequence dataset: inputs are frozen Encoder latents plus the observed
goal/ranges, and labels are future states from the Actor's own successful
trajectory.
"""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from algos.td3.networks import Actor
from benchmarks.forest_benchmark import FAMILIES, install_map, load_split
from envs.ForestAviary import CustomForestAviary
from envs.preprocess import preprocess_state
from gym_pybullet_drones.utils.enums import ActionType, ObservationType
from high_level_planner import FUTURE_TIMES, extract_planner_input, save_reference_dataset
from models.encoder import Encoder
from scripts.evaluate_native_latent_actor import array_stats, clean_json, terminal_reason, write_json
from scripts.freeze_logical_async import file_record


def load_frozen_native(checkpoint_prefix: str):
    encoder = Encoder(263, 16)
    actor = Actor(16, 4, 1.0)
    records = {}
    for name, module in (("encoder", encoder), ("actor", actor)):
        path = Path(str(checkpoint_prefix) + "_" + name)
        state = torch.load(path, map_location="cpu", weights_only=True)
        result = module.load_state_dict(state, strict=True)
        module.eval().requires_grad_(False)
        records[name] = {
            **file_record(path),
            "missing_keys": result.missing_keys,
            "unexpected_keys": result.unexpected_keys,
            "parameter_count": sum(p.numel() for p in module.parameters()),
        }
    return encoder, actor, records


def _canonical_axis_record(record):
    value = dict(record)
    value["start"] = [-3.5, 0.0, 1.0]
    value["goal"] = [3.5, 0.0, 1.0]
    value["initial_yaw"] = 0.0
    return value


def make_env(record, episode_len_sec: float, *, pose_mode: str):
    env_record = _canonical_axis_record(record) if pose_mode == "canonical_axis" else record
    yaw = 0.0 if pose_mode == "zero_yaw" else env_record["initial_yaw"]
    with contextlib.redirect_stdout(io.StringIO()):
        env = CustomForestAviary(
            obs=ObservationType.KIN,
            act=ActionType.RPM,
            pyb_freq=240,
            ctrl_freq=120,
            gui=False,
            curriculum=False,
            num_trees=0,
            route_blocking_tree=False,
            initial_xyzs=np.asarray([env_record["start"]], dtype=np.float32),
            initial_rpys=np.asarray([[0.0, 0.0, yaw]], dtype=np.float32),
            target_pos=np.asarray(env_record["goal"], dtype=np.float32),
            episode_len_sec=float(episode_len_sec),
        )
    return env, env_record


def rollout(record, encoder, actor, *, split: str, map_index: int, high_level_interval: int, episode_len_sec: float, pose_mode: str):
    env, env_record = make_env(record, episode_len_sec, pose_mode=pose_mode)
    try:
        observation, _ = env.reset(seed=int(env_record["seed"]))
        observation, _ = install_map(env, env_record)
        raw = np.asarray(observation, dtype=np.float32).reshape(-1)
        if raw.shape != (263,):
            raise ValueError(f"Expected 263D observation, got {raw.shape}")
        offsets = np.rint(FUTURE_TIMES * int(env.CTRL_FREQ)).astype(np.int32)
        positions = [env.pos[0].copy()]
        velocities = [env.vel[0].copy()]
        decisions = []
        rewards = []
        min_clearance = float(env._computeNearestTreeClearance(env.pos[0]))
        terminated = truncated = False
        info = {}
        step = 0
        with torch.inference_mode():
            while not (terminated or truncated):
                state = preprocess_state(raw)
                z = encoder(torch.from_numpy(state).reshape(1, -1))
                action = actor(z)[0].numpy().astype(np.float32)
                if step % int(high_level_interval) == 0:
                    decisions.append(
                        {
                            "step": int(step),
                            "planner_input": extract_planner_input(z[0].numpy(), state),
                            "position": env.pos[0].copy(),
                        }
                    )
                observation, reward, terminated, truncated, info = env.step(action.reshape(1, 4))
                raw = np.asarray(observation, dtype=np.float32).reshape(-1)
                positions.append(env.pos[0].copy())
                velocities.append(env.vel[0].copy())
                rewards.append(float(reward))
                min_clearance = min(min_clearance, float(env._computeNearestTreeClearance(env.pos[0])))
                step += 1
        positions = np.asarray(positions, dtype=np.float32)
        velocities = np.asarray(velocities, dtype=np.float32)
        samples = []
        if bool(info.get("success", False)):
            for decision in decisions:
                indices = int(decision["step"]) + offsets
                if int(indices[-1]) >= positions.shape[0]:
                    continue
                target = np.concatenate(
                    [
                        positions[indices] - decision["position"][None, :],
                        velocities[indices],
                    ],
                    axis=1,
                ).astype(np.float32)
                samples.append(
                    {
                        "split": split,
                        "map_index": int(map_index),
                        "map_id": record["map_id"],
                        "family": record["family"],
                        "step_id": int(decision["step"]),
                        "planner_input": decision["planner_input"],
                        "reference_target": target,
                    }
                )
        summary = {
            "split": split,
            "map_index": int(map_index),
            "map_id": record["map_id"],
            "family": record["family"],
            "seed": int(record["seed"]),
            "pose_mode": pose_mode,
            "success": bool(info.get("success", False)),
            "collision": bool(info.get("collision", False)),
            "done_reason": terminal_reason(env, info, terminated, truncated),
            "steps": int(step),
            "return": float(np.sum(rewards)),
            "final_distance": float(np.linalg.norm(env.TARGET_POS - env.pos[0])),
            "minimum_clearance": float(min_clearance),
            "sample_count": int(len(samples)),
            "initial_observation_stats": array_stats(raw),
        }
        print(
            f"{split} {record['family']} map={map_index}: "
            f"{summary['done_reason']}, dist={summary['final_distance']:.3f}, "
            f"samples={len(samples)}",
            flush=True,
        )
        return summary, samples
    finally:
        env.close()


def selected_records(benchmark_root, split: str, per_family: int, *, purpose: str):
    records = load_split(benchmark_root, split, purpose=purpose)
    selected = []
    for family in FAMILIES:
        selected.extend([record for record in records if record["family"] == family][:per_family])
    return selected


def write_reference_dataset(root, samples, metadata, *, split_seed: int):
    arrays = {
        "episode_id": [],
        "step_id": [],
        "planner_input": [],
        "reference_target": [],
    }
    episode_by_map = {}
    for row in samples:
        episode_id = episode_by_map.setdefault(row["map_id"], len(episode_by_map))
        arrays["episode_id"].append(episode_id)
        arrays["step_id"].append(row["step_id"])
        arrays["planner_input"].append(row["planner_input"])
        arrays["reference_target"].append(row["reference_target"])
    return save_reference_dataset(
        root,
        episode_id=arrays["episode_id"],
        step_id=arrays["step_id"],
        planner_input=arrays["planner_input"],
        reference_target=arrays["reference_target"],
        metadata=metadata,
        split_seed=split_seed,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/latent_only_affordance_dim16_legacy_reward_rerun/model_best")
    parser.add_argument("--benchmark", default="data/benchmarks/forest_map_isolation_v1")
    parser.add_argument("--output", default="analysis_outputs/native_actor_map_isolated_legacy_best_v1")
    parser.add_argument("--dataset-output", default="data/reference_planner/native_actor_map_isolated_legacy_best_v1")
    parser.add_argument("--splits", nargs="+", choices=("train", "validation", "test"), default=("train", "validation"))
    parser.add_argument("--maps-per-family", type=int, default=2)
    parser.add_argument("--high-level-interval", type=int, default=8)
    parser.add_argument("--episode-len-sec", type=float, default=20.0)
    parser.add_argument(
        "--pose-mode",
        choices=("benchmark", "zero_yaw", "canonical_axis"),
        default="benchmark",
        help="benchmark is the formal map contract; other modes are diagnostic controls only.",
    )
    parser.add_argument("--final-evaluation", action="store_true", help="Allow reading the sealed test split.")
    parser.add_argument("--no-dataset", action="store_true")
    args = parser.parse_args()
    if "test" in args.splits and not args.final_evaluation:
        raise ValueError("Refusing to read test maps without --final-evaluation.")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    encoder, actor, networks = load_frozen_native(args.checkpoint)
    summaries = []
    samples = []
    purpose = "final_evaluation" if args.final_evaluation else "diagnostic"
    for split in args.splits:
        for map_index, record in enumerate(selected_records(args.benchmark, split, args.maps_per_family, purpose=purpose)):
            summary, rows = rollout(
                record,
                encoder,
                actor,
                split=split,
                map_index=map_index,
                high_level_interval=args.high_level_interval,
                episode_len_sec=args.episode_len_sec,
                pose_mode=args.pose_mode,
            )
            summaries.append(summary)
            samples.extend(rows)
    families = sorted({row["family"] for row in summaries})
    by_family = {}
    for family in families:
        rows = [row for row in summaries if row["family"] == family]
        by_family[family] = {
            "episodes": len(rows),
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "collision_rate": float(np.mean([row["collision"] for row in rows])),
            "mean_final_distance": float(np.mean([row["final_distance"] for row in rows])),
            "successful_reference_samples": int(sum(row["sample_count"] for row in rows)),
        }
    report = {
        "checkpoint_prefix": args.checkpoint,
        "benchmark": file_record(Path(args.benchmark) / "manifest.json"),
        "splits": list(args.splits),
        "test_maps_read": bool("test" in args.splits),
        "maps_per_family": int(args.maps_per_family),
        "high_level_frequency_hz": 120.0 / int(args.high_level_interval),
        "episode_len_sec": float(args.episode_len_sec),
        "pose_mode": args.pose_mode,
        "formal_map_isolated": bool(args.pose_mode == "benchmark"),
        "networks": networks,
        "summary_by_family": by_family,
        "episodes": summaries,
        "total_successes": int(sum(row["success"] for row in summaries)),
        "total_episodes": int(len(summaries)),
        "total_reference_samples": int(len(samples)),
        "teacher": "frozen_native_latent_actor_no_pid_no_oracle",
    }
    if samples and not args.no_dataset:
        dataset_metadata = write_reference_dataset(
            args.dataset_output,
            samples,
            {
                "teacher_checkpoint": args.checkpoint,
                "teacher_controller": "frozen_native_latent_actor_no_pid_no_oracle",
                "encoder_frozen": True,
                "actor_frozen": True,
                "map_isolated": True,
                "benchmark": str(Path(args.benchmark)),
                "benchmark_manifest": file_record(Path(args.benchmark) / "manifest.json"),
                "source_splits": list(args.splits),
                "test_maps_read": bool("test" in args.splits),
                "successful_episodes_only": True,
                "control_frequency_hz": 120,
                "high_level_frequency_hz": 120.0 / int(args.high_level_interval),
                "target_layout": "relative_position_xyz_then_world_velocity_xyz",
            },
            split_seed=20260907,
        )
        report["dataset"] = {**dataset_metadata, "path": str(Path(args.dataset_output))}
    elif not samples:
        report["dataset"] = None
    write_json(output / "report.json", clean_json(report))
    print(
        f"wrote {output}: success={report['total_successes']}/{report['total_episodes']}, "
        f"samples={report['total_reference_samples']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
