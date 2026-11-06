"""Collect a provenance-complete paired instant/temporal BC cohort.

This is the fallback when an old instantaneous archive lacks the raw temporal
observations and immutable map provenance needed to reconstruct them.  It keeps
the original first-round *training cohort* (10 successful episodes per
scenario), frozen Encoder, privileged A*/DSLPID teacher, labels,
and 15-Hz decision cadence.  It writes two views of *exactly the same rows*:
the B0 27-D archive and the B1/B2 archive with an 8x16 history.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.ForestAviary import CustomForestAviary
from envs.preprocess import preprocess_state
from gym_pybullet_drones.utils.enums import ActionType, ObservationType
from high_level_planner import save_reference_dataset, save_temporal_reference_dataset
from scripts.build_temporal_reference_dataset import replay_candidate
from scripts.collect_reference_planner_dataset import SCENARIO_CONFIGS, load_teacher


def make_seeded_env(scenario: str, layout_seed: int, episode_len_sec: float):
    with contextlib.redirect_stdout(io.StringIO()):
        return CustomForestAviary(
            obs=ObservationType.KIN,
            act=ActionType.RPM,
            pyb_freq=240,
            ctrl_freq=120,
            gui=False,
            curriculum=False,
            episode_len_sec=float(episode_len_sec),
            seed=int(layout_seed),
            **SCENARIO_CONFIGS[scenario],
        )


def tree_record(spec: dict) -> dict:
    return {
        "xy": np.asarray(spec["xy"], dtype=float).tolist(),
        "radius": float(spec["radius"]),
        "height": float(spec["height"]),
        **{name: float(spec[name]) for name in ("route_fraction", "route_lateral_offset") if name in spec},
        **({"route_blocking": bool(spec["route_blocking"])} if "route_blocking" in spec else {}),
    }


def apply_split_template(root: Path, episode_id: np.ndarray, reference_target: np.ndarray, template: Path) -> dict:
    """Preserve the historical B0 episode split for the paired replacement cohort."""

    with np.load(template, allow_pickle=False) as values:
        splits = {name: np.asarray(values[name], dtype=np.int32).copy() for name in values.files}
    expected = set(range(int(np.unique(episode_id).size)))
    found = set(np.concatenate(list(splits.values())).tolist())
    if set(splits) != {"train", "validation", "test"} or found != expected:
        raise ValueError("Split template is not a complete disjoint episode partition.")
    train_mask = np.isin(episode_id, splits["train"])
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["target_mean"] = reference_target[train_mask].mean(axis=0).tolist()
    metadata["target_std"] = np.maximum(reference_target[train_mask].std(axis=0), 1e-4).tolist()
    metadata["split_episode_counts"] = {name: int(values.size) for name, values in splits.items()}
    metadata["split_contract"] = "historical_latent_affordance_margin_v2_episode_partition"
    metadata["split_template"] = str(template)
    np.savez_compressed(root / "splits.npz", **splits)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return splits


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instant-output", default="data/reference_planner/temporal_controlled_v1_instant")
    parser.add_argument("--temporal-output", default="data/reference_planner/temporal_controlled_v1_temporal8")
    parser.add_argument("--checkpoint", default="checkpoints/latent_only_affordance_dim16/model_final")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--episode-len-sec", type=float, default=12.0)
    parser.add_argument("--high-level-interval", type=int, default=8)
    parser.add_argument(
        "--split-template",
        default="data/reference_planner/latent_affordance_margin_v2/splits.npz",
        help="Historical B0 episode partition to reuse for all paired variants.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.high_level_interval <= 0 or args.episode_len_sec <= 0.5:
        raise ValueError("Invalid high-level interval or episode length.")
    device = torch.device("cpu")
    arrays = {name: [] for name in (
        "episode_id", "step_id", "planner_input", "environment_history",
        "history_padding_count", "reference_target",
    )}
    scenario_stats = {}
    accepted_maps = []
    episode_id = 0
    agent = None
    # With per-map deterministic layouts, seeds 201003/7/10/12--16 do not
    # finish under the unchanged oracle/PID controller.  Eighteen candidate
    # maps are needed to retain the historical 10 successful forest examples;
    # rejected trajectories are not saved or used for BC.
    budgets = {"clear": (10, 10), "single": (10, 10), "forest24": (10, 18)}
    for scenario_index, scenario in enumerate(("clear", "single", "forest24")):
        wanted, budget = budgets[scenario]
        accepted = collisions = 0
        for attempt in range(budget):
            layout_seed = int(args.seed) + scenario_index * 100000 + attempt
            env = make_seeded_env(scenario, layout_seed, args.episode_len_sec)
            try:
                initial, _ = env.reset(seed=layout_seed)
                if agent is None:
                    agent = load_teacher(
                        args.checkpoint,
                        int(preprocess_state(initial.reshape(-1)).size),
                        device,
                    )
                replay = replay_candidate(env, agent, layout_seed, args.high_level_interval)
                collisions += int(replay["collision"])
                if not replay["success"]:
                    continue
                if accepted >= wanted:
                    continue
                for decision in replay["decisions"]:
                    arrays["episode_id"].append(episode_id)
                    arrays["step_id"].append(decision["step"])
                    arrays["planner_input"].append(decision["planner_input"])
                    arrays["environment_history"].append(decision["history"])
                    arrays["history_padding_count"].append(decision["history_padding_count"])
                    arrays["reference_target"].append(decision["reference_target"])
                accepted_maps.append({
                    "episode_id": int(episode_id), "scenario": scenario,
                    "layout_seed": layout_seed,
                    "tree_specs": [tree_record(spec) for spec in env._tree_specs],
                })
                accepted += 1
                episode_id += 1
                print(f"{scenario}: accepted={accepted}/{wanted}, seed={layout_seed}, samples={len(replay['decisions'])}")
            finally:
                env.close()
        scenario_stats[scenario] = {
            "attempted_episodes": budget,
            "successful_episodes": accepted,
            "collision_episodes": collisions,
            "samples": int(sum(1 for ep in arrays["episode_id"] if ep >= episode_id - accepted)),
        }
        if accepted != wanted:
            raise RuntimeError(f"{scenario}: only {accepted}/{wanted} successful teacher episodes within budget {budget}.")
    instant_arrays = {name: np.asarray(arrays[name]) for name in ("episode_id", "step_id", "planner_input", "reference_target")}
    instant_metadata = save_reference_dataset(
        args.instant_output,
        **instant_arrays,
        metadata={
            "teacher_checkpoint": args.checkpoint,
            "teacher_controller": "oracle_astar_reference_plus_dslpid",
            "encoder_frozen": True,
            "actor_frozen": True,
            "control_frequency_hz": 120,
            "high_level_frequency_hz": 15.0,
            "scenario_stats": scenario_stats,
            "successful_episodes_only": True,
            "collection_contract": "controlled_temporal_pair_v1",
            "layout_seed_per_episode": accepted_maps,
            "same_rows_as_temporal_archive": True,
            "no_successful_cohort_expansion": True,
            "forest24_candidate_maps_screened": 18,
        },
        split_seed=args.seed,
    )
    splits = apply_split_template(
        Path(args.instant_output),
        np.asarray(arrays["episode_id"], dtype=np.int32),
        np.asarray(arrays["reference_target"], dtype=np.float32),
        Path(args.split_template),
    )
    temporal_metadata = save_temporal_reference_dataset(
        args.temporal_output,
        **{name: np.asarray(values) for name, values in arrays.items()},
        splits=splits,
        metadata={
            "source_instant_dataset": str(args.instant_output),
            "teacher_checkpoint": args.checkpoint,
            "teacher_controller": "oracle_astar_reference_plus_dslpid",
            "encoder_frozen": True,
            "actor_frozen": True,
            "control_frequency_hz": 120,
            "high_level_frequency_hz": 15.0,
            "scenario_stats": scenario_stats,
            "successful_episodes_only": True,
            "collection_contract": "controlled_temporal_pair_v1",
            "layout_seed_per_episode": accepted_maps,
            "same_rows_as_instant_archive": True,
            "no_successful_cohort_expansion": True,
            "forest24_candidate_maps_screened": 18,
        },
    )
    if not np.array_equal(np.asarray(arrays["planner_input"], dtype=np.float32), np.load(Path(args.instant_output) / "samples.npz")["planner_input"]):
        raise AssertionError("Paired instant and temporal archives do not share planner inputs.")
    print(json.dumps({
        "instant_output": args.instant_output,
        "temporal_output": args.temporal_output,
        "instant_samples": instant_metadata["sample_count"],
        "temporal_samples": temporal_metadata["sample_count"],
        "episodes": episode_id,
        "scenario_stats": scenario_stats,
        "history_padding_histogram": np.bincount(np.asarray(arrays["history_padding_count"], dtype=np.int8), minlength=8).tolist(),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
