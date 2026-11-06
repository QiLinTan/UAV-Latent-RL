"""Reconstruct explicit environmental history for the frozen BC dataset.

The source v2 archive stored only a 27-D instantaneous planner input.  This
tool replays the *same* deterministic oracle teacher candidates, matches every
accepted trajectory back to its original episode by input/target equality, and
adds eight high-level environmental frames.  It refuses to write an augmented
archive if any source row cannot be verified; it is not a data-expansion tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.preprocess import preprocess_state
from high_level_planner import (
    FUTURE_TIMES,
    append_temporal_frame,
    extract_environment_temporal_frame,
    extract_planner_input,
    save_temporal_reference_dataset,
)
from scripts.collect_reference_planner_dataset import (
    SCENARIO_CONFIGS,
    OracleReferencePIDTeacher,
    latent_and_action,
    load_teacher,
    make_env,
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_source(root: Path) -> tuple[dict, dict, dict]:
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    with np.load(root / "samples.npz", allow_pickle=False) as values:
        data = {name: np.asarray(values[name]).copy() for name in values.files}
    with np.load(root / "splits.npz", allow_pickle=False) as values:
        splits = {name: np.asarray(values[name], dtype=np.int32).copy() for name in values.files}
    required = {"episode_id", "step_id", "planner_input", "reference_target"}
    if required - set(data):
        raise ValueError(f"Source archive missing {sorted(required - set(data))}")
    return metadata, data, splits


def replay_candidate(env, agent, seed: int, high_level_interval: int) -> dict:
    """Replay exactly the old oracle/PID control trajectory and record history."""

    observation, _ = env.reset(seed=int(seed))
    state = preprocess_state(observation.reshape(-1))
    teacher = OracleReferencePIDTeacher(env)
    positions = [np.asarray(env.pos[0], dtype=np.float32).copy()]
    velocities = [np.asarray(env.vel[0], dtype=np.float32).copy()]
    decisions = []
    history: list[np.ndarray] = []
    step = 0
    info = {}
    terminated = truncated = False
    while not (terminated or truncated):
        # Match source collection's frozen-Encoder calls, even though only the
        # latent at high-level decision instants is retained.
        latent, _ = latent_and_action(agent, state)
        if step % int(high_level_interval) == 0:
            temporal_frame = extract_environment_temporal_frame(state)
            temporal_history, padding = append_temporal_frame(history, temporal_frame)
            decisions.append(
                {
                    "step": int(step),
                    "planner_input": extract_planner_input(latent, state),
                    "position": positions[-1].copy(),
                    "history": temporal_history,
                    "history_padding_count": int(padding),
                }
            )
        action = teacher.action()
        observation, _, terminated, truncated, info = env.step(action.reshape(1, -1))
        state = preprocess_state(observation.reshape(-1))
        positions.append(np.asarray(env.pos[0], dtype=np.float32).copy())
        velocities.append(np.asarray(env.vel[0], dtype=np.float32).copy())
        step += 1
    offsets = np.rint(FUTURE_TIMES * int(env.CTRL_FREQ)).astype(np.int32)
    valid = []
    for decision in decisions:
        indices = int(decision["step"]) + offsets
        if int(indices[-1]) >= len(positions):
            continue
        target = np.concatenate(
            [
                np.asarray(positions, dtype=np.float32)[indices] - decision["position"][None, :],
                np.asarray(velocities, dtype=np.float32)[indices],
            ],
            axis=1,
        ).astype(np.float32)
        valid.append({**decision, "reference_target": target})
    return {
        "success": bool(info.get("success", False)),
        "collision": bool(info.get("collision", False)),
        "done_reason": str(info.get("done_reason", "unknown")),
        "decisions": valid,
    }


def match_episode(source: dict, episode_id: int, replay: dict, atol: float) -> tuple[bool, dict]:
    rows = np.flatnonzero(source["episode_id"] == int(episode_id))
    expected_steps = source["step_id"][rows]
    actual_steps = np.asarray([item["step"] for item in replay["decisions"]], dtype=np.int32)
    if not replay["success"] or not np.array_equal(expected_steps, actual_steps):
        return False, {"source_rows": int(rows.size), "replay_rows": int(actual_steps.size)}
    actual_input = np.asarray([item["planner_input"] for item in replay["decisions"]], dtype=np.float32)
    actual_target = np.asarray([item["reference_target"] for item in replay["decisions"]], dtype=np.float32)
    input_abs = float(np.max(np.abs(actual_input - source["planner_input"][rows])))
    target_abs = float(np.max(np.abs(actual_target - source["reference_target"][rows])))
    return input_abs <= atol and target_abs <= atol, {
        "source_rows": int(rows.size),
        "replay_rows": int(actual_steps.size),
        "input_max_abs_error": input_abs,
        "target_max_abs_error": target_abs,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/reference_planner/latent_affordance_margin_v2")
    parser.add_argument("--output", default="data/reference_planner/latent_affordance_margin_v2_temporal8")
    parser.add_argument("--checkpoint", default="checkpoints/latent_only_affordance_dim16/model_final")
    parser.add_argument("--collection-seed", type=int, default=1000)
    parser.add_argument("--high-level-interval", type=int, default=8)
    parser.add_argument("--atol", type=float, default=2e-5)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.high_level_interval <= 0 or args.atol <= 0.0:
        raise ValueError("high-level interval and atol must be positive.")
    source_root = Path(args.source)
    metadata, source, splits = load_source(source_root)
    source_episode_ids = np.unique(source["episode_id"])
    expected_count = int(metadata["episode_count"])
    if source_episode_ids.size != expected_count or not np.array_equal(source_episode_ids, np.arange(expected_count)):
        raise ValueError("Source episode ids must be contiguous for verified replay matching.")
    if float(metadata.get("high_level_frequency_hz", 0.0)) != 15.0:
        raise ValueError("This augmentation contract requires the source 15-Hz dataset.")
    if str(metadata.get("teacher_controller")) != "oracle_astar_reference_plus_dslpid":
        raise ValueError("Temporal augmentation only supports the frozen oracle/PID source contract.")

    device = torch.device("cpu")
    arrays = {name: [] for name in ("episode_id", "step_id", "planner_input", "environment_history", "history_padding_count", "reference_target")}
    replay_provenance = []
    next_episode_id = 0
    scenario_stats = metadata["scenario_stats"]
    for scenario_index, scenario in enumerate(("clear", "single", "forest24")):
        attempts = int(scenario_stats[scenario]["attempted_episodes"])
        success_target = int(scenario_stats[scenario]["successful_episodes"])
        env = make_env(scenario, 12.0)
        try:
            # The original collector initializes each newly created scenario
            # environment with its unshifted command-line seed before the
            # scenario-specific rollout seeds begin.  Preserve that detail so
            # replay verification catches no false data mismatch.
            initial_observation, _ = env.reset(seed=int(args.collection_seed))
            agent = load_teacher(args.checkpoint, int(preprocess_state(initial_observation.reshape(-1)).size), device)
            accepted = 0
            for attempt in range(attempts):
                seed = int(args.collection_seed) + scenario_index * 100000 + attempt
                replay = replay_candidate(env, agent, seed, args.high_level_interval)
                if not replay["success"]:
                    replay_provenance.append({"scenario": scenario, "seed": seed, "accepted": False, "reason": replay["done_reason"]})
                    continue
                if next_episode_id >= expected_count:
                    raise RuntimeError("Replay found more source-success candidates than expected.")
                matched, detail = match_episode(source, next_episode_id, replay, args.atol)
                if not matched:
                    raise RuntimeError(
                        f"Verified replay mismatch for source episode {next_episode_id}, scenario={scenario}, seed={seed}: {detail}"
                    )
                rows = np.flatnonzero(source["episode_id"] == next_episode_id)
                arrays["episode_id"].extend(source["episode_id"][rows])
                arrays["step_id"].extend(source["step_id"][rows])
                arrays["planner_input"].extend(source["planner_input"][rows])
                arrays["reference_target"].extend(source["reference_target"][rows])
                arrays["environment_history"].extend([item["history"] for item in replay["decisions"]])
                arrays["history_padding_count"].extend([item["history_padding_count"] for item in replay["decisions"]])
                replay_provenance.append({"scenario": scenario, "seed": seed, "accepted": True, "source_episode_id": next_episode_id, **detail})
                accepted += 1
                next_episode_id += 1
            if accepted != success_target:
                raise RuntimeError(f"{scenario}: matched {accepted} source episodes, expected {success_target}.")
        finally:
            env.close()
    if next_episode_id != expected_count:
        raise RuntimeError(f"Matched {next_episode_id} source episodes, expected {expected_count}.")
    for name, values in arrays.items():
        if len(values) != int(metadata["sample_count"]):
            raise RuntimeError(f"{name} count {len(values)} does not match source samples {metadata['sample_count']}.")

    result = save_temporal_reference_dataset(
        args.output,
        **arrays,
        splits=splits,
        metadata={
            "source_dataset": str(source_root),
            "source_samples_sha256": digest(source_root / "samples.npz"),
            "source_metadata_sha256": digest(source_root / "metadata.json"),
            "source_teacher_controller": metadata["teacher_controller"],
            "source_teacher_checkpoint": metadata["teacher_checkpoint"],
            "source_sample_count": int(metadata["sample_count"]),
            "source_episode_count": int(metadata["episode_count"]),
            "collection_seed_reconstructed": int(args.collection_seed),
            "high_level_interval_low_level_steps": int(args.high_level_interval),
            "high_level_frequency_hz": 15.0,
            "verification_atol": float(args.atol),
            "verified_replay_provenance": replay_provenance,
            "no_new_teacher_episodes": True,
        },
    )
    print(json.dumps({
        "output": args.output,
        "samples": result["sample_count"],
        "episodes": result["episode_count"],
        "history_shape": [8, 16],
        "verified_source_samples": result["sample_count"],
        "padding_count_histogram": np.bincount(np.asarray(arrays["history_padding_count"], dtype=np.int8), minlength=8).tolist(),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
