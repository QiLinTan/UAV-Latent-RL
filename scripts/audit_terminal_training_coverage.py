#!/usr/bin/env python3
"""Audit successful terminal transitions and success replay from training logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _scalar_series(accumulators, tag: str):
    values = []
    for accumulator in accumulators:
        if tag in accumulator.Tags().get("scalars", []):
            values.extend(accumulator.Scalars(tag))
    return sorted(values, key=lambda item: (item.step, item.wall_time))


def _sum_binary(accumulators, tag: str) -> int:
    return int(round(sum(float(item.value) for item in _scalar_series(accumulators, tag))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {run_dir}")

    accumulators = []
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file))
        accumulator.Reload()
        accumulators.append(accumulator)

    successes = _sum_binary(accumulators, "env/success")
    done_reasons = {
        reason: _sum_binary(accumulators, f"env/done_reason/{reason}")
        for reason in (
            "success",
            "collision",
            "xy_bound",
            "height_bound",
            "attitude_bound",
            "timeout",
        )
    }
    episode_count = sum(done_reasons.values())

    scalar_tags = sorted(
        {
            tag
            for accumulator in accumulators
            for tag in accumulator.Tags().get("scalars", [])
        }
    )
    size_tags = [
        tag
        for tag in scalar_tags
        if tag.startswith("success_replay/") and tag.endswith("_size")
    ]
    sample_tags = [
        tag
        for tag in scalar_tags
        if tag.startswith("success_replay/") and tag.endswith("_sample_count")
    ]

    def tag_stats(tag):
        series = _scalar_series(accumulators, tag)
        return {
            "observation_count": int(len(series)),
            "last": float(series[-1].value) if series else None,
            "maximum": float(max(item.value for item in series)) if series else None,
            "sum": float(sum(item.value for item in series)) if series else None,
        }

    success_replay_sizes = {tag: tag_stats(tag) for tag in size_tags}
    success_replay_samples = {tag: tag_stats(tag) for tag in sample_tags}
    main_replay = tag_stats("env/replay_size")
    replay_snapshots = sorted(
        str(path.relative_to(checkpoint_dir))
        for path in checkpoint_dir.rglob("*")
        if path.is_file()
        and ("replay" in path.name.lower() or "buffer" in path.name.lower())
    )

    success_replay_max_size = int(
        round(sum((item["maximum"] or 0.0) for item in success_replay_sizes.values()))
    )
    success_replay_total_samples = int(
        round(sum((item["sum"] or 0.0) for item in success_replay_samples.values()))
    )
    if successes == 0 and success_replay_max_size == 0:
        conclusion = (
            "No successful training episode occurred, so no successful terminal "
            "transition or successful episode trajectory entered success replay."
        )
    else:
        conclusion = (
            "Successful episodes or success replay entries exist; inspect per-tag "
            "statistics and a replay snapshot for transition-level composition."
        )

    payload = {
        "schema_version": "upper_terminal_training_coverage_v1",
        "sources": {
            "run_dir": str(run_dir),
            "event_files": [str(path) for path in event_files],
            "checkpoint_dir": str(checkpoint_dir),
        },
        "training_episode_coverage": {
            "episode_count": int(episode_count),
            "successful_episode_count": int(successes),
            "successful_episode_fraction": (
                float(successes / episode_count) if episode_count else None
            ),
            "successful_terminal_transition_count": int(successes),
            "done_reason_counts": done_reasons,
        },
        "success_replay_coverage": {
            "maximum_logged_size": success_replay_max_size,
            "logged_sample_count_sum": success_replay_total_samples,
            "size_tags": success_replay_sizes,
            "sample_count_tags": success_replay_samples,
        },
        "main_replay": main_replay,
        "replay_checkpoint": {
            "available": bool(replay_snapshots),
            "matching_files": replay_snapshots,
        },
        "recoverability": {
            "successful_terminal_transition_coverage": "exact_from_episode_logs",
            "success_replay_coverage": (
                "exact_zero_from_logged_stratum_sizes"
                if success_replay_max_size == 0
                else "periodic_size_and_sampling_observations_only"
            ),
            "historical_main_replay_near_goal_composition": (
                "unavailable_without_replay_snapshot"
                if not replay_snapshots
                else "available_from_replay_snapshot"
            ),
        },
        "conclusion": conclusion,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
