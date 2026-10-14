"""Extract per-episode start-curriculum coverage from TensorBoard events."""

from collections import Counter, defaultdict
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


NAME = "upper_dslpid_recurrent_multitree_terminal_v2_noroute_startcurr_v1_h8_k1_seed"
OUTPUT = Path("analysis_outputs/startcurr_multiseed_20261009")
OUTCOMES = ("success", "collision", "height_bound", "xy_bound", "attitude_bound", "timeout")


def load_episodes(run):
    table = defaultdict(dict)
    for path in sorted(run.glob("events.out.tfevents.*")):
        events = EventAccumulator(str(path), size_guidance={"scalars": 0})
        events.Reload()
        for tag in events.Tags()["scalars"]:
            if tag.startswith(("env/", "episode/")):
                for item in events.Scalars(tag):
                    table[item.step][tag] = float(item.value)
    rows = [dict(step=step, **values) for step, values in sorted(table.items())
            if "env/done_reason/success" in values]
    if not rows:
        raise ValueError(f"No episode scalars found in {run}")
    return rows


def count(rows, tag):
    return sum(row.get(tag, 0.0) > 0.5 for row in rows)


def source_level(rows, level):
    group = [row for row in rows
             if int(round(row["env/start_curriculum_episode_level"])) == level]
    band = [row for row in group if row.get("env/start_curriculum_is_band", 0) > 0.5]
    original = [row for row in group if row.get("env/start_curriculum_is_original", 0) > 0.5]
    return {
        "level": level,
        "episodes": len(group),
        "band_episodes": len(band),
        "original_episodes": len(original),
        "fallback_episodes": len(group) - len(band) - len(original),
        "successes": count(group, "env/done_reason/success"),
        "band_successes": count(band, "env/done_reason/success"),
        "original_successes": count(original, "env/done_reason/success"),
        "guidance_entries": count(group, "env/training_episode/entered_guidance_1p2"),
        "band_guidance_entries": count(band, "env/training_episode/entered_guidance_1p2"),
        "original_guidance_entries": count(original, "env/training_episode/entered_guidance_1p2"),
        "first_step": min((row["step"] for row in group), default=None),
        "last_step": max((row["step"] for row in group), default=None),
    }


def audit_seed(seed):
    name = f"{NAME}{seed}"
    run = Path("runs") / name
    episodes = load_episodes(run)
    periodic = [json.loads(line) for line in (run / "evaluation_metrics.jsonl").read_text().splitlines()
                if line.strip()]
    raw_by_step = {}
    for row in periodic:
        raw_by_step.setdefault(row["step"], row)  # exclude restored-final duplicate
    raw = list(raw_by_step.values())
    selection = json.loads((Path("checkpoints") / name / "model_final_selection.json").read_text())
    success_tag = "env/done_reason/success"
    original_tag = "env/start_curriculum_is_original"
    result = {
        "seed": seed,
        "episodes": len(episodes),
        "last_episode_step": episodes[-1]["step"],
        "last_eval_step": raw[-1]["step"],
        "first_success_step": next((row["step"] for row in episodes if row[success_tag] > 0.5), None),
        "first_original_success_step": next((row["step"] for row in episodes
                                             if row[success_tag] > 0.5 and row.get(original_tag, 0) > 0.5), None),
        "first_full_start_step": next((row["step"] for row in episodes
                                       if row["env/start_curriculum_episode_level"] >= 4), None),
        "training_successes": count(episodes, success_tag),
        "guidance_entries": count(episodes, "env/training_episode/entered_guidance_1p2"),
        "train_outcomes": dict(Counter(next((outcome for outcome in OUTCOMES
                                              if row.get(f"env/done_reason/{outcome}", 0) > 0.5),
                                             "unknown") for row in episodes)),
        "source_counts": {
            "band": count(episodes, "env/start_curriculum_is_band"),
            "original": count(episodes, original_tag),
            "fallback": count(episodes, "env/start_curriculum_is_fallback"),
        },
        "stage": [source_level(episodes, level) for level in range(5)],
        "selected_checkpoint": selection["selected_checkpoint"],
        "selected_checkpoint_step": selection["selected_checkpoint_step"],
        "periodic_eval": {
            "count": len(raw),
            "first_success_step": next((row["step"] for row in raw
                                        if row["summary"].get("eval/success_rate", 0) > 0), None),
            "best_success": max(row["summary"].get("eval/success_rate", 0) for row in raw),
            "final_success": raw[-1]["summary"].get("eval/success_rate"),
            "first_guidance_step": next((row["step"] for row in raw if row["summary"].get(
                "eval/terminal_coverage/guidance_entered_episode_fraction", 0) > 0), None),
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / f"training_seed{seed}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(seed, result["episodes"], result["training_successes"], result["first_success_step"],
          result["first_original_success_step"], flush=True)


if __name__ == "__main__":
    for training_seed in (42, 43, 44, 45):
        audit_seed(training_seed)
