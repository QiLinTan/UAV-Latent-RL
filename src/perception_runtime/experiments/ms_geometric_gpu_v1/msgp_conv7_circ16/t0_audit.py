"""Export the four historical 40k runs from their raw logs."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = (
    ("Directional comparison", "directional_comparison_seed43_20261119a"),
    ("E11 mixed", "directional_mixed_seed43_40k_20261119a"),
)
MODELS = (("A4", "original"), ("Directional", "directional"))


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def csv_write(path, records):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def stats(group, name):
    values = np.array([r[name] for r in group], np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"nonfinite {name}")
    return dict(mean=float(values.mean()), p95=float(np.quantile(values, .95)),
                maximum=float(values.max()))


def diagnose(source, evaluation):
    episodes = rows(source / "episodes.jsonl")
    updates = rows(source / "updates.jsonl")
    metrics = rows(source / "metrics.jsonl")
    if [r["step"] for r in updates] != list(range(5000, 40001)) or metrics[-1]["step"] != 40000:
        raise RuntimeError(f"incomplete 40k history: {source}")
    if [r["episode"] for r in episodes] != list(range(len(episodes))):
        raise RuntimeError(f"episode discontinuity: {source}")
    bins = []
    for end in range(1000, 40001, 1000):
        batch = [r for r in updates if end - 1000 < r["step"] <= end]
        completed = [r for r in episodes if end - 1000 < r["step"] <= end]
        bins.append(dict(
            step=end, episodes=len(completed), updates=len(batch),
            episode_reward=float(np.mean([r["reward"] for r in completed])) if completed else None,
            critic_loss=stats(batch, "critic_loss")["mean"] if batch else None,
            q_mean=float(np.mean([(r["q1_mean"] + r["q2_mean"]) / 2 for r in batch])) if batch else None,
            collision_rate=sum(r["collision"] for r in completed) / len(completed) if completed else None,
            timeout_rate=sum(r["reason"] == "timeout" for r in completed) / len(completed) if completed else None,
        ))
    summary = dict(
        episodes=len(episodes), training_episode_reasons=dict(Counter(r["reason"] for r in episodes)),
        training_collision_observed=sum(r["collision"] for r in episodes),
        updates=len(updates), critic_loss=stats(updates, "critic_loss"), q_abs_max=stats(updates, "q_abs_max"),
        frozen_reasons=dict(Counter(r["reason"] for r in evaluation)),
        frozen_collision_observed=sum(r["collision_observed"] for r in evaluation),
        frozen_by_stage={
            str(i): dict(Counter(r["reason"] for r in evaluation if r["stage"] == i))
            for i in range(4)
        },
    )
    return bins, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    labels = dict(episode_reward="Mean episode reward", critic_loss="Critic loss (MSE)",
                  q_mean="Twin Q means", collision_rate="Observed collision rate", timeout_rate="Timeout rate")
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), layout="constrained")
    axes.flat[-1].axis("off")
    combined, summaries = [], {}
    for course, dirname in RUNS:
        run = args.runs / dirname
        frozen = json.loads((run / "evaluation_40000/results.json").read_text())
        for model, mode in MODELS:
            label = f"{course} / {model}"
            trial = [r for r in frozen["forest"] if r["mode"] == mode]
            if len(trial) != 80:
                raise RuntimeError(f"incomplete frozen trial: {label}")
            bins, summary = diagnose(run / mode, trial)
            summaries[label] = dict(source=f"{dirname}/{mode}", **summary)
            combined.extend(dict(course=course, model=model, **r) for r in bins)
            for ax, (metric, title) in zip(axes.flat, labels.items()):
                ax.plot([r["step"] / 1000 for r in bins], [r[metric] for r in bins], label=label)
                ax.set(xlabel="Training steps (thousands)", title=title)
                ax.grid(alpha=.2)
    axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(), loc="center")
    fig.suptitle("Seed 43, 1,000-step bins. Episode metrics use completed episodes.")
    fig.savefig(args.output / "t0_curves.png", dpi=170)
    csv_write(args.output / "t0_curves.csv", combined)
    (args.output / "t0_historical.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
