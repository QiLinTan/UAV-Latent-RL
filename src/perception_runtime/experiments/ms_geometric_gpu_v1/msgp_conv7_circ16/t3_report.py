"""Build the T3 160k decision packet from verified saved evaluations."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from t2_eval import read_json, read_lines, write_json
from t3_eval import E12_NODES, NODES, T2_NODES, checkpoints
from train import sha

COLORS = {"t3": "#126a82", "t2": "#b85330", "a1": "#5d8072", "a4": "#7c6399"}
LABELS = {"t3": "T3 dual", "t2": "T2 near-only", "a1": "RV1 A1", "a4": "Original A4"}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_final(root, timeline):
    run = root / "train"
    files = checkpoints(run)
    if set(files) != set(range(40000, 160001, 10000)) or not all(files.values()):
        raise RuntimeError("incomplete requested 40k–160k checkpoint series")
    if [r["step"] for r in timeline] != list(range(2000, 160001, 2000)):
        raise RuntimeError("incomplete 2k probe timeline")
    if not all(r["fixed_actor"]["n"] == 2048 and r["behavior"]["n"] == 2000 for r in timeline):
        raise RuntimeError("incomplete per-window action probe")
    if [p["updates"] for p in timeline[:3]] != [0, 0, 1001] or any(p["updates"] != 2000 for p in timeline[3:]):
        raise RuntimeError("unexpected TD3 update cadence")
    path = run / "checkpoint_160000.pt"
    if sha(path) != files[160000]:
        raise RuntimeError("final checkpoint digest mismatch")
    pack = torch.load(path, map_location="cpu", weights_only=False)
    c = pack["contract"]
    a = pack["agent"]
    if (pack["trainer"]["step"], c["seed"], c["course"], c["use_far_branch"],
            pack["replay"].main.total, a["updates"], a["actor_updates"]) != (
            160000, 43, "original", True, 160000, 155001, 77500):
        raise RuntimeError("160k final checkpoint contract mismatch")
    keys = ("environment", "observation", "rng", "cuda_rng", "action_rng", "replay", "agent")
    if not all(key in pack for key in keys):
        raise RuntimeError("incomplete resume state")
    return dict(step=160000, sha256=files[160000], updates=a["updates"], actor_updates=a["actor_updates"],
                replay_capacity=pack["replay"].main.capacity,
                success_capacity=sum(r.capacity for r in pack["replay"].success), resume_state_fields=list(keys),
                filename=str(path), all_saved_checkpoints=files)


def load_results(output):
    entries = {}
    reference = None
    for model in LABELS:
        scores = {}
        for step in (T2_NODES if model == "t2" else NODES):
            directory = output if model == "t3" else output / "history"
            doc = read_json(directory / f"{model}_{step:06d}.json")
            init = [(r["stage"], r["case"], r["init"]) for r in doc["rows"]]
            if reference is None:
                reference = init
            if len(init) != 80 or init != reference or (doc["step"], doc["model"]) != (step, model):
                raise RuntimeError("T3 comparison failed fixed pairing")
            scores[step] = dict(stages={k: x["counts"] for k, x in doc["stages"].items()},
                                totals=doc["totals"]["counts"],
                                exits=None if model == "t2" else doc["totals"]["xy_exits_by_side"],
                                timeout_median_stall_share=doc["totals"]["timeout_median_stall_window_share"],
                                checkpoint_sha256=doc["checkpoint_sha256"])
        entries[model] = scores
    return entries


def comparisons_figure(entries, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained", sharex=True, sharey=True)
    for stage, ax in enumerate(axes.flat):
        for model, records in entries.items():
            x = np.array(list(records)) / 1000
            y = [row["stages"][str(stage)]["success"] for row in records.values()]
            ax.plot(x, y, "-o", label=LABELS[model], color=COLORS[model], lw=2 if model == "t3" else 1.4)
        ax.set(title=f"Stage {stage}", xlabel="Training steps (thousands)", ylabel="Arrivals out of 20",
               ylim=(-.5, 20.5), xticks=np.array(NODES) / 1000)
        ax.grid(alpha=.24)
    axes[0, 0].legend(loc="best")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def probe_figure(timeline, promotions, path):
    fig, axes = plt.subplots(4, 2, figsize=(13, 13), layout="constrained", sharex=True)
    course, vel, saturation, critic, actor_grad, critic_grad, feature, ratio = axes.flat
    x = np.array([p["step"] for p in timeline]) / 1000
    for header, label in (("start_curriculum_level", "Start level"), ("obstacle_curriculum_stage", "Obstacle stage")):
        course.step(x, [p["course"][header] for p in timeline], where="post", label=label)
    course.set(title="Original curriculum", ylabel="Level / stage")
    course.legend()
    for j, name in enumerate(("vx", "vy", "vz")):
        vel.plot(x, [p["behavior"]["mean"][j] for p in timeline], label=name)
    vel.plot(x, [p["behavior"]["mean_abs_vy"] for p in timeline], "--", label="Mean |vy|")
    vel.set(title="Executed training commands", ylabel="Yaw-frame m/s")
    vel.legend(ncol=2)
    saturation.plot(x, [p["behavior"]["xy_saturated_fraction"] for p in timeline], label="Training")
    saturation.plot(x, [p["fixed_actor"]["xy_saturated_fraction"] for p in timeline], label="Fixed Actor panel")
    sat_y = saturation.twinx()
    sat_y.plot(x, [p["actor_mean_vy_drift_from_step0"] for p in timeline], "--", color="#bf5630", label="Mean vy drift")
    saturation.set(title="XY speed saturation and fixed-panel mean vy", ylabel="Fraction near XY cap")
    sat_y.set_ylabel("vy drift from step 0 (m/s)")
    sat_y.legend(loc="lower right")
    saturation.legend(loc="upper left")
    critic.plot(x, [p["critic_loss_mean"] for p in timeline], color="#126a82", label="Critic loss")
    q = critic.twinx()
    q.plot(x, [p["twin_q_mean"] for p in timeline], color="#bf5630", label="Twin Q mean")
    critic.set(title="Critic training statistics", ylabel="Loss (MSE)")
    q.set_ylabel("Twin Q mean")
    for ax, name in ((actor_grad, "actor"), (critic_grad, "critic")):
        for side, color in (("near", "#126a82"), ("far", "#bf5630")):
            samples = [p["branches"].get(name, {}).get("averaged", {}).get(f"{side}_gradient_l2") for p in timeline]
            ax.plot(x, samples, label=side, color=color)
        ax.set(title=f"{name.title()} branch gradient L2", ylabel="Gradient L2", yscale="log")
        ax.legend()
    for net, style in (("actor", "-"), ("critic", "--")):
        for side, color in (("near", "#126a82"), ("far", "#bf5630")):
            feature.plot(x, [p["branches"].get(net, {}).get("averaged", {}).get(f"{side}_feature_l2") for p in timeline],
                         style, label=f"{net} {side}", color=color)
        ratio.plot(x, [p["branches"].get(net, {}).get("averaged", {}).get("near_over_far_gradient_l2") for p in timeline],
                   style, label=net)
    feature.set(title="Latent output L2", ylabel="Feature L2")
    feature.legend(ncol=2)
    ratio.axhline(1, color="#777777", alpha=.7)
    ratio.set(title="Near / far gradient ratio", ylabel="Ratio", yscale="log")
    ratio.legend()
    for ax in axes.flat:
        for event in promotions:
            ax.axvline(event["step"] / 1000, color="#666666", linestyle=":", alpha=.32)
        ax.grid(alpha=.23)
        ax.set_xlabel("Training steps (thousands)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def e12_figure(e12, path):
    fig, (counter, real) = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    names = [f"{node//1000}k" for node in E12_NODES]
    indices = np.arange(len(names))
    for j, horizon in enumerate(("1", "3")):
        counter.bar(indices + (j - .5) * .35, [r["closed_loop"][horizon]["correct_switch"] for r in e12.values()],
                    .35, label=f"{horizon} s")
    for j, (key, label) in enumerate((("passed", "Passed opening"), ("arrived", "Arrived"))):
        real.bar(indices + (j - .5) * .35, [r["actual_apertures"][key] for r in e12.values()], .35, label=label)
    counter.set(title="Far-only displacement switch", ylabel="Correct pairs out of 12", ylim=(0, 12.5))
    real.set(title="Physical wall scenes", ylabel="Episodes out of 24", ylim=(0, 25))
    for ax in (counter, real):
        ax.set_xticks(indices, names)
        ax.legend(loc="best")
        ax.grid(axis="y", alpha=.2)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    out = args.output
    timeline = read_lines(args.run / "probes_002000.jsonl")
    promotions = read_lines(args.run / "curriculum_promotions.jsonl")
    final = verify_final(args.run, timeline)
    entries = load_results(out)
    e12 = {}
    for step in E12_NODES:
        doc = read_json(out / f"e12_{step:06d}.json")
        if doc["checkpoint_sha256"] != final["all_saved_checkpoints"][step] or len(doc["cases"]) != 12:
            raise RuntimeError("E12 model digest or case count mismatch")
        e12[step] = doc["summary"]
    rows = []
    for model, scores in entries.items():
        for step, row in scores.items():
            values = {f"stage{stage}_arrival": row["stages"][str(stage)]["success"] for stage in range(4)}
            positive_y = row["exits"].get("y+", 0) if row["exits"] is not None else None
            rows.append(dict(step=step, model=model, **values, **row["totals"], world_positive_y_exits=positive_y))
    rows.sort(key=lambda r: (r["step"], r["model"]))
    write_csv(out / "all_node_comparison.csv", rows)
    (out / "probes_002000.jsonl").write_bytes((args.run / "probes_002000.jsonl").read_bytes())
    comparisons_figure(entries, out / "forest_comparison.png")
    probe_figure(timeline, promotions, out / "probe_trends.png")
    e12_figure(e12, out / "e12_trends.png")
    last = entries["t3"][160000]
    stage2, stage3 = [last["stages"][str(s)]["success"] for s in (2, 3)]
    recent = [r for r in timeline if r["step"] >= 120000]
    checkpoint_report = dict(
        checkpoint=final, comparisons=entries, e12=e12, promotions=promotions,
        training_exposure=timeline[-1]["exposure"],
        gate=dict(stage2_arrivals=stage2, stage2_required=14, stage3_arrivals=stage3, stage3_required=8,
                  forest_numeric_gate_met=stage2 >= 14 and stage3 >= 8,
                  t3_late_forest={step: entries["t3"][step] for step in (120000, 140000, 160000)},
                  final_e12=e12[160000], final_gradient_probes=timeline[-1]["branches"],
                  late_behavior_xy_saturation_minmax=[min(r["behavior"]["xy_saturated_fraction"] for r in recent),
                                                     max(r["behavior"]["xy_saturated_fraction"] for r in recent)],
                  late_fixed_panel_xy_saturation_minmax=[min(r["fixed_actor"]["xy_saturated_fraction"] for r in recent),
                                                        max(r["fixed_actor"]["xy_saturated_fraction"] for r in recent)],
                  status="160k reached; decision awaits user confirmation"))
    write_json(out / "gate_160k.json", checkpoint_report)
    write_json(args.run / "gate_160k.json", checkpoint_report)


if __name__ == "__main__":
    main()
