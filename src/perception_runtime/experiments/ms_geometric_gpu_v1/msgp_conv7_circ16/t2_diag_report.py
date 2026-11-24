"""Aggregate the T2-Diag evidence and render comparison figures."""
import argparse
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from t2_eval import read_json, write_json

BLUE = "#146c94"
ORANGE = "#c35329"
GREEN = "#45845d"
PROMOTION = 83745


def fixed_nodes(run, output):
    docs = {step: read_json(run / f"evaluation_{step:06d}.json") for step in (40000, 80000, 120000)}
    for step in (90000, 100000, 110000):
        docs[step] = read_json(output / f"fixed_{step:06d}.json")
    baseline = [(r["stage"], r["case"], r["init"]) for r in docs[80000]["rows"]]
    result = {}
    for step, doc in sorted(docs.items()):
        if [(r["stage"], r["case"], r["init"]) for r in doc["rows"]] != baseline:
            raise RuntimeError(f"non-paired evaluation at {step}")
        totals = {name: sum(s["counts"][name] for s in doc["stages"].values())
                  for name in ("success", "collision", "timeout", "bounds")}
        result[str(step)] = dict(by_stage={k: s["counts"] for k, s in doc["stages"].items()},
                                 totals=totals, reasons=dict(Counter(r["reason"] for r in doc["rows"])),
                                 start_course_level=doc["training"]["course"]["start_curriculum_level"],
                                 exposure=doc["training"]["exposure"])
    return result


def training_figure(replay, nodes, output):
    data = replay["timeline_2k"]
    x = np.array([r["step"] for r in data]) / 1000
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), layout="constrained")
    a, b, c, d, e, f = axes.flat
    for key, label, color in (("success_rate", "Success", GREEN), ("collision_rate", "Collision", ORANGE),
                              ("xy_bound_rate", "XY bound", BLUE)):
        a.plot(x, [r[key] for r in data], label=label, color=color)
    a.set(title="Training episode outcomes per 2,000 steps", ylabel="Fraction of completed episodes")
    a.legend()
    ns = np.array(list(nodes), dtype=int) / 1000
    for key, label, color in (("success", "Success", GREEN), ("collision", "Collision", ORANGE),
                              ("bounds", "Bound", BLUE)):
        b.plot(ns, [r["totals"][key] for r in nodes.values()], "-o", label=label, color=color)
    b.set(title="Fixed evaluation", ylabel="Cases out of 80")
    b.legend()
    for key, label in (("critic_loss_mean", "Mean"), ("critic_loss_median", "Median")):
        c.plot(x, [r[key] for r in data], label=label)
    c.set(title="Critic loss per 2,000 steps", ylabel="MSE")
    c.legend()
    for key, label in (("q_mean", "Twin Q mean"), ("q_absolute_batch_max_p99", "P99 of batch max |Q|")):
        d.plot(x, [r[key] for r in data], label=label)
    d.set(title="Training Q statistics", ylabel="Scaled return")
    d.legend()
    for i, comp in enumerate(("vx", "vy", "vz")):
        e.plot(x, [r["behavior_velocity"]["velocity_mean"][i] for r in data], label=comp)
    e.plot(x, [r["behavior_velocity"]["abs_vy_mean"] for r in data], "--", label="Mean |vy|")
    e.set(title="Executed training commands, with exploration", ylabel="Yaw-frame m/s")
    e.legend(ncol=2)
    panel = sorted((int(k), v["all"]) for k, v in replay["fixed_actor_panel"].items())
    panel_x = np.array([n for n, _ in panel]) / 1000
    f.plot(panel_x, [v["velocity_mean"][1] for _, v in panel], "-o", color=BLUE, label="Mean vy")
    f.set(title="Actor outputs on 2,048 identical inputs", ylabel="Mean yaw-frame vy (m/s)")
    f.legend(loc="upper left")
    sat = f.twinx()
    sat.plot(panel_x, [v["horizontal_saturated_fraction"] for _, v in panel], "--s", color=ORANGE,
             label="XY saturation")
    sat.set_ylabel("Fraction ≥99% of XY speed cap")
    sat.legend(loc="lower left")
    for ax in axes.flat:
        ax.axvline(PROMOTION / 1000, color="#6b7280", linestyle=":", linewidth=1.4)
        ax.grid(alpha=.2)
        ax.set(xlabel="Training steps (thousands)")
    fig.suptitle("T2-Diag | Level 2 → 3 at step 83,745 (dotted line)", fontsize=15)
    fig.savefig(output / "training_and_policy.png", dpi=160)
    plt.close(fig)


def replay_figure(replay, output):
    r0, r1 = (replay["replay"][str(n)] for n in (80000, 120000))
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(14, 4.3), layout="constrained")
    levels = ("Level0", "Level1", "Level2", "Level3")
    cohorts = (r0["main"], r1["main"], r0["success"]["0"], r1["success"]["0"])
    labels = ["80k main", "120k main", "80k success", "120k success"]
    offset = np.zeros(4)
    for color, level in zip(("#88aadd", "#395e84", "#45845d", "#c35329"), levels):
        rates = np.array([g["fractions"]["episode_start_level"][level] for g in cohorts]) * 100
        ax.barh(labels, rates, left=offset, label=level, color=color)
        offset += rates
    ax.set(title="Replay start-course composition", xlabel="% of transitions")
    ax.legend(ncol=2, loc="lower right")
    ax.invert_yaxis()
    x = np.arange(4)
    keys = list(r0["main"]["counts"]["distance"])
    for i, (node, group, color) in enumerate((("80k", r0["main"], BLUE), ("120k", r1["main"], ORANGE))):
        o = -.2 + i * .4
        bx.bar(x + o, [100 * group["fractions"]["distance"][k] for k in keys], .4, label=node, color=color)
        reasons = ("collision", "xy_bound", "height_bound", "timeout")
        cx.bar(x + o, [group["counts"]["terminal_reason"][k] for k in reasons], .4, label=node, color=color)
    bx.set(xticks=x, xticklabels=keys, title="Distance to goal in main Replay", ylabel="% of transitions")
    cx.set(xticks=x, xticklabels=["Collision", "XY bound", "Z bound", "Timeout"],
           title="Failed endings in main Replay", ylabel="Transitions")
    for panel in (bx, cx):
        panel.legend()
        panel.grid(axis="y", alpha=.2)
    fig.savefig(output / "replay_mix.png", dpi=160)
    plt.close(fig)


def traces_figure(entries, traces, output):
    fig, axes = plt.subplots(3, 5, figsize=(15, 8.5), layout="constrained", sharex=True, sharey=True)
    for ax, item, data in zip(axes.flat, entries, traces):
        for key, color in (("old", BLUE), ("new", ORANGE)):
            xy = np.array([r["position_m"][:2] for r in data[key]["rows"]] +
                          [data[key]["rows"][-1]["next_position_m"][:2]])
            ax.plot(*xy.T, color=color, lw=1.4, label="80k" if key == "old" else "120k")
            ax.scatter(*xy[-1], c=color, marker="x", s=24)
        bx, by = data["new"]["bounds_xy_m"]
        ax.plot([-bx, bx, bx, -bx, -bx], [-by, -by, by, by, -by], color="#8b9095", linestyle=":", lw=.9)
        ax.scatter(*data["new"]["goal_m"][:2], marker="*", color=GREEN, s=40)
        ax.set(title=f"S{item['stage']} / seed {item['case']:02d} / {item['failure']}", aspect="equal")
        ax.label_outer()
    axes[0, 0].legend(loc="upper left")
    fig.supxlabel("World X (m)")
    fig.supylabel("World Y (m)")
    fig.suptitle("Same starting state, 80k success and 120k failure")
    fig.savefig(output / "paired_paths.png", dpi=165)
    plt.close(fig)


def critics_figure(entries, traces, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained", sharey=True)
    for ax, reason, title in zip(axes, ("xy_bound", "collision"), ("XY bound (10 cases)", "Collision (5 cases)")):
        tails = [data["new"]["rows"][-15:] for item, data in zip(entries, traces) if item["failure"] == reason]
        if any(len(t) != 15 for t in tails):
            raise RuntimeError("critic graph needs 15 steps per selected failure")
        x = np.arange(-14, 1)
        for node, color in (("80000", BLUE), ("120000", ORANGE)):
            array = np.array([[r["critics"][node]["minimum"] for r in seq] for seq in tails])
            ax.plot(x, array.mean(axis=0), color=color, label=f"{int(node)//1000}k minimum Q")
        actual = np.array([[r["realized_discounted_return"] for r in seq] for seq in tails])
        ax.plot(x, actual.mean(axis=0), "--", color="#535960", label="Observed discounted return")
        ax.set(title=title, xlabel="Upper control steps before terminal action", ylabel="Return on training scale")
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle("Critics scored the same 120k executed actions and observations")
    fig.savefig(output / "critic_cross_scores.png", dpi=160)
    plt.close(fig)


def characterize(items, traces):
    groups = {}
    for reason in ("xy_bound", "collision"):
        cases = [r["summary"] for r in items if r["failure"] == reason]
        divergences = [r["first_half_meter_lateral_divergence"] for r in cases]
        present = [d for d in divergences if d is not None]
        last_vy = [np.mean([r["command_world_mps"][1] for r in data["new"]["rows"][-15:]])
                   for item, data in zip(items, traces) if item["failure"] == reason]
        groups[reason] = dict(n=len(cases), crossed_world_bounds=dict(Counter(
                              side for r in cases for side in r["crossed_world_bounds"])),
                              divergence_median_s=float(np.median([d["time_s"] for d in present])),
                              divergence_over_8m_from_goal=sum(d["goal_distance_m"] > 8 for d in present),
                              divergence_before_first_near_pass=sum(d is not None and (
                                  r["first_clearance_at_most_half_m_s"] is None or
                                  d["time_s"] < r["first_clearance_at_most_half_m_s"])
                                  for d, r in zip(divergences, cases)),
                              first15_mean_lateral_command_positive=sum(
                                  r["initial_commands"]["120000"]["lateral_world_first_15_mps"] > 0 for r in cases),
                              last15_mean_world_vy_positive=sum(bool(v > 0) for v in last_vy),
                              last15_world_vy_mean_mps=float(np.mean(last_vy)),
                              terminal_remaining_over_8m=sum(r["terminal_goal_distance_m"] > 8 for r in cases))
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output
    replay = read_json(output / "replay_and_timeline.json")
    trace_summary = read_json(output / "paired_trace_summary.json")
    entries = trace_summary["cases"]
    traces = [read_json(output / "paired_trajectories" / item["trajectory"]) for item in entries]
    nodes = fixed_nodes(args.run, output)
    training_figure(replay, nodes, output)
    replay_figure(replay, output)
    traces_figure(entries, traces, output)
    critics_figure(entries, traces, output)
    q_late = [r for r in replay["timeline_2k"] if r["step"] >= 80000]
    doc = dict(
        scope="Saved checkpoints and logs only, plus evaluation rollouts; original offline diagnostics on an online TD3 run",
        level2_to_level3_step=PROMOTION, fixed_evaluation=nodes, paired_trajectory_findings=characterize(entries, traces),
        critic_cross_scores=trace_summary["critic_groups"],
        late_training_q_batch_abs_max_p99_range=[min(r["q_absolute_batch_max_p99"] for r in q_late),
                                                  max(r["q_absolute_batch_max_p99"] for r in q_late)],
        replay_composition=replay["replay"],
        fixed_panel_actor={k: r for k, r in replay["fixed_actor_panel"].items() if int(k) >= 80000},
        assessment={
            "onset": "固定评估在 80k 到 90k 首次回落，100k 越界暂时消失，110k 和 120k 继续恶化。课程晋升与首次回退处于同一检查点区间。",
            "replay": "Level0–1 和近目标覆盖显著减少，碰撞与水平越界终止样本合计只少了两条。Replay 的分布变化真实存在，现有计数不足以证明负样本遗忘。",
            "policy": "配对失效轨迹出现方向性侧逸，越界抽样集中从世界坐标 +Y 离场。第一秒的侧向指令普遍没有统一的正向符号。",
            "critic": "120k Critic 在全部 15 个失败末段的平均最低 Q 均低于 80k。两版在终止动作上仍对实得终止回报偏乐观。",
            "decision": "现有证据支持后期策略退化，Replay、课程、观测和优化的因果影响尚未分离。沿用已触发的 T2 门禁。"
        },
        evidence_limits=[
            "Checkpoint intervals bracket onset; they cannot establish an abrupt change at step 83,745.",
            "Each fixed test begins far from the goal, and there is no paired short-start intervention.",
            "Near-obstacle proxy excludes possible floor voxels; archived occupancy does not identify hit objects.",
            "Wide training-Q statistics are replay-conditional and do not test every state-action stratum.",
            "Failure tails were deliberately selected; the critic comparison is conditional on those rollouts.",
            "Both critics can be optimistic on terminal actions even when 120k estimates are lower than 80k.",
            "No counterfactual training was run to isolate curriculum, replay, observation, or optimizer effects."
        ],
        files=dict(timeline="replay_and_timeline.json", pairs="paired_trace_summary.json",
                   trajectories="paired_trajectories/", charts=["training_and_policy.png", "replay_mix.png",
                                                                "paired_paths.png", "critic_cross_scores.png"])
    )
    write_json(output / "diagnostic_report.json", doc)


if __name__ == "__main__":
    main()
