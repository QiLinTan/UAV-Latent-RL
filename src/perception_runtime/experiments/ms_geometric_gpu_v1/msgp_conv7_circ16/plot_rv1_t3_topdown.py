"""Replay and draw matched top-down forest trajectories from frozen checkpoints."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle
import numpy as np

from evaluate import make_validation_env, realtime_decisions
from t0_eval import follow, initial_info
from t2_diag_traces import xy_bounds
from t2_eval import read_json
import t3_eval as evaluator
from train import sha

POLICIES = (("a1", 200000), ("t3", 200000), ("a1", 240000), ("t3", 240000))
NAMES = {"a1": "RV1 A1", "t3": "T3 dual"}
COLORS = {"a1": "#007079", "t3": "#2858b6"}
STATUS = {"success": ("Arrived", "#267748"), "collision": ("Collision", "#cc4236"),
          "xy_bound": ("Horizontal bound", "#ac5728"), "height_bound": ("Height bound", "#ac5728"),
          "timeout": ("Timeout", "#a66a29")}


def select_cases(results):
    def first(stage, reasons):
        for case in range(20):
            actual = tuple(results[key][stage, case]["reason"] for key in POLICIES)
            if actual == reasons:
                return stage, case
        raise RuntimeError(f"No matching Stage {stage} trajectory example")

    return (
        first(0, ("success", "success", "success", "xy_bound")),
        first(3, ("collision", "success", "success", "success")),
    )


def trace(model, node, checkpoint, fingerprint, expected):
    evaluator.init_worker(model, checkpoint, fingerprint)
    stage, case = expected["stage"], expected["case"]
    env = make_validation_env(evaluator.contract)
    try:
        env.obstacle_curriculum.stage = stage
        obs, info = env.reset(seed=case)
        if initial_info(env, info) != expected["init"]:
            raise RuntimeError("trajectory initialization did not match the saved evaluation")
        plant = env.base_env
        spec = plant.MAP_SPEC
        trees = [(o.position[0], o.position[1], o.radius) for o in spec.obstacles if o.kind == "cylinder"]
        if len(trees) != len(spec.obstacles):
            raise RuntimeError("unsupported forest obstacle footprint")
        path = [plant.pos[0].tolist()]
        original = env.step

        def tracked(action):
            outcome = original(action)
            path.extend(t["position"] for t in env.last_ticks)
            return outcome

        env.step = tracked
        with realtime_decisions():
            row, _ = follow(env, evaluator.agent, evaluator.builder, obs, info)
        if (evaluator.step != node or row["reason"] != expected["reason"]
                or row["upper_steps"] != expected["upper_steps"]
                or not np.allclose(path[-1], expected["final_position_m"], atol=1e-6, rtol=0)):
            raise RuntimeError(f"trajectory did not reproduce {model}/{node}/{stage}/{case}")
        return dict(path=path, start=path[0], goal=plant.TARGET_POS.tolist(), trees=trees,
                    bounds=xy_bounds(plant).tolist(), reason=row["reason"])
    finally:
        env.close()


def render(stage, case, curves, output):
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.5), layout="constrained", sharex=True, sharey=True)
    for ax, (model, node) in zip(axes.flat, POLICIES):
        row = curves[model, node]
        path = np.asarray(row["path"])
        x_bound, y_bound = row["bounds"]
        ax.set_facecolor("#f7fafb")
        for x, y, radius in row["trees"]:
            ax.add_patch(Circle((x, y), radius, facecolor="#aebbb4", edgecolor="#648075", lw=.5, zorder=2))
        ax.add_patch(Rectangle((-x_bound, -y_bound), 2 * x_bound, 2 * y_bound, fill=False,
                               edgecolor="#778188", linestyle=(0, (3, 3)), lw=1, zorder=2))
        ax.plot(path[:, 0], path[:, 1], color=COLORS[model], linewidth=1.75, zorder=3)
        ax.scatter(*row["start"][:2], color="#21272b", marker="o", s=22, zorder=4)
        ax.scatter(*row["goal"][:2], color="#23a269", edgecolors="#1d6f4f", linewidths=.6,
                   marker="*", s=100, zorder=4)
        label, color = STATUS[row["reason"]]
        if row["reason"] == "success":
            ax.scatter(*path[-1, :2], facecolors="none", edgecolors=COLORS[model],
                       marker="o", s=102, lw=1.4, zorder=5)
        else:
            ax.scatter(*path[-1, :2], color=color, marker="X", edgecolors="white", linewidths=.7, s=65, zorder=5)
        ax.set_title(f"{NAMES[model]} · {node//1000}k · {label}", color=color, fontweight="semibold", fontsize=11)
        ax.set(xlim=(-x_bound - .55, x_bound + .55), ylim=(-y_bound - .35, y_bound + .35),
               aspect="equal", xticks=(-8, -4, 0, 4, 8), yticks=(-4, -2, 0, 2, 4))
        ax.grid(alpha=.22, color="#9da9ae")
    for ax in axes[-1]:
        ax.set_xlabel("World X (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel("World Y (m)")
    handles = [Patch(facecolor="#aebbb4", edgecolor="#648075", label="Trees"),
               Line2D([], [], color="#778188", linestyle="--", label="Map boundary"),
               Line2D([], [], color="#21272b", linestyle="", marker="o", label="Start"),
               Line2D([], [], color="#23a269", linestyle="", marker="*", markersize=10, label="Goal"),
               Line2D([], [], color="#ac5728", linestyle="", marker="X", label="Failed endpoint")]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle(f"Matched forest trajectories · Stage {stage} · Scenario seed {case:02d}",
                 fontsize=14, fontweight="semibold")
    fig.savefig(output / f"stage{stage}_seed{case:02d}_topdown.png", dpi=170, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    comparison = read_json(args.results / "rv1_comparison/comparison.json")
    pool = read_json(args.run / "train/metadata.json")["contract"]["pool_fingerprint"]
    entries, checks = {}, {}
    for model, node in POLICIES:
        summary = comparison["nodes"][str(node)][model]
        if model == "t3":
            checkpoint = args.run / f"train/checkpoint_{node:06d}.pt"
            rows = read_json(args.results / f"t3_{node:06d}.json")["rows"]
        else:
            checkpoint = args.run.parent / f"formal_gpu_a1_seed43_240k_20261118a/checkpoint_{node:06d}.pt"
            rows = read_json(args.results / f"rv1_comparison/a1_{node:06d}.json")["forest"]
        if sha(checkpoint) != summary["checkpoint_sha256"]:
            raise RuntimeError("trajectory checkpoint checksum mismatch")
        checks[model, node] = checkpoint
        entries[model, node] = {(row["stage"], row["case"]): row for row in rows}
    selected = select_cases(entries)
    paired = {key: {} for key in selected}
    with ProcessPoolExecutor(max_workers=8, mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = {}
        for key in selected:
            initial = [entries[policy][key]["init"] for policy in POLICIES]
            if any(x != initial[0] for x in initial):
                raise RuntimeError("model trajectories are not paired")
            for model, node in POLICIES:
                policy = (model, node)
                job = executor.submit(trace, model, node, checks[policy], pool, entries[policy][key])
                futures[job] = key, policy
        for job in as_completed(futures):
            key, policy = futures[job]
            paired[key][policy] = job.result()
    for (stage, case), curves in paired.items():
        render(stage, case, curves, args.output)
        print(f"Verified all four trajectories for Stage {stage}, seed {case:02d}", flush=True)


if __name__ == "__main__":
    main()
