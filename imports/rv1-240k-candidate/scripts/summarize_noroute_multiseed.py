"""Summarize the four no_route runs and their matched validation diagnostics."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.summarize_spatial_q import analyze


BASE = "upper_dslpid_recurrent_multitree_terminal_v2_noroute_h8_k1"
OUT = Path("analysis_outputs/noroute_multiseed_20261008")


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for seed in (42, 43, 44, 45):
        name = BASE if seed == 42 else f"{BASE}_seed{seed}"
        formal = f"{name}_formal_eval" + ("" if seed == 42 else "_20261008")
        qdir = Path("analysis_outputs") / (
            "spatial_q_scan_noroute_20260924" if seed == 42 else
            f"spatial_q_scan_noroute_seed{seed}_20261008")
        evaluation = lines(f"runs/{formal}/evaluation_metrics.jsonl")[-1]
        coverage = read(f"analysis_outputs/{formal}_coverage/terminal_coverage_latest.json")
        timeout = read(f"analysis_outputs/{formal}_timeout/timeout_diagnostics_latest.json")
        training = read(OUT / f"training_seed{seed}.json")
        selection = read(f"checkpoints/{name}/model_final_selection.json")
        periodic = {}
        for row in lines(f"runs/{name}/evaluation_metrics.jsonl"):
            periodic.setdefault(row["step"], row)
        periodic = list(periodic.values())
        qrecords = read(qdir / "q_scan.json")
        cal = lines(qdir / "calibration.jsonl")
        actions = lines(qdir / "actor_actions.jsonl")
        diagnostics = analyze(qrecords, cal, actions)
        if seed != 42:
            (qdir / "analysis.json").write_text(json.dumps(diagnostics, indent=2, allow_nan=False)+"\n")
        summary = read(qdir / "summary.json")
        speeds = [np.linalg.norm(r["command"][:2]) for r in actions]
        norms = [np.linalg.norm(r["actor_action"][:2]) for r in actions]
        actual = [np.linalg.norm(r["actual_velocity"][:2]) for r in actions]
        closest = [r for r in qrecords if "closest_preterminal" in r["case"]["labels"]]
        selected = {}
        for key in ("q1", "q_min"):
            selected[key] = dict(
                closest_best_family_counts=dict(Counter(max(r["candidates"], key=lambda c:c[key])["family"] for r in closest)),
                closest_best_z_counts=dict(Counter(str(round(max(r["candidates"], key=lambda c:c[key])["z_command_mps"], 2)) for r in closest)))
        values = dict(
            training_seed=seed, checkpoint_step=selection["selected_checkpoint_step"],
            selection=selection["selected_checkpoint"],
            outcomes=dict(Counter(r["done_reason"] for r in evaluation["episodes"])),
            metrics={k:v for k,v in evaluation["summary"].items() if any(t in k for t in (
                "success_rate", "collision_rate", "final_goal_distance", "final_z", "mean_abs_z_error", "length"))},
            coverage=coverage["summary"],
            timeout_classes=dict(Counter(r["classification"]["code"] for r in timeout["timeouts"])),
            training=training["training_episode_coverage"],
            success_replay=training["success_replay_coverage"],
            periodic_eval_count=len(periodic), last_periodic_step=periodic[-1]["step"],
            periodic_best_success=max(r["summary"]["eval/success_rate"] for r in periodic),
            periodic_best_guidance=max(r["summary"].get("eval/terminal_coverage/guidance_entered_episode_fraction", 0) for r in periodic),
            training_curriculum_stages=sorted(set(r["curriculum"]["curriculum_stage"] for r in periodic)),
            action_audit=dict(transition_count=len(actions), maximum_mapping_error=summary["maximum_mapping_error"],
                mean_command_xy=float(np.mean(speeds)), mean_actual_xy=float(np.mean(actual)),
                saturated_fraction=float(np.mean(np.array(norms)>=.95)),
                away_command_fraction=float(np.mean([r["command_closing_speed"]<0 for r in actions])),
                mean_z_command=float(np.mean([r["command"][2] for r in actions]))),
            q_summary=summary, q_diagnostics=diagnostics, closest_q_preferences=selected)
        results.append(values)
        axes[0].plot([r["step"] for r in periodic], [r["summary"]["eval/success_rate"] for r in periodic], label=str(seed))
        axes[1].plot([r["step"] for r in periodic], [r["summary"].get("eval/terminal_coverage/guidance_entered_episode_fraction", 0) for r in periodic], label=str(seed))
    successes = [r["outcomes"].get("success", 0)/20 for r in results]
    combined = dict(runs=results, aggregate=dict(
        training_seed_count=4, success_rates=successes,
        mean_success_rate=float(np.mean(successes)),
        sample_std_success_rate=float(np.std(successes, ddof=1)),
        median_success_rate=float(np.median(successes)),
        caveat="Four training runs, same validation seeds 0..19; selected checkpoints. Do not treat 80 episodes as independent training replications. Validation seeds 0..9 also used for checkpoint selection."))
    (OUT / "summary.json").write_text(json.dumps(combined, indent=2, allow_nan=False)+"\n")
    axes[2].bar([str(r["training_seed"]) for r in results], successes)
    axes[0].set(title="Periodic validation success", xlabel="Training steps", ylabel="Fraction", ylim=(-.03, 1.03))
    axes[1].set(title="Periodic guidance entry", xlabel="Training steps", ylabel="Fraction", ylim=(-.03, 1.03))
    axes[2].set(title="Selected checkpoint: 20 episodes", xlabel="Training seed", ylabel="Success fraction", ylim=(0, 1))
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].legend(title="Training seed")
    axes[1].legend(title="Training seed")
    fig.savefig(OUT / "multiseed_comparison.png", dpi=160)
    plt.close(fig)
    print(json.dumps(combined["aggregate"], indent=2))


if __name__ == "__main__":
    main()
