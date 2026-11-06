"""Summarize start-distance curriculum training and matched 20-seed evaluations.

Training-episode JSON inputs are produced from TensorBoard scalar events by the
start-curriculum audit. Validation episodes 0..9 were used for checkpoint
selection; 10..19 are held out from that selection, but share the map family.
"""

from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("analysis_outputs/startcurr_multiseed_20261009")
NAME = "upper_dslpid_recurrent_multitree_terminal_v2_noroute_startcurr_v1_h8_k1_seed"


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def summarize():
    ROOT.mkdir(parents=True, exist_ok=True)
    previous = read("analysis_outputs/noroute_multiseed_20261008/summary.json")
    old_rates = {int(row["training_seed"]): row["outcomes"].get("success", 0) / 20
                 for row in previous["runs"]}
    runs = []
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True)
    for seed in (42, 43, 44, 45):
        name = f"{NAME}{seed}"
        formal = f"{name}_formal_eval_20261009"
        training = read(ROOT / f"training_seed{seed}.json")
        selection = read(Path("checkpoints") / name / "model_final_selection.json")
        rows = jsonl(Path("runs") / name / "evaluation_metrics.jsonl")
        selected_periodic = next(row for row in rows
                                 if row["step"] == selection["selected_checkpoint_step"])
        periodic = {}
        for row in rows:
            periodic.setdefault(row["step"], row)  # final restored evaluation shares a step
        periodic = list(periodic.values())
        evaluation = jsonl(Path("runs") / formal / "evaluation_metrics.jsonl")[-1]
        coverage = read(Path("analysis_outputs") / f"{formal}_coverage" /
                        "terminal_coverage_latest.json")
        stage0 = read(ROOT / f"stage0_original_seed{seed}.json")
        old_stage0 = read(Path("analysis_outputs/stage0_feasibility_20261008") /
                          f"frozen_seed{seed}.json")
        collision = read(Path("analysis_outputs") / f"{formal}_collision" /
                         "collision_windows_latest.json")
        outcomes = dict(Counter(row["done_reason"] for row in evaluation["episodes"]))
        withheld = dict(Counter(row["done_reason"] for row in evaluation["episodes"]
                                if 10 <= row["episode"] < 20))
        q_dir = Path(f"analysis_outputs/startcurr_spatial_q_seed{seed}_20261009")
        q_scan = None
        if (q_dir / "summary.json").exists() and (q_dir / "analysis.json").exists():
            q = read(q_dir / "summary.json")
            a = read(q_dir / "analysis.json")
            q_scan = {
                "state_count": q["all_states"]["state_count"],
                "calibration_state_count": q["calibration_state_count"],
                "calibration_rollout_count": q["calibration_rollout_count"],
                "actor_transition_count": q["actor_transition_count"],
                "maximum_mapping_error": q["maximum_mapping_error"],
                "calibration_outcomes": q["calibration_outcomes"],
                "q1_concordant_pairs": a["calibration"]["q1"]["concordant_pairs"],
                "q1_discordant_pairs": a["calibration"]["q1"]["discordant_pairs"],
                "qmin_concordant_pairs": a["calibration"]["q_min"]["concordant_pairs"],
                "qmin_discordant_pairs": a["calibration"]["q_min"]["discordant_pairs"],
                "action_distance_bins": a["action_distance_bins"],
            }
        runs.append({
            "training_seed": seed,
            "selected_step": selection["selected_checkpoint_step"],
            "selected_checkpoint": selection["selected_checkpoint"],
            "selected_start_curriculum_level": selected_periodic["curriculum"]["start_curriculum_level"],
            "training": {key: training[key] for key in (
                "episodes", "last_episode_step", "last_eval_step", "first_success_step",
                "first_full_start_step", "training_successes", "guidance_entries",
                "train_outcomes", "source_counts", "stage", "periodic_eval")},
            "periodic_raw_count": len(periodic),
            "periodic_last_raw_success_rate": periodic[-1]["summary"]["eval/success_rate"],
            "periodic_raw_zero_success_count": sum(
                row["summary"]["eval/success_rate"] == 0 for row in periodic),
            "formal_outcomes": outcomes,
            "formal_heldout_10_19_outcomes": withheld,
            "formal_success_rate": outcomes.get("success", 0) / 20,
            "formal_mean_final_distance_m": evaluation["summary"]["eval/final_goal_distance"],
            "formal_mean_abs_z_error_m": evaluation["summary"]["eval/mean_abs_z_error"],
            "formal_guidance_entry_count": coverage["summary"]["guidance_entered_episode_count"],
            "formal_success_transition_count": coverage["summary"]["successful_terminal_transition_count"],
            "stage0_original_outcomes": stage0["summary"]["outcomes"],
            "stage0_original_guidance_entries": stage0["summary"]["entered_1p2"],
            "old_stage0_original_outcomes": old_stage0["summary"]["outcomes"],
            "collision_classes": dict(Counter(item["classification"]["code"]
                                              for item in collision["collisions"])),
            "q_scan": q_scan,
        })
        axes[0].plot([row["step"] / 1000 for row in periodic],
                     [row["summary"]["eval/success_rate"] for row in periodic],
                     label=f"seed {seed}")
        axes[1].step([row["step"] / 1000 for row in periodic],
                     [row["curriculum"]["start_curriculum_level"] for row in periodic],
                     where="post", label=f"seed {seed}")
    seeds = [str(row["training_seed"]) for row in runs]
    xx = np.arange(4)
    axes[2].bar(xx - .18, [old_rates[int(seed)] for seed in seeds], width=.36,
                label="no_route")
    axes[2].bar(xx + .18, [row["formal_success_rate"] for row in runs], width=.36,
                label="start curriculum")
    axes[2].set_xticks(xx, seeds)
    axes[0].set(title="Periodic validation: raw networks", xlabel="Train steps (k)",
                ylabel="Success fraction", ylim=(-.04, 1.04))
    axes[1].set(title="Training start-distance level", xlabel="Train steps (k)",
                ylabel="Level (4 = original 7 m)", ylim=(-.2, 4.3))
    axes[2].set(title="Selected checkpoint: 20 episodes", xlabel="Training seed",
                ylabel="Success fraction", ylim=(0, 1.08))
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(ROOT / "multiseed_comparison.png", dpi=160)
    plt.close(fig)
    rates = [row["formal_success_rate"] for row in runs]
    result = {
        "runs": runs,
        "aggregate": {
            "training_seed_count": 4,
            "formal_success_rates": rates,
            "mean_formal_success_rate": float(np.mean(rates)),
            "sample_std_formal_success_rate": float(np.std(rates, ddof=1)),
            "old_no_route_success_rates": [old_rates[int(seed)] for seed in seeds],
            "formal_successes": sum(row["formal_outcomes"].get("success", 0) for row in runs),
            "formal_episode_count": 80,
            "heldout_10_19_successes": sum(row["formal_heldout_10_19_outcomes"].get("success", 0)
                                           for row in runs),
            "heldout_10_19_episode_count": 40,
            "caveat": "Only four training seeds; 20 evaluation seeds share one validation map family. "
                      "Evaluation seeds 0..9 selected checkpoints. New training pinned Stage 0; "
                      "old seed 42 advanced obstacle stages, so that pair is not a pure "
                      "start-distribution ablation. Raw periodic networks can regress.",
        },
    }
    (ROOT / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    summarize()
