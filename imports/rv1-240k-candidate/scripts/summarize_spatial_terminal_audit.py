"""Produce compact, same-state comparisons and figures from the forest audit."""

import argparse
from collections import Counter
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_dir", type=Path)
    cli = parser.parse_args()
    directory = cli.audit_dir
    summary = json.loads((directory/"summary.json").read_text())
    rows = [json.loads(line) for line in (directory/"rollouts.jsonl").read_text().splitlines()]
    traces = np.load(directory/"traces.npz")
    groups = {c["id"]: [r for r in rows if r["case_id"]==c["id"]] for c in summary["cases"]}
    comparisons = []
    late = []
    for case in summary["cases"]:
        group = groups[case["id"]]
        successes = [r for r in group if r["outcome"]=="success"]
        actor = next(r for r in group if r["behavior"]=="actor")
        # Optimistic bound even ignoring acceleration, horizontal travel and trees.
        if max(0.0,case["z_error"]-0.2)/0.3 > case["remaining_seconds"]+1/15:
            late.append(case["id"])
        if not successes:
            continue
        best = max(successes, key=lambda r:r["G_baseline"])
        comparisons.append(dict(case_id=case["id"], success_behavior=best["behavior"],
            baseline_success=best["G_baseline"], baseline_actor=actor["G_baseline"],
            no_route_success=best["G_no_route"], no_route_actor=actor["G_no_route"],
            success_first_1s=best["G_baseline_first_1s"], actor_first_1s=actor["G_baseline_first_1s"],
            success_first_2s=best["G_baseline_first_2s"], actor_first_2s=actor["G_baseline_first_2s"]))
    total_running = outward_positive = outward_positive_net = 0
    for case in summary["cases"]:
        if case["step"] != 0:
            continue
        trace = traces[f'{case["id"]}_actor']
        previous = np.r_[case["d_3d"], trace[:-1,7]]
        running = trace[:,17] == 0
        outward = trace[:,7] > previous+1e-6
        positive_progress = trace[:,12]+trace[:,13] > 1e-6
        total_running += int(running.sum())
        outward_positive += int(np.sum(running & outward & positive_progress))
        outward_positive_net += int(np.sum(running & outward & (trace[:,8]>0)))
    report = dict(
        case_count=len(summary["cases"]), rollout_count=len(rows),
        policy_outcomes=dict(Counter(r["outcome"] for r in summary["policy_episodes"])),
        reward_rankings=summary["rankings"],
        cases_vertically_unreachable_with_remaining_time_even_at_max_speed=late,
        actor_running_steps=total_running,
        actor_steps_distance_increases_but_progress_positive=outward_positive,
        actor_steps_distance_increases_but_total_reward_positive=outward_positive_net,
        successful_alternative_vs_actor=comparisons,
        success_has_lower_first_1s_return_than_actor=sum(c["success_first_1s"]<c["actor_first_1s"] for c in comparisons),
        success_has_lower_first_2s_return_than_actor=sum(c["success_first_2s"]<c["actor_first_2s"] for c in comparisons),
    )
    (directory/"comparisons.json").write_text(json.dumps(report,indent=2)+"\n")
    if comparisons:
        fig,axes = plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
        for ax, prefix in zip(axes,("baseline","no_route")):
            x = np.array([c[f"{prefix}_actor"] for c in comparisons])
            y = np.array([c[f"{prefix}_success"] for c in comparisons])
            low, high = min(x.min(),y.min())-1, max(x.max(),y.max())+1
            ax.scatter(x,y,s=24)
            ax.plot([low,high],[low,high],"k--",linewidth=1)
            ax.set(title=prefix, xlabel="Actor continuation: discounted return",
                   ylabel="Successful candidate: discounted return")
            ax.grid(alpha=.25)
        fig.savefig(directory/"return_comparison.png",dpi=170)
        plt.close(fig)
        chosen = next((c for c in comparisons if c["case_id"]=="seed00_step000"),comparisons[0])
        fig,axes = plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
        for behavior in ("actor", chosen["success_behavior"], "brake_to_stop"):
            trace = traces[f'{chosen["case_id"]}_{behavior}']
            axes[0].plot(trace[:,0],trace[:,7],label=behavior)
            axes[1].plot(trace[:,0],np.cumsum(trace[:,10]),label=behavior)
            axes[2].plot(trace[:,0],np.cumsum(trace[:,11]),label=behavior)
        axes[0].axhline(1.2,color="gray",linestyle="--")
        axes[0].axhline(.2,color="black",linestyle=":")
        for ax,title,ylabel in zip(axes,("3-D approach","Baseline reward","No-route reward"),
                                  ("Goal distance (m)","Discounted return","Discounted return")):
            ax.set(title=title,xlabel="Time since branch (s)",ylabel=ylabel)
            ax.grid(alpha=.25)
        axes[0].legend(fontsize=8)
        fig.savefig(directory/"example_trajectory.png",dpi=170)
        plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k!="successful_alternative_vs_actor"},indent=2))


if __name__ == "__main__":
    main()
