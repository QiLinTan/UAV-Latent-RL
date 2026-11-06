"""Summarize Q1/Q2/min scans separately from same-policy rollout returns."""

import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def mean(values):
    return float(np.mean(values)) if values else None


def analyze(records, calibrations, actions):
    summary={}
    groups={
        "all":records,
        "height_mismatch":[r for r in records if r["category"]=="xy_near_height_mismatch"],
        "height_mismatch_clear":[r for r in records if r["category"]=="xy_near_height_mismatch" and r["clearance_above_0_35"]],
        "above_target_by_0_5m":[r for r in records if r["case"]["position"][2]>1.5],
    }
    for name,group in groups.items():
        item=dict(state_count=len(group))
        for key in ("q1","q2","q_min"):
            best=[max(r["candidates"],key=lambda c:c[key]) for r in group]
            vertical=[max([c for c in r["candidates"] if c["family"]=="vertical_only"],key=lambda c:c[key]) for r in group]
            item[key]=dict(best_family_counts=dict(Counter(c["family"] for c in best)),
                best_z_counts=dict(Counter(f'{c["z_command_mps"]:.2f}' for c in best)),
                best_xy_speed_counts=dict(Counter(f'{c["xy_speed_mps"]:.2f}' for c in best)),
                fixed_actor_xy_best_z_counts=dict(Counter(f'{c["z_command_mps"]:.2f}' for c in vertical)),
                mean_actor_gap=mean([max(c[key] for c in r["candidates"])-r["candidates"][0][key] for r in group]),
                mean_within_state_range=mean([max(c[key] for c in r["candidates"])-min(c[key] for c in r["candidates"]) for r in group]))
            pairs=[(next(c for c in r["candidates"] if c["name"]=="actor_xy_z-1.0"),r["candidates"][0]) for r in group]
            item[key]["down_better_than_actor_count"]=sum(d[key]>a[key]+.01 for d,a in pairs)
            item[key]["mean_down_minus_actor"]=mean([d[key]-a[key] for d,a in pairs])
        summary[name]=item
    calibrated={}
    for row in calibrations:
        calibrated.setdefault(row["case_id"],[]).append(row)
    comparison={"state_count":len(calibrated),"trajectory_count":len(calibrations),
                "outcomes":dict(Counter(r["outcome"] for r in calibrations))}
    for key in ("q1","q2","q_min"):
        regrets, signed_errors, errors=[],[],[]
        concordant=discordant=0
        for group in calibrated.values():
            best=max(group,key=lambda r:r[key])
            regrets.append(max(r["actual_return"] for r in group)-best["actual_return"])
            for i,a in enumerate(group):
                signed_errors.append(a[key]-a["actual_return"])
                errors.append(abs(a[key]-a["actual_return"]))
                for b in group[i+1:]:
                    dq=a[key]-b[key]
                    dg=a["actual_return"]-b["actual_return"]
                    if abs(dq)<=.01 or abs(dg)<=.01:
                        continue
                    concordant+=int(dq*dg>0)
                    discordant+=int(dq*dg<0)
        comparison[key]=dict(mean_Q_minus_realized_return=mean(signed_errors),
            mean_absolute_error=mean(errors),mean_selected_candidate_regret=mean(regrets),
            max_selected_candidate_regret=max(regrets,default=None),
            nontrivial_pair_count=concordant+discordant,concordant_pairs=concordant,
            discordant_pairs=discordant)
    down_pairs=[]
    for group in calibrated.values():
        actor=next(r for r in group if r["name"]=="actor")
        down=next(r for r in group if r["name"]=="actor_xy_z-1.0")
        down_pairs.append(dict(case_id=actor["case_id"],category=actor["category"],
            delta_q1=down["q1"]-actor["q1"],delta_qmin=down["q_min"]-actor["q_min"],
            delta_return=down["actual_return"]-actor["actual_return"],
            actor_return=actor["actual_return"],down_return=down["actual_return"],
            actor_outcome=actor["outcome"],down_outcome=down["outcome"]))
    comparison["down_vs_actor"]=down_pairs
    summary["calibration"]=comparison
    bins={}
    for low,high in ((0,.2),(.2,.5),(.5,.8),(.8,1.2),(1.2,2.),(2.,3.),(3.,float("inf"))):
        selected=[a for a in actions if low<=a["goal_distance"]<high]
        norms=[float(np.linalg.norm(a["actor_action"][:2])) for a in selected]
        bins[f"[{low},{high})"]=dict(count=len(selected),mean_actor_xy_norm=mean(norms),
            saturated_xy_fraction=mean([v>=.95 for v in norms]),
            mean_command_xy_speed=mean([float(np.linalg.norm(a["command"][:2])) for a in selected]),
            mean_actual_xy_speed=mean([float(np.linalg.norm(a["actual_velocity"][:2])) for a in selected]),
            mean_z_command=mean([a["command"][2] for a in selected]),
            away_command_fraction=mean([a["command_closing_speed"]<0 for a in selected]))
    summary["action_distance_bins"]=bins
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    cli=parser.parse_args()
    p=cli.directory
    records=json.loads((p/"q_scan.json").read_text())
    calibrations=[json.loads(s) for s in (p/"calibration.jsonl").read_text().splitlines()]
    actions=[json.loads(s) for s in (p/"actor_actions.jsonl").read_text().splitlines()]
    result=analyze(records,calibrations,actions)
    (p/"analysis.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    # All three critics are retained; actor training follows Q1 rather than min.
    group=[r for r in records if r["category"]=="xy_near_height_mismatch"]
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    for key in ("q1","q2","q_min"):
        ys=[]
        for z in (-1.,-.5,0.,.5,1.):
            ys.append(mean([next(c[key] for c in r["candidates"] if c["name"]==f"actor_xy_z{z:+.1f}")-
                            r["candidates"][0][key] for r in group]))
        axes[0].plot(np.array([-1.,-.5,0.,.5,1.])*.3,ys,marker="o",label=key)
    axes[0].axhline(0,color="gray",linestyle="--")
    axes[0].set(xlabel="Candidate vertical command (m/s)",ylabel="Mean Q(candidate) - Q(actor)",
                title=f"XY-near / height-mismatch states (n={len(group)})")
    axes[0].legend()
    pairs=result["calibration"]["down_vs_actor"]
    for category in sorted(set(r["category"] for r in pairs)):
        subset=[r for r in pairs if r["category"]==category]
        axes[1].scatter([r["delta_q1"] for r in subset],[r["delta_return"] for r in subset],label=category)
    axes[1].axhline(0,color="gray",linestyle="--")
    axes[1].axvline(0,color="gray",linestyle="--")
    axes[1].set(xlabel="Q1(down once) - Q1(actor)",ylabel="Realized return difference",
                title="One descent action, then original actor")
    axes[1].legend(fontsize=7)
    for ax in axes: ax.grid(alpha=.25)
    fig.savefig(p/"vertical_q_and_calibration.png",dpi=170)
    plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k not in ("calibration",)},indent=2))


if __name__=="__main__":
    main()
