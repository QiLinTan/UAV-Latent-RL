"""Aggregate matched T4 rollouts, seed statistics, curriculum, and efficiency."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch

from envs.map_atlas.forest_pool import ForestPool
from experiments.direct_velocity_full_v1.task_factory import ROOT, load_config
from experiments.rv1_range_cnn.environment import make_course_env
from t2_eval import read_json, read_lines, write_json
from t3_report import write_csv
from t4_evaluate import FINAL_NODES, MODELS, SEEDS, load_worker, run_path
from t4_opening_audit import overshoot
from t4_suite import load_suite
from t4_train import NODES

TERMINALS = ("success", "collision", "timeout", "xy_bound", "low_height_bound", "high_height_bound")


def split_groups(rows, standard):
    groups = defaultdict(list)
    for row in rows:
        if standard:
            group = (f"Stage{row['stage']}", "all")
        elif row["suite"] == "A_forest":
            group = ("A_forest", f"A_Stage{row['stage']}", "A_all")
        elif row["suite"] == "A_aperture":
            group = ("A_aperture", "A_all")
        else:
            group = (row["suite"],)
        for key in group:
            groups[key].append(row)
    return groups


def grouped_metrics(rows):
    totals = Counter(r["terminal_class"] for r in rows)
    n = len(rows)
    won = [r for r in rows if r["success"]]
    clear = [r["min_clearance_m"] for r in rows if r["min_clearance_m"] is not None]
    windows = sum(r["windows_at_least_4s"] for r in rows)
    out = dict(episodes=n, arrival_rate=totals["success"]/n, collision_rate=totals["collision"]/n,
               collision_observed_rate=sum(r["collision_observed"] for r in rows)/n,
               timeout_rate=totals["timeout"]/n,
               boundary_rate=sum(totals[key] for key in TERMINALS[3:])/n,
               horizontal_boundary_rate=totals["xy_bound"]/n, low_boundary_rate=totals["low_height_bound"]/n,
               high_boundary_rate=totals["high_height_bound"]/n, other_terminal_rate=sum(
                   count for reason, count in totals.items() if reason not in TERMINALS)/n,
               successful_path_mean_m=mean(r["path_length_m"] for r in won) if won else None,
               successful_time_mean_s=mean(r["duration_s"] for r in won) if won else None,
               minimum_clearance_p10_m=float(np.percentile(clear, 10)) if clear else None,
               minimum_clearance_mean_m=mean(clear) if clear else None,
               e03_stall_window_fraction=sum(r["stall_windows"] for r in rows)/windows if windows else None,
               terminal_counts=dict(totals))
    out.update({f"boundary_{label}_rate": sum(edge in r.get("xy_exits", ()) for r in rows)/n
                for edge, label in (("x+", "x_positive"), ("x-", "x_negative"),
                                    ("y+", "y_positive"), ("y-", "y_negative"))})
    if all(r.get("passed") is not None for r in rows):
        passed = [r for r in rows if r["passed"]]
        both = sum(r["success"] for r in passed)
        out.update(opening_pass_rate=len(passed)/n, pass_and_arrival_rate=both/n,
                   arrival_given_pass_rate=both/len(passed) if passed else None)
        for label in ("jamb", "wall_face"):
            out[f"opening_{label}_failure_rate"] = sum(
                not r["success"] and r["terminal_class"] == "collision"
                and (r.get("opening") or {}).get("contact_class") == label for r in rows)/n
        out["opening_other_contact_failure_rate"] = sum(
            r["terminal_class"] == "collision" and (r.get("opening") or {}).get("contact_class") is None
            for r in rows)/n
    if all(r.get("corridor") is not None for r in rows):
        corridor = [r["corridor"] for r in rows]
        out.update(dense_entry_rate=sum(r["entered_dense"] for r in corridor)/n,
                   dense_exit_rate=sum(r["exited_dense"] for r in corridor)/n)
        for i, label in enumerate(("entry", "dense", "exit")):
            clear = [r["phase_min_clearance_m"][i] for r in corridor if r["phase_min_clearance_m"][i] is not None]
            out[f"{label}_minimum_clearance_mean_m"] = mean(clear) if clear else None
            out[f"{label}_failure_rate"] = sum(not row["success"] and phase["terminal_phase"] == label
                                               for row, phase in zip(rows, corridor))/n
        for label in ("entry", "exit"):
            out[f"{label}_boundary_band_failure_rate"] = sum(
                not row["success"] and phase.get("terminal_boundary") == label
                for row, phase in zip(rows, corridor))/n
    return out


def moments(rows):
    result = {}
    for key in rows[0]:
        if key == "terminal_counts":
            continue
        values = [r[key] for r in rows if r[key] is not None]
        result[key] = dict(mean=mean(values) if values else None, sd=stdev(values) if len(values) > 1 else None,
                           n=len(values))
    return result


def sign_permutation(deltas):
    frequencies = Counter({0: 1})
    for difference in deltas:
        new = Counter()
        for previous, n in frequencies.items():
            new[previous + difference] += n
            new[previous - difference] += n
        frequencies = new
    observed = sum(deltas)
    return dict(units=len(deltas), observed_difference_sum=observed,
                two_sided_p=sum(n for x, n in frequencies.items() if abs(x) >= abs(observed)) / 2**len(deltas))


def paired(groups, standard):
    result = {}
    for scope in groups[43, "t3"]:
        metrics = ["success", "collision"]
        if scope == "A_aperture":
            metrics.append("passed")
        scope_result = {}
        for metric in metrics:
            seed_deltas = []
            maps = Counter()
            per_seed = {}
            for seed in SEEDS:
                dual = groups[seed, "t3"][scope]
                rv1 = groups[seed, "a1"][scope]
                key = (lambda r: (r["stage"], r["case"])) if standard else (lambda r: r["id"])
                controls = {key(r): r for r in rv1}
                outcomes = Counter()
                diff = 0
                if len(dual) != len(controls):
                    raise RuntimeError("paired metric cohort count mismatch")
                for row in dual:
                    other = controls[key(row)]
                    if row["init"] != other["init"]:
                        raise RuntimeError("paired case initialization mismatch")
                    a = row["reason"] == "collision" if metric == "collision" else bool(row[metric])
                    b = other["reason"] == "collision" if metric == "collision" else bool(other[metric])
                    d = int(a) - int(b)
                    diff += d
                    # Stage variants share a forest seed; mirrored apertures share a layout seed.
                    map_key = row["case"] if standard else (row["suite"], row["map_seed"])
                    maps[map_key] += d
                    outcomes[("both+" if b else "t3_only") if a else ("rv1_only" if b else "both-")] += 1
                seed_deltas.append(diff)
                per_seed[seed] = dict(episodes=len(dual), delta_rate=diff/len(dual), outcomes=dict(outcomes))
            scope_result[metric] = dict(per_seed=per_seed, seed_level_exact=sign_permutation(seed_deltas),
                                        map_cluster_exact_conditional_on_three_runs=sign_permutation(list(maps.values())))
        result[scope] = scope_result
    return result


def summarize_files(out, standard):
    folder = out / ("standard_results" if standard else "heldout_results")
    table, stats, tests = [], {}, {}
    for node in (NODES if standard else FINAL_NODES):
        groups = {}
        stats[node] = {}
        for seed in SEEDS:
            for model in MODELS:
                doc = read_json(folder / f"{model}_{seed}_{node}.json")
                groups[seed, model] = split_groups(doc["rows"], standard)
        for model in MODELS:
            values = {}
            for scope in groups[43, model]:
                per_seed = {seed: grouped_metrics(groups[seed, model][scope]) for seed in SEEDS}
                summary = moments(list(per_seed.values()))
                values[scope] = dict(per_seed=per_seed, mean_sd=summary)
                for metric, aggregate in summary.items():
                    table.append(dict(step=node, model=model, scope=scope, metric=metric, **aggregate,
                                      **{f"seed{seed}": data[metric] for seed, data in per_seed.items()}))
            stats[node][model] = values
        tests[node] = paired(groups, standard)
    return table, stats, tests


def match_archive_course(env, expected):
    actual = env.course_state()
    return all(actual[key] == expected[key] for key in (
        "draws", "visits", "pool_rng", "start_rng", "start_scheduler", "obstacle_scheduler"))


def reconstruct_archived_a1(run):
    snapshots = {}
    for node in NODES:
        pack = torch.load(run / f"checkpoint_{node:06d}.pt", map_location="cpu", weights_only=False)
        snapshots.setdefault(pack["trainer"]["episode_id"], []).append(pack["environment"]["course"])
    env = make_course_env(ForestPool(ROOT / load_config()["forest_pool_path"]))
    env._observation = lambda state: state
    transitions = []
    reproduced = True
    try:
        env.reset(seed=43)
        for index, row in enumerate(read_lines(run / "episodes.jsonl")):
            if index in snapshots:
                reproduced &= all(match_archive_course(env, s) for s in snapshots[index])
            previous = env.start_curriculum.level, env.obstacle_curriculum.stage
            reason = row["done_reason"]
            env.record_episode_outcome(reason == "success", collision=reason == "collision", done_reason=reason)
            current = env.start_curriculum.level, env.obstacle_curriculum.stage
            if current != previous:
                transitions.append(dict(step=row["step"], old_start_level=previous[0], start_level=current[0],
                                        old_obstacle_stage=previous[1], obstacle_stage=current[1]))
            env.reset()
        if env._pool_draw_count - 1 in snapshots:
            reproduced &= all(match_archive_course(env, s) for s in snapshots[env._pool_draw_count - 1])
    finally:
        env.close()
    if not reproduced:
        audit = read_json(ROOT / "MSGP_RESEARCH_REPORT/data/e01_e06_detailed.json")
        return dict(source="Historical E01 audit; early start-level nodes are snapshots.",
                    exact_obstacle_stage_changes=[dict(step=r["begin_after_step"], obstacle_stage=r["stage"])
                        for r in audit["exposure"] if r["model"] == "a1" and r["stage"] > 0],
                    exact_stage0_eligibility_unlock=next(r["eligible_episode_start_after_step"]
                        for r in audit["exposure"] if r["model"] == "a1" and r["stage"] == 0),
                    first_snapshot_by_start_level=audit["course"]["a1"]["first_snapshot_each_start_level"])
    return dict(source="Episode-ending events reconstructed; course state matched archived checkpoint snapshots.",
                transitions=transitions)


def course_history(runs):
    result = {}
    for seed in SEEDS:
        result[seed] = {}
        for model in MODELS:
            run = run_path(runs, seed, model)
            if seed == 43 and model == "a1":
                result[seed][model] = reconstruct_archived_a1(run)
            else:
                result[seed][model] = dict(source="Recorded during training",
                                           transitions=read_lines(run.parent / "curriculum_promotions.jsonl"))
    return result


def late_run_diagnostics(runs, out, fingerprint):
    result = {}
    for seed in SEEDS:
        result[seed] = {}
        for model in MODELS:
            run = run_path(runs, seed, model)
            rows = read_json(out / f"standard_results/{model}_{seed}_240000.json")["rows"]
            candidates = [r for r in rows if "x+" in r.get("xy_exits", ())]
            traces = []
            if candidates:
                load_worker(model, seed, 240000, run / "checkpoint_240000.pt", fingerprint)
                traces = [dict(stage=r["stage"], case=r["case"], **overshoot(r)) for r in candidates]
            result[seed][model] = dict(fixed_240k_positive_x_exit_traces=traces)
            if model == "t3":
                probes = [r for r in read_lines(run.parent / "probes_002000.jsonl")
                          if r["step"] in (160000, 200000, 240000)]
                if len(probes) != 3:
                    raise RuntimeError("missing T3 late-run probe")
                result[seed][model]["late_probes"] = [
                    {key: row[key] for key in ("step", "behavior", "fixed_actor",
                                              "actor_mean_vy_drift_from_step0", "branches")}
                    for row in probes]
    return result


def plot_node_arrivals(out, standard):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(10, 4.2), constrained_layout=True)
    steps = np.asarray(NODES) / 1000
    for ax, model, name, color in zip(axes, MODELS, ("T3", "RV1"), ("#007479", "#9d4619")):
        cohorts = [standard[node][model]["all"] for node in NODES]
        for seed, mark in zip(SEEDS, ("o", "s", "^")):
            arrivals = [100 * data["per_seed"][seed]["arrival_rate"] for data in cohorts]
            ax.plot(steps, arrivals, marker=mark, linewidth=1, alpha=.55, label=f"Seed {seed}")
        stats = [data["mean_sd"]["arrival_rate"] for data in cohorts]
        ax.errorbar(steps, [100 * row["mean"] for row in stats],
                    yerr=[100 * row["sd"] for row in stats], color=color,
                    capsize=4, marker="D", label="Mean ± sample SD")
        ax.set(title=name, xlabel="Training transitions (thousands)", xticks=steps, ylim=(-5, 112))
        ax.grid(axis="y", alpha=.3)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("Arrivals on fixed forest set (%)")
    fig.savefig(out / "t4_fixed_forest_training_curves.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    out = args.output
    std_table, std, std_tests = summarize_files(out, True)
    held_table, held, held_tests = summarize_files(out, False)
    efficiency = read_json(out / "efficiency.json")
    timings = efficiency["latency"]
    efficiency_table = []
    for test, model in (("t3", "t3"), ("a1_shared_scanner", "a1"), ("a1_native_pipeline", "a1")):
        memory = efficiency["replay"][model]
        efficiency_table.append(dict(model=model, scanner=timings[test]["lidar"],
            **efficiency["parameters"][model], replay_theory_bytes=memory["theory_bytes"],
            replay_measured_resident_delta_bytes=memory["measured_private_rss_increase_bytes"],
            replay_measured_cuda_tensor_delta_bytes=memory["measured_cuda_tensor_delta_bytes"],
            mean_ms=timings[test]["mean_ms"], p99_ms=timings[test]["p99_ms"]))
    write_csv(out / "t4_standard_comparison.csv", std_table)
    write_csv(out / "t4_heldout_comparison.csv", held_table)
    write_csv(out / "t4_efficiency_comparison.csv", efficiency_table)
    plot_node_arrivals(out, std)
    manifest = load_suite(out / "heldout_suite/manifest.json")
    write_json(out / "t4_summary_report.json", dict(
        seeds=SEEDS, standard_nodes=NODES, heldout_nodes=FINAL_NODES,
        sealed_suite=dict(fingerprint=manifest["fingerprint"], sizes=manifest["sizes"]),
        course_history=course_history(args.runs),
        late_run_diagnostics=late_run_diagnostics(args.runs, out, manifest["frozen_training_pool_fingerprint"]),
        standard=std, standard_paired=std_tests, heldout=held, heldout_paired=held_tests, efficiency=efficiency,
        inference_scope="Means and sample SD are over three paired training seeds. Seed-level exact tests have low power; map cluster tests are conditional on the three trained pairs.",
        diagnostics_scope=dict(
            clearance="P10 and mean of the finite per-episode minimum obstacle clearances.",
            corridor="Terminal x defines entry (< -4 m), dense ([-4, 4) m), and exit (>= 4 m). Boundary bands extend 1 m on either side of each density boundary.",
            opening="A terminal collision with a panel contact within 0.06 m of either opening edge is a jamb collision. Other panel contacts are wall-face collisions.",
            fixed_positive_x="All 240k fixed-set positive X exits are reproduced against the recorded outcome and checked for a missed goal sphere."),
        e03="Sliding 4 s windows at 15 Hz, net goal progress below 0.12 m.",
        status="completed; awaiting user decision before Phase 3"))
    print("T4 report complete.", flush=True)


if __name__ == "__main__":
    main()
