"""Report fully paired Suite D outcomes, seed dispersion, and failure evidence."""
import argparse
from collections import Counter
import csv
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from contract import HERE, digest, load_manifest, read, write
from geometry import TOPOLOGIES
from metrics import FAILURE_BUCKETS

MODELS = ("a1", "t3")
SEEDS = (43, 44, 45)
NODES = (200000, 240000)
TERMINALS = ("success", "collision", "timeout", "xy_bound", "low_height_bound", "high_height_bound", "attitude_bound")
SCALARS = ("geodesic_progress_ratio", "max_geodesic_progress_ratio", "clearance_min_m",
           "clearance_mean_m", "clearance_p10_m", "successful_path_ratio",
           "successful_effective_speed_mps", "successful_actual_mean_speed_mps",
           "horizontal_saturated_fraction", "vertical_usage_mean", "z_saturated_fraction")


def compact(row):
    output = {key: row[key] for key in SCALARS}
    output.update({f"rate_{key}": int(row["terminal"] == key) for key in TERMINALS})
    output["rate_out_of_bounds"] = sum(output[key] for key in
                                       ("rate_xy_bound", "rate_low_height_bound", "rate_high_height_bound"))
    output["successful_duration_s"] = row["duration_s"] if row["terminal"] == "success" else None
    output.update({f"milestone_{key}": int(value) for key, value in row["milestone"].items()
                   if isinstance(value, bool)})
    output.update({f"approach_{key}": row["approach"][key] for key in
                   ("horizontal_mean_mps", "horizontal_saturated_fraction", "vertical_usage_mean")})
    output["stall_ratio"] = row["stalled_windows"] / row["stall_windows"] if row["stall_windows"] else None
    output.update({f"failure_rate_{key}": int(row["failure_bucket"] == key) for key in FAILURE_BUCKETS})
    return output


def aggregate(rows):
    compacted = [compact(row) for row in rows]
    keys = sorted({key for row in compacted for key in row})
    result = {}
    for key in keys:
        values = [row[key] for row in compacted if row.get(key) is not None]
        result[key] = float(np.mean(values)) if values else None
    minimums = [row["clearance_min_m"] for row in rows]
    result["across_episode_min_clearance_p10_m"] = float(np.percentile(minimums, 10))
    return result


def moments(values):
    observed = [value for value in values if value is not None]
    return dict(mean=float(np.mean(observed)) if observed else None,
                sd=float(np.std(observed, ddof=1)) if len(observed) >= 2 else None, n=len(observed))


def build(root, manifest, allow_partial):
    meta = read(root / "run_manifest.json")
    if meta["manifest"] != manifest["fingerprint"]:
        raise RuntimeError("run manifest and freeze do not match")
    if meta["mode"] != "formal" and not allow_partial:
        raise RuntimeError("pilot cannot be reported as formal results")
    cases = {case["id"]: case for case in manifest["cases"]}
    rows = {}
    for run in meta["checkpoints"]:
        key = run["model"], run["seed"], run["checkpoint"]
        folder = root / run["id"]
        if not folder.is_dir():
            continue
        for path in folder.glob("suite_d_*.json"):
            row = read(path)
            case = cases[row["id"]]
            if (row["run"] != run or row["topology"] != case["topology"]
                    or row["geometry_sha256"] != case["geometry_sha256"]
                    or row["initial_digest"] != digest(case["initial"])
                    or row["suite_fingerprint"] != manifest["fingerprint"]):
                raise RuntimeError(f"pairing contract drift: {path}")
            if (row["terminal"] not in TERMINALS
                    or (row["terminal"] == "success") != (row["failure_bucket"] is None)
                    or row["terminal"] != "success" and row["failure_bucket"] not in FAILURE_BUCKETS):
                raise RuntimeError(f"invalid terminal accounting: {path}")
            ticks = round(row["duration_s"] * manifest["physics_hz"])
            deadline = case["timeout_physics_ticks"]
            if (ticks > deadline or (row["terminal"] == "timeout" and ticks != deadline)
                    or (row["terminal"] == "success" and row["goal_distance_m"] >= manifest["reach_radius_m"])
                    or (case["closed"] and row["terminal"] in ("xy_bound", "low_height_bound", "high_height_bound"))
                    or not (folder / row["trace"]).is_file()):
                raise RuntimeError(f"deadline, boundary, arrival or trace validation failed: {path}")
            rows.setdefault(key, {})[row["id"]] = row
    actual = sum(map(len, rows.values()))
    complete = (meta["mode"] == "formal" and set(rows) == set(product(MODELS, SEEDS, NODES))
                and all(set(selected) == set(cases) for selected in rows.values()))
    if not complete and not allow_partial:
        raise RuntimeError(f"formal paired grid incomplete: {actual}/960")
    if not actual:
        raise RuntimeError("no completed episodes to report")
    summaries, paired_summaries, paired_rows = {}, {}, []
    for checkpoint, topology in product(NODES, (*TOPOLOGIES, "all")):
        key = f"{checkpoint}_{topology}"
        per_seed = {model: {} for model in MODELS}
        pair_seed = {}
        for seed in SEEDS:
            group_a = rows.get(("a1", seed, checkpoint), {})
            group_t = rows.get(("t3", seed, checkpoint), {})
            for model, group in (("a1", group_a), ("t3", group_t)):
                observed = [row for row in group.values() if topology in ("all", row["topology"])]
                if observed:
                    per_seed[model][seed] = aggregate(observed)
            pairs = []
            for identifier in sorted(group_a.keys() & group_t.keys()):
                left, right = group_a[identifier], group_t[identifier]
                if topology not in ("all", left["topology"]):
                    continue
                left_metrics, right_metrics = compact(left), compact(right)
                deltas = {metric: right_metrics[metric] - left_metrics[metric]
                          if right_metrics.get(metric) is not None and left_metrics.get(metric) is not None else None
                          for metric in sorted(left_metrics.keys() | right_metrics.keys())}
                pairs.append(deltas)
                if topology != "all":
                    paired_rows.append(dict(checkpoint=checkpoint, seed=seed, id=identifier, topology=topology,
                                            a1_terminal=left["terminal"], t3_terminal=right["terminal"],
                                            a1_failure=left["failure_bucket"], t3_failure=right["failure_bucket"], **deltas))
            if pairs:
                pair_seed[seed] = {metric: moments([row.get(metric) for row in pairs])
                                   for metric in sorted({metric for row in pairs for metric in row})}
        summaries[key] = {model: dict(per_seed=seed_map,
                                      across_seeds={metric: moments([value.get(metric) for value in seed_map.values()])
                                                    for metric in sorted({metric for value in seed_map.values()
                                                                          for metric in value})})
                          for model, seed_map in per_seed.items()}
        paired_summaries[key] = dict(per_seed=pair_seed, across_seed_mean_deltas={
            metric: moments([values.get(metric, {}).get("mean") for values in pair_seed.values()])
            for metric in sorted({metric for values in pair_seed.values() for metric in values})})
    failure = {f"{node}_{model}_{topology}": dict(Counter(
        row["failure_bucket"] for seed in SEEDS for row in rows.get((model, seed, node), {}).values()
        if row["topology"] == topology and row["failure_bucket"] is not None))
        for node, model, topology in product(NODES, MODELS, TOPOLOGIES)}
    output = dict(status="formal_complete" if complete else "partial_not_formal",
                  manifest=manifest["fingerprint"], episodes=actual, expected_episodes=960,
                  number_of_pairs=len(paired_rows), seed_stats=summaries, paired_t3_minus_a1=paired_summaries,
                  failures=failure, phase3="not_applicable: the frozen A1 and T3 checkpoints do not use Phase 3")
    revision = root / "deadline_revision.json"
    if revision.is_file():
        audit = read(revision)
        output["deadline_audit"] = {name: audit[name] for name in
                                    ("correction", "unchanged_predeadline_rollouts", "rerun_timeout_rollouts")}
    return output, paired_rows


def chart(output, target):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True, layout="constrained")
    colors = ("#537da2", "#68a4aa", "#d28b59", "#ad5e75", "#786ca7", "#a6a6a6")
    names = [f"{topology}\n{model.upper()}" for topology, model in product(TOPOLOGIES, MODELS)]
    for axis, node in zip(axes, NODES):
        bottom = np.zeros(len(names))
        for bucket, color in zip(FAILURE_BUCKETS, colors):
            values = [output["failures"][f"{node}_{model}_{topology}"].get(bucket, 0)
                      for topology, model in product(TOPOLOGIES, MODELS)]
            axis.bar(names, values, bottom=bottom, label=bucket, color=color)
            bottom += values
        axis.set_title(f"{node // 1000}k checkpoint")
        axis.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Failed episodes counted")
    axes[1].legend(loc="upper center", bbox_to_anchor=(.5, -.18), ncol=6, fontsize=8)
    figure.suptitle("Suite D — failure signatures" if output["status"] == "formal_complete"
                    else f"Suite D — partial QA only ({output['episodes']}/960 episodes)")
    figure.savefig(target, dpi=170)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    parser.add_argument("--results", type=Path, default=HERE / "results_verified")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest, verify_runtime=False)
    output, pairs = build(args.results, manifest, args.allow_partial)
    stem = "suite_d" if output["status"] == "formal_complete" else "suite_d_partial"
    write(args.results / f"{stem}_summary.json", output, mode="w")
    if pairs:
        with (args.results / f"{stem}_paired.csv").open("w", newline="", encoding="utf-8") as stream:
            keys = list(dict.fromkeys(key for row in pairs for key in row))
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(pairs)
    chart(output, args.results / f"{stem}_failure_buckets.png")
    print(f"{output['status']}: {output['episodes']} episodes, {len(pairs)} paired T3−A1 differences")


if __name__ == "__main__":
    main()
