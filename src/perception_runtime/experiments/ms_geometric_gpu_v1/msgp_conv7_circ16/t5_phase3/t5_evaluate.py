"""Evaluate completed dynamic checkpoints on the frozen T4 and Suite D cohorts."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import contextlib
from itertools import product
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from dynamic_v22 import DynamicAgent, DynamicBuilder, DynamicReplay, MODE, WIDTH, attach_frame_observer
from modulation_v22 import RESULT_FIELDS
from experiments.rv1_range_cnn.lidar import LidarConfig
from t2_eval import read_json, read_lines, write_json
from t3_report import write_csv
from t4_evaluate import CHECKED_CONFIG, FINAL_NODES, SEEDS
import t4_evaluate as fixed
import t4_report as report
from t5_train import NODES, RUN_NAME, SCHEMA
from train import sha

HERE = Path(__file__).resolve().parent
T4 = HERE.parent / "t4_benchmark"
D = T4 / "suite_d"
BASELINES = ("t3", "a1")


def dynamic_path(runs, seed):
    return runs / RUN_NAME.format(seed=seed) / "train"


def identify(runs, fingerprint):
    baseline = read_json(fixed.run_path(runs, 43, "t3") / "metadata.json")["config"]
    registry = {}
    for seed in SEEDS:
        run = dynamic_path(runs, seed)
        meta = read_json(run / "metadata.json")
        config, contract = meta["config"], meta["contract"]
        if (any(config[key] != baseline[key] for key in CHECKED_CONFIG) or config["seed"] != seed
                or contract["pool_fingerprint"] != fingerprint or contract["fusion_mode"] != MODE):
            raise RuntimeError("T5 training contract does not match T4")
        events = {r["step"]: r["sha256"] for r in read_lines(run / "events.jsonl") if r["event"] == "checkpoint"}
        for node in NODES:
            path = run / f"checkpoint_{node:06d}.pt"
            if (digest := sha(path)) != events[node]:
                raise RuntimeError("T5 checkpoint digest mismatch")
            registry[seed, node] = dict(id=f"dynamic_{seed}_{node}", model="dynamic", seed=seed,
                                        checkpoint=node, path=str(path), sha256=digest)
    return registry


def load_policy(path, device, seed, node, fingerprint):
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    contract = checkpoint["contract"]
    if (contract["schema"], contract["fusion_mode"], contract["observation_width"], contract["seed"],
            contract["pool_fingerprint"], checkpoint["trainer"]["step"]) != (
            SCHEMA, MODE, WIDTH, seed, fingerprint, node):
        raise RuntimeError("unexpected T5 checkpoint")
    agent = DynamicAgent(contract["td3"], True, device)
    agent.actor.load_state_dict(checkpoint["agent"]["actor"])
    agent.actor.eval()
    return agent, DynamicBuilder(device, LidarConfig(**contract["sensor"])), contract


def load_worker(model, seed, node, path, fingerprint):
    if model != "dynamic":
        raise ValueError("this runner is for T5")
    torch.set_num_threads(1)
    fixed.base.agent, fixed.base.builder, fixed.base.contract = load_policy(path, "cpu", seed, node, fingerprint)
    fixed.base.step = node
    original_world = fixed.base.make_validation_env
    fixed.base.make_validation_env = lambda contract: attach_frame_observer(original_world(contract))
    original_range = fixed.RangeImageEnv
    fixed.RangeImageEnv = lambda *args, **kwargs: attach_frame_observer(original_range(*args, **kwargs))


def heldout_group(cases):
    original = fixed.follow
    rows = []
    for case in cases:
        trace = []
        if case["suite"] in ("A_aperture", "C"):
            def follow(env, agent, builder, obs, info, wall):
                def record(raw):
                    position = env.base_env.pos[0]
                    trace.append(dict(time_s=env.upper_tick / 15, x_m=float(position[0]), z_m=float(position[2]),
                                      d_min_m=float(raw["modulation_d_min"]),
                                      **dict(zip(RESULT_FIELDS, map(float, raw["modulation"])))))

                step = env.step
                record(obs)

                def track(action):
                    result = step(action)
                    if not (result[2] or result[3]):
                        record(result[0])
                    return result

                env.step = track
                return original(env, agent, builder, obs, info, wall)

            fixed.follow = follow
        try:
            row = fixed.heldout_case(case)
        finally:
            fixed.follow = original
        if trace:
            row["modulation_trace"] = trace
        rows.append(row)
    return rows


def eval_abc(registry, manifest, workers):
    context = multiprocessing.get_context("spawn")
    fingerprint = manifest["frozen_training_pool_fingerprint"]
    reference = [(r["stage"], r["case"], r["init"]) for r in fixed.archived_fixed(T4, "t3", 240000)["rows"]]
    for (seed, node), entry in registry.items():
        for held in (False, True) if node in FINAL_NODES else (False,):
            folder = "heldout_results" if held else "standard_results"
            target = HERE / folder / f"{entry['id']}.json"
            target.parent.mkdir(exist_ok=True)
            if target.is_file():
                document = read_json(target)
                if document["checkpoint_sha256"] != entry["sha256"]:
                    raise RuntimeError("T5 evaluation cache has a different checkpoint")
                rows = document["rows"]
            else:
                with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=load_worker,
                                         initargs=("dynamic", seed, node, Path(entry["path"]), fingerprint)) as pool:
                    if held:
                        maps = manifest["maps"]
                        jobs = [pool.submit(heldout_group, maps[i:i+4]) for i in range(0, len(maps), 4)]
                    else:
                        jobs = [pool.submit(fixed.fixed_forest, stage, range(i, i+5))
                                for stage in range(4) for i in range(0, 20, 5)]
                    rows = [row for job in jobs for row in job.result()]
                rows.sort(key=(lambda r: r["id"]) if held else (lambda r: (r["stage"], r["case"])))
            if held:
                archived = read_json(T4 / folder / f"t3_{seed}_{node}.json")
                if (archived["suite_fingerprint"] != manifest["fingerprint"] or len(rows) != 144
                        or [(r["id"], r["init"]) for r in rows] !=
                        [(r["id"], r["init"]) for r in archived["rows"]]):
                    raise RuntimeError("T5 sealed evaluations do not pair against T4")
            else:
                fixed.base.verify_initializations(rows, reference)
            write_json(target, dict(entry=entry, checkpoint_sha256=entry["sha256"], rows=rows,
                                    suite_fingerprint=manifest["fingerprint"] if held else None))
            print(f"{entry['id']} {folder}: {len(rows)} matched", flush=True)


def suite_d_modules():
    sys.path[:0] = [str(D), str(T4)]
    import suite_d.run as runner
    import suite_d.report as summary
    return runner, summary


def load_d_worker(entry, fingerprint):
    runner, _ = suite_d_modules()
    load_worker("dynamic", entry["seed"], entry["checkpoint"], Path(entry["path"]), fingerprint)
    original = runner.make_env
    runner.make_env = lambda *args, **kwargs: attach_frame_observer(original(*args, **kwargs))


def eval_d_case(case, folder, entry, fingerprint):
    runner, _ = suite_d_modules()
    with open(os.devnull, "w") as stream, contextlib.redirect_stdout(stream):
        return runner.evaluate_case(case, folder, entry, fingerprint)


def eval_d(registry, workers):
    runner, summary = suite_d_modules()
    manifest = runner.load_manifest(D / "manifest.json")
    fingerprint = manifest["fingerprint"]
    root = HERE / "suite_d_results"
    root.mkdir(exist_ok=True)
    contexts = multiprocessing.get_context("spawn")
    cases = manifest["cases"]
    aggregations, per_case = {}, []
    for (seed, node), entry in registry.items():
        if node not in FINAL_NODES:
            continue
        folder = root / entry["id"]
        folder.mkdir(exist_ok=True)
        with ProcessPoolExecutor(max_workers=workers, mp_context=contexts, initializer=load_d_worker,
                                 initargs=(entry, manifest["training_pool_fingerprint"])) as pool:
            jobs = [pool.submit(eval_d_case, case, folder, entry, fingerprint) for case in cases]
            dynamic = [job.result() for job in jobs]
        for model in BASELINES:
            archived = [read_json(D / "results_verified" / f"{model}_{seed}_{node}" / f"{c['id']}.json")
                        for c in cases]
            for left, right in zip(archived, dynamic):
                key = lambda row: (row["id"], row["initial_digest"], row["geometry_sha256"], row["suite_fingerprint"])
                if key(left) != key(right):
                    raise RuntimeError("Suite D initial state pairing failed")
                per_case.append(dict(seed=seed, node=node, model=model, id=left["id"],
                                     topology=left["topology"], dynamic_terminal=right["terminal"],
                                     reference_terminal=left["terminal"], dynamic_failure=right["failure_bucket"],
                                     reference_failure=left["failure_bucket"],
                                     dynamic_geo=right["geodesic_progress_ratio"],
                                     reference_geo=left["geodesic_progress_ratio"]))
            for topology in (*summary.TOPOLOGIES, "all"):
                groups = aggregations.setdefault(node, {}).setdefault(topology, {}).setdefault(model, {})
                lhs, rhs = ([r for r in rows if topology in ("all", r["topology"])]
                            for rows in (archived, dynamic))
                groups[seed] = dict(reference=summary.aggregate(lhs), dynamic=summary.aggregate(rhs))
        print(f"{entry['id']} Suite D: {len(dynamic)} matched", flush=True)
    for node in aggregations.values():
        for group in node.values():
            for model, values in group.items():
                means = {metric: report.moments([seed[metric] for seed in values.values()])
                         for metric in ("reference", "dynamic")}
                group[model] = dict(per_seed=values, across_seeds=means)
    write_csv(HERE / "t5_suite_d_pairs.csv", per_case)
    write_json(HERE / "suite_d_summary.json", dict(fingerprint=fingerprint, results=aggregations))


def efficiency(registry, manifest, rss):
    import t4_efficiency as eff
    from train import cuda_setup
    fingerprint = manifest["frozen_training_pool_fingerprint"]
    entry = registry[43, 240000]
    cuda_setup(43)
    agent, builder, contract = load_policy(entry["path"], "cuda", 43, 240000, fingerprint)
    original = eff.make_validation_env
    eff.make_validation_env = lambda spec: attach_frame_observer(original(spec))
    tape = read_json(T4 / "efficiency_action_tape.json")
    timing, _ = eff.latency(agent, builder, contract, tape, False)
    return dict(dynamic=dict(parameters=eff.parameter_totals(agent), latency=timing, replay=rss),
                frozen_t4=read_json(T4 / "efficiency.json"))


def memory():
    from t4_efficiency import private_resident_bytes
    from train import load_config
    torch.cuda.init()
    cuda_before = torch.cuda.memory_allocated()
    before = private_resident_bytes()
    replay = DynamicReplay(load_config()["td3"], WIDTH, 43)
    arrays = [v for ring in (replay.main, *replay.success) for v in vars(ring).values()
              if isinstance(v, np.ndarray)]
    for array in arrays:
        view = array.view(np.uint8).reshape(-1)
        view[::4096] = 1
        view[-1] = 1
    return dict(backing_array_bytes=sum(a.nbytes for a in arrays),
                modulation_dense_bytes=sum(r.modulation.nbytes for r in (replay.main, *replay.success)),
                resident_delta_bytes=private_resident_bytes() - before,
                cuda_tensor_delta_bytes=torch.cuda.memory_allocated() - cuda_before,
                scope="Dense buffers page-touched; sparse terminal and live-edge payload and in-flight episode excluded.")


def comparisons(held):
    folder = "heldout_results" if held else "standard_results"
    output, csv_rows = {}, []
    for node in FINAL_NODES if held else NODES:
        grouped = {(seed, model): report.split_groups(
            read_json((HERE if model == "dynamic" else T4) / folder / f"{model}_{seed}_{node}.json")["rows"],
            not held) for seed, model in product(SEEDS, ("dynamic", *BASELINES))}
        by_model = {}
        for model in ("dynamic", *BASELINES):
            by_model[model] = {}
            for scope in grouped[43, model]:
                values = {seed: report.grouped_metrics(grouped[seed, model][scope]) for seed in SEEDS}
                stats = report.moments(list(values.values()))
                by_model[model][scope] = dict(per_seed=values, mean_sd=stats)
                csv_rows.extend(dict(node=node, model=model, scope=scope, metric=metric, **summary)
                                for metric, summary in stats.items())
        paired = {model: report.paired({(seed, key): grouped[seed, source] for seed in SEEDS
                                       for key, source in (("t3", "dynamic"), ("a1", model))}, not held)
                  for model in BASELINES}
        output[node] = dict(models=by_model, dynamic_vs_reference=paired)
    write_csv(HERE / f"t5_{folder}_comparison.csv", csv_rows)
    return output


def curves():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [row for seed in SEEDS
            for row in read_json(HERE / "heldout_results" / f"dynamic_{seed}_240000.json")["rows"]
            if "modulation_trace" in row]
    for suite in ("C", "A_aperture"):
        successful = lambda r: r["success"] and (suite != "A_aperture" or r["passed"])
        selected = [next((r for r in rows if r["suite"] == suite and successful(r) == success), None)
                    for success in (True, False)]
        fig, axes = plt.subplots(3, 1, figsize=(9, 8), layout="constrained")
        for row, style in zip(selected, ("-", "--")):
            if row is None:
                continue
            trace = row["modulation_trace"]
            times = [r["time_s"] for r in trace]
            for ax, keys in zip(axes, (("d_min_m", "z_m"), RESULT_FIELDS[:4], RESULT_FIELDS[4:])):
                for key in keys:
                    ax.plot(times, [r[key] for r in trace], style, label=f"{key} · {row['id']}")
                ax.legend(fontsize=7)
                ax.grid(alpha=.25)
        axes[0].set_title(f"T5 v2.2 {suite} — representative closed-loop rollouts")
        axes[-1].set_xlabel("Time (s)")
        fig.savefig(HERE / f"t5_modulation_{suite}.png", dpi=170)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--phase", choices=("all", "suite-d", "memory"), default="all")
    args = parser.parse_args()
    if args.phase == "memory":
        write_json(HERE / "t5_replay_memory.json", memory())
        return
    manifest = fixed.load_suite(T4 / "heldout_suite/manifest.json")
    registry = identify(args.runs, manifest["frozen_training_pool_fingerprint"])
    if args.phase == "suite-d":
        eval_d(registry, args.workers)
        return
    eval_abc(registry, manifest, args.workers)
    for phase in ("suite-d", "memory"):
        subprocess.run([sys.executable, str(Path(__file__)), str(args.runs), "--phase", phase,
                        "--workers", str(args.workers)], check=True)
    measured = efficiency(registry, manifest, read_json(HERE / "t5_replay_memory.json"))
    write_csv(HERE / "t5_efficiency.csv", [
        dict(model=model, mean_ms=latency["mean_ms"], p99_ms=latency["p99_ms"])
        for model, latency in {"dynamic": measured["dynamic"]["latency"],
                               **measured["frozen_t4"]["latency"]}.items()])
    curves()
    logs = {seed: dict(curriculum=read_lines(dynamic_path(args.runs, seed).parent / "curriculum_promotions.jsonl"),
                      probes=read_lines(dynamic_path(args.runs, seed).parent / "probes_002000.jsonl"))
            for seed in SEEDS}
    write_json(HERE / "t5_summary_report.json", dict(
        step0=read_json(HERE / "preflight_v22_check.json"), interface=read_json(HERE / "interface_check.json"),
        sealed_abc_fingerprint=manifest["fingerprint"], fixed=comparisons(False), heldout=comparisons(True),
        suite_d=read_json(HERE / "suite_d_summary.json"), training=logs, efficiency=measured))


if __name__ == "__main__":
    main()
