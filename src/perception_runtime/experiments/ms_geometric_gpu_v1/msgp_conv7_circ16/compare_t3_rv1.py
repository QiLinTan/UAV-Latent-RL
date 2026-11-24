"""Pair T3 continuation results with archived RV1 A1 on the same scenes."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pathlib import Path

from experiments.rv1_range_cnn.environment import RangeImageEnv

from compare_rv1 import audit_node, match
from evaluate import realtime_decisions
from sensor import use_parallel_lidar
from t0_eval import APERTURES, follow, initial_info, paired_map, through
from t2_eval import read_json, write_json
import t3_eval as evaluator


def physical_apertures(case):
    wall, offset, width = APERTURES[case]
    rows = []
    for side in (-1, 1):
        env = use_parallel_lidar(RangeImageEnv(map_spec=paired_map(case, side)))
        try:
            obs, info = env.reset(seed=1000 + case)
            init = initial_info(env, info)
            with realtime_decisions():
                row, trace = follow(env, evaluator.agent, evaluator.builder, obs, info, wall=wall)
            rows.append(dict(case=case, side=side, init=init,
                             passed=through(trace, wall, side * offset, width), **row))
        finally:
            env.close()
    return evaluator.step, rows


def evaluate_rv1(node, checkpoint, fingerprint, workers):
    forest, apertures = [], []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=evaluator.init_worker,
                             initargs=("a1", checkpoint, fingerprint)) as pool:
        tasks = {pool.submit(evaluator.forest, stage, range(case, case + 5)): forest
                 for stage in range(4) for case in range(0, 20, 5)}
        if node == 240000:
            tasks.update({pool.submit(physical_apertures, case): apertures for case in range(12)})
        for job in as_completed(tasks):
            step, rows = job.result()
            if step != node:
                raise RuntimeError(f"evaluated incorrect checkpoint {step}")
            tasks[job].extend(rows)
    forest.sort(key=lambda row: (row["stage"], row["case"]))
    apertures.sort(key=lambda row: (row["case"], row["side"]))
    return forest, apertures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("t3_results", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    runs = args.run.parent
    baseline = runs / "formal_gpu_a1_seed43_240k_20261118a"
    a1_config = read_json(baseline / "metadata.json")["config"]
    t3_meta = read_json(args.run / "train/metadata.json")
    settings = ("seed", "td3", "reward_scale", "reward_kwargs", "control_spec", "curriculum")
    if any(a1_config[key] != t3_meta["config"][key] for key in settings):
        raise RuntimeError("RV1 and T3 training settings differ")
    summary = read_json(args.t3_results / "summary_240k.json")
    aperture_t3 = read_json(args.t3_results / "e12_240000.json")
    comparison = {}
    actual = None
    for node in (200000, 240000):
        checkpoint, archive, course = audit_node(runs, baseline, node)
        fingerprint = archive["pool_fingerprint"]
        if fingerprint != t3_meta["contract"]["pool_fingerprint"]:
            raise RuntimeError("RV1 and T3 validation pools differ")
        forest, apertures = evaluate_rv1(node, checkpoint, fingerprint, args.workers)
        t3 = read_json(args.t3_results / f"t3_{node:06d}.json")
        if t3["checkpoint_sha256"] != summary["checkpoints"][str(node)]:
            raise RuntimeError("mismatched T3 checkpoint in existing result")
        paired = {stage: {"t3_only" if name == "t2_only" else name: count for name, count in group.items()}
                  for stage, group in match(t3["rows"], forest).items()}
        a1_stages = {str(stage): evaluator.count_outcomes([r for r in forest if r["stage"] == stage])
                     for stage in range(4)}
        comparison[node] = dict(
            a1=dict(stages=a1_stages, totals=evaluator.count_outcomes(forest), course=course,
                    checkpoint_sha256=archive["checkpoint_sha256"]),
            t3=dict(stages=t3["stages"], totals=t3["totals"], checkpoint_sha256=t3["checkpoint_sha256"]),
            paired_arrival=paired)
        write_json(args.output / f"a1_{node:06d}.json", dict(step=node, forest=forest, apertures=apertures))
        print(f"Verified {node}: {len(forest)} paired forest scenes", flush=True)
        if node == 240000:
            t3_real = {(pair["case"], row["side"]): row["init"]
                       for pair in aperture_t3["cases"] for row in pair["real"]}
            if len(apertures) != len(t3_real) or any(
                    row["init"] != t3_real[row["case"], row["side"]] for row in apertures):
                raise RuntimeError("mismatched physical aperture starts")
            actual = dict(a1=dict(episodes=len(apertures), passed=sum(r["passed"] for r in apertures),
                                  arrived=sum(r["success"] for r in apertures),
                                  passed_and_arrived=sum(r["passed"] and r["success"] for r in apertures),
                                  reasons=dict(Counter(r["reason"] for r in apertures))),
                          t3=aperture_t3["summary"]["actual_apertures"])
    write_json(args.output / "comparison.json",
               dict(seed=43, nodes=comparison, physical_apertures_240k=actual,
                    t3_training_exposure=summary["final_training"]["exposure"]))


if __name__ == "__main__":
    main()
