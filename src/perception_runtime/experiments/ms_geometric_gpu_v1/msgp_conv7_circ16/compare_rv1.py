"""Compare T2 with archived RV1 at matched training steps and fixed forest cases."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pathlib import Path

import torch

from experiments.direct_velocity_full_v1.task_factory import load_config
from experiments.rv1_range_cnn.td3 import RangeTD3

from t0_eval import identity
import t2_eval as harness
from train import sha


def evaluate_rv1(checkpoint, fingerprint, stage, cases):
    torch.set_num_threads(1)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = load_config()
    agent = RangeTD3(config["td3"], state["contract"], device="cpu")
    agent.actor.load_state_dict(state["agent"]["actor"])
    agent.actor.eval()
    harness.agent = agent
    harness.builder = identity
    harness.checkpoint_step = state["trainer"]["step"]
    harness.contract = dict(sensor=state["contract"]["sensor"], pool_fingerprint=fingerprint,
                            lidar_extraction="parallel_pybullet_v1")
    return harness.evaluate_cases(stage, cases)


def audit_node(runs, baseline, step):
    checkpoint = baseline / f"checkpoint_{step:06d}.pt"
    archive = runs / "formal_gpu_seed43_seven_nodes_eval_20261118a" / f"a1_{step:06d}"
    recorded = harness.read_json(archive / "protocol.json")
    if sha(checkpoint) != recorded["checkpoint_sha256"]:
        raise RuntimeError(f"RV1 checkpoint checksum mismatch at {step}")
    pack = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if pack["trainer"]["step"] != step or pack["contract"]["latent_dim"] != 64:
        raise RuntimeError(f"unexpected RV1 checkpoint at {step}")
    course = pack["environment"]["course"]
    course_stats = dict(start_level=course["start_scheduler"]["level"],
                        obstacle_stage=course["obstacle_scheduler"]["stage"],
                        eligible_start_episodes=course["start_scheduler"]["level_episodes"],
                        eligible_obstacle_episodes=course["obstacle_scheduler"]["stage_episodes"])
    return checkpoint, recorded, course_stats


def aggregate(rows):
    return {str(stage): harness.group_stats([r for r in rows if r["stage"] == stage]) for stage in range(4)}


def match(t2, rv1):
    t2_by_key = {(r["stage"], r["case"]): r for r in t2}
    rv1_by_key = {(r["stage"], r["case"]): r for r in rv1}
    if len(t2_by_key) != 80 or len(rv1_by_key) != 80 or t2_by_key.keys() != rv1_by_key.keys():
        raise RuntimeError("expected 80 common forest cases")
    comparison = {}
    for key, row in t2_by_key.items():
        other = rv1_by_key[key]
        if row["init"] != other["init"]:
            raise RuntimeError(f"map, target, or starting state mismatch at {key}")
        stage = str(key[0])
        counts = comparison.setdefault(stage, Counter())
        label = (
            "both_success" if row["success"] and other["success"]
            else "t2_only" if row["success"]
            else "rv1_only" if other["success"]
            else "neither_success"
        )
        counts[label] += 1
    return comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    runs = args.run.parent
    baseline = runs / "formal_gpu_a1_seed43_240k_20261118a"
    metadata = harness.read_json(baseline / "metadata.json")
    t2_meta = harness.read_json(args.run / "train/metadata.json")
    settings = ("seed", "td3", "reward_scale", "reward_kwargs", "control_spec", "curriculum")
    if any(metadata["config"][key] != t2_meta["config"][key] for key in settings):
        raise RuntimeError("RV1 and T2 frozen settings mismatch")
    audited = {step: audit_node(runs, baseline, step) for step in harness.NODES}
    rows = {step: [] for step in harness.NODES}
    args.output.mkdir(parents=True, exist_ok=False)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        jobs = {}
        for step, (checkpoint, recorded, _) in audited.items():
            if recorded["pool_fingerprint"] != t2_meta["contract"]["pool_fingerprint"]:
                raise RuntimeError("RV1 and T2 validation pool mismatch")
            for stage in range(4):
                for case in range(0, 20, 5):
                    job = pool.submit(evaluate_rv1, checkpoint, recorded["pool_fingerprint"],
                                      stage, range(case, case + 5))
                    jobs[job] = step
        for future in as_completed(jobs):
            step, group = future.result()
            if step != jobs[future]:
                raise RuntimeError("incorrect RV1 evaluation node")
            rows[step].extend(group)
            print(f"RV1 {step} {len(rows[step])}/80", flush=True)

    comparison = {}
    for step, raw in rows.items():
        _, recorded, course = audited[step]
        raw.sort(key=lambda r: (r["stage"], r["case"]))
        t2 = harness.read_json(args.run / f"evaluation_{step:06d}.json")
        rv1 = aggregate(raw)
        comparison[str(step)] = dict(rv1=rv1, t2=t2["stages"], paired=match(t2["rows"], raw),
                                     rv1_course=course, t2_course=t2["training"]["course"],
                                     t2_exposure=t2["training"]["exposure"],
                                     rv1_checkpoint_sha256=recorded["checkpoint_sha256"],
                                     t2_checkpoint_sha256=t2["checkpoint_sha256"])
        harness.write_json(args.output / f"rv1_{step:06d}.json", dict(step=step, stages=rv1, rows=raw))
    harness.write_json(args.output / "rv1_t2_comparison.json", dict(seed=43, evaluation_device="cpu",
                       cases_per_stage=20, case_match="map, geometry hash, target, and full initial state",
                       nodes=comparison))


if __name__ == "__main__":
    main()
