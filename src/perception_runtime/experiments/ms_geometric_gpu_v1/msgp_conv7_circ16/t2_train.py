"""Run the T2 original-course trainer and log its live curriculum counters."""
import argparse
import json
from pathlib import Path
import sys

import torch

import train
from agent import Actor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    inputs = Path("/opt/uav-training-20261103/runs")
    decision = json.loads((inputs / "msgp_conv7_t0_rv1_e11_seed43_20261120a/t0_decision.json").read_text())
    if decision["status"] != "complete" or decision["selected_course_for_t2_t3"] != "original_rv1":
        raise RuntimeError("T2 requires the audited original course")
    accepted = inputs / "msgp_conv7_t1_seed43_20261120b"
    near = json.loads((accepted / "near-only/metadata.json").read_text())
    tests = json.loads((accepted / "acceptance.json").read_text())
    contract = near["contract"]
    cfg = train.load_config()
    if (cfg["seed"], contract["seed"], contract["course"], contract["observation_width"],
            contract["use_far_branch"], Actor(False).policy[0].in_features) != (
            43, 43, "original", 704, False, 50):
        raise RuntimeError("T1 near-only checkpoint contract mismatch")
    if not (tests["cuda_smoke"]["exact_128_vs_64_plus_64"] and tests["latency"]["p99_ms"] < 1000 / 15
            and tests["isolation"]["exact"] and tests["geometry"]["e06_mirror_pairs"] == 9):
        raise RuntimeError("T1 verification is incomplete")
    for name, expected in contract["source_sha256"].items():
        if train.sha(Path(__file__).with_name(name)) != expected:
            raise RuntimeError(f"T1 source changed: {name}")
    baseline = json.loads((inputs / "msgp_conv7_t0_rv1_e11_seed43_20261120a/metadata.json").read_text())["config"]
    audited = ("td3", "reward_scale", "reward_kwargs", "control_spec", "curriculum")
    if any(json.loads(json.dumps(cfg[k])) != baseline[k] for k in audited):
        raise RuntimeError("RV1 task or TD3 config changed after T0")
    if not torch.cuda.is_available():
        raise RuntimeError("training needs CUDA")

    original_factory, original_log = train.make_course_env, train.log
    env = None

    def remember_env(*params, **kwargs):
        nonlocal env
        env = original_factory(*params, **kwargs)
        return env

    def log_course(path, record):
        if path.name in ("episodes.jsonl", "metrics.jsonl"):
            details = env.course_info()
            course = {key: value for key, value in details.items()
                      if key.startswith(("start_curriculum_", "density_curriculum_",
                                         "obstacle_curriculum_"))}
            course.update(density_stage=details["density_stage"],
                          active_tree_count=details["active_tree_count"])
            record = dict(record, course=course)
        original_log(path, record)

    train.make_course_env = remember_env
    train.log = log_course
    sys.argv = [str(train.__file__), "--run-dir", str(args.run_dir), "--end", "120000",
                "--interval", "10000", "--course", "original", "--no-use-far-branch"]
    if args.resume:
        sys.argv += ["--resume", str(args.resume)]
    print("T2 preflight passed: Seed43, original course, near-only 50D, CUDA, 120,000 steps", flush=True)
    train.main()


if __name__ == "__main__":
    main()
