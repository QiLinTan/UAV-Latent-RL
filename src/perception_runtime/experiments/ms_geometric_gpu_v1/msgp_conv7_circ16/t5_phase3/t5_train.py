"""Run v2.2 with the original T3 rollout, course, TD3, and checkpoint loop."""

import argparse
import copy
import gc
import json
from pathlib import Path
import sys

import numpy as np

from dynamic_v22 import DynamicAgent, DynamicBuilder, DynamicReplay, MODE, WIDTH, attach_frame_observer
from t3_train import WINDOW, SPEED_SCALE, velocity_stats, verify_contract
import train

HERE = Path(__file__).resolve().parent
RUN_NAME = "msgp_conv7_circ16_dyn_v22_seed{seed}_240k"
SCHEMA = "msgp_conv7_circ16_dyn_v22"
NODES = (80000, 120000, 160000, 200000, 240000)


class Telemetry:
    def __init__(self, root):
        self.root = root
        self.factory = train.make_course_env
        self.write = train.log
        self.step = None
        self.actions = []
        self.gates = []
        self.updates = {}

    def make_env(self, *args, **kwargs):
        self.env = attach_frame_observer(self.factory(*args, **kwargs))
        step = self.env.step
        reset = self.env.reset

        def collect(action):
            result = step(action)
            if self.step is not None:
                self.step += 1
                self.actions.append(result[-1]["projected_action"] * SPEED_SCALE)
                self.gates.append(self.current_gate)
                self.current_gate = result[0]["modulation"]
            return result

        def restart(*args, **kwargs):
            if gc.isenabled():
                gc.collect()
                gc.disable()
            result = reset(*args, **kwargs)
            self.current_gate = result[0]["modulation"]
            return result

        observation = self.env._observation

        def remember(state):
            raw = observation(state)
            if self.step is None and not self.env._defer_scan:
                self.current_gate = raw["modulation"]
            return raw

        self.env._observation = remember
        self.env.step = collect
        self.env.reset = restart
        return self.env

    def course(self):
        return self.env.start_curriculum.level, self.env.obstacle_curriculum.stage

    def log(self, path, row):
        name = path.name
        if name == "events.jsonl" and row["event"] in ("start", "resume"):
            self.step = row["step"]
            self.previous_course = self.course()
        if name == "episodes.jsonl":
            current = self.course()
            if current != self.previous_course:
                self.write(self.root / "curriculum_promotions.jsonl", dict(
                    step=row["step"], old_start_level=self.previous_course[0], start_level=current[0],
                    old_obstacle_stage=self.previous_course[1], obstacle_stage=current[1]))
                self.previous_course = current
        if name == "updates.jsonl" and "branches" in row:
            for key, value in row["branches"].items():
                self.updates[key] = value
        if name == "metrics.jsonl" and row["step"] % WINDOW == 0:
            self.flush(row["step"])
        self.write(path, row)

    def flush(self, step):
        if self.step != step or len(self.actions) != WINDOW or len(self.gates) != WINDOW:
            raise RuntimeError("incomplete T5 telemetry window")
        values = np.asarray(self.gates, dtype=float)
        gates = {}
        for name, column in zip(("alpha_t", "s_t"), (4, 5)):
            data = values[:, column]
            p05, p95 = np.percentile(data, [5, 95])
            gates[name] = dict(mean=float(data.mean()), std=float(data.std()), p05=float(p05), p95=float(p95))
        branches = {}
        for key, probe in self.updates.items():
            branches[key] = dict(probe, near_over_far_gradient=probe["near_gradient_l2"] / max(
                1e-12, probe["far_gradient_l2"]), near_over_far_output=probe["near_feature_l2"] / max(
                1e-12, probe["far_feature_l2"]), near_over_far_modulated=probe["near_modulated_l2"] / max(
                1e-12, probe["far_modulated_l2"]))
        self.write(self.root / "probes_002000.jsonl",
                   dict(step=step, begin_step=step - WINDOW + 1, course=self.course(), gates=gates,
                        behavior=velocity_stats(np.asarray(self.actions)), branches=branches))
        self.actions.clear()
        self.gates.clear()
        self.updates.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int, choices=(43, 44, 45))
    parser.add_argument("runs", type=Path)
    parser.add_argument("--end", type=int, choices=NODES, default=240000)
    parser.add_argument("--resume", type=int, choices=NODES)
    parser.add_argument("--smoke-dir", type=Path)
    args = parser.parse_args()
    verify_contract()
    gate = json.loads((HERE / "preflight_v22_check.json").read_text(encoding="utf-8"))
    smoke = args.smoke_dir is not None
    if smoke and args.resume:
        parser.error("smoke run starts fresh")
    end = 64 if smoke else args.end
    if args.resume is not None and args.resume >= end:
        parser.error("resume node must precede end")
    root = args.smoke_dir if smoke else args.runs / RUN_NAME.format(seed=args.seed)
    if args.resume is None:
        root.mkdir(parents=True, exist_ok=False)
    run = root / "train"
    original_config = train.load_config
    original_specifications = train.specifications
    t4 = json.loads((args.runs / "msgp_conv7_circ16_dual_seed43_240k/train/metadata.json").read_text())

    def load_config():
        config = copy.deepcopy(original_config())
        fields = ("td3", "reward_scale", "reward_kwargs", "control_spec", "curriculum")
        if any(json.loads(json.dumps(config[k])) != t4["config"][k] for k in fields):
            raise RuntimeError("T5 configuration does not match frozen T4")
        config["seed"] = args.seed
        if smoke:
            config["td3"].update(buffer_size=80, success_replay_size=32)
        return config

    def specifications(config, env, pool, use_far, course, short):
        spec = original_specifications(config, env, pool, use_far, course, short)
        if spec["pool_fingerprint"] != t4["contract"]["pool_fingerprint"]:
            raise RuntimeError("T5 training pool does not match frozen T4")
        return dict(spec, schema=SCHEMA, observation_width=WIDTH, fusion_mode=MODE,
                    replay_contract="float32_six_per_observation_plus_exact_ring_edges",
                    modulation_constants=gate["constants"],
                    preflight_sha256=train.sha(HERE / "preflight_v22_check.json"),
                    preflight_passed=gate["passed"], authorized_start_20261124=True,
                    t5_source_sha256={name: train.sha(HERE / name) for name in
                                      ("modulation_v22.py", "dynamic_v22.py", "t5_train.py")})

    recorder = Telemetry(root)
    train.make_course_env, train.log = recorder.make_env, recorder.log
    train.specifications, train.load_config = specifications, load_config
    train.MSGPAgent, train.ObservationBuilder, train.BalancedReplay = DynamicAgent, DynamicBuilder, DynamicReplay
    sys.argv = [train.__file__, "--run-dir", str(run), "--end", str(end), "--interval",
                "64" if smoke else ("80000" if args.resume is None else "40000"), "--course", "original"]
    if smoke:
        sys.argv += ["--short-check"]
    if args.resume is not None:
        sys.argv += ["--resume", str(run / f"checkpoint_{args.resume:06d}.pt")]
    try:
        train.main()
    finally:
        gc.enable()


if __name__ == "__main__":
    main()
