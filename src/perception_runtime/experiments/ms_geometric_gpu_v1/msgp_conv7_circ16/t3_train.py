"""Launch the frozen dual-branch trainer with read-only T3 telemetry hooks."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch

from agent import Actor, project
from evaluate import load_policy, make_validation_env, realtime_decisions
from perception import ObservationBuilder, unpack
from t2_eval import write_json
import train

RUNS = Path("/opt/uav-training-20261103/runs")
T2 = RUNS / "msgp_conv7_t2_near_seed43_120k_20261120a"
T1 = RUNS / "msgp_conv7_t1_seed43_20261120b"
WINDOW = 2000
SPEED_SCALE = np.array([.8, .8, .3])


def fixed_panel(root):
    """Create one balanced panel of real observations before the first T3 update."""
    path = root / "fixed_actor_panel.npy"
    provenance = root / "fixed_actor_panel.json"
    if path.is_file() and provenance.is_file():
        expected = json.loads(provenance.read_text())
        if train.sha(path) != expected["sha256"]:
            raise RuntimeError("fixed Actor panel digest mismatch")
        panel = np.load(path, allow_pickle=False)
        if panel.shape != (2048, 784) or not np.isfinite(panel).all():
            raise RuntimeError("invalid fixed Actor panel")
        return panel
    torch.set_num_threads(1)
    checkpoint = T2 / "train/checkpoint_080000.pt"
    old, old_builder, contract, step = load_policy(checkpoint, "cpu")
    report = json.loads((T2 / "evaluation_080000.json").read_text())
    digest = train.sha(checkpoint)
    if step != 80000 or digest != report["checkpoint_sha256"] or contract["use_far_branch"]:
        raise RuntimeError("unexpected T2 reference for fixed panel")
    env = make_validation_env(contract)
    builder = ObservationBuilder("cpu", env.lidar.config, True)
    features, cases = [], []
    try:
        for stage in range(4):
            env.obstacle_curriculum.stage = stage
            for seed in (0, 5, 10, 15):
                taken = 0
                active_seed = seed
                pieces = []
                while taken < 128:
                    obs, info = env.reset(seed=active_seed)
                    if info["density_stage"] != stage or info["start_curriculum_source"] != "original":
                        raise RuntimeError("fixed panel initialization mismatch")
                    begin = taken
                    with realtime_decisions():
                        while taken < 128:
                            features.append(builder(obs).numpy().copy())
                            obs, _, terminal, truncated, _ = env.step(old.action(old_builder(obs)))
                            taken += 1
                            if terminal or truncated:
                                break
                    pieces.append(dict(seed=active_seed, observations=taken - begin))
                    active_seed = (active_seed + 1) % 20
                cases.append(dict(stage=stage, observations=128, pieces=pieces))
    finally:
        env.close()
    panel = np.stack(features)
    if panel.shape != (2048, 784) or not np.isfinite(panel).all():
        raise RuntimeError("failed to build fixed Actor panel")
    with path.open("xb") as stream:
        np.save(stream, panel, allow_pickle=False)
    write_json(provenance, dict(n=len(panel), sha256=train.sha(path), reference_t2_step=step,
                               reference_t2_checkpoint_sha256=digest, cases=cases))
    return panel


def velocity_stats(actions):
    speed = np.linalg.norm(actions[:, :2], axis=1)
    return dict(n=len(actions), mean=actions.mean(axis=0).tolist(), std=actions.std(axis=0).tolist(),
                mean_abs_vy=float(np.abs(actions[:, 1]).mean()),
                xy_saturated_fraction=float((speed >= .99 * .8).mean()))


class Telemetry:
    def __init__(self, root, panel):
        self.root = root
        self.panel = panel
        self.panel_cuda = None
        self.env = self.agent = None
        self.initial_actor_panel = None
        self.step = None
        self.actions = []
        self.updates = []
        self.branches = defaultdict(list)
        self.last_promotion = None
        self.factory = train.make_course_env
        self.agent_factory = train.MSGPAgent
        self.log = train.log

    def make_env(self, *args, **kwargs):
        self.env = self.factory(*args, **kwargs)
        original = self.env.step

        def step(action):
            outcome = original(action)
            if self.step is not None:
                self.step += 1
                self.actions.append(outcome[-1]["projected_action"] * SPEED_SCALE)
            return outcome

        self.env.step = step
        return self.env

    def make_agent(self, *args, **kwargs):
        self.agent = self.agent_factory(*args, **kwargs)
        if not self.agent.use_far_branch or self.agent.actor.policy[0].in_features != 82:
            raise RuntimeError("T3 requires the dual-branch 82D policy")
        return self.agent

    @torch.no_grad()
    def actor_stats(self):
        if self.panel_cuda is None:
            self.panel_cuda = torch.from_numpy(self.panel).to(self.agent.device)
        commands = torch.cat([project(self.agent.actor(unpack(batch, True)))
                              for batch in self.panel_cuda.split(128)]).cpu().numpy() * SPEED_SCALE
        return velocity_stats(commands)

    def course(self):
        info = self.env.course_info()
        return {key: val for key, val in info.items() if key.startswith((
            "start_curriculum_", "obstacle_curriculum_", "density_stage", "active_tree_count"))}

    def records(self, path, row):
        name = path.name
        if name == "events.jsonl" and row["event"] in ("start", "resume"):
            self.step = row["step"]
            c = self.course()
            self.last_promotion = c["start_curriculum_level"], c["obstacle_curriculum_stage"]
            if self.step == 0:
                self.initial_actor_panel = self.actor_stats()
                write_json(self.root / "actor_panel_step0.json", self.initial_actor_panel)
            else:
                self.initial_actor_panel = json.loads((self.root / "actor_panel_step0.json").read_text())
        if name == "episodes.jsonl":
            c = self.course()
            current = c["start_curriculum_level"], c["obstacle_curriculum_stage"]
            if current != self.last_promotion:
                self.log(self.root / "curriculum_promotions.jsonl", dict(
                    step=row["step"], old_start_level=self.last_promotion[0], start_level=current[0],
                    old_obstacle_stage=self.last_promotion[1], obstacle_stage=current[1]))
                self.last_promotion = current
            row = dict(row, course=c)
        if name == "updates.jsonl":
            self.updates.append((row["critic_loss"], (row["q1_mean"] + row["q2_mean"]) / 2, row["q_abs_max"]))
            for network, values in row.get("branches", {}).items():
                self.branches[network].append(values)
        if name == "metrics.jsonl":
            row = dict(row, course=self.course())
            if row["step"] % WINDOW == 0:
                self.flush(row)
        self.log(path, row)

    def flush(self, row):
        step = row["step"]
        if self.step != step or len(self.actions) != WINDOW:
            raise RuntimeError("incomplete T3 action window")
        actor = self.actor_stats()
        q = np.asarray(self.updates).reshape(-1, 3)
        branch_probe = {}
        for name, snapshots in self.branches.items():
            keys = [k for k in snapshots[0] if k != "step"]
            values = {k: float(np.mean([r[k] for r in snapshots])) for k in keys}
            near, far = values["near_gradient_l2"], values["far_gradient_l2"]
            values["near_over_far_gradient_l2"] = near / far if far else None
            values["larger_over_smaller_gradient_l2"] = max(near, far) / min(near, far) if min(near, far) else None
            branch_probe[name] = dict(averaged=values, samples=snapshots)
        self.log(self.root / "probes_002000.jsonl", dict(
            step=step, begin_step=step - WINDOW + 1, course=row["course"],
            exposure=row["exposure"], sampled_main=row["sampled_main"], sampled_success=row["sampled_success"],
            behavior=velocity_stats(np.asarray(self.actions)), fixed_actor=actor,
            actor_mean_vy_drift_from_step0=actor["mean"][1] - self.initial_actor_panel["mean"][1],
            updates=len(q), critic_loss_mean=float(q[:, 0].mean()) if len(q) else None,
            twin_q_mean=float(q[:, 1].mean()) if len(q) else None,
            q_absolute_batch_max_p99=float(np.quantile(q[:, 2], .99)) if len(q) else None,
            branches=branch_probe))
        self.actions.clear()
        self.updates.clear()
        self.branches.clear()


def verify_contract():
    archived = json.loads((T1 / "smoke/metadata.json").read_text())["contract"]
    acceptance = json.loads((T1 / "acceptance.json").read_text())
    flags = (archived["use_far_branch"], archived["course"], archived["seed"],
             archived["observation_width"], Actor(True).policy[0].in_features)
    if flags != (True, "original", 43, 784, 82) or not acceptance["cuda_smoke"]["exact_128_vs_64_plus_64"]:
        raise RuntimeError("T1 dual acceptance contract changed")
    for name, expected in archived["source_sha256"].items():
        if train.sha(Path(__file__).with_name(name)) != expected:
            raise RuntimeError(f"accepted T1 model or trainer source changed: {name}")
    t2 = json.loads((T2 / "train/metadata.json").read_text())
    cfg = train.load_config()
    fields = ("seed", "td3", "reward_scale", "reward_kwargs", "control_spec", "curriculum")
    if any(json.loads(json.dumps(cfg[k])) != t2["config"][k] for k in fields):
        raise RuntimeError("RV1 or T2 control variables changed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--end", type=int, choices=(40000, 160000), required=True)
    parser.add_argument("--resume", type=int)
    args = parser.parse_args()
    if args.resume is not None and (args.resume % 10000 or not 40000 <= args.resume < args.end):
        raise ValueError("expected a resumable T3 10k node")
    if args.end == 160000 and args.resume is None:
        raise ValueError("160k stage must resume a saved T3 node")
    verify_contract()
    panel = fixed_panel(args.root)
    tracker = Telemetry(args.root, panel)
    train.make_course_env, train.MSGPAgent, train.log = tracker.make_env, tracker.make_agent, tracker.records
    run = args.root / "train"
    interval = 40000 if args.end == 40000 else 10000
    sys.argv = [str(train.__file__), "--run-dir", str(run), "--end", str(args.end),
                "--interval", str(interval), "--course", "original", "--use-far-branch"]
    if args.resume is not None:
        sys.argv += ["--resume", str(run / f"checkpoint_{args.resume:06d}.pt")]
    train.main()


if __name__ == "__main__":
    main()
