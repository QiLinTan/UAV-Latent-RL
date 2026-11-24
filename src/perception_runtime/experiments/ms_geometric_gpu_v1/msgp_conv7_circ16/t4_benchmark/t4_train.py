"""Launch new-seed T4 controls through the existing frozen trainers."""
import argparse
import copy
import json
from functools import partial
from pathlib import Path
import sys

from experiments.rv1_range_cnn import train_formal
import t3_train
import train

NODES = (80000, 120000, 160000, 200000, 240000)
ARCHIVED_DUAL = Path("/opt/uav-training-20261103/runs/msgp_conv7_circ16_dual_seed43_240k")
RUN_NAMES = {"t3": "msgp_conv7_circ16_dual_seed{seed}_240k", "a1": "rv1_baseline_seed{seed}_240k"}


class BaselineCourseLog:
    def __init__(self, root):
        self.root = root
        self.factory = train_formal.make_course_env
        self.log = train_formal.log
        self.previous = None

    def make_env(self, *args, **kwargs):
        self.env = self.factory(*args, **kwargs)
        return self.env

    def record(self, path, row):
        name = path.name
        if name in ("episodes.jsonl", "metrics.jsonl") or (
                name == "events.jsonl" and row["event"] in ("start", "resume")):
            info = self.env.course_info()
            course = {key: value for key, value in info.items()
                      if key.startswith(("start_curriculum_", "obstacle_curriculum_"))}
            course.update(density_stage=info["density_stage"], active_tree_count=info["active_tree_count"])
            current = course["start_curriculum_level"], course["obstacle_curriculum_stage"]
            if self.previous is None:
                self.previous = current
            if name == "episodes.jsonl" and self.previous != current:
                self.log(self.root / "curriculum_promotions.jsonl", dict(
                    step=row["step"], old_start_level=self.previous[0], start_level=current[0],
                    old_obstacle_stage=self.previous[1], obstacle_stage=current[1]))
                self.previous = current
            row = dict(row, course=course)
        self.log(path, row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=tuple(RUN_NAMES))
    parser.add_argument("seed", type=int, choices=(44, 45))
    parser.add_argument("runs", type=Path)
    parser.add_argument("--end", type=int, choices=(80000, 240000), required=True)
    args = parser.parse_args()
    t3_train.verify_contract()
    root = args.runs / RUN_NAMES[args.model].format(seed=args.seed)
    run = root / "train"
    initial = args.end == 80000
    if initial:
        root.mkdir(exist_ok=False)
    else:
        metadata = json.loads((run / "metadata.json").read_text())
        if metadata["config"]["seed"] != args.seed:
            raise RuntimeError("resume seed mismatch")
    original_config = train.load_config

    def config():
        cfg = copy.deepcopy(original_config())
        cfg["seed"] = args.seed
        return cfg

    interval = "80000" if initial else "40000"
    resume = [] if initial else ["--resume", str(run / "checkpoint_080000.pt")]
    print(f"T4 {args.model}, Seed{args.seed}, original course, end={args.end}", flush=True)
    if args.model == "t3":
        probe_source = ARCHIVED_DUAL / "fixed_actor_panel.npy"
        if not probe_source.is_file() or not probe_source.with_suffix(".json").is_file():
            raise RuntimeError("archived standard-validation panel missing")
        tracker = t3_train.Telemetry(root, t3_train.fixed_panel(ARCHIVED_DUAL))
        train.make_course_env, train.MSGPAgent, train.log = tracker.make_env, tracker.make_agent, tracker.records
        train.load_config = config
        sys.argv = [str(train.__file__), "--run-dir", str(run), "--end", str(args.end),
                    "--interval", interval, "--course", "original", "--use-far-branch", *resume]
        train.main()
    else:
        recorder = BaselineCourseLog(root)
        train.cuda_setup(args.seed)
        train_formal.load_config = config
        if initial:
            train_formal.SuccessReplay = partial(train_formal.SuccessReplay, seed=args.seed)
        train_formal.make_course_env, train_formal.log = recorder.make_env, recorder.record
        sys.argv = [str(train_formal.__file__), "--run-dir", str(run), "--device", "cuda",
                    "--stop-after", str(args.end), "--checkpoint-interval", interval, *resume]
        train_formal.main()


if __name__ == "__main__":
    main()
