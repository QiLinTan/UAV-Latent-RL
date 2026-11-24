"""Continue the approved T3 run from 160k to 240k, then evaluate saved nodes."""
import argparse
from pathlib import Path
import sys

from t2_eval import read_json, read_lines, write_json
from t3_eval import checkpoints, evaluate
from t3_train import Telemetry, fixed_panel, verify_contract
import train

START = 160000
END = 240000
EVALUATIONS = (200000, END)


def continue_training(root):
    run = root / "train"
    gate = read_json(root / "gate_160k.json")
    checkpoint = run / "checkpoint_160000.pt"
    saved = checkpoints(run)
    if (not gate["gate"]["forest_numeric_gate_met"] or gate["checkpoint"]["step"] != START
            or max(saved) != START or saved[START] != gate["checkpoint"]["sha256"]
            or train.sha(checkpoint) != saved[START]):
        raise RuntimeError("the approved 160k checkpoint does not match the saved run")
    if read_lines(run / "metrics.jsonl")[-1]["step"] != START:
        raise RuntimeError("run metrics extend past the approved starting checkpoint")
    if read_lines(root / "probes_002000.jsonl")[-1]["step"] != START:
        raise RuntimeError("the 160k probe timeline is missing or already advanced")
    verify_contract()
    tracker = Telemetry(root, fixed_panel(root))
    train.make_course_env, train.MSGPAgent, train.log = tracker.make_env, tracker.make_agent, tracker.records
    sys.argv = [str(train.__file__), "--run-dir", str(run), "--end", str(END),
                "--interval", "10000", "--course", "original", "--use-far-branch",
                "--resume", str(checkpoint)]
    train.main()


def evaluate_after_training(root, output, reference_160k, workers):
    from t3_report import probe_figure, write_csv

    output.mkdir(parents=True, exist_ok=True)
    run = root / "train"
    saved = checkpoints(run)
    required = range(START, END + 1, 10000)
    if max(saved) != END or any(not saved.get(step) for step in required):
        raise RuntimeError("training has not saved every 10k continuation checkpoint through 240k")
    last = read_lines(run / "metrics.jsonl")[-1]
    if (last["step"], last["updates"], last["actor_updates"]) != (END, 235001, 117500):
        raise RuntimeError("unexpected final training or update counts")
    reference = read_json(reference_160k / "t3_160000.json")
    if len(reference["rows"]) != 80 or reference["checkpoint_sha256"] != saved[START]:
        raise RuntimeError("the 160k evaluation does not match this run")
    scenes = [(r["stage"], r["case"], r["init"]) for r in reference["rows"]]
    metadata = read_json(run / "metadata.json")
    pool = metadata["contract"]["pool_fingerprint"]
    forests = {START: reference}
    for step in EVALUATIONS:
        forest_path = output / f"t3_{step:06d}.json"
        run_e12 = step == END and not (output / f"e12_{step:06d}.json").is_file()
        if not forest_path.is_file() or run_e12:
            evaluate("t3", step, run / f"checkpoint_{step:06d}.pt", saved[step], pool, scenes,
                     workers, output, complete_forest=not forest_path.is_file(), e12=run_e12)
        result = read_json(forest_path)
        if (result["checkpoint_sha256"] != saved[step]
                or [(r["stage"], r["case"], r["init"]) for r in result["rows"]] != scenes):
            raise RuntimeError(f"fixed forest evaluation does not match at {step}")
        forests[step] = result
    e12 = read_json(output / "e12_240000.json")
    before = read_json(reference_160k / "e12_160000.json")
    for step, result in ((START, before), (END, e12)):
        if (result["checkpoint_sha256"] != saved[step] or len(result["cases"]) != 12
                or result["summary"]["actual_apertures"]["episodes"] != 24):
            raise RuntimeError(f"incomplete E12 at {step}")
    timeline = read_lines(root / "probes_002000.jsonl")
    if [row["step"] for row in timeline] != list(range(2000, END + 1, 2000)):
        raise RuntimeError("incomplete 2k probe timeline")
    promotions = read_lines(root / "curriculum_promotions.jsonl")
    summary = {}
    for step, result in forests.items():
        totals = result["totals"]
        summary[step] = dict(stages={key: value["counts"] for key, value in result["stages"].items()},
                             totals=totals["counts"], xy_exits_by_side=totals["xy_exits_by_side"],
                             checkpoint_sha256=saved[step])
    table = [
        dict(step=step, **{f"stage{key}_arrivals": value["success"] for key, value in row["stages"].items()},
             **row["totals"])
        for step, row in summary.items()
    ]
    write_csv(output / "forest_comparison.csv", table)
    (output / "probes_002000.jsonl").write_bytes((root / "probes_002000.jsonl").read_bytes())
    probe_figure(timeline, promotions, output / "probe_trends.png")
    report = dict(status="completed", step=END, checkpoint_sha256=saved[END],
                  checkpoints={step: saved[step] for step in required}, forest=summary,
                  e12={START: before["summary"], END: e12["summary"]},
                  promotions=promotions, final_training=last, final_probe=timeline[-1])
    write_json(output / "summary_240k.json", report)
    write_json(root / "summary_240k.json", report)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("train")
    start.add_argument("root", type=Path)
    assess = commands.add_parser("evaluate")
    assess.add_argument("root", type=Path)
    assess.add_argument("output", type=Path)
    assess.add_argument("reference_160k", type=Path)
    assess.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.command == "train":
        continue_training(args.root)
    else:
        evaluate_after_training(args.root, args.output, args.reference_160k, args.workers)


if __name__ == "__main__":
    main()
