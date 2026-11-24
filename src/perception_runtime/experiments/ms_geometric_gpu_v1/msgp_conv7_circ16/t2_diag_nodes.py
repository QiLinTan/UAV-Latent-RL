"""Re-evaluate archived intermediate T2 checkpoints on the original fixed pool."""
import argparse
from pathlib import Path

import t2_eval as evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run = args.run / "train"
    baseline, _ = evaluation.historical_a2(args.run.parent)
    events = {r["step"]: r["sha256"] for r in evaluation.read_lines(run / "events.jsonl")
              if r["event"] == "checkpoint"}
    fixed = evaluation.read_json(args.run / "evaluation_080000.json")
    reference = {(r["stage"], r["case"]): r["init"] for r in fixed["rows"]}
    for step in (90000, 100000, 110000):
        doc = evaluation.evaluate(step, run / f"checkpoint_{step:06d}.pt", events[step], run,
                                  args.output / f"fixed_{step:06d}.json", baseline, args.workers)
        if any(r["init"] != reference[r["stage"], r["case"]] for r in doc["rows"]):
            raise RuntimeError(f"80k fixed initial state mismatch at {step}")


if __name__ == "__main__":
    main()
