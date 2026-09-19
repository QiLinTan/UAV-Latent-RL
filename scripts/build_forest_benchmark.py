"""Freeze 120/40/60 disjoint maps without evaluating a policy on the test set."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmarks.forest_benchmark import save_benchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="data/benchmarks/forest_map_isolation_v1")
    parser.add_argument("--seed",type=int,default=20260907)
    args = parser.parse_args()
    result = save_benchmark(args.output,seed=args.seed)
    print(json.dumps(result["validation"]))
