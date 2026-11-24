"""Verify the public freeze and replay a selected archived checkpoint on CPU."""

import argparse
from hashlib import sha256
from importlib.util import find_spec
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "src/perception_runtime"
DOCS = ROOT / "docs/perception_freeze"
CONV = RUNTIME / "experiments/ms_geometric_gpu_v1/msgp_conv7_circ16"
SUITE_D = CONV / "t4_benchmark/suite_d"
sys.path[:0] = list(map(str, (RUNTIME, CONV, CONV / "t4_benchmark", CONV / "t5_phase3", SUITE_D)))
os.environ["SUITE_D_PROJECT"] = str(RUNTIME)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def hash_file(path):
    with path.open("rb") as stream:
        digest = sha256()
        while data := stream.read(8 * 1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def materialize_pool():
    folder = DOCS / "manifests/fixed_forest_pool"
    index = read(folder / "index.json")
    raw = b"".join((folder / name).read_bytes() for name in index["pieces"])
    if len(raw) != index["bytes"] or sha256(raw).hexdigest() != index["sha256"]:
        raise ValueError("fixed forest pool fingerprint mismatch")
    path = RUNTIME / index["runtime_path"]
    if not path.exists():
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_bytes(raw)
    if hash_file(path) != index["sha256"]:
        raise ValueError("existing fixed forest pool differs from the frozen pool")
    # The legacy controller resolves its URDF through this historical location.
    installed = Path(find_spec("gym_pybullet_drones").origin).resolve().parent
    slot = RUNTIME / ".wsl2/drones-cpu/lib/python3.10/site-packages/gym_pybullet_drones"
    if not slot.exists():
        slot.parent.mkdir(parents=True, exist_ok=True)
        slot.symlink_to(installed, target_is_directory=True)
    if slot.resolve() != installed:
        raise ValueError("legacy model path is bound to another Gym-PyBullet-Drones installation")
    return index


def verify(archive):
    from contract import load_manifest
    from t4_suite import load_suite

    pool = materialize_pool()
    abc = load_suite(DOCS / "manifests/suite_abc.json")
    suite_d = load_manifest(DOCS / "manifests/suite_d.json")
    results = read(DOCS / "t5_summary_report.json")
    if (abc["fingerprint"] != results["sealed_abc_fingerprint"]
            or suite_d["fingerprint"] != results["suite_d"]["fingerprint"]
            or abc["frozen_training_pool_fingerprint"] != pool["fingerprint"]
            or suite_d["training_pool_fingerprint"] != pool["fingerprint"]):
        raise ValueError("inconsistent frozen suite fingerprints")
    rows = read(DOCS / "checkpoint_hashes.json")["checkpoints"]
    if len(rows) != 18:
        raise ValueError("expected 18 frozen checkpoint records")
    if archive:
        for row in rows:
            if hash_file(archive / row["archive_relative_path"]) != row["sha256"]:
                raise ValueError(f"checkpoint hash mismatch: {row['archive_relative_path']}")
    print(json.dumps(dict(fixed_pool=pool["fingerprint"], suite_abc=abc["fingerprint"],
                          suite_d=suite_d["fingerprint"], checkpoints_verified=len(rows) if archive else 0)))


def replay(args):
    import torch
    torch.set_num_threads(1)
    import t4_evaluate as t4
    from t2_eval import write_json

    pool = materialize_pool()
    model = {"rv1": "a1", "static": "t3", "phys_dynamic_v22": "dynamic"}[args.model]
    checkpoint, = [row for row in read(DOCS / "checkpoint_hashes.json")["checkpoints"]
                   if (row["model"], row["seed"], row["step"]) == (model, args.seed, args.step)]
    path = args.archive / checkpoint["archive_relative_path"]
    if hash_file(path) != checkpoint["sha256"]:
        raise ValueError("checkpoint hash mismatch")
    if model == "dynamic":
        from t5_evaluate import load_worker, heldout_group
        load_worker(model, args.seed, args.step, path, pool["fingerprint"])
    else:
        t4.load_worker(model, args.seed, args.step, path, pool["fingerprint"])
    args.output.mkdir(parents=True, exist_ok=False)
    if args.suite == "fixed":
        limit = args.limit or 80
        rows = [row for stage in range(min(4, (limit + 19) // 20)) for row in t4.fixed_forest(
            stage, range(min(20, limit - stage * 20)))]
        fingerprint = pool["fingerprint"]
    elif args.suite == "abc":
        manifest = t4.load_suite(DOCS / "manifests/suite_abc.json")
        cases = manifest["maps"][:args.limit]
        rows = heldout_group(cases) if model == "dynamic" else t4.heldout_group(cases)
        fingerprint = manifest["fingerprint"]
    else:
        import suite_d.run as runner
        manifest = runner.load_manifest(DOCS / "manifests/suite_d.json")
        fingerprint = manifest["fingerprint"]
        if model == "dynamic":
            from dynamic_v22 import attach_frame_observer
            original = runner.make_env
            runner.make_env = lambda *a, **kw: attach_frame_observer(original(*a, **kw))
        entry = dict(id=f"{model}_{args.seed}_{args.step}", model=model, seed=args.seed,
                     checkpoint=args.step, path=str(path), sha256=checkpoint["sha256"])
        rows = [runner.quiet_case(case, args.output, entry, fingerprint) for case in manifest["cases"][:args.limit]]
    write_json(args.output / "results.json", dict(checkpoint=checkpoint, fingerprint=fingerprint, rows=rows,
                                                 pilot=args.limit is not None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    check = commands.add_parser("verify")
    check.add_argument("--archive", type=Path)
    run = commands.add_parser("replay")
    run.add_argument("--archive", type=Path, required=True)
    run.add_argument("--model", choices=("rv1", "static", "phys_dynamic_v22"), required=True)
    run.add_argument("--seed", type=int, choices=(43, 44, 45), required=True)
    run.add_argument("--step", type=int, choices=(200000, 240000), required=True)
    run.add_argument("--suite", choices=("fixed", "abc", "d"), required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--limit", type=int, choices=range(1, 145), help="Limit cases for a smoke replay.")
    args = parser.parse_args()
    if args.action == "verify":
        verify(args.archive)
    else:
        replay(args)


if __name__ == "__main__":
    main()
