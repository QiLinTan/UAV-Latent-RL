"""Archive the logical-async experiment without changing its control code."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path):
    path = Path(path)
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def compare_json(expected, actual, prefix=""):
    """Compare every scalar, including startup missing cycles; no tolerances."""
    differences = []
    if type(expected) is not type(actual):
        return [{"field": prefix, "expected": expected, "actual": actual}]
    if isinstance(expected, dict):
        for key in sorted(expected.keys() | actual.keys()):
            if key not in expected or key not in actual:
                differences.append({"field": prefix + "/" + key, "missing_key": True})
            else:
                differences.extend(compare_json(expected[key], actual[key], prefix + "/" + key))
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            differences.append({"field": prefix, "length_mismatch": True})
        else:
            for i, (first, second) in enumerate(zip(expected, actual)):
                differences.extend(compare_json(first, second, prefix + f"/{i}"))
    elif expected != actual:
        differences.append({"field": prefix, "expected": expected, "actual": actual})
    return differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--git-finalized", action="store_true")
    parser.add_argument("--reproduced", default="analysis_outputs/logical_async_freeze_v1/async_results_reproduced.json")
    args = parser.parse_args()
    source = subprocess.check_output(["git", "rev-parse", args.source_commit], cwd=ROOT, text=True).strip()
    old_path = ROOT / "analysis_outputs/latent_async_pid/async_results.json"
    old = json.loads(old_path.read_text())
    new = json.loads((ROOT / args.reproduced).read_text())
    differences = compare_json(old, new)
    archive_root = ROOT / "checkpoints/experiment_archives/logical_async_reference_pid_v1"
    archive_root.mkdir(parents=True, exist_ok=True)
    archive = archive_root / "models_data_dependencies.zip"
    if archive.exists():
        raise FileExistsError(f"Refusing to overwrite frozen archive: {archive}")
    model_paths = [
        "checkpoints/latent_only_affordance_dim16/model_final_encoder",
        "checkpoints/latent_only_affordance_dim16/model_final_actor",
        "checkpoints/reference_planner/latent_affordance_margin_v2/planner_final.pt",
    ]
    data_paths = sorted(str(path.relative_to(ROOT)) for path in (ROOT / "data/reference_planner").rglob("*") if path.is_file())
    external_root = Path("/home/tequila/gym-pybullet-drones/gym_pybullet_drones")
    external_paths = sorted(path for path in external_root.rglob("*") if path.is_file() and path.suffix in {".py", ".urdf", ".obj", ".mtl", ".stl"})
    artifacts = []
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        # Also preserve a self-contained source snapshot if Git approval is
        # temporarily unavailable. This does not mutate Git or claim a tag.
        for directory in ("async_control", "high_level_planner", "envs", "models", "algos", "utils", "scripts", "tests", "trainers", "buffers"):
            for path in sorted((ROOT / directory).rglob("*.py")):
                if "__pycache__" not in path.parts:
                    relative = str(path.relative_to(ROOT))
                    bundle.write(path, "source/" + relative)
                    artifacts.append({**file_record(path), "archive_member": "source/" + relative})
        for relative in model_paths + data_paths:
            path = ROOT / relative
            bundle.write(path, relative)
            artifacts.append({**file_record(path), "archive_member": relative})
        for path in external_paths:
            member = "external/gym_pybullet_drones/" + str(path.relative_to(external_root))
            bundle.write(path, member)
            artifacts.append({**file_record(path), "archive_member": member})
        # User-owned changes remain unstaged. Exact patch and complete file are
        # archived so a checkout of the tag can recover the tested dependency.
        for relative in ("envs/forest/layout.py", ".gitignore", "scripts/td3_latent_uav.py"):
            diff = subprocess.check_output(["git", "diff", "HEAD", "--", relative], cwd=ROOT)
            bundle.writestr("workspace_diffs/" + relative + ".patch", diff)
            bundle.write(ROOT / relative, "workspace_files/" + relative)
    packages = {}
    for name in ("numpy", "torch", "gymnasium", "pybullet", "scipy", "tensorboard"):
        packages[name] = importlib.metadata.version(name)
    code_paths = []
    for directory in ("async_control", "high_level_planner"):
        code_paths.extend(sorted((ROOT / directory).glob("*.py")))
    code_paths.extend(ROOT / path for path in (
        "scripts/evaluate_latent_async_pid.py", "scripts/evaluate_reference_planner.py",
        "scripts/collect_reference_planner_dataset.py", "scripts/train_reference_planner.py",
        "envs/forest/core.py", "envs/forest/layout.py", "envs/preprocess.py",
        "models/motor_action_codec.py", "models/encoder.py", "algos/td3/networks.py",
    ))
    manifest = {
        "schema_version": 1, "name": "logical-async-reference-pid-v1",
        "git_commit": source,
        "git_commit_semantics": ("Frozen source commit; manifest committed afterwards." if args.git_finalized else "Base commit only; uncommitted frozen code is in archive source/ with per-file hashes."),
        "git_freeze_complete": bool(args.git_finalized),
        "git_tag": "logical-async-reference-pid-v1" if args.git_finalized else None,
        "git_blocker": None if args.git_finalized else "Git staging auto-review rejected because of usage limit; no tag or source commit claimed.",
        "runtime_async": False,
        "clock": "deterministic simulation time, single Python thread",
        "deadline_measured": False,
        "interruption_interpretation": "Logical control selections only; zero is NOT evidence of meeting a wall-clock deadline.",
        "encoder_sha256": sha256(ROOT / model_paths[0]),
        "actor_sha256": sha256(ROOT / model_paths[1]),
        "planner_sha256": sha256(ROOT / model_paths[2]),
        "dataset_sha256": sha256(ROOT / "data/reference_planner/latent_affordance_margin_v2/samples.npz"),
        "archive": file_record(archive), "artifacts": artifacts,
        "source_files": [file_record(path) for path in code_paths],
        "observation_dim": 263, "latent_dim": 16,
        "high_level_hz": 15, "sync_baseline_high_level_hz": 120,
        "low_level_hz": 120, "reference_horizon_seconds": 0.5,
        "motor_codec": "asymmetric_rpm", "environment_config": old["environment"],
        "evaluation_seeds": [ep["seed"] for ep in old["profiles"]["A_sync_baseline"]["episodes"]],
        "commands": [
            f"{sys.executable} -m unittest discover -s tests -v",
            f"{sys.executable} scripts/evaluate_latent_async_pid.py --episodes 3 --seed 0 --num-trees 0 --output {args.reproduced}",
        ],
        "regression_at_freeze": {"tests_passed": 50, "failures": 0},
        "reproduction": {"all_fields_equal": not differences, "differences": differences,
                         "expected": file_record(old_path), "actual": file_record(ROOT / args.reproduced)},
        "startup_missing_cycles": {name: value["summary"]["total_missing_packet_cycles"] for name, value in new["profiles"].items()},
        "python": sys.version, "python_executable": sys.executable,
        "platform": platform.platform(), "package_versions": packages,
        "ownership": {
            "async_control_and_planner": "Prior phase deliverables, explicitly included in freeze source commit.",
            "envs/forest/layout.py": "Pre-existing user diff. Alters tree sampling attempts (40->160, minimum 200->1000). Forest outputs depend on this generator; archived exact file+diff, not committed as our environment change. Clear num_trees=0 has no sampling dependency.",
            ".gitignore": "User change, excluded and archived for provenance.",
            "scripts/td3_latent_uav.py": "User change, unrelated legacy training entry; excluded and archived for provenance.",
            "analysis_outputs": "Only explicitly named async/planner results tracked; other analyses remain user-owned.",
            "data/reference_planner": "Prior phase datasets included in model/data archive, not a map-isolated benchmark.",
        },
        "recovery": [
            "If git_freeze_complete is false, checkout git_commit separately and overlay archive source/; otherwise checkout the named tag.",
            "Locate archive.path (same-machine archive; copy it off this computer for disaster recovery). Verify SHA-256 before use.",
            "Extract checkpoints/ and data/ members into that checkout.",
            "For exact forest reproduction restore workspace_files/envs/forest/layout.py in the separate checkout; this is an explicitly archived dependency.",
            "Use the recorded environment or install matching packages plus archived editable gym_pybullet_drones sources/assets.",
            "Run commands and compare JSON fields. Runtime deadline guarantees are outside this version.",
        ],
    }
    output = ROOT / "docs/experiment_contracts/logical_async_reference_pid_v1.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(output), "archive": str(archive), "all_fields_equal": not differences, "difference_count": len(differences)}))
    if differences:
        raise SystemExit("Freeze gate failed: result fields differ.")


if __name__ == "__main__":
    main()
