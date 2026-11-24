"""Freeze/verify Suite D independently of model outcomes."""
import argparse
from collections import Counter
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
from types import MethodType

import numpy as np

from geometry import HEIGHT, MARGIN, RADIUS, RESOLUTION, TOPOLOGIES, generate, solve

HERE = Path(__file__).resolve().parent
BASE = Path(os.environ.get("SUITE_D_BASE", "/opt/uav-training-20261103"))
PROJECT = Path(os.environ.get("SUITE_D_PROJECT", str(BASE / "rv1_project")))
sys.path[:0] = [str(HERE.parent), str(PROJECT),
                str(PROJECT / "experiments/ms_geometric_gpu_v1/msgp_conv7_circ16")]

from envs.map_atlas.forest_pool import ForestPool
from envs.map_atlas.spec import MapSpec
from experiments.direct_velocity_full_v1.task_factory import ROOT, load_config
from experiments.rv1_range_cnn.environment import RangeImageEnv
from experiments.rv1_range_cnn.lidar import LidarConfig
from sensor import use_parallel_lidar
from t4_suite import geometry_hash, load_suite
from train import sha

SCHEMA = "suite_d_v1"
DECISION_HZ = 15
CONTROL_HZ = 120
PHYSICS_HZ = 240
GOAL_RADIUS = .20


def digest(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, document, mode="x"):
    with path.open(mode, encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")


def binding_hashes():
    files = (
        "envs/forest/core.py",
        "envs/map_atlas/runtime.py",
        "experiments/direct_velocity_full_v1/envs/direct_velocity_full.py",
        "experiments/direct_velocity_full_v1/controls/velocity_tracking.py",
        "experiments/rv1_range_cnn/environment.py",
        "experiments/rv1_range_cnn/lidar.py",
        "experiments/ms_geometric_gpu_v1/msgp_conv7_circ16/sensor.py",
    )
    return {file: sha(PROJECT / file) for file in files}


def reason_for(base, state, *, collision=None):
    if base._checkCollision() if collision is None else bool(collision):
        return "collision"
    position = state[:3]
    if float(np.linalg.norm(base.TARGET_POS - position)) < GOAL_RADIUS:
        return "success"
    if not base.suite_d_closed:
        if np.any(abs(position[:2]) > base.MAP_SPEC.bounds_xy):
            return "xy_bound"
        if position[2] < np.float32(HEIGHT[0]) or position[2] > np.float32(HEIGHT[1]):
            return "height_bound"
    if abs(state[7]) > .65 or abs(state[8]) > .65:
        return "attitude_bound"
    if base.physics_ticks >= base.suite_d_timeout_ticks:
        return "timeout"
    return "running"


def make_env(case, sensor=None):
    spec = MapSpec.from_dict(case["map"], root=PROJECT)
    env = use_parallel_lidar(RangeImageEnv(map_spec=spec, lidar_config=sensor or LidarConfig()))
    base = env.base_env
    base.GOAL_TOLERANCE = GOAL_RADIUS
    base.suite_d_closed = case["closed"]
    base.suite_d_timeout_ticks = case["timeout_physics_ticks"]
    base._classifyDoneReason = MethodType(reason_for, base)
    if base.CTRL_FREQ != CONTROL_HZ or base.PYB_FREQ != PHYSICS_HZ or env.lidar.config.shape != (32, 128):
        env.close()
        raise RuntimeError("control/physics or LiDAR contract mismatch")
    return env


def capture_initial(env, seed):
    obs, info = env.reset(seed=seed)
    base = env.base_env
    if info["collision"] or base.physics_ticks != 0:
        raise RuntimeError("initial state is already in collision or advanced in time")
    initial = dict(state20=base._getDroneStateVector(0).tolist(), state18=obs["state"].tolist(),
                   target=base.TARGET_POS.tolist(), initial_command=base.accepted_world.tolist())
    return obs, initial


def qa_milestone(topology, graph, details, record):
    if topology == "D1":
        route = graph.route
        crossings = np.flatnonzero((route[:-1, 0] <= details["x"]) & (route[1:, 0] > details["x"]))
        if not len(crossings):
            raise RuntimeError("beam QA lacks a plane crossing")
        levels = route[crossings, 2]
        if ((details["intended_pass"] == "above" and not (levels > details["z"]).all())
                or (details["intended_pass"] == "below" and not (levels < details["z"]).all())):
            raise RuntimeError("beam QA used the wrong vertical route")
        return dict(crossing=details["intended_pass"])
    if topology == "D2":
        return dict(aperture_only_length_m=solve(record, aperture_only=True).length)
    if topology == "D3":
        if not np.any(graph.route[:, 0] > details["first_bend_x"]):
            raise RuntimeError("cave QA lacks first-bend crossing")
    if topology == "D4":
        route = graph.route
        if not np.any((route[:, 1] > details["junction_south"]) & (abs(route[:, 0]) < details["arm_gate_x"])):
            raise RuntimeError("mine QA lacks junction entry")
    return {}


def freeze(output):
    config = load_config()
    pool = ForestPool(ROOT / config["forest_pool_path"])
    suite_abc = load_suite(HERE.parent / "heldout_suite/manifest.json")
    seen = {geometry_hash(entry["map"]) for entry in suite_abc["maps"]}
    for split in pool.manifest["maps"].values():
        for entry in split:
            record = entry["record"]
            for stage in entry["stages"]:
                seen.add(geometry_hash(dict(record, obstacles=[record["obstacles"][index]
                                                               for index in stage["active_indices"]])))
    cases = []
    for topology in TOPOLOGIES:
        for index in range(20):
            record, details, seed, closed = generate(topology, index)
            graph = solve(record)
            witness = qa_milestone(topology, graph, details, record)
            key = geometry_hash(record)
            if key in seen:
                raise RuntimeError(f"duplicate geometry/start/goal: {record['map_id']}")
            seen.add(key)
            timeout = max(20., 3.0 * graph.length / .8)
            record["episode_len_sec"] = timeout
            case = dict(id=record["map_id"], topology=topology, index=index, seed=seed,
                        reset_seed=900000 + 20 * TOPOLOGIES.index(topology) + index,
                        map=record, geometry_sha256=key, closed=closed, details=details,
                        l_geo_m=graph.length, timeout_s=timeout, timeout_physics_ticks=2 * int(np.ceil(timeout * CONTROL_HZ)),
                        shortest_route_xyz_m=graph.route.tolist(), qa=witness)
            with open(os.devnull, "w", encoding="utf-8") as stream, contextlib.redirect_stdout(stream):
                env = make_env(case)
                try:
                    _, case["initial"] = capture_initial(env, case["reset_seed"])
                    _, repeated = capture_initial(env, case["reset_seed"])
                    if digest(repeated) != digest(case["initial"]):
                        raise RuntimeError(f"non-deterministic reset: {case['id']}")
                finally:
                    env.close()
            cases.append(case)
            print(f"{case['id']}: L_geo={graph.length:.3f} m T_max={timeout:.3f} s", flush=True)
    manifest = dict(schema=SCHEMA, suite="D", sizes=dict(Counter(case["topology"] for case in cases)),
                    training_pool_fingerprint=pool.fingerprint, suite_abc_fingerprint=suite_abc["fingerprint"],
                    control_spec=config["control_spec"], bound_sources=binding_hashes(),
                    decision_hz=DECISION_HZ, control_hz=CONTROL_HZ, physics_hz=PHYSICS_HZ,
                    reach_radius_m=GOAL_RADIUS, qa=dict(radius_m=RADIUS, margin_m=MARGIN, resolution_m=RESOLUTION,
                                                       graph="conservative inflated boxes, weighted 26-neighbour, 3D"),
                    generation="Fixed parameter grid, no actor execution or outcome selection during freezing.",
                    cases=cases)
    manifest["fingerprint"] = digest(manifest)
    if output.exists():
        raise FileExistsError(output)
    write(output, manifest)
    return manifest


def load_manifest(path, verify_runtime=True):
    manifest = read(path)
    payload = {key: value for key, value in manifest.items() if key != "fingerprint"}
    cases = manifest["cases"]
    if (manifest["schema"] != SCHEMA or digest(payload) != manifest["fingerprint"]
            or Counter(case["topology"] for case in cases) != dict.fromkeys(TOPOLOGIES, 20)
            or len({case["id"] for case in cases}) != 80
            or len({case["geometry_sha256"] for case in cases}) != 80
            or any(geometry_hash(case["map"]) != case["geometry_sha256"] for case in cases)):
        raise RuntimeError("Suite D manifest fingerprint or roster mismatch")
    if verify_runtime and (manifest["bound_sources"] != binding_hashes()
                           or manifest["control_spec"] != load_config()["control_spec"]):
        raise RuntimeError("frozen runtime/controller no longer matches")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest) if args.verify else freeze(args.manifest)
    print(f"{manifest['sizes']}: {manifest['fingerprint']}")


if __name__ == "__main__":
    main()
