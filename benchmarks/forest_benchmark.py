"""Explicit, map-isolated forest geometries and read-only split access."""

from __future__ import annotations

from collections import deque
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.map_contract import canonical_geometry, map_id, validate_partitions

FAMILIES = ("clear", "single", "sparse_forest", "dense_forest", "route_blocking", "symmetric", "narrow_corridor", "dead_end", "multiple_turns")


def grid_route(record, *, resolution=.1, inflation=.16, bound=4.0):
    """2D constant-altitude reachability check; never fed to a policy.

    Four-connected BFS uses a conservative extra half-cell diagonal padding,
    so adjacent free cells have collision-free line segments for cylinders.
    """
    coords = np.arange(-bound, bound + resolution/2, resolution)
    xx, yy = np.meshgrid(coords, coords, indexing="ij")
    free = np.ones(xx.shape, dtype=bool)
    for tree in record["trees"]:
        free &= (xx-tree["xy"][0])**2 + (yy-tree["xy"][1])**2 > (tree["radius"]+inflation+resolution*np.sqrt(2)/2)**2
    def cell(position):
        return tuple(np.rint((np.asarray(position[:2])+bound)/resolution).astype(int))
    start, goal = cell(record["start"]), cell(record["goal"])
    if any(i<0 or i>=len(coords) for i in start+goal) or not free[start] or not free[goal]:
        return None
    parents = {start: None}
    queue = deque([start])
    while queue and goal not in parents:
        current = queue.popleft()
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            following = current[0]+dx, current[1]+dy
            if 0<=following[0]<len(coords) and 0<=following[1]<len(coords) and free[following] and following not in parents:
                parents[following] = current
                queue.append(following)
    if goal not in parents:
        return None
    route, current = [], goal
    while current is not None:
        route.append([coords[current[0]],coords[current[1]]])
        current = parents[current]
    path = np.asarray(route[::-1])
    path[0], path[-1] = record["start"][:2], record["goal"][:2]
    return path


def generate_map(family, seed):
    rng = np.random.default_rng(seed)
    start = np.array([-rng.uniform(2.9,3.35), rng.uniform(-.2,.2), 1.0])
    goal = np.array([rng.uniform(2.9,3.35), rng.uniform(-.2,.2), 1.0])
    trees = []
    def tree(x,y,r=None):
        trees.append({"xy":[float(x),float(y)], "radius": float(r if r is not None else rng.uniform(.12,.20)), "height":float(rng.uniform(1.8,2.4))})
    if family in ("single", "symmetric"):
        tree(rng.uniform(-.5,.5), rng.uniform(-.2,.2), rng.uniform(.18,.28))
        if family == "symmetric":
            side = rng.choice([-1,1])
            for x in (-.55,0,.55):
                tree(x, side*.85)
    elif family == "route_blocking":
        for x in (-1.6,0,1.6):
            tree(x+rng.uniform(-.2,.2),rng.uniform(-.3,.3))
    elif family == "narrow_corridor":
        width = rng.uniform(.55,.72)
        for x in np.linspace(-2.3,2.3,12):
            for y in (-width,width):
                tree(x,y,.18)
    elif family == "dead_end":
        for y in np.linspace(-.9,.9,7):
            tree(.5,y,.18)
        for x in np.linspace(-1.2,.2,6):
            tree(x,-.9,.18)
            tree(x,.9,.18)
    elif family == "multiple_turns":
        for x,side in ((-1.5,1),(0,-1),(1.5,1)):
            for y in np.linspace(-.45,2.35,9)*side:
                tree(x,y,.18)
    if family in ("sparse_forest", "dense_forest", "route_blocking"):
        target = 8 if family == "sparse_forest" else 24
        attempts = 0
        while len(trees)<target and attempts<10000:
            attempts += 1
            xy = rng.uniform(-2.55,2.55,2)
            radius = rng.uniform(.12,.20)
            if any(np.linalg.norm(xy-p[:2])<radius+.6 for p in (start,goal)):
                continue
            if any(np.linalg.norm(xy-t["xy"])<radius+t["radius"]+.15 for t in trees):
                continue
            tree(*xy,radius)
        if len(trees)!=target:
            raise RuntimeError("Map sampler exhausted its placement budget")
    yaw = rng.uniform(-np.pi,np.pi)
    c,s = np.cos(yaw),np.sin(yaw)
    rotation = np.array([[c,-s],[s,c]])
    translation = rng.uniform(-.12,.12,2)
    start[:2] = rotation@start[:2]+translation
    goal[:2] = rotation@goal[:2]+translation
    for t in trees:
        t["xy"] = rotation@t["xy"]+translation
    geometry = canonical_geometry(start,goal,trees)
    return {**geometry, "map_id":map_id(**geometry), "family":family, "seed":seed,
            "initial_yaw":float(yaw+rng.uniform(-.25,.25)), "corridor_width": float(2*width-.36) if family=="narrow_corridor" else None,
            "required_route":None, "required_route_note":"Not inferred from family; both sides may be feasible.",
            "generator_version":"forest_map_isolation_v1"}


def build_partitions(counts=(120,40,60), seed=20260907):
    partitions = {}
    rejected = []
    all_ids = set()
    for split_index,(split,count) in enumerate(zip(("train","validation","test"), counts)):
        records = []
        attempt = 0
        while len(records)<count:
            family = FAMILIES[len(records)%len(FAMILIES)]
            value = seed+split_index*100000+attempt
            attempt += 1
            record = generate_map(family,value)
            if record["map_id"] in all_ids or grid_route(record) is None:
                rejected.append({"seed":value,"family":family,"reason":"duplicate_or_grid_unreachable"})
                continue
            record["reachability"] = {"method":"4-neighbour BFS at z=1m", "resolution":.1, "inflation":.16, "reachable":True,
                                      "purpose":"geometry validation only; never a training label or policy input"}
            all_ids.add(record["map_id"])
            records.append(record)
            if attempt>count*100:
                raise RuntimeError("Benchmark generation budget exhausted")
        partitions[split] = records
    return partitions, rejected


def save_benchmark(root, counts=(120,40,60), seed=20260907):
    root = Path(root)
    root.mkdir(parents=True,exist_ok=False)
    partitions,rejected = build_partitions(counts,seed)
    validation = validate_partitions(partitions)
    files = {}
    for split,records in partitions.items():
        payload = json.dumps(records,indent=2,sort_keys=True,allow_nan=False)+"\n"
        (root/f"{split}.json").write_text(payload)
        files[split] = {"path":str(root/f"{split}.json"), "sha256":hashlib.sha256(payload.encode()).hexdigest()}
    manifest = {"schema_version":1,"seed":seed,"families":FAMILIES,"validation":validation,"files":files,
                "rejected_layouts":rejected,"test_policy":"Sealed for final navigation evaluation; never used for probe fitting, early stopping or selection.",
                "coordinate_contract":"docs/experiment_contracts/planner_local_coordinates_v1.md",
                "reachability_warning":"Planar conservative grid feasibility only, not a certificate of dynamic flyability or unique required route."}
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    return manifest


def load_split(root, split, *, purpose="diagnostic"):
    if split not in ("train","validation","test"):
        raise ValueError("Unknown split")
    if split=="test" and purpose!="final_evaluation":
        raise ValueError("Test maps are sealed; diagnostics may use train/validation only")
    root = Path(root)
    manifest = json.loads((root/"manifest.json").read_text())
    data = (root/f"{split}.json").read_bytes()
    if hashlib.sha256(data).hexdigest()!=manifest["files"][split]["sha256"]:
        raise ValueError("Benchmark split changed after freezing")
    records = json.loads(data)
    validate_partitions({split:records})
    return records


def install_map(env, record):
    """Install explicit cylinders through the existing environment geometry API.

    Call only immediately after reset, with matching configured start/goal.
    Keeps observations, physics, rewards and terminal logic unchanged.
    """
    import pybullet as p
    if not np.allclose(env.START_POS,record["start"],atol=1e-6) or not np.allclose(env.TARGET_POS,record["goal"],atol=1e-6):
        raise ValueError("Environment start/goal do not match map contract")
    for body in env._tree_ids:
        p.removeBody(body,physicsClientId=env.CLIENT)
    env._tree_ids,env._tree_specs = [],[]
    for tree in record["trees"]:
        body = env._layout_generator.create_tree_body(client_id=env.CLIENT,**tree)
        env._tree_ids.append(body)
        env._tree_specs.append({**tree,"xy":np.array(tree["xy"],dtype=np.float32),"id":body})
    env._last_clearance = env._computeNearestTreeClearance(env.pos[0])
    env._last_collision = env._checkCollision()
    return env._computeObs(),env._computeInfo()
