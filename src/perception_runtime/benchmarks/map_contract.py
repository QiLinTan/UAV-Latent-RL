"""Canonical map identity and auditable map-level partitions."""

import hashlib
import json

import numpy as np


def _vector(value, length):
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise ValueError(f"Expected a finite {length}D vector")
    return np.round(vector, 6).tolist()


def canonical_geometry(start, goal, trees):
    records = []
    for tree in trees:
        radius, height = float(tree["radius"]), float(tree["height"])
        if not np.isfinite([radius, height]).all() or min(radius, height) <= 0:
            raise ValueError("Tree radius/height must be finite and positive")
        records.append({"xy": _vector(tree["xy"], 2), "radius": round(radius, 6), "height": round(height, 6)})
    records.sort(key=lambda item: (*item["xy"], item["radius"], item["height"]))
    return {"start": _vector(start, 3), "goal": _vector(goal, 3), "trees": records}


def map_id(start, goal, trees):
    content = json.dumps(canonical_geometry(start, goal, trees), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode()).hexdigest()


def from_environment(env, *, family, seed):
    geometry = canonical_geometry(env.START_POS, env.TARGET_POS, env._tree_specs)
    return {**geometry, "map_id": map_id(**geometry), "family": family, "seed": int(seed),
            "initial_yaw": float(env.INIT_RPYS[0][2]), "corridor_width": None, "required_route": None}


def validate_partitions(partitions):
    seen = {}
    for split, maps in partitions.items():
        for record in maps:
            identity = map_id(record["start"], record["goal"], record["trees"])
            if identity != record["map_id"]:
                raise ValueError("Map geometry does not match its canonical identity")
            if identity in seen:
                raise ValueError(f"Map leakage/duplicate between {seen[identity]} and {split}: {identity}")
            seen[identity] = split
    return {"disjoint": True, "counts": {key: len(value) for key, value in partitions.items()}, "unique_maps": len(seen)}
