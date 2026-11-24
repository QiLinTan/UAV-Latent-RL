"""Generate and fingerprint independent T4 scenes before policy evaluation."""
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from envs.map_atlas.ego_forest import EgoForestConfig, clearance_route, generate_ego_forest
from envs.map_atlas.forest_pool import COUNTS, ForestPool, digest
from envs.map_atlas.spec import MapSpec
from experiments.direct_velocity_full_v1.task_factory import ROOT, load_config
from t2_eval import read_json

FOREST = EgoForestConfig()
RANGES = {"A_forest": (440000, 441000), "B": (442000, 443000), "C": (444000, 445000),
          "A_aperture": (446000, 446012)}
ZONES = (-10., -4., 4., 10.)


def geometry_hash(record):
    objects = [{k: obs[k] for k in ("kind", "position", "radius", "height", "half_extents") if k in obs}
               for obs in record["obstacles"]]
    return digest(dict(bounds=record.get("xy_half_extents", [record["xy_half_extent"]] * 2),
                       start=record["start_pos"], goal=record["target_pos"], obstacles=objects))


def corridor(seed, index):
    rng = np.random.default_rng(seed)
    chosen, obstacles = [], []

    def tree(xy):
        xy = np.floor(np.asarray(xy) * 10) / 10 + .05
        radius = float(rng.uniform(.2, .3))
        if (np.any(abs(xy) + radius >= [10., 5.])
                or any(np.linalg.norm(xy - point[:2]) < radius + .8 for point in (FOREST.start, FOREST.goal))
                or any(np.linalg.norm(xy - other) < 1 for other in chosen)):
            return False
        height = float(rng.uniform(3., 4.))
        chosen.append(xy)
        obstacles.append(dict(kind="cylinder", name=f"tree_{len(obstacles):03d}",
                              position=[float(xy[0]), float(xy[1]), height/2], radius=radius, height=height))
        return True

    def fill(count, x_min, x_max):
        end = len(obstacles) + count
        for _ in range(12000):
            if len(obstacles) == end:
                return
            tree(rng.uniform((x_min, -4.7), (x_max, 4.7)))
        raise RuntimeError("could not place corridor tree count")

    # Stagger the central stands across the direct route without sealing it.
    for x in (-3.25, -1.65, 0., 1.65, 3.25):
        if not tree((x + rng.uniform(-.15, .15), rng.uniform(-.28, .28))):
            raise RuntimeError("central corridor tree spacing failed")
    fill(17, -3.95, 3.95)
    if len({int((c[1] + 5) // 2.5) for c in chosen}) < 4:
        raise RuntimeError("center trees did not span all lateral bands")
    fill(1 + index % 2, -9.65, -4.05)
    fill(2 - index % 2, 4.05, 9.65)
    return dict(map_id=f"t4_c_{seed}", source="T4 fixed local generator", scenario="density transition",
                split="test", start_pos=list(FOREST.start), target_pos=list(FOREST.goal),
                xy_half_extent=10., xy_half_extents=[10., 5.], obstacle_sensor_range=FOREST.sensor_range,
                episode_len_sec=FOREST.episode_len_sec, obstacles=obstacles)


def opening(seed, side):
    rng = np.random.default_rng(seed)
    wall, offset, width = (float(x) for x in rng.uniform((1.72, .39, .46), (3.4, .69, .77)))
    middle = side * offset
    panels = [dict(kind="box", name=f"panel_{i}", position=[wall, (left + right) / 2, 1.4],
                   half_extents=[.09, (right - left) / 2, 1.4])
              for i, (left, right) in enumerate(((-1.45, middle-width/2), (middle+width/2, 1.45)))]
    record = dict(map_id=f"t4_open_{seed}_{side}", source="T4 fixed local generator",
                  scenario="mirrored aperture", split="test", start_pos=[0., 0., 1.],
                  target_pos=[4.35, 0., 1.], xy_half_extent=5., obstacle_sensor_range=8.,
                  episode_len_sec=30., obstacles=panels)
    return record, dict(wall_center_x_m=wall, opening_center_y_m=middle, opening_width_m=width, side=side)


def freeze(output):
    pool = ForestPool(ROOT / load_config()["forest_pool_path"])
    seen = set()
    for entries in pool.manifest["maps"].values():
        for entry in entries:
            for stage in entry["stages"]:
                record = entry["record"]
                active = [record["obstacles"][i] for i in stage["active_indices"]]
                seen.add(geometry_hash(dict(record, obstacles=active)))
    rows, rejected = [], []

    def check(suite, index, seed, record, stage=None, aperture=None):
        record = dict(record, split="test")
        spec = MapSpec.from_dict(record, root=Path("."))
        route = clearance_route(spec) if aperture is None else None
        if aperture is None and route is None:
            raise RuntimeError("geometric QA found no route")
        if aperture and aperture["opening_width_m"] <= 2 * (.06 + .1):
            raise RuntimeError("aperture QA width too narrow")
        return dict(id=f"{suite}_{stage if stage is not None else 'map'}_{index:02d}", suite=suite, stage=stage,
                    seed=seed, reset_seed=seed, geometry_sha256=geometry_hash(record),
                    map=record, aperture=aperture,
                    qa_route_length_m=float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum())
                    if route is not None else None)

    def commit(group):
        signatures = {r["geometry_sha256"] for r in group}
        if len(signatures) != len(group) or seen.intersection(signatures):
            raise RuntimeError("duplicate geometry")
        seen.update(signatures)
        rows.extend(group)

    for name in ("A_forest", "B", "C"):
        count = 0
        for seed in range(*RANGES[name]):
            try:
                if name == "C":
                    record = corridor(seed, count)
                else:
                    cfg = replace(FOREST, tree_count=55) if name == "B" else FOREST
                    record = generate_ego_forest(cfg, seed=seed)[0]
                if name == "A_forest":
                    group = [check(name, count, seed, dict(record, map_id=f"t4_a_{seed}_{stage}",
                                                         obstacles=record["obstacles"][:n]), stage)
                             for stage, n in enumerate(COUNTS)]
                else:
                    group = [check(name, count, seed, record, 4 if name == "B" else None)]
                commit(group)
            except RuntimeError as error:
                rejected.append(dict(suite=name, seed=seed, reason=str(error)))
                continue
            count += 1
            if count == 20:
                break
        if count != 20:
            raise RuntimeError(f"too few geometrically valid scenes for {name}")
    for i, seed in enumerate(range(*RANGES["A_aperture"])):
        group = []
        for j, side in enumerate((-1, 1)):
            record, aperture = opening(seed, side)
            group.append(check("A_aperture", 2*i+j, seed, record, aperture=aperture))
        commit(group)
    sizes = dict(Counter(r["suite"] for r in rows))
    expected = dict(A_forest=80, B=20, C=20, A_aperture=24)
    if sizes != expected or len({r["id"] for r in rows}) != 144:
        raise RuntimeError("held-out suite manifest count mismatch")
    manifest = dict(schema="t4_heldout_v1", sizes=sizes, seed_ranges=RANGES, rejected=rejected,
                    frozen_training_pool_fingerprint=pool.fingerprint, maps=rows,
                    generation_rule="First geometry-QA accepted seeds; no policy execution or test-score selection.",
                    corridor=dict(x_zone_edges_m=ZONES, middle_count=22, outside_count=[1, 2], arena_area_m2=200,
                                  density_equivalent="22/80 m² = 55 per full arena; outer zones average 5 per full arena."))
    manifest["fingerprint"] = digest(manifest)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return manifest


def load_suite(path):
    manifest = read_json(path)
    payload = dict(manifest)
    frozen = payload.pop("fingerprint")
    if digest(payload) != frozen or any(geometry_hash(r["map"]) != r["geometry_sha256"]
                                       for r in manifest["maps"]):
        raise RuntimeError("held-out suite fingerprint/geometry mismatch")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    suite = freeze(args.output)
    assert load_suite(args.output / "manifest.json")["sizes"] == suite["sizes"]
    print(suite["sizes"], suite["fingerprint"], flush=True)
