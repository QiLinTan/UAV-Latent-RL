"""EGO-inspired rectangular forests represented as exact collision cylinders.

This is a Python scene generator, not a point-for-point port of EGO's C++ RNG
or voxelized cloud. Positions are grid snapped; diameter and height are explicit
physical quantities. The returned route is privileged QA data only.
"""
from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .spec import MapSpec

SOURCE = 'https://github.com/ZJU-FAST-Lab/ego-planner-swarm/blob/master/src/uav_simulator/map_generator/src/random_forest_sensing.cpp'


@dataclass(frozen=True)
class EgoForestConfig:
    size_x: float = 20.
    size_y: float = 10.
    tree_count: int = 40
    diameter_range: tuple = (.4, .6)
    height_range: tuple = (3., 4.)
    minimum_center_distance: float = 1.
    position_resolution: float = .1
    endpoint_clearance: float = .8
    start: tuple = (-8., 0., 1.)
    goal: tuple = (8., 0., 1.)
    sensor_range: float = 1.75
    episode_len_sec: float = 40.

    def validate(self):
        if self.tree_count < 1 or int(self.tree_count) != self.tree_count:
            raise ValueError('tree_count must be a positive integer')
        for name in ('size_x', 'size_y', 'minimum_center_distance', 'position_resolution',
                     'endpoint_clearance', 'sensor_range', 'episode_len_sec'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for limits in (self.diameter_range, self.height_range):
            if len(limits) != 2 or not np.isfinite(limits).all() or not 0 < limits[0] <= limits[1]:
                raise ValueError('diameter/height limits must be ordered positive values')
        for point in (self.start, self.goal):
            if len(point) != 3 or not np.isfinite(point).all():
                raise ValueError('endpoints must be finite 3D vectors')
            if np.any(np.abs(point[:2]) >= np.array([self.size_x, self.size_y]) / 2):
                raise ValueError('endpoints must be inside the rectangle')
        if self.start[2] != self.goal[2] or self.start[2] <= 0:
            raise ValueError('This forest QA route requires equal positive endpoint altitude')


def clearance_route(spec, *, resolution=.1, vehicle_radius=.06, margin=.2):
    """Conservative 4-neighbour path at the task altitude, inside the rectangle."""
    half_x, half_y = spec.bounds_xy
    # Keep even cell centers clear of domain boundaries by a vehicle envelope.
    xs = np.arange(-half_x + vehicle_radius + margin, half_x - vehicle_radius - margin + 1e-9, resolution)
    ys = np.arange(-half_y + vehicle_radius + margin, half_y - vehicle_radius - margin + 1e-9, resolution)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    free = np.ones(xx.shape, dtype=bool)
    padding = vehicle_radius + margin + resolution * np.sqrt(2) / 2
    for obstacle in spec.obstacles:
        if obstacle.kind != 'cylinder':
            raise ValueError('Forest QA route supports cylinders only')
        free &= (xx-obstacle.position[0])**2 + (yy-obstacle.position[1])**2 > (obstacle.radius+padding)**2
    def index(point):
        return int(np.rint((point[0]-xs[0])/resolution)), int(np.rint((point[1]-ys[0])/resolution))
    start, goal = index(spec.start_pos), index(spec.target_pos)
    for cell in (start, goal):
        if not (0 <= cell[0] < len(xs) and 0 <= cell[1] < len(ys)) or not free[cell]:
            return None
    parents = {start: None}
    queue = deque([start])
    while queue and goal not in parents:
        current = queue.popleft()
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            candidate = current[0]+dx, current[1]+dy
            if (0 <= candidate[0] < len(xs) and 0 <= candidate[1] < len(ys)
                    and candidate not in parents and free[candidate]):
                parents[candidate] = current
                queue.append(candidate)
    if goal not in parents:
        return None
    path = []
    current = goal
    while current is not None:
        path.append((xs[current[0]], ys[current[1]], spec.start_pos[2]))
        current = parents[current]
    path = np.asarray(path[::-1])
    path[0], path[-1] = spec.start_pos, spec.target_pos
    return path


def generate_ego_forest(config=EgoForestConfig(), *, seed=43):
    config.validate()
    rng = np.random.default_rng(seed)
    centers = []
    obstacles = []
    attempts = 0
    budget = config.tree_count * 500
    half = np.array([config.size_x, config.size_y]) / 2
    while len(obstacles) < config.tree_count and attempts < budget:
        attempts += 1
        radius = float(rng.uniform(*config.diameter_range) / 2)
        xy = rng.uniform(-half, half)
        xy = np.floor(xy/config.position_resolution)*config.position_resolution + config.position_resolution/2
        if np.any(np.abs(xy) + radius >= half):
            continue
        if any(np.linalg.norm(xy-np.asarray(p[:2])) < radius + config.endpoint_clearance
               for p in (config.start, config.goal)):
            continue
        if any(np.linalg.norm(xy-c) < config.minimum_center_distance for c in centers):
            continue
        height = float(rng.uniform(*config.height_range))
        centers.append(xy)
        obstacles.append(dict(name=f'tree_{len(obstacles):03d}', kind='cylinder',
                              position=[float(xy[0]),float(xy[1]),height/2],
                              radius=radius, height=height, color=[.42,.26,.12,1.]))
    if len(obstacles) != config.tree_count:
        raise RuntimeError(f'Placed {len(obstacles)}/{config.tree_count} trees within {budget} attempts')
    record = dict(map_id=f'ego_forest_{config.size_x:g}x{config.size_y:g}_seed{seed}_v1',
                  source='local_ego_inspired_analytic_generator_v1', scenario='forest', split='development',
                  start_pos=list(config.start), target_pos=list(config.goal),
                  xy_half_extent=float(max(half)), xy_half_extents=half.tolist(),
                  obstacle_sensor_range=config.sensor_range, episode_len_sec=config.episode_len_sec,
                  environment_overrides=dict(curriculum=False, route_count_curriculum=False,
                                             route_blocking_tree=False, route_blocking_tree_count=0,
                                             curriculum_simple_mix_probability=0.),
                  source_url=SOURCE, raw_asset_status='local_generated_frozen_instance',
                  notes='EGO-inspired bounded uniform centers with minimum center distance; analytic cylinders; no forced route trees or protected corridor. Development map, not a held-out benchmark.',
                  obstacles=obstacles)
    spec = MapSpec.from_dict(record, root=Path('.'))
    route = clearance_route(spec)
    if route is None:
        raise RuntimeError(f'Seed {seed} has no certified route; choose another seed explicitly')
    payload = json.dumps({'bounds': record['xy_half_extents'], 'start': record['start_pos'],
                          'goal': record['target_pos'], 'obstacles': obstacles}, sort_keys=True, separators=(',',':'))
    metadata = dict(generator_version='ego_inspired_analytic_v1', seed=seed, config=config.__dict__,
                    source_url=SOURCE, sampling_attempts=attempts,
                    geometry_sha256=hashlib.sha256(payload.encode()).hexdigest(),
                    route_validation=dict(method='4-neighbour BFS at task altitude', resolution_m=.1,
                                          vehicle_radius_m=.06, additional_clearance_m=.2,
                                          additional_cell_padding_m=float(.1*np.sqrt(2)/2),
                                          reachable=True, path_length_m=float(np.linalg.norm(np.diff(route,axis=0),axis=1).sum()),
                                          purpose='Geometry QA only; never observation or policy input'))
    return record, metadata, route
