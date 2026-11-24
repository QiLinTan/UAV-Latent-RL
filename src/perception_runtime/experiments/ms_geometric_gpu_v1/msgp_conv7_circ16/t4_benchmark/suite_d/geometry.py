"""Policy-blind geometry, configuration-space graph, and projection for Suite D."""
from dataclasses import dataclass
from functools import cached_property
import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

RADIUS = .062
MARGIN = .10
SAFE = RADIUS + MARGIN
RESOLUTION = .10
HEIGHT = (.08, 2.75)
SEED_BASE = 470000
TOPOLOGIES = ("D1", "D2", "D3", "D4")


def box(name, lower, upper):
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if np.any(lower >= upper):
        raise ValueError(f"invalid box: {name}")
    return dict(name=name, kind="box", position=((lower + upper) / 2).tolist(),
                half_extents=((upper - lower) / 2).tolist())


def generate(topology, index):
    variant = index // 2
    seed = SEED_BASE + 1000 * TOPOLOGIES.index(topology) + variant
    rng = np.random.default_rng(seed)
    side = (-1, 1)[index % 2]
    bounds = (4.4, 2.8)
    objects = []
    start = (-3.0, 0., 1.35)
    goal = (3.0, 0., 1.35)
    closed = topology in ("D3", "D4")

    def add(name, lower, upper):
        objects.append(box(name, lower, upper))

    def caps():
        limit_x, limit_y = bounds
        add("floor", [-limit_x, -limit_y, 0.], [limit_x, limit_y, .12])
        add("roof", [-limit_x, -limit_y, 2.59], [limit_x, limit_y, 2.78])

    if topology == "D1":
        thickness = float(np.linspace(.05, .20, 10)[variant])
        x = float(rng.uniform(-.30, .30))
        z = (0.36 + .014 * variant) if side < 0 else (2.43 - .014 * variant)
        start = (-3., 0., z)
        goal = (3., 0., z)
        add("thin_beam", [x - .09, -bounds[1], z - thickness / 2],
            [x + .09, bounds[1], z + thickness / 2])
        details = dict(x=x, front_x=x - .09, rear_x=x + .09, z=z, thickness=thickness,
                       intended_pass="above" if side < 0 else "below")
    elif topology == "D2":
        x = float(rng.uniform(-.25, .25))
        width = float(np.linspace(1.02, .60, 10)[variant])
        height = float(np.linspace(.98, .60, 10)[variant])
        y = side * float(rng.uniform(.34, .57))
        z_offset = 0.0 if variant < 2 else float(np.linspace(.4, .8, 8)[variant - 2]) * side
        z = 1.35 + z_offset
        left, right = y - width / 2, y + width / 2
        low, high = z - height / 2, z + height / 2
        wall_y = 1.65
        front, rear = x - .09, x + .09
        for name, y0, y1, z0, z1 in (
            ("window_left", -wall_y, left, 0, 2.85),
            ("window_right", right, wall_y, 0, 2.85),
            ("window_sill", left, right, 0, low),
            ("window_lintel", left, right, high, 2.85),
        ):
            add(name, [front, y0, z0], [rear, y1, z1])
        details = dict(x=x, front_x=front, rear_x=rear, y_window=[left, right], z_window=[low, high],
                       y_wall_half=wall_y, width=width, height=height, z_offset=z_offset)
    elif topology == "D3":
        bounds = (7.5, 4.2)
        caps()
        width = float(np.linspace(3.2, 1.8, 10)[variant])
        amplitude = float(np.linspace(1.25, 2.25, 10)[variant])
        turns = 1 if index % 2 == 0 else 2
        edges = np.linspace(-bounds[0], bounds[0], 61)
        centers = (edges[1:] + edges[:-1]) / 2
        origin = float(rng.uniform(-.25, .25))
        ys = origin + amplitude * np.sin(turns * np.pi * (centers + bounds[0]) / (2 * bounds[0]))
        for slab, (left, right, y) in enumerate(zip(edges, edges[1:], ys)):
            add(f"cave_{slab:02}_south", [left, -bounds[1], .12], [right, y - width / 2, 2.59])
            add(f"cave_{slab:02}_north", [left, y + width / 2, .12], [right, bounds[1], 2.59])
        add("end_west", [-7.5, -4.2, .12], [-7.3, 4.2, 2.59])
        add("end_east", [7.3, -4.2, .12], [7.5, 4.2, 2.59])
        start_x, goal_x = -6.65, 6.65
        start = (start_x, float(np.interp(start_x, centers, ys)), 1.30)
        goal = (goal_x, float(np.interp(goal_x, centers, ys)), 1.30)
        details = dict(shape="single_bend" if turns == 1 else "S_bend", width=width, amplitude=amplitude,
                       nominal_min_curvature_radius=(15 / (turns * np.pi)) ** 2 / amplitude,
                       first_bend_x=-7.5 + 7.5 / turns, centerline_xy=np.column_stack((centers, ys)).tolist())
    else:
        bounds = (7., 5.5)
        caps()
        width = float(np.linspace(3.0, 1.8, 10)[variant])
        half = width / 2
        y = float(rng.uniform(.35, 1.15))
        x_end = float(rng.uniform(5.7, 6.5))
        stem_end = -5.10
        add("mine_north", [-7, y + half, .12], [7, 5.5, 2.59])
        add("mine_south", [-7, -5.5, .12], [7, stem_end, 2.59])
        add("mine_west_mass", [-7, stem_end, .12], [-half, y - half, 2.59])
        add("mine_east_mass", [half, stem_end, .12], [7, y - half, 2.59])
        add("mine_west_end", [-7, y - half, .12], [-x_end, y + half, 2.59])
        add("mine_east_end", [x_end, y - half, .12], [7, y + half, 2.59])
        start = (0., stem_end + .65, 1.30)
        goal = (side * (x_end - .58), y, 1.30)
        details = dict(width=width, junction_y=y, junction_south=y - half, junction_north=y + half,
                       arm_gate_x=half + .30, target_arm="left" if side < 0 else "right", target_sign=side)
    record = dict(map_id=f"suite_d_{topology.lower()}_{index:02}", source="Suite D v1 policy-blind generator",
                  scenario=topology, split="test", start_pos=list(start), target_pos=list(goal),
                  xy_half_extent=max(bounds), xy_half_extents=list(bounds),
                  navigation_z_bounds=list(HEIGHT), obstacle_sensor_range=8.,
                  box_collision_mode="triangles" if topology == "D3" else "primitive", obstacles=objects)
    return record, details, seed, closed


def extents(record):
    centers = np.array([entry["position"] for entry in record["obstacles"]])
    halves = np.array([entry["half_extents"] for entry in record["obstacles"]])
    return centers - halves, centers + halves


def blocked(starts, ends, lower, upper, pad=0.):
    """Intersect closed segment(s) with the union of the expanded boxes."""
    starts = np.atleast_2d(starts)[:, None, :]
    delta = np.atleast_2d(ends)[:, None, :] - starts
    active = abs(delta) > 1e-12
    divisor = np.where(active, delta, 1.)
    enter = (lower[None, :, :] - pad - starts) / divisor
    leave = (upper[None, :, :] + pad - starts) / divisor
    inside = ((starts >= lower - pad) & (starts <= upper + pad)) | active
    near = np.where(active, np.minimum(enter, leave), -np.inf).max(axis=2)
    far = np.where(active, np.maximum(enter, leave), np.inf).min(axis=2)
    return (inside.all(axis=2) & (np.maximum(near, 0.) <= np.minimum(far, 1.))).any(axis=1)


@dataclass
class Graph:
    record: dict
    positions: np.ndarray
    remaining: np.ndarray
    route: np.ndarray
    length: float

    @cached_property
    def tree(self):
        return cKDTree(self.positions)

    def progress(self, positions):
        lower, upper = extents(self.record)
        distances, candidates = self.tree.query(positions, k=min(16, len(self.positions)))
        good = np.zeros(len(positions), dtype=bool)
        progress = np.zeros(len(positions))
        for index, position in enumerate(positions):
            indices = np.atleast_1d(candidates[index])
            visible = ~blocked(position, self.positions[indices], lower, upper)
            if visible.any():
                nearest = float(np.atleast_1d(distances[index])[visible][0])
                near = visible & (np.atleast_1d(distances[index]) <= nearest + RESOLUTION + 1e-9)
                additional = np.atleast_1d(distances[index])[near]
                progress[index] = self.length - np.min(self.remaining[indices[near]] + additional)
                good[index] = True
            elif index:
                progress[index] = progress[index - 1]
        progress[0] = 0.
        return progress, good


def solve(record, aperture_only=False):
    """Euclidean weighted 26-neighbour shortest path on a certified 3D grid."""
    bound_x, bound_y = record["xy_half_extents"]
    if aperture_only:
        bound_y = 1.64
    low = np.array([-bound_x, -bound_y, HEIGHT[0]])
    high = np.array([bound_x, bound_y, HEIGHT[1]])
    lower, upper = extents(record)
    pad = SAFE + RESOLUTION * math.sqrt(3) / 2
    axes = [(lo + hi) / 2 + np.arange(-count, count + 1) * RESOLUTION
            for lo, hi in zip(low, high) for count in [int(np.floor((hi - lo) / 2 / RESOLUTION - pad / RESOLUTION))]]
    shape = tuple(len(axis) for axis in axes)
    grid = np.ones(shape, dtype=bool)
    for begin, end in zip(lower - pad, upper + pad):
        spans = tuple(slice(axis.searchsorted(start), axis.searchsorted(stop, side="right"))
                      for axis, start, stop in zip(axes, begin, end))
        grid[spans] = False
    ijk = np.nonzero(grid)
    positions = np.column_stack([axis[indices] for axis, indices in zip(axes, ijk)])
    if not len(positions):
        raise RuntimeError("no certified free grid nodes")
    labels = np.full(shape, -1, dtype=np.int32)
    labels[ijk] = np.arange(len(positions), dtype=np.int32)
    row, col, weights = [], [], []

    def edge(source, destination, weight):
        row.extend((source, destination))
        col.extend((destination, source))
        weights.extend((weight, weight))

    for shift in np.ndindex((3, 3, 3)):
        delta = tuple(value - 1 for value in shift)
        if delta <= (0, 0, 0):
            continue
        source = labels[tuple(slice(max(0, -d), size - max(0, d)) for d, size in zip(delta, shape))]
        target = labels[tuple(slice(max(0, d), size - max(0, -d)) for d, size in zip(delta, shape))]
        valid = (source >= 0) & (target >= 0)
        left, right = source[valid], target[valid]
        edge(left, right, np.full(len(left), RESOLUTION * np.linalg.norm(delta)))

    endpoints = np.array([record["start_pos"], record["target_pos"]])
    tree = cKDTree(positions)
    for index, point in enumerate(endpoints):
        candidates = np.asarray(tree.query_ball_point(point, 3 * RESOLUTION), dtype=int)
        if not len(candidates) or np.any(point <= low + SAFE) or np.any(point >= high - SAFE):
            raise RuntimeError("endpoint outside certified domain")
        visible = candidates[~blocked(point, positions[candidates], lower, upper, SAFE)]
        if not len(visible):
            raise RuntimeError("endpoint lacks certified connector")
        edge(np.full(len(visible), len(positions) + index), visible,
             np.maximum(1e-12, np.linalg.norm(positions[visible] - point, axis=1)))
    size = len(positions) + len(endpoints)
    matrix = coo_matrix((np.concatenate(weights), (np.concatenate(row), np.concatenate(col))),
                        shape=(size, size)).tocsr()
    source, goal = size - 2, size - 1
    distance, predecessor = dijkstra(matrix, directed=True, indices=goal, return_predecessors=True)
    if not np.isfinite(distance[source]):
        raise RuntimeError("start and goal are not connected in 3D")
    positions = np.vstack((positions, endpoints))
    route = [source]
    while route[-1] != goal:
        route.append(int(predecessor[route[-1]]))
    valid = np.isfinite(distance)
    return Graph(record, positions[valid], distance[valid], positions[route], float(distance[source]))
