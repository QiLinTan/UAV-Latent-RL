"""Conservative 3D geometry QA; paths are privileged data, never policy inputs."""
from heapq import heappop, heappush
from functools import lru_cache
import numpy as np
from .spec import MapValidationReport


@lru_cache(maxsize=16)
def _box_arrays(obstacles):
    return np.array([o.position for o in obstacles]), np.array([o.half_extents for o in obstacles])


def signed_clearance(spec, points):
    """Exact point signed distance to the union of axis-aligned primitives."""
    points = np.asarray(points, dtype=float)
    if spec.obstacles and all(o.kind == 'box' for o in spec.obstacles):
        centers, extents = _box_arrays(spec.obstacles)
        flat = points.reshape(-1, 3)
        result = np.empty(len(flat))
        for begin in range(0, len(flat), 512):
            d = np.abs(flat[begin:begin+512, None, :] - centers) - extents
            distances = np.linalg.norm(np.maximum(d, 0), axis=-1) + np.minimum(np.max(d, axis=-1), 0)
            result[begin:begin+512] = distances.min(axis=1)
        return result.reshape(points.shape[:-1])
    best = np.full(points.shape[:-1], np.inf)
    for obstacle in spec.obstacles:
        q = points - obstacle.position
        if obstacle.kind == 'box':
            d = np.abs(q) - obstacle.half_extents
            distance = np.linalg.norm(np.maximum(d, 0), axis=-1) + np.minimum(np.max(d, axis=-1), 0)
        elif obstacle.kind == 'cylinder':
            d = np.stack((np.linalg.norm(q[..., :2], axis=-1) - obstacle.radius,
                          np.abs(q[..., 2]) - obstacle.height / 2), axis=-1)
            distance = np.linalg.norm(np.maximum(d, 0), axis=-1) + np.minimum(np.max(d, axis=-1), 0)
        else:
            raise ValueError('3D primitive QA does not certify mesh geometry')
        best = np.minimum(best, distance)
    return best


def find_route_3d(spec, *, vehicle_radius=.06, margin=.10, resolution=.20, fixed_z=None):
    """6-neighbour A* with cell-diagonal padding and checked endpoint connectors.

    The finite grid is capped to bound resource usage. A failed search is a
    conservative discretized result, not proof that no continuous path exists.
    fixed_z is a diagnostic plane restriction, not used for regular validation.
    """
    if not np.isfinite([vehicle_radius, margin, resolution]).all() or vehicle_radius < 0 or margin < 0 or resolution <= 0:
        raise ValueError('Invalid QA radius, margin or resolution')
    if spec.navigation_z_bounds is None:
        raise ValueError('3D QA requires explicit navigation_z_bounds')
    if spec.uses_uncertified_mesh:
        raise ValueError('3D primitive QA does not certify mesh geometry')
    lo = np.array([-spec.bounds_xy[0], -spec.bounds_xy[1], spec.navigation_z_bounds[0]])
    hi = np.array([spec.bounds_xy[0], spec.bounds_xy[1], spec.navigation_z_bounds[1]])
    safe = vehicle_radius + margin
    start, goal = np.array(spec.start_pos), np.array(spec.target_pos)
    if fixed_z is not None:
        if not np.isfinite(fixed_z):
            raise ValueError('fixed_z must be finite')
        start[2] = goal[2] = fixed_z
    def valid(points, pad=safe):
        return (np.all(points >= lo + pad, axis=-1) & np.all(points <= hi - pad, axis=-1)
                & (signed_clearance(spec, points) > pad))
    def report(ok, reachable, message, path=None):
        return MapValidationReport(spec.map_id, ok, reachable, message), np.empty((0, 3)) if path is None else path
    if not valid(np.array([start, goal])).all():
        return report(False, False, '3D endpoints violate bounds or clearance')
    counts = np.maximum(0, np.floor((hi - lo - 2 * safe) / resolution).astype(int) + 1)
    if fixed_z is not None:
        counts[2] = 1
    if not counts.all():
        return report(True, False, 'No room for a conservative 3D grid')
    if int(np.prod(counts)) > 2_000_000:
        raise ValueError('3D QA grid exceeds 2 million cells; increase resolution')
    axes = [lo[i] + safe + np.arange(counts[i]) * resolution for i in range(3)]
    if fixed_z is not None:
        axes[2] = np.array([fixed_z])
    xyz = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
    # Lipschitz distance bound certifies entire edges/cells, not only nodes.
    padding = safe + resolution * np.sqrt(2 if fixed_z is not None else 3) / 2
    free = np.all(xyz >= lo + padding, axis=-1) & np.all(xyz <= hi - padding, axis=-1)
    # Rasterize expanded primitives locally instead of evaluating every obstacle
    # at every cell. Box inflation is deliberately conservative at corners.
    for obstacle in spec.obstacles:
        center = np.asarray(obstacle.position)
        extent = np.array(obstacle.half_extents if obstacle.kind == 'box' else
                          [obstacle.radius, obstacle.radius, obstacle.height / 2]) + padding
        slices = tuple(slice(int(np.searchsorted(axes[i], center[i]-extent[i], side='left')),
                             int(np.searchsorted(axes[i], center[i]+extent[i], side='right'))) for i in range(3))
        if obstacle.kind == 'box':
            free[slices] = False
        else:
            local = xyz[slices] - center
            occupied = ((np.linalg.norm(local[..., :2], axis=-1) <= obstacle.radius + padding)
                        & (np.abs(local[..., 2]) <= obstacle.height/2 + padding))
            free[slices] &= ~occupied
    shape = free.shape
    def nearest(point):
        return tuple(int(np.argmin(abs(axis - point[i]))) for i, axis in enumerate(axes))
    first, last = nearest(start), nearest(goal)
    for point, index in ((start, first), (goal, last)):
        if not free[index] or not valid(np.linspace(point, xyz[index], 17), safe + resolution / 16).all():
            return report(True, False, 'Endpoint cannot connect safely to conservative grid')
    queue = [(0., 0, first)]
    costs = {first: 0}
    parent = {}
    while queue:
        _, cost, node = heappop(queue)
        if cost != costs[node]:
            continue
        if node == last:
            indices = [node]
            while node != first:
                node = parent[node]
                indices.append(node)
            path = np.vstack((start, [xyz[i] for i in reversed(indices)], goal))
            return report(True, True, 'Conservative 3D primitive route; geometric QA only', path)
        for axis in range(3):
            for step in (-1, 1):
                neighbour = list(node)
                neighbour[axis] += step
                neighbour = tuple(neighbour)
                if not all(0 <= neighbour[i] < shape[i] for i in range(3)) or not free[neighbour]:
                    continue
                new = cost + 1
                if new < costs.get(neighbour, np.inf):
                    costs[neighbour] = new
                    parent[neighbour] = node
                    h = sum(abs(neighbour[i] - last[i]) for i in range(3))
                    heappush(queue, (new + h, new, neighbour))
    return report(True, False, 'No route on conservative 3D grid')
