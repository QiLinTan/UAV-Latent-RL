"""Parallel PyBullet ray extraction with the original RV1 scan contract."""
import numpy as np
import pybullet as p

from experiments.direct_velocity_full_v1.contracts import yaw_from_quaternion, yaw_rotation
from experiments.rv1_range_cnn.lidar import Lidar, Scan


class ParallelLidar(Lidar):
    def scan(self, *, client_id, position, quaternion_xyzw, timestamp_s, cache_key=None,
             scanned_mask=None, self_ids=(), ground_ids=(), obstacle_ids=()):
        config = self.config
        position = np.asarray(position, dtype=float)
        quaternion = np.asarray(quaternion_xyzw, dtype=float)
        if (position.shape != (3,) or quaternion.shape != (4,) or not np.isfinite(position).all()
                or not np.isfinite(quaternion).all() or abs(np.linalg.norm(quaternion) - 1) > 1e-5
                or not np.isfinite(timestamp_s) or timestamp_s < 0):
            raise ValueError("finite SI pose, unit xyzw quaternion and nonnegative timestamp required")
        mask = np.ones(config.shape, bool) if scanned_mask is None else np.asarray(scanned_mask, dtype=bool)
        if mask.shape != config.shape:
            raise ValueError("scan mask shape")
        if cache_key is not None and cache_key == self.cache_key:
            if not np.array_equal(mask, self.cache.scanned_mask):
                raise ValueError("cache key reused for different scan mask")
            return self.cache
        directions = self.directions_local @ yaw_rotation(yaw_from_quaternion(quaternion)).T
        origin = position + [0., 0., config.origin_height_m]
        scanned = mask.ravel()
        rays = directions.reshape(-1, 3)[scanned]
        starts = (origin + rays * config.origin_offset_m).tolist()
        ends = (origin + rays * config.max_range_m).tolist()
        results = []
        calls = 0
        limit = p.MAX_RAY_INTERSECTION_BATCH_SIZE
        for first in range(0, len(rays), limit):
            results.extend(p.rayTestBatch(starts[first:first + limit], ends[first:first + limit],
                                         numThreads=4, physicsClientId=client_id))
            calls += 1
        count = len(scanned)
        hit_ids = np.fromiter((r[0] for r in results), dtype=np.int32, count=len(results))
        fractions = np.fromiter((r[2] for r in results), dtype=np.float64, count=len(results))
        ranges = np.full(count, np.nan, dtype=np.float32)
        ranges[scanned] = config.origin_offset_m + fractions * (config.max_range_m - config.origin_offset_m)
        body = np.full(count, -1, dtype=np.int32)
        body[scanned] = hit_ids
        hit = body >= 0
        xyz = np.full((count, 3), np.nan)
        xyz[hit] = np.asarray([r[3] for r in results if r[0] >= 0], dtype=np.float64).reshape(-1, 3)
        types = np.zeros(count, dtype=np.uint8)
        types[hit] = 4
        types[hit & np.isin(body, obstacle_ids)] = 3
        types[hit & np.isin(body, ground_ids)] = 2
        types[hit & np.isin(body, self_ids)] = 1
        scan = Scan(config, float(timestamp_s), origin, directions, ranges.reshape(config.shape),
                    hit.reshape(config.shape), mask.copy(), body.reshape(config.shape),
                    types.reshape(config.shape), xyz.reshape(*config.shape, 3))
        for array in (scan.origin_world, scan.directions_world, scan.range_m, scan.hit_mask, scan.scanned_mask,
                      scan.hit_body, scan.hit_type, scan.raw_hit_world):
            array.setflags(write=False)
        self.cache = scan
        self.cache_key = cache_key
        self.scan_count += 1
        self.batch_calls += calls
        return scan


def use_parallel_lidar(env):
    env.lidar = ParallelLidar(env.lidar.config)
    return env
