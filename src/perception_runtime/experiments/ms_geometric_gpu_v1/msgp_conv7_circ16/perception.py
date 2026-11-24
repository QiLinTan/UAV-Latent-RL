"""GPU aggregation of yaw-frame RV1 scans into near and far observations."""
import numpy as np
import torch

from experiments.rv1_range_cnn.lidar import LidarConfig

GRID = 7
CELLS = GRID ** 3
NEAR_WIDTH = 2 * CELLS
FAR_WIDTH = 5 * 16
HALF = 1.4
VOXEL = 0.4
BOUNDARY_EPS = 1e-6


def observation_width(use_far_branch):
    return NEAR_WIDTH + FAR_WIDTH * use_far_branch + 18


def unpack(batch, use_far_branch):
    """View the float32 Replay row as its original observation tensors."""
    near = batch[..., :NEAR_WIDTH].reshape(-1, 2, GRID, GRID, GRID)
    far = batch[..., NEAR_WIDTH:-18].reshape(-1, 5, 16) if use_far_branch else None
    return near, far, batch[..., -18:]


def ray_segments(directions, sensor):
    """Precompute the grid cells traversed by each static LiDAR ray."""
    origin = np.array([0., 0., sensor.origin_height_m])
    planes = np.linspace(-HALF, HALF, GRID + 1)
    numerator = planes[None, :] - origin[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        crossings = numerator / directions[:, :, None]
        exits = (np.copysign(HALF, directions) - origin) / directions
    exit_range = np.min(np.where(abs(directions) > 1e-12, exits, np.inf), axis=1)
    crossings = np.sort(np.clip(crossings.reshape(len(directions), -1), sensor.origin_offset_m,
                                exit_range[:, None]), axis=1)
    crossings = np.column_stack((np.full(len(directions), sensor.origin_offset_m), crossings, exit_range))
    start, end = crossings[:, :-1], crossings[:, 1:]
    center = origin + directions[:, None, :] * ((start + end) / 2)[..., None]
    idx = np.clip(np.floor((center + HALF) / VOXEL).astype(np.int64), 0, GRID - 1)
    cell = (idx[..., 0] * GRID + idx[..., 1]) * GRID + idx[..., 2]
    return cell, start, end - start > BOUNDARY_EPS, exit_range


class ObservationBuilder:
    def __init__(self, device, sensor=None, use_far_branch=True):
        self.device = torch.device(device)
        self.sensor = sensor or LidarConfig()
        self.use_far_branch = use_far_branch
        if not self.sensor.circular or self.sensor.max_range_m <= HALF:
            raise ValueError("360-degree RV1 scan with far coverage required")
        directions = self.sensor.directions().reshape(-1, 3)
        cells, start, valid, exits = ray_segments(directions, self.sensor)
        self.cells = self._tensor(cells, torch.int64).flatten()
        self.start = self._tensor(start)
        self.segment_valid = self._tensor(valid, torch.bool)
        self.exits = self._tensor(exits)
        self.origin = self._tensor([0., 0., self.sensor.origin_height_m])
        self.directions = self._tensor(directions)
        # Sector 0 faces forward. Round-to-even at exact bisectors preserves reflection.
        angles = (self.sensor.azimuth_start_deg + np.arange(self.sensor.width) * 360 / self.sensor.width)
        sectors = np.rint(angles / 22.5).astype(np.int64) % 16
        self.sectors = self._tensor(np.tile(sectors, self.sensor.height), torch.int64)
        center = np.arange(16) * (np.pi / 8)
        self.heading = self._tensor(np.stack((np.cos(center), np.sin(center))))

    def _tensor(self, value, dtype=torch.float32):
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def _counts(self, size, indices, mask):
        return torch.zeros(size, dtype=torch.int32, device=self.device).scatter_add_(
            0, indices, mask.to(torch.int32))

    @torch.no_grad()
    def __call__(self, obs):
        # PyBullet supplies first-return scans on the CPU. All spatial aggregation stays on device.
        if any(obs[key].shape != self.sensor.shape for key in ("range_m", "hit_mask", "scanned_mask")):
            raise ValueError("scan shape mismatch")
        data = self._tensor(np.stack((obs["range_m"], obs["hit_mask"], obs["scanned_mask"]))).reshape(3, -1)
        distance, hit, scanned = data
        scanned = scanned.bool()
        distance = torch.where(scanned, distance, 0.)
        hit = hit.bool() & scanned
        state = self._tensor(obs["state"]).reshape(18)
        occupied_hit = hit & (distance <= self.exits + BOUNDARY_EPS)
        point = self.origin + self.directions * distance[:, None]
        idx = ((point + HALF) / VOXEL).floor().long().clamp_(0, GRID - 1)
        occupied_idx = (idx[:, 0] * GRID + idx[:, 1]) * GRID + idx[:, 2]
        occupied = self._counts(CELLS, occupied_idx, occupied_hit) > 0
        traversed = self.segment_valid & scanned[:, None] & (self.start < distance[:, None] - BOUNDARY_EPS)
        free = (self._counts(CELLS, self.cells, traversed.flatten()) > 0) & ~occupied
        near = torch.stack((occupied, free)).float().flatten()
        if not self.use_far_branch:
            return torch.cat((near, state))

        # A near first return occludes its far sector ray. Keep the fixed scanned-ray
        # denominator and classify the occluded ray as unknown in both numerators.
        far_eligible = scanned & (self.exits < self.sensor.max_range_m)
        far_hit = far_eligible & hit & ~occupied_hit
        far_free = far_eligible & ~hit & (distance > self.exits + BOUNDARY_EPS)
        valid = self._counts(16, self.sectors, far_eligible).clamp_min_(1)
        occ_ratio = self._counts(16, self.sectors, far_hit) / valid
        free_ratio = self._counts(16, self.sectors, far_free) / valid
        closest = torch.full((16,), self.sensor.max_range_m, device=self.device)
        closest.scatter_reduce_(0, self.sectors,
                                torch.where(far_hit, distance, self.sensor.max_range_m), reduce="amin")
        norm_dist = ((closest - HALF) / (self.sensor.max_range_m - HALF)).clamp_(0, 1)
        goal = state[:2]
        goal_align = (goal / torch.linalg.vector_norm(goal).clamp_min(1e-6)) @ self.heading
        # State18 already applies the audited 0.8 m/s normalization.
        vel_align = state[4:6] @ self.heading
        far = torch.stack((occ_ratio, free_ratio, norm_dist, goal_align, vel_align))
        return torch.cat((near, far.flatten(), state))
