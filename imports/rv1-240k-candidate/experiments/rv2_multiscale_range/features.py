"""Deterministic first-return features. No scene truth or simulator API."""
import numpy as np
import torch
from torch import nn
from experiments.rv1_range_cnn.codec import RangeCodec
from .contracts import PerceptionSpec

CODEC = RangeCodec((32, 128), 8.)

def canonical_image(raw):
    """Same arithmetic as frozen rv1 Replay, used for online inference too."""
    for key in ('hit_mask', 'scanned_mask'):
        mask = np.asarray(raw[key])
        if np.any((mask != 0) & (mask != 1)):
            raise ValueError('binary raw masks required')
    q, m = CODEC.encode(raw['range_m'], raw['hit_mask'], raw['scanned_mask'])
    r = np.asarray(raw['range_m']); s = np.asarray(raw['scanned_mask'], bool)
    if np.any(~np.isnan(r[~s])):
        raise ValueError('unscanned raw distances must be NaN')
    if np.any(r[s] < .08):
        raise ValueError('return inside unobserved radial blind zone')
    shape = CODEC.shape
    h, s = [np.unpackbits(x, bitorder='little', count=CODEC.size).reshape(shape) for x in m]
    return np.stack((np.where(s, q.astype(np.float32)/65534., 0), h, s)).astype(np.float32)

def validate_image(image):
    if image.ndim != 4 or tuple(image.shape[1:]) != (3, 32, 128) or not image.is_floating_point():
        raise ValueError('B x 3 x 32 x 128 canonical image required')
    if not bool(torch.isfinite(image).all()):
        raise ValueError('nonfinite canonical image')
    d, h, s = image.unbind(1)
    if bool(((h != 0) & (h != 1)).any() | ((s != 0) & (s != 1)).any()):
        raise ValueError('binary masks required')
    if bool(((h > s) | (d < 0) | (d > 1) | ((s == 0) & (d != 0)) | ((s == 1) & (h == 0) & (d != 1))).any()):
        raise ValueError('first-return masks/range contract')
    # Quantization can put a legal .08m return half a quantum below .08.
    if bool(((s == 1) & (d*8 < .08 - CODEC.quantum_m/2 - 1e-7)).any()):
        raise ValueError('blind-zone return')
    return d*8, h.bool(), s.bool()

def radial_images(image, spec=PerceptionSpec(), validate=True):
    """B x 3 x 5 x 32 x 128; free fractions refer to ray intervals only."""
    if validate:
        d, h, s = validate_image(image)
    else:
        d, h, s = image[:, 0]*8, image[:, 1].bool(), image[:, 2].bool()
    d = torch.where(s, d.clamp_min(.08), torch.full_like(d, 8.))
    bands = []
    for i, (low, high) in enumerate(zip(spec.band_edges_m, spec.band_edges_m[1:])):
        surface = s & h & (d >= low) & ((d <= high) if i == 2 else (d < high))
        local = torch.where(surface, ((d-low)/(high-low)).clamp(0, 1), torch.zeros_like(d))
        free = (torch.minimum(d, torch.full_like(d, high)) - max(low, .08)).clamp_min(0)/(high-low)
        free = free * s
        blocked = s & h & (d < low)
        bands.append(torch.stack((surface.to(d.dtype), local, free, blocked.to(d.dtype), s.to(d.dtype)), 1))
    return torch.stack(bands, 1)

def sector_indices(directions, spec):
    """Actual yaw-frame unit directions, not guessed row splits."""
    directions = np.asarray(directions, np.float64)
    if directions.shape != (32, 128, 3) or not np.isfinite(directions).all():
        raise ValueError('actual direction grid required')
    if not np.allclose(np.linalg.norm(directions, axis=-1), 1, atol=1e-12):
        raise ValueError('unit directions')
    a = np.rad2deg(np.arctan2(directions[..., 1], directions[..., 0]))
    e = np.rad2deg(np.arctan2(directions[..., 2], np.linalg.norm(directions[..., :2], axis=-1)))
    width = 360/spec.horizontal_sectors
    horizontal = np.floor(np.mod(a+width/2, 360)/width + 1e-12).astype(int) % spec.horizontal_sectors
    edges = np.asarray(spec.vertical_edges_deg)
    if np.any((e < edges[0]-1e-9) | (e > edges[-1]+1e-9)):
        raise ValueError('directions outside sector VFOV')
    vertical = np.searchsorted(edges[1:-1], np.round(e, 10), side='right')
    return (vertical*spec.horizontal_sectors+horizontal).astype(np.int64)

class NearBypass(nn.Module):
    """Per-sector minimum first return, then scan coverage; no safety action."""
    def __init__(self, directions, spec=PerceptionSpec(bypass=True)):
        super().__init__()
        self.spec = spec
        indices = sector_indices(directions, spec).ravel()
        self.count = spec.horizontal_sectors*(len(spec.vertical_edges_deg)-1)
        self.register_buffer('indices', torch.from_numpy(indices), persistent=False)
        denominators = np.bincount(indices, minlength=self.count).astype(np.float32)
        if np.any(denominators == 0):
            raise ValueError('empty predefined angular sector')
        self.register_buffer('denominators', torch.from_numpy(denominators), persistent=False)

    def forward(self, image, validate=True):
        if validate:
            d, h, s = validate_image(image)
        else:
            d, h, s = image[:, 0]*8, image[:, 1].bool(), image[:, 2].bool()
        valid = h & s & (d < self.spec.near_threshold_m)
        distances = torch.where(valid, d.clamp_min(.08)/self.spec.near_threshold_m, torch.ones_like(d)).flatten(1)
        idx = self.indices[None].expand(image.shape[0], -1)
        nearest = torch.ones(image.shape[0], self.count, device=image.device, dtype=image.dtype)
        nearest.scatter_reduce_(1, idx, distances, reduce='amin', include_self=True)
        coverage = torch.zeros_like(nearest).scatter_add_(1, idx, s.flatten(1).to(image.dtype))/self.denominators
        return torch.stack((nearest, coverage), -1).flatten(1)
