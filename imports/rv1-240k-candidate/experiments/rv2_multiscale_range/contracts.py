"""Versioned canonical sensor, radial features, bypass and network contract."""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json

SCHEMA = 'rv2_multiscale_range_td3_v1'
SENSOR = dict(height=32, width=128, horizontal_fov_deg=360., vertical_min_deg=-30.,
              vertical_max_deg=30., azimuth_start_deg=0., max_range_m=8.,
              origin_offset_m=.08, origin_height_m=.02, batch_size=1024)
CHANNELS = ['hit_in_bin', 'local_hit_depth', 'visible_free_fraction', 'blocked_before_bin', 'scanned']

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

@dataclass(frozen=True)
class PerceptionSpec:
    encoder: str = 'multiscale'
    bypass: bool = False
    mode: str = 'full'
    horizontal_sectors: int = 8
    vertical_edges_deg: tuple = (-30., -10., 10., 30.)
    band_edges_m: tuple = (0., 1.5, 4., 8.)
    near_threshold_m: float = 1.5
    latent_dims: tuple = (24, 24, 16)

    def __post_init__(self):
        if self.encoder not in ('single', 'multiscale', 'joint_bands', 'single_large'):
            raise ValueError('unknown encoder')
        if self.mode != 'full':
            raise ValueError('angular input downsampling deferred; full 32x128 only')
        if self.horizontal_sectors not in (4, 8, 16):
            raise ValueError('supported sector ablations: 4/8/16')
        if len(self.vertical_edges_deg) < 2 or self.vertical_edges_deg[0] != -30. or self.vertical_edges_deg[-1] != 30.:
            raise ValueError('partition must cover actual VFOV')
        if any(a >= b for a, b in zip(self.vertical_edges_deg, self.vertical_edges_deg[1:])):
            raise ValueError('ordered elevation edges')
        if len(self.band_edges_m) != 4 or self.band_edges_m[0] != 0. or self.band_edges_m[-1] != 8.:
            raise ValueError('three radial bins over original sensor range')
        if any(a >= b for a, b in zip(self.band_edges_m, self.band_edges_m[1:])):
            raise ValueError('ordered radial bins')
        if self.near_threshold_m != self.band_edges_m[1] or self.latent_dims != (24, 24, 16):
            raise ValueError('freeze threshold and 64D latent allocation')

    @property
    def bypass_dim(self):
        return 2 * self.horizontal_sectors * (len(self.vertical_edges_deg) - 1) if self.bypass else 0

def make_contract(spec, parent_contract, sources_sha256):
    if parent_contract.get('sensor') != SENSOR or parent_contract.get('state_dim') != 18:
        raise ValueError('exact frozen rv1 sensor and State18 required')
    return dict(schema=SCHEMA, sensor=SENSOR, parent_contract=parent_contract,
                perception=asdict(spec), sources_sha256=sources_sha256,
                state_dim=18, action_dim=3, latent_dim=64, policy_input_dim=82 + spec.bypass_dim,
                sensor_frame='yaw_horizontal', scan_frequency_hz=15, control_hz=120, physics_hz=240,
                channels=CHANNELS, canonicalization='rv1 uint16 quantum 8/65534 then float32 q/65534; online and Replay identical',
                radial_endpoints='left closed right open; final bin includes 8m; near hit excludes 1.5m',
                bypass_order='vertical zone then horizontal sector; interleaved nearest/coverage',
                azimuth_sectors='centered on k*360/N degrees; half-open; wrap at -180/180',
                vertical_sectors='actual elevation angles, left closed right open, final includes +30',
                blindness='[0,.08)m unknown even when scanned; no volume interpolation',
                task_semantics='inherit exact frozen parent reward/done/control/course; terminated OR truncated masks bootstrap')

def validate_contract(actual, expected):
    if actual.get('schema') != SCHEMA or digest(actual) != digest(expected):
        raise ValueError('rv2 schema/config/source mismatch; old 114D, rv1 and other rv2 variants rejected')

def source_manifest(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(root).rglob('*.py')) if 'outputs' not in p.parts and '__pycache__' not in p.parts}
