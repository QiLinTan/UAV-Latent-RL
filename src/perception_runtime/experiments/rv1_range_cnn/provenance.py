"""Version contracts include rv1 sensor/codec/model and frozen dv0 hashes."""
from pathlib import Path
from dataclasses import asdict
import json,hashlib
from experiments.direct_velocity_full_v1.task_factory import load_config,source_hashes,gate_a_check,digest
from .codec import RangeCodec
EXP=Path(__file__).resolve().parent
ROOT=EXP.parents[1]
SCHEMA='rv1_range_image_td3_v1'

def sources():
    own={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(EXP.rglob('*.py')) if 'outputs' not in p.parts}
    return {**source_hashes(),**own}

def contract(sensor,catalog_fingerprint,latent_dim=64):
    c=load_config();gate_a_check(c)
    frozen={k:c[k] for k in ('control_spec','reward_kwargs','reward_scale','curriculum','td3')}
    return dict(schema=SCHEMA,state_dim=18,action_dim=3,latent_dim=latent_dim,sensor=asdict(sensor),
        sensor_frame='yaw_horizontal',scan_frequency_hz=15,control_hz=120,physics_hz=240,
        codec=RangeCodec.schema,quantization='range_max / 65534 meters; 65535 unscanned',
        policy_channels=['range/max_range with unscanned=0','first_hit_mask','scanned_mask'],
        catalog_fingerprint=catalog_fingerprint,frozen_task_sha256=digest(frozen),sources_sha256=digest(sources()))

def validate_contract(actual,expected):
    if actual!=expected:raise ValueError('rv1 schema/config/source mismatch; dv0 114D/old Replay are rejected')
