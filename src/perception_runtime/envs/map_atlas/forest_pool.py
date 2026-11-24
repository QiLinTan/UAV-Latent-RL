"""Frozen, disjoint geometry pools. QA routes are reset-only privileged data."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from .ego_forest import EgoForestConfig, generate_ego_forest, clearance_route
from .spec import MapSpec

COUNTS = (10, 20, 30, 40)
DEFAULT_MANIFEST = 'data/map_atlas/generated/ego20x10_curr_v1/manifest.json'
SPLITS = {'train': (128, 1000, 2000), 'validation': (20, 2000, 2500), 'test': (40, 3000, 3500)}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def geometry(record):
    return {k: record[k] for k in ('xy_half_extents', 'start_pos', 'target_pos', 'obstacles')}


def build_pool():
    result = dict(schema='ego20x10_curr_v1', config=EgoForestConfig().__dict__,
                  counts=list(COUNTS), sampling='first geometrically accepted candidates; nested generation-order prefixes',
                  qa=dict(resolution=.1, vehicle_radius=.06, margin=.2, cell_padding=.1 * np.sqrt(2) / 2),
                  candidate_ranges=SPLITS, maps={}, rejected=[])
    seen = set()
    for split, (count, begin, end) in SPLITS.items():
        accepted = []
        for seed in range(begin, end):
            try:
                record, meta, full_route = generate_ego_forest(seed=seed)
                record['split'] = split
                record['notes'] = 'Frozen multi-map pool; QA paths used only for reset placement.'
                spec = MapSpec.from_dict(record, root=Path('.'))
                stages = []
                for stage, active in enumerate(COUNTS):
                    active_record = {**record, 'obstacles': record['obstacles'][:active]}
                    route = full_route if active == 40 else clearance_route(replace(spec, obstacles=spec.obstacles[:active]))
                    if route is None:
                        raise RuntimeError(f'Stage {stage} has no QA route')
                    stages.append(dict(active_indices=list(range(active)), geometry_sha256=digest(geometry(active_record)),
                                       path=route.tolist(), path_length=float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum())))
                full_hash = digest(geometry(record))
                if full_hash in seen:
                    raise RuntimeError('Duplicate geometry')
                seen.add(full_hash)
                accepted.append(dict(seed=seed, record=record, generator=meta, geometry_sha256=full_hash, stages=stages))
            except RuntimeError as exc:
                result['rejected'].append(dict(split=split, seed=seed, reason=str(exc)))
                continue
            if len(accepted) == count:
                break
        if len(accepted) != count:
            raise RuntimeError(f'{split}: only {len(accepted)} of {count} geometrically accepted')
        result['maps'][split] = accepted
    result['fingerprint'] = digest(result)
    return result


class ForestPool:
    def __init__(self, path=DEFAULT_MANIFEST):
        self.path = Path(path)
        self.manifest = json.loads(self.path.read_text())
        payload = dict(self.manifest)
        self.fingerprint = payload.pop('fingerprint')
        if payload['schema'] != 'ego20x10_curr_v1' or digest(payload) != self.fingerprint:
            raise ValueError('Forest manifest fingerprint/schema mismatch')
        if tuple(payload['counts']) != COUNTS:
            raise ValueError('Forest density protocol changed')
        seen = set()
        self._specs = {}
        for split, entries in payload['maps'].items():
            for index, entry in enumerate(entries):
                full_hash = digest(geometry(entry['record']))
                if full_hash != entry['geometry_sha256'] or full_hash in seen:
                    raise ValueError('Duplicate or changed pool geometry')
                seen.add(full_hash)
                for stage, active in enumerate(COUNTS):
                    data = entry['stages'][stage]
                    record = {**entry['record'], 'obstacles': [entry['record']['obstacles'][i] for i in data['active_indices']]}
                    if len(record['obstacles']) != active or digest(geometry(record)) != data['geometry_sha256']:
                        raise ValueError('Active geometry mismatch')
                    self._specs[split, index, stage] = MapSpec.from_dict(record, root=self.path.parent)

    def entries(self, split):
        return self.manifest['maps'][split]

    def spec(self, split, index, stage):
        return self._specs[split, index, stage]

    def path_for(self, split, index, stage):
        return np.asarray(self.entries(split)[index]['stages'][stage]['path'], dtype=np.float32)
