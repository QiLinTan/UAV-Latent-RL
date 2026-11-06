"""rv2 raw uint16 rings, no stored learned latent/bypass, tensor ZIP archive."""
from pathlib import Path
import copy
import numpy as np
import torch
from experiments.rv1_range_cnn.replay import SuccessReplay as FrozenSuccessSampling
from experiments.rv1_range_cnn.codec import RangeCodec
from .contracts import SCHEMA, validate_contract
from .features import canonical_image

FORMAT = 'rv2_raw_uint16_tensor_replay_v1'
NAMES = ('states', 'next_states', 'ranges', 'next_ranges', 'masks', 'next_masks', 'actions', 'rewards', 'terminated', 'truncated', 'episode_ids', 'sensor_times')

class Replay:
    def __init__(self, capacity, contract, seed=43):
        if contract.get('schema') != SCHEMA or capacity < 1:
            raise ValueError('rv2 raw Replay required')
        self.capacity = int(capacity); self.contract = copy.deepcopy(contract)
        self.codec = RangeCodec((32, 128), 8.)
        self.states = np.zeros((capacity, 18), np.float32); self.next_states = np.zeros_like(self.states)
        self.ranges = np.zeros((capacity, 32, 128), np.uint16); self.next_ranges = np.zeros_like(self.ranges)
        self.masks = np.zeros((capacity, 2, 512), np.uint8); self.next_masks = np.zeros_like(self.masks)
        self.actions = np.zeros((capacity, 3), np.float32); self.rewards = np.zeros((capacity, 1), np.float32)
        self.terminated = np.zeros((capacity, 1), bool); self.truncated = np.zeros_like(self.terminated)
        self.episode_ids = np.zeros(capacity, np.int64); self.sensor_times = np.zeros((capacity, 2), np.float64)
        self.size = self.pointer = self.total = 0; self.rng = np.random.default_rng(seed)

    def encode_observation(self, obs):
        if set(obs) != {'state', 'range_m', 'hit_mask', 'scanned_mask'}:
            raise ValueError('raw observation keys')
        s = np.asarray(obs['state'], np.float32)
        if s.shape != (18,) or not np.isfinite(s).all():
            raise ValueError('State18 contract')
        canonical_image(obs)  # Same validation before any mutation.
        q, m = self.codec.encode(obs['range_m'], obs['hit_mask'], obs['scanned_mask'])
        return s, q, m

    def add(self, obs, action, next_obs, reward, terminated, truncated, *, episode_id=0, timestamps=(0., 1/15)):
        s, q, m = self.encode_observation(obs); ns, nq, nm = self.encode_observation(next_obs)
        a = np.asarray(action, np.float32); ts = np.asarray(timestamps, np.float64)
        if a.shape != (3,) or not np.isfinite(a).all() or np.max(abs(a)) > 1.000001 or np.linalg.norm(a[:2]) > 1.000001:
            raise ValueError('projected local action required')
        if not np.isfinite(reward) or ts.shape != (2,) or not np.isfinite(ts).all() or ts[0] < 0 or ts[1] <= ts[0]:
            raise ValueError('reward/timestamps')
        if not isinstance(episode_id, (int, np.integer)) or episode_id < 0 or episode_id > np.iinfo(np.int64).max:
            raise ValueError('nonnegative int64 episode ID')
        if type(terminated) not in (bool, np.bool_) or type(truncated) not in (bool, np.bool_):
            raise ValueError('explicit boolean terminal flags')
        values = (s, ns, q, nq, m, nm, a, reward, terminated, truncated, episode_id, ts)
        for name, value in zip(NAMES, values):
            getattr(self, name)[self.pointer] = value
        self.pointer = (self.pointer+1) % self.capacity; self.size = min(self.size+1, self.capacity); self.total += 1

    def _batch(self, indices, next_state=False):
        states = self.next_states if next_state else self.states
        ranges = self.next_ranges if next_state else self.ranges
        masks = self.next_masks if next_state else self.masks
        q, m = ranges[indices], masks[indices]
        shape = (len(indices), 32, 128)
        h = np.unpackbits(m[:, 0], axis=-1, bitorder='little', count=4096).reshape(shape)
        s = np.unpackbits(m[:, 1], axis=-1, bitorder='little', count=4096).reshape(shape)
        image = np.stack((np.where(s, q.astype(np.float32)/65534., 0), h, s), 1).astype(np.float32)
        return dict(state=torch.from_numpy(states[indices]), image=torch.from_numpy(image))

    def sample(self, batch_size):
        if self.size < 1:
            raise ValueError('empty Replay')
        i = self.rng.integers(self.size, size=batch_size)
        return (self._batch(i), torch.from_numpy(self.actions[i]), self._batch(i, True),
                torch.from_numpy(self.rewards[i]), torch.from_numpy((~(self.terminated[i] | self.truncated[i])).astype(np.float32)))

    def payload(self):
        return dict(capacity=self.capacity, size=self.size, pointer=self.pointer, total=self.total,
                    rng=copy.deepcopy(self.rng.bit_generator.state), arrays={n: torch.from_numpy(getattr(self, n)[:self.size]) for n in NAMES})

    def save(self, path):
        _save_new(dict(format=FORMAT, schema=SCHEMA, contract=self.contract, kind='ring', ring=self.payload()), path)

    @classmethod
    def load(cls, path, expected_contract):
        data = _load(path, expected_contract, 'ring')
        obj = cls(data['ring']['capacity'], expected_contract); _restore(obj, data['ring']); return obj

class SuccessReplay(FrozenSuccessSampling):
    """Inherit only frozen balancing/pending/sampling equations, use rv2 rings."""
    def __init__(self, contract, capacity=60000, success_capacity=20000, fraction=.2, seed=43):
        if success_capacity < 4 or success_capacity % 4 or not 0 <= fraction <= 1:
            raise ValueError('success strata/fraction')
        self.contract = copy.deepcopy(contract); self.fraction = fraction
        self.main = Replay(capacity, contract, seed)
        self.success = {d: Replay(success_capacity//4, contract, seed+d) for d in (10, 20, 30, 40)}
        self.pending = []; self.last_sample_counts = {}
    def save(self, path):
        _save_new(dict(format=FORMAT, schema=SCHEMA, contract=self.contract, kind='balanced', fraction=self.fraction,
                       main=self.main.payload(), success={d:r.payload() for d,r in self.success.items()}, pending=self.pending), path)
    @classmethod
    def load(cls, path, expected_contract):
        data = _load(path, expected_contract, 'balanced')
        if set(data['success']) != {10, 20, 30, 40}:
            raise ValueError('success strata schema')
        obj = cls(expected_contract, data['main']['capacity'], sum(r['capacity'] for r in data['success'].values()), data['fraction'])
        _restore(obj.main, data['main'])
        for d, ring in data['success'].items():
            _restore(obj.success[d], ring)
        obj.pending = data['pending']; return obj

def _save_new(data, path):
    # Synchronous archive: caller must not mutate rings concurrently.
    path = Path(path)
    with path.open('xb') as stream:
        torch.save(data, stream, pickle_protocol=4)

def _load(path, contract, kind):
    data = torch.load(path, map_location='cpu', weights_only=False)
    if data.get('format') != FORMAT or data.get('kind') != kind:
        raise ValueError('rv2 Replay envelope required')
    validate_contract(data['contract'], contract); return data

def _restore(ring, payload):
    size, pointer, total = payload['size'], payload['pointer'], payload['total']
    if not 0 <= size <= ring.capacity or not 0 <= pointer < ring.capacity or total < size or pointer != total % ring.capacity or size != min(total, ring.capacity):
        raise ValueError('corrupt ring lifecycle')
    if set(payload['arrays']) != set(NAMES):
        raise ValueError('array schema')
    for name, value in payload['arrays'].items():
        target = getattr(ring, name)
        if not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or value.shape != target[:size].shape or value.numpy().dtype != target.dtype:
            raise ValueError('array dtype/shape')
    for prefix in ('', 'next_'):
        q=payload['arrays'][prefix+'ranges'].numpy(); m=payload['arrays'][prefix+'masks'].numpy()
        h=np.unpackbits(m[:,0],axis=-1,bitorder='little',count=4096).reshape(size,32,128).astype(bool)
        s=np.unpackbits(m[:,1],axis=-1,bitorder='little',count=4096).reshape(size,32,128).astype(bool)
        if np.any(h & ~s) or not np.array_equal(q==65535,~s) or np.any(q[s & ~h]!=65534):
            raise ValueError('corrupt raw sensor masks/sentinel')
    for name, value in payload['arrays'].items():
        getattr(ring, name)[:size] = value.numpy()
    ring.size = size; ring.pointer = pointer; ring.total = total; ring.rng.bit_generator.state = payload['rng']
