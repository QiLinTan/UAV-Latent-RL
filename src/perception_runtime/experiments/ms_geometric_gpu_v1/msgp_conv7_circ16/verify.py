"""Geometry, isolation, real-time and CUDA recovery acceptance for T1."""
import argparse
import csv
import gc
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from agent import MSGPConvEncoder
from evaluate import load_policy, make_validation_env, realtime_decisions
from perception import ObservationBuilder, unpack
from t0_eval import read_jsonl
from train import cuda_setup, sha
from verify_directional import assert_exact, check_resume

from envs.map_atlas.spec import MapSpec, StaticObstacleSpec
from experiments.direct_velocity_full_v1.task_factory import ROOT
from experiments.rv1_range_cnn.environment import RangeImageEnv
from experiments.rv1_range_cnn.lidar import Lidar


def raw_from_archive(path, state):
    with np.load(path) as frame:
        return dict(state=state.copy(), **{k: frame[k] for k in ("range_m", "hit_mask", "scanned_mask")})


def geometry(builder):
    state = np.zeros(18, dtype=np.float32)
    state[[1, 5]] = [1, -.5]
    reverse = state.copy()
    reverse[[1, 5]] *= -1
    data = ROOT / "MSGP_RESEARCH_REPORT/data"
    archive = data / "e04_e06_raw"
    with (data / "e06_mirror_pairs.csv").open(newline="") as stream:
        cases = [(r["case_a"], r["case_b"]) for r in csv.DictReader(stream)]
    pairs = [(p.stem, p.stem.replace("+0.12", "-0.12")) for p in sorted(archive.glob("door_*_+0.12_*.npz"))]
    reflection = (-torch.arange(16, device=builder.device)) % 16
    affected_near, affected_far = 0, 0
    for left, right in cases + pairs:
        first = builder(raw_from_archive(archive / f"{left}.npz", state))[None]
        second = builder(raw_from_archive(archive / f"{right}.npz", reverse))[None]
        near, far, unchanged = unpack(first, True)
        mirror_near, mirror_far, mirror_state = unpack(second, True)
        if not torch.equal(near.flip(-2), mirror_near):
            raise AssertionError(f"{left}/{right}: near mirror differs at {int((near.flip(-2) != mirror_near).sum())} values")
        try:
            torch.testing.assert_close(far[:, :, reflection], mirror_far, atol=1e-6, rtol=0)
        except AssertionError as ex:
            raise AssertionError(f"{left}/{right}: far mirror failed\n{ex}") from ex
        torch.testing.assert_close(unchanged[0], torch.as_tensor(state, device=builder.device), rtol=0, atol=0)
        torch.testing.assert_close(mirror_state[0], torch.as_tensor(reverse, device=builder.device), rtol=0, atol=0)
        torch.testing.assert_close(far[0, 3:, 4], torch.tensor([1., -.5], device=builder.device), rtol=0, atol=1e-7)
        torch.testing.assert_close(mirror_far[0, 3:, 4], torch.tensor([-1., .5], device=builder.device), rtol=0, atol=1e-7)
        affected_near += not torch.equal(near, mirror_near)
        affected_far += not torch.equal(far[:, :3], mirror_far[:, :3])
        assert not torch.any(near[:, 0].bool() & near[:, 1].bool())
    assert cases and pairs and affected_near and affected_far
    for enabled, width in ((False, 50), (True, 82)):
        encoder = MSGPConvEncoder(enabled).to(builder.device)
        encoded = encoder(near, far if enabled else None, unchanged)
        assert encoded.shape == (1, width) and torch.equal(encoded[:, -18:], unchanged)
        assert sum(p.numel() for p in encoder.parameters()) == (48800 if enabled else 44832)
    return dict(e06_mirror_pairs=len(cases), e06_aperture_pairs=len(pairs),
                spatial_near_pairs=affected_near, spatial_far_pairs=affected_far)


def isolation(builder):
    def synth(distance=None):
        shape = builder.sensor.shape
        ranges = np.full(shape, builder.sensor.max_range_m, dtype=np.float32)
        hit = np.zeros(shape, dtype=bool)
        if distance is not None:
            # Central forward ray meets an obstacle surface at the specified range.
            ranges[shape[0] // 2, 0] = distance
            hit[shape[0] // 2, 0] = True
        raw = dict(range_m=ranges, hit_mask=hit, scanned_mask=np.ones_like(hit),
                   state=np.zeros(18, dtype=np.float32))
        return unpack(builder(raw)[None], True)[:2]

    clean, clean_far = synth()
    close, close_far = synth(.9)
    edge, edge_far = synth(1.4)
    distant, distant_far = synth(3.)
    assert close[0, 0].sum().item() == edge[0, 0].sum().item() == 1
    assert torch.equal(clean_far[:, [0, 2]], close_far[:, [0, 2]])
    assert torch.equal(clean_far[:, [0, 2]], edge_far[:, [0, 2]])
    assert torch.equal(clean, distant)
    assert distant_far[0, 0, 0] > 0 and distant_far[0, 2, 0] < 1

    def scan_case(front):
        obstacles = () if front is None else (
            StaticObstacleSpec(kind="box", name="isolation_box", position=(front + .05, 0, 1.14),
                               half_extents=(.05, .12, .07)),
        )
        scene = MapSpec(map_id="isolation", source="T1", scenario="range isolation", split="probe",
                        obstacles=obstacles, start_pos=(0., 0., 1.), target_pos=(4.5, 0., 1.),
                        xy_half_extent=5., obstacle_sensor_range=8., episode_len_sec=10.)
        env = RangeImageEnv(map_spec=scene)
        try:
            obs, _ = env.reset(seed=43)
            return unpack(builder(obs)[None], True)[:2]
        finally:
            env.close()

    empty, empty_far = scan_case(None)
    near_box, near_box_far = scan_case(.9)
    far_box, far_box_far = scan_case(3.)
    assert not torch.equal(empty[0, 0], near_box[0, 0])
    assert not torch.equal(empty_far[:, [0, 2]], far_box_far[:, [0, 2]])
    assert torch.equal(empty_far[:, [0, 2]], near_box_far[:, [0, 2]])
    assert torch.equal(empty, far_box)
    return dict(near_distances_m=[.9, 1.4], far_distance_m=3., exact=True,
                physical_scene_and_ray_probe=True)


def latency(checkpoint):
    agent, builder, contract, _ = load_policy(checkpoint, "cuda")
    assert contract["use_far_branch"]
    env = make_validation_env(contract)
    env.obstacle_curriculum.stage = 3
    measured, control_loop = [], []
    last_action = None
    record = False
    original = env._observation

    def choose(raw_state):
        nonlocal last_action
        start = time.perf_counter()
        obs = original(raw_state)
        if not env._defer_scan:
            extracted = time.perf_counter()
            last_action = agent.action(builder(obs))
            if record:
                end = time.perf_counter()
                measured.append(((end - start) * 1000, (extracted - start) * 1000,
                                 (end - extracted) * 1000))
        return obs

    env._observation = choose
    try:
        seed = 0
        env.reset(seed=seed)
        with realtime_decisions():
            for i in range(1020):
                record = i >= 20
                tick = time.perf_counter()
                _, _, terminal, truncated, _ = env.step(last_action)
                if record:
                    control_loop.append((time.perf_counter() - tick) * 1000)
                record = False
                if terminal or truncated:
                    gc.collect()
                    seed = (seed + 1) % 20
                    env.reset(seed=seed)
    finally:
        env.close()
    assert len(measured) == 1000
    full, lidar, policy = np.asarray(measured).T
    result = dict(samples=len(measured), mean_ms=float(np.mean(full)),
                  p99_ms=float(np.percentile(full, 99)), max_ms=float(np.max(full)),
                  lidar_p99_ms=float(np.percentile(lidar, 99)), policy_p99_ms=float(np.percentile(policy, 99)),
                  including_simulation_p99_ms=float(np.percentile(control_loop, 99)),
                  threshold_ms=1000 / 15, stage=3)
    print(json.dumps(result), flush=True)
    assert result["p99_ms"] < result["threshold_ms"], result
    return result


def verify_sensor(checkpoint):
    _, _, contract, _ = load_policy(checkpoint, "cuda")
    env = make_validation_env(contract)
    reference = Lidar(env.lidar.config)
    tested = 0
    fields = ("range_m", "hit_mask", "scanned_mask", "hit_body", "hit_type",
              "raw_hit_world", "origin_world", "directions_world")
    try:
        for stage in range(4):
            env.obstacle_curriculum.stage = stage
            for seed in (0, 7, 19):
                env.reset(seed=seed)
                base = env.base_env
                args = dict(client_id=base.CLIENT, position=base.pos[0], quaternion_xyzw=base.quat[0],
                            timestamp_s=base.physics_ticks / 240, self_ids=tuple(map(int, base.DRONE_IDS)),
                            ground_ids=(int(base.PLANE_ID),), obstacle_ids=tuple(base._tree_ids))
                partial = np.random.default_rng(seed).random(env.lidar.config.shape) > .3
                for mask in (None, partial):
                    old, new = reference.scan(scanned_mask=mask, **args), env.lidar.scan(scanned_mask=mask, **args)
                    for key in fields:
                        np.testing.assert_array_equal(getattr(old, key), getattr(new, key))
                    tested += 1
    finally:
        env.close()
    return dict(exact_scans=tested)


def smoke(run):
    check_resume(run)
    paths = [run / f"smoke/checkpoint_{step:06d}.pt" for step in (6000, 6001)]
    events = read_jsonl(run / "smoke/events.jsonl")
    assert any(r["event"] == "resume" and r["step"] == 6000 and r["observation_match"] for r in events)
    first = None
    for step, path in zip((6000, 6001), paths):
        assert [r["sha256"] for r in events if r["event"] == "checkpoint" and r["step"] == step] == [sha(path)]
        state = torch.load(path, map_location="cpu", weights_only=False)
        assert state["trainer"]["step"] == state["replay"].main.total == step
        assert state["agent"]["updates"] == step - 4999
        assert state["agent"]["actor_updates"] == (step - 4999) // 2
        assert state["agent"]["actor_optimizer"]["state"] and state["agent"]["critic_optimizer"]["state"]
        assert state["environment"]["sensor_scan_count"] == state["environment"]["upper_tick"] + 1
        assert state["observation"].shape == (784,) and np.isfinite(state["observation"]).all()
        assert state["cuda_rng"] and state["contract"]["gpu"]
        if first is None:
            first = state
    probes = first["branch_probes"]
    ratios = {}
    for name, data in probes.items():
        assert all(math.isfinite(data[k]) and data[k] > 0
                   for k in ("near_feature_l2", "far_feature_l2", "near_gradient_l2", "far_gradient_l2"))
        ratio = data["near_gradient_l2"] / data["far_gradient_l2"]
        assert .1 <= ratio <= 10., data
        ratios[name] = max(ratio, 1 / ratio)
    assert set(ratios) == {"actor", "critic"}
    checkpoints = [torch.load(run / d / "checkpoint_000128.pt", map_location="cpu", weights_only=False)
                   for d in ("continuous", "split")]
    for key in ("contract", "environment", "branch_probes"):
        assert_exact(checkpoints[0][key], checkpoints[1][key])
    return dict(gpu=first["contract"]["gpu"], updates=first["agent"]["updates"],
                actor_updates=first["agent"]["actor_updates"], branch_probes=probes,
                gradient_larger_over_smaller=ratios,
                exact_128_vs_64_plus_64=True, resumed_through=6001)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--geometry-only", action="store_true")
    args = parser.parse_args()
    cuda_setup(43)
    builder = ObservationBuilder("cuda")
    results = dict(geometry=geometry(builder), isolation=isolation(builder))
    if not args.geometry_only:
        checkpoint = args.run / "smoke/checkpoint_006000.pt"
        results["scan_contract"] = verify_sensor(checkpoint)
        results["latency"] = latency(checkpoint)
        results["cuda_smoke"] = smoke(args.run)
        (args.run / "acceptance.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(json.dumps(results, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
