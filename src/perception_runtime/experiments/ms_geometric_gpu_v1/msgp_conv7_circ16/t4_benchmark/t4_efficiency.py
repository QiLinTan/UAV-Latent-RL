"""Measure Replay allocations and paired Stage3 sensing/inference latency."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from evaluate import load_policy, make_validation_env, realtime_decisions
from experiments.ms_geometric_directional.directional_agent import BalancedReplay
from experiments.rv1_range_cnn.lidar import Lidar, LidarConfig
from experiments.rv1_range_cnn.replay import SuccessReplay
from experiments.rv1_range_cnn.td3 import RangeTD3
from t0_eval import identity
from t2_eval import read_json, write_json
from t4_evaluate import run_path
from train import cuda_setup, load_config, sha


def private_resident_bytes():
    rows = Path("/proc/self/smaps_rollup").read_text().splitlines()
    return sum(int(row.split()[1]) * 1024 for row in rows
               if row.startswith(("Private_Clean:", "Private_Dirty:")))


def replay_memory(model, checkpoint):
    cfg = load_config()["td3"]
    torch.cuda.init()
    cuda_before = torch.cuda.memory_allocated()
    if model == "t3":
        per = 2 * 784 * 4 + 3 * 4 + 4 + 1 + 1 + 1
        baseline = private_resident_bytes()
        replay = BalancedReplay(cfg, 784, 43)
        rings = [replay.main, *replay.success]
    else:
        per = 2 * (18 * 4 + 32 * 128 * 2 + 2 * 32 * 128 // 8) + 3 * 4 + 4 + 1 + 1 + 8 + 16
        contract = torch.load(checkpoint, map_location="cpu", weights_only=False)["contract"]
        baseline = private_resident_bytes()
        replay = SuccessReplay(contract, cfg["buffer_size"], cfg["success_replay_size"],
                               cfg["success_replay_fraction"])
        rings = [replay.main, *replay.success.values()]
    arrays = [value for r in rings for value in vars(r).values() if isinstance(value, np.ndarray)]
    actual = sum(x.nbytes for x in arrays)
    expected = per * (cfg["buffer_size"] + cfg["success_replay_size"])
    if actual != expected:
        raise RuntimeError("Replay backing-array calculation disagrees with allocation")
    for array in arrays:
        view = array.view(np.uint8).reshape(-1)
        view[::4096] = 1
        view[-1] = 1
    resident = private_resident_bytes() - baseline
    if resident < .97 * expected:
        raise RuntimeError("Replay resident-memory benchmark did not touch all capacity")
    return dict(model=model, transition_bytes=per, theory_bytes=expected, backing_array_bytes=actual,
                measured_private_rss_increase_bytes=resident, configured_capacities=[60000, 20000],
                measured_cuda_tensor_delta_bytes=torch.cuda.memory_allocated() - cuda_before,
                scope="Fully page-touched Replay storage in an isolated process. CUDA tensor allocation delta excludes the CUDA context, transient sampling batches and the model.")


def policies(runs, device):
    t3 = run_path(runs, 43, "t3") / "checkpoint_240000.pt"
    a1 = run_path(runs, 43, "a1") / "checkpoint_240000.pt"
    dual, builder, contract, _ = load_policy(t3, device)
    pack = torch.load(a1, map_location="cpu", weights_only=False)
    base = RangeTD3(load_config()["td3"], pack["contract"], device=device)
    base.actor.load_state_dict(pack["agent"]["actor"])
    base.actor.eval()
    return {"t3": (dual, builder), "a1": (base, identity)}, contract


def make_tape(agent, builder, contract):
    env = make_validation_env(contract)
    env.obstacle_curriculum.stage = 3
    tape = []
    try:
        for start_seed in (0, 5, 10, 15):
            remaining, seed = 250, start_seed
            while remaining:
                obs, _ = env.reset(seed=seed)
                segment = []
                with realtime_decisions():
                    while remaining:
                        action = agent.action(builder(obs))
                        obs, _, done, truncated, _ = env.step(action)
                        remaining -= 1
                        segment.append(action.tolist())
                        if done or truncated:
                            break
                tape.append(dict(seed=seed, actions=segment))
                seed = (seed + 1) % 20
    finally:
        env.close()
    return tape


def latency(agent, builder, contract, tape, native):
    env = make_validation_env(contract)
    if native:
        env.lidar = Lidar(env.lidar.config)
    env.obstacle_curriculum.stage = 3
    captured, scans, positions = [], [], []
    record = False
    original = env._observation

    def timed(state):
        start = time.perf_counter()
        obs = original(state)
        if record and not env._defer_scan:
            split = time.perf_counter()
            agent.action(builder(obs))
            captured.append((time.perf_counter() - start) * 1000)
            scans.append((split - start) * 1000)
        return obs

    env._observation = timed
    try:
        obs, _ = env.reset(seed=tape[0]["seed"])
        for _ in range(32):
            agent.action(builder(obs))
        torch.cuda.synchronize()
        for entry in tape:
            record = False
            env.reset(seed=entry["seed"])
            with realtime_decisions():
                record = True
                for i, raw in enumerate(entry["actions"]):
                    _, _, done, truncated, _ = env.step(np.asarray(raw, dtype=np.float32))
                    positions.append(env.base_env.pos[0].copy())
                    if (done or truncated) and i != len(entry["actions"]) - 1:
                        raise RuntimeError("latency fixture lost simulation synchronization")
    finally:
        env.close()
    if len(captured) != 1000 or len(positions) != 1000:
        raise RuntimeError("expected exactly 1,000 Stage3 sensing/inference ticks")
    return dict(samples=1000, mean_ms=float(np.mean(captured)), p99_ms=float(np.percentile(captured, 99)),
                sensing_mean_ms=float(np.mean(scans)), sensing_p99_ms=float(np.percentile(scans, 99)),
                per_tick_ms=captured, lidar="native_serial" if native else "parallel_pybullet_v1"), np.asarray(positions)


def parameter_totals(agent):
    values = {}
    for name in ("actor", "critic"):
        net = getattr(agent, name)
        values[name] = sum(p.numel() for p in net.parameters())
        values[name + "_encoder"] = sum(p.numel() for p in net.encoder.parameters())
    values["online_actor_and_critic"] = values["actor"] + values["critic"]
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--memory", choices=("t3", "a1"))
    args = parser.parse_args()
    if args.memory:
        result = replay_memory(args.memory, run_path(args.runs, 43, args.memory) / "checkpoint_240000.pt")
        write_json(args.output, result)
        return
    cuda_setup(43)
    frozen = read_json(args.output / "frozen_seed43.json")
    for model, record in frozen["models"].items():
        path = run_path(args.runs, 43, model) / "checkpoint_240000.pt"
        if sha(path) != record["checkpoints"]["240000"]["sha256"]:
            raise RuntimeError(f"historical model {model} checksum mismatch")
    agents, contract = policies(args.runs, "cuda")
    tape = make_tape(*agents["t3"], contract)
    write_json(args.output / "efficiency_action_tape.json", tape)
    reference = None
    timings = {}
    for name, model, native in (("t3", "t3", False), ("a1_shared_scanner", "a1", False),
                                 ("a1_native_pipeline", "a1", True)):
        timings[name], pos = latency(*agents[model], contract, tape, native)
        if reference is not None and not np.array_equal(pos, reference):
            raise RuntimeError("paired timing runs did not visit the same states")
        reference = pos
        print(f"{name}: {timings[name]['p99_ms']:.3f} ms P99", flush=True)
    params = {name: parameter_totals(pair[0]) for name, pair in agents.items()}
    del agents
    gc.collect()
    torch.cuda.empty_cache()
    memory = {}
    for model in ("t3", "a1"):
        detail = args.output / f"efficiency_memory_{model}.json"
        subprocess.run([sys.executable, str(Path(__file__)), str(args.runs), str(detail),
                        "--memory", model], check=True)
        memory[model] = read_json(detail)
    write_json(args.output / "efficiency.json", dict(
        device=torch.cuda.get_device_name(), samples=1000, stage=3,
        parameters=params, replay=memory, latency=timings,
        scope="Wall clock fresh LiDAR plus observation preprocessing and synchronized Actor inference on the same 1,000 paired physical states; simulation and resets excluded."))


if __name__ == "__main__":
    main()
