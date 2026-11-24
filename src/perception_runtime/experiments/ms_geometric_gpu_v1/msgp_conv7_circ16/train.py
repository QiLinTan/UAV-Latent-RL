"""CUDA training and full-state checkpointing for MSGP-Conv7-Circ16."""
import argparse
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from envs.map_atlas.forest_pool import ForestPool
from experiments.direct_velocity_full_v1.rl.td3_adapter import project
from experiments.direct_velocity_full_v1.task_factory import ROOT, digest, load_config
from experiments.ms_geometric_directional.directional_agent import BalancedReplay
from experiments.ms_geometric_mixed.mixed_course import describe, make_mixed_env
from experiments.rv1_range_cnn.checkpoint import atomic_save, capture, restore_environment, rng_state, restore_rng
from experiments.rv1_range_cnn.environment import make_course_env
from experiments.rv1_range_cnn.train_formal import log, sha

from agent import MSGPAgent
from perception import ObservationBuilder, observation_width
from sensor import use_parallel_lidar

SCHEMA = "msgp_conv7_circ16_v1"


def cuda_setup(seed):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def specifications(config, env, pool, use_far, course, short):
    files = [Path(__file__).with_name(file) for file in ("train.py", "agent.py", "perception.py", "sensor.py")]
    frozen = {k: config[k] for k in ("control_spec", "reward_kwargs", "reward_scale", "curriculum", "td3")}
    return dict(schema=SCHEMA, seed=config["seed"], td3=config["td3"], use_far_branch=use_far,
                observation_width=observation_width(use_far), course=course,
                mixed_course=describe() if course == "e11" else None, short=short,
                sensor=asdict(env.lidar.config), pool_fingerprint=pool.fingerprint,
                lidar_extraction="parallel_pybullet_v1",
                frozen_task_sha256=digest(frozen), torch_version=str(torch.__version__),
                cuda_version=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                source_sha256={file.name: sha(file) for file in files})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--course", choices=("original", "e11"), required=True)
    parser.add_argument("--use-far-branch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--interval", type=int, default=20000)
    parser.add_argument("--short-check", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = load_config()
    maximum = 128 if args.short_check else config["total_steps"]
    if not (1 <= args.end <= maximum and args.interval > 0 and torch.cuda.is_available()):
        raise ValueError("training budget, interval, or CUDA unavailable")
    run = args.run_dir.resolve()
    if args.resume is None:
        run.mkdir(parents=True, exist_ok=False)
    elif args.resume.resolve().parent != run:
        raise ValueError("resume must use a checkpoint in this run")
    td = config["td3"]
    if args.short_check:
        config = copy.deepcopy(config)
        td = config["td3"]
        td.update(warmup=32, update_after=32, batch_size=32)
    seed = config["seed"]
    cuda_setup(seed)
    action_rng = np.random.default_rng(seed)
    pool = ForestPool(ROOT / config["forest_pool_path"])
    env = use_parallel_lidar(make_mixed_env(pool) if args.course == "e11" else make_course_env(pool))
    try:
        contract = specifications(config, env, pool, args.use_far_branch, args.course, args.short_check)
        builder = ObservationBuilder("cuda", env.lidar.config, args.use_far_branch)
        agent = MSGPAgent(td, args.use_far_branch, "cuda")
        if args.resume:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
            if checkpoint["contract"] != contract:
                raise ValueError("checkpoint contract mismatch")
            core = checkpoint["trainer"]
            replay = checkpoint["replay"]
            agent.restore(checkpoint["agent"])
            agent.probes = checkpoint["branch_probes"]
            state = builder(restore_environment(env, checkpoint["environment"]))
            if not np.array_equal(state.cpu().numpy(), checkpoint["observation"]):
                raise RuntimeError("restored observation mismatch")
            action_rng.bit_generator.state = checkpoint["action_rng"]
            restore_rng(checkpoint["rng"])
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        else:
            core = dict(step=0, episode_id=0, episode_return=0., episode_length=0, severe=0)
            replay = BalancedReplay(td, agent.dim, seed)
            state = builder(env.reset(seed=seed)[0])
            (run / "metadata.json").write_text(json.dumps(dict(contract=contract, config=config), indent=2))
        start = core["step"]
        if start >= args.end or replay.main.total != start:
            raise ValueError("invalid resume step or inconsistent replay")
        log(run / "events.jsonl", dict(event="resume" if start else "start", step=start,
                                     cuda_device=str(agent.device), observation_match=True))
        stage = env._density_episode_stage
        before = time.perf_counter()
        cpu_state = state.cpu().numpy()
        for step in range(start + 1, args.end + 1):
            if step <= td["warmup"]:
                action = action_rng.uniform(-1, 1, 3)
            else:
                sigma = max(td["exploration_end"], td["exploration_start"] -
                            (td["exploration_start"] - td["exploration_end"]) * step / td["noise_decay_steps"])
                action = agent.action(state) + action_rng.normal(0, sigma, 3)
            action = project(torch.as_tensor(action, dtype=torch.float32)).numpy()
            stamp = env.lidar.cache.timestamp_s
            raw, reward, terminal, truncated, info = env.step(action)
            if env.lidar.cache.timestamp_s <= stamp or env.lidar.scan_count != env.upper_tick + 1:
                raise RuntimeError("invalid LiDAR cadence")
            state = builder(raw)
            next_cpu = state.cpu().numpy()
            replay.add(cpu_state, np.asarray(info["projected_action"], np.float32), next_cpu,
                       reward * config["reward_scale"], terminal, truncated, stage)
            cpu_state = next_cpu
            core["episode_return"] += reward
            core["episode_length"] += 1
            if step >= td["update_after"]:
                metrics = agent.update(replay)
                limits = config["diagnostics"]
                severe = metrics["q_abs_max"] > limits["severe_q_abs"] or metrics["critic_loss"] > limits["severe_loss"]
                core["severe"] = core["severe"] + 1 if severe else 0
                log(run / "updates.jsonl", dict(step=step, **metrics))
                if core["severe"] >= limits["severe_consecutive_updates"]:
                    raise FloatingPointError("severe learning diagnostics")
            if terminal or truncated:
                reason = info["done_reason"]
                success = reason == "success"
                replay.finish(success, stage)
                env.record_episode_outcome(success, collision=info["collision"], done_reason=reason)
                log(run / "episodes.jsonl", dict(core, step=step, stage=stage, reason=reason))
                core.update(episode_id=core["episode_id"] + 1, episode_return=0., episode_length=0)
                raw, info = env.reset()
                stage = info["density_stage"]
                state = builder(raw)
                cpu_state = state.cpu().numpy()
            core["step"] = step
            if step % 500 == 0 or step == args.end:
                metrics = dict(step=step, updates=agent.updates, actor_updates=agent.actor_updates,
                               steps_per_second=(step - start) / (time.perf_counter() - before),
                               cuda_peak_bytes=torch.cuda.max_memory_allocated(), **replay.stats())
                log(run / "metrics.jsonl", metrics)
                print(json.dumps(metrics), flush=True)
            if step % args.interval == 0 or step == args.end:
                path = run / f"checkpoint_{step:06d}.pt"
                if path.exists():
                    raise FileExistsError(path)
                atomic_save(path, dict(contract=contract, trainer=core, agent=agent.state(), replay=replay,
                                       branch_probes=agent.probes, environment=capture(env),
                                       observation=cpu_state, action_rng=action_rng.bit_generator.state,
                                       rng=rng_state(), cuda_rng=torch.cuda.get_rng_state_all()))
                log(run / "events.jsonl", dict(event="checkpoint", step=step, filename=path.name, sha256=sha(path)))
    finally:
        env.close()


if __name__ == "__main__":
    main()
