"""Fixed original-start forest evaluation for either MSGP checkpoint variant."""
import argparse
from contextlib import contextmanager
import gc
import json
from pathlib import Path

import torch

from agent import MSGPAgent
from perception import ObservationBuilder, observation_width
from sensor import use_parallel_lidar
from t0_eval import follow, initial_info
from train import SCHEMA, cuda_setup

from envs.map_atlas.forest_pool import ForestPool
from experiments.direct_velocity_full_v1.task_factory import ROOT, load_config
from experiments.rv1_range_cnn.environment import make_course_env
from experiments.rv1_range_cnn.lidar import LidarConfig
from experiments.rv1_range_cnn.train_formal import sha


@contextmanager
def realtime_decisions():
    """Defer cyclic collection across control ticks. References still release normally."""
    if not gc.isenabled():
        yield
        return
    gc.disable()
    try:
        yield
    finally:
        gc.enable()
        gc.collect()


def load_policy(checkpoint, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    contract = state["contract"]
    use_far = contract["use_far_branch"]
    if contract["schema"] != SCHEMA or contract["observation_width"] != observation_width(use_far):
        raise ValueError("incompatible Conv7 checkpoint")
    agent = MSGPAgent(contract["td3"], use_far, device)
    agent.actor.load_state_dict(state["agent"]["actor"])
    agent.actor.eval()
    builder = ObservationBuilder(device, LidarConfig(**contract["sensor"]), use_far)
    return agent, builder, contract, state["trainer"]["step"]


def make_validation_env(contract):
    pool = ForestPool(ROOT / load_config()["forest_pool_path"])
    if pool.fingerprint != contract["pool_fingerprint"]:
        raise RuntimeError("checkpoint validation pool mismatch")
    env = make_course_env(pool, training=False, lidar_config=LidarConfig(**contract["sensor"]))
    return use_parallel_lidar(env) if contract.get("lidar_extraction") == "parallel_pybullet_v1" else env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--stages", nargs="+", choices=range(4), type=int, default=list(range(4)))
    parser.add_argument("--cases", type=int, choices=range(1, 21), default=20)
    args = parser.parse_args()
    if args.device == "cuda":
        cuda_setup(43)
    else:
        torch.set_num_threads(1)
    agent, builder, contract, step = load_policy(args.checkpoint, args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    env = make_validation_env(contract)
    rows = []
    try:
        for stage in args.stages:
            env.obstacle_curriculum.stage = stage
            for case in range(args.cases):
                obs, info = env.reset(seed=case)
                if (info["density_stage"], info["start_curriculum_source"]) != (stage, "original"):
                    raise RuntimeError("fixed evaluation initialization mismatch")
                initial = initial_info(env, info)
                with realtime_decisions():
                    row, _ = follow(env, agent, builder, obs, info)
                rows.append(dict(stage=stage, case=case, init=initial, **row))
    finally:
        env.close()
    document = dict(step=step, checkpoint_sha256=sha(args.checkpoint), split="validation",
                    use_far_branch=contract["use_far_branch"], device=args.device, rows=rows)
    (args.output / "results.json").write_text(json.dumps(document, indent=2, allow_nan=False))
    print(json.dumps(dict(cases=len(rows), use_far_branch=contract["use_far_branch"], device=args.device)))


if __name__ == "__main__":
    main()
