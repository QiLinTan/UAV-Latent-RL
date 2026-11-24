"""Bounded CUDA train/resume for the A4 and direction-enhanced comparators."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import numpy as np
import torch

from envs.map_atlas.forest_pool import ForestPool
from experiments.direct_velocity_full_v1.task_factory import ROOT, load_config
from experiments.direct_velocity_full_v1.rl.td3_adapter import project
from experiments.rv1_range_cnn.environment import make_course_env
from experiments.rv1_range_cnn.checkpoint import capture,restore_environment,rng_state,restore_rng,atomic_save
from fast_multiscale import __file__ as fast_source
from multiscale import __file__ as voxel_source
from directional_features import DIMS,SCHEMA,ObservationEncoder
from directional_agent import Agent,BalancedReplay

HERE=Path(__file__).resolve().parent


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def log(path,data):
    with path.open("a") as stream:
        stream.write(json.dumps(data,allow_nan=False)+"\n")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--mode",choices=DIMS,required=True)
    parser.add_argument("--end",type=int,required=True)
    parser.add_argument("--interval",type=int,default=20000)
    parser.add_argument("--resume",type=Path)
    parser.add_argument("--short-check",action="store_true")
    args=parser.parse_args()
    cap=128 if args.short_check else 40000
    if not 1<=args.end<=cap or args.interval<1 or not torch.cuda.is_available():
        raise ValueError("valid GPU diagnostic budget and CUDA required")
    run=args.run_dir.resolve()
    if args.resume is None:
        run.mkdir(exist_ok=False,parents=True)
    elif args.resume.resolve().parent!=run or not run.is_dir():
        raise ValueError("resume must use a checkpoint in this run")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    config=load_config()
    td=copy.deepcopy(config["td3"])
    seed=config["seed"]
    if args.short_check:
        td.update(warmup=32,update_after=32,batch_size=32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    generator=np.random.default_rng(seed)
    pool=ForestPool(ROOT/config["forest_pool_path"])
    env=make_course_env(pool,training=True)
    encoder=ObservationEncoder(args.mode,env.lidar.config)
    agent=Agent(td,DIMS[args.mode],"cuda")
    sources=[HERE/x for x in ("train_directional.py","directional_features.py","directional_agent.py")]
    sources += [Path(fast_source),Path(voxel_source)]
    contract=dict(schema=SCHEMA[args.mode],mode=args.mode,dimension=DIMS[args.mode],device="cuda",
                  seed=seed,td3=td,short_check=args.short_check,torch=torch.__version__,torch_cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0),pool_fingerprint=pool.fingerprint,sensor=vars(env.lidar.config),
                  source_sha256={p.name:sha(p) for p in sources},
                  task_config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True,default=str).encode()).hexdigest())
    ep_length=0
    try:
        if args.resume:
            if sorted(run.glob("checkpoint_*.pt"))[-1] != args.resume.resolve():
                raise ValueError("only the latest checkpoint can resume")
            saved=torch.load(args.resume,map_location="cpu",weights_only=False)
            if saved["contract"]!=contract:
                raise ValueError("checkpoint contract mismatch")
            agent.restore(saved["agent"])
            replay=saved["replay"]
            raw=restore_environment(env,saved["environment"])
            state=encoder(raw)
            np.testing.assert_array_equal(state,saved["observation"])
            restore_rng(saved["rng"])
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
            generator.bit_generator.state=saved["action_rng"]
            start,episode_id,episode_return,severe,ep_length=[
                saved["trainer"][k] for k in ("step","episode_id","episode_return","severe","episode_length")]
            del saved
        else:
            replay=BalancedReplay(td,encoder.dim,seed)
            raw,_=env.reset(seed=seed)
            state=encoder(raw)
            start=episode_id=severe=0
            episode_return=0.
            (run/"metadata.json").write_text(json.dumps(dict(contract=contract,config=config),indent=2,default=str))
        if args.end<=start:
            raise ValueError("target does not extend checkpoint")
        stage=int(env._density_episode_stage)
        begin=time.perf_counter()
        log(run/"events.jsonl",dict(event="resume" if start else "start",step=start,target=args.end,
                                     observation_match=True,stage=stage,time=time.time()))
        for step in range(start+1,args.end+1):
            if step<=td["warmup"]:
                action=generator.uniform(-1,1,3)
            else:
                sigma=max(td["exploration_end"],td["exploration_start"]-
                          (td["exploration_start"]-td["exploration_end"])*step/td["noise_decay_steps"])
                action=agent.action(state)+generator.normal(0,sigma,3)
            action=project(torch.as_tensor(action,dtype=torch.float32)).numpy()
            previous_time=env.lidar.cache.timestamp_s
            raw,reward,terminal,truncated,info=env.step(action)
            if env.lidar.cache.timestamp_s<=previous_time or env.lidar.scan_count!=env.upper_tick+1:
                raise RuntimeError("invalid scan cadence")
            nxt=encoder(raw)
            executed=np.asarray(info["projected_action"],dtype=np.float32)
            replay.add(state,executed,nxt,reward*config["reward_scale"],terminal,truncated,stage)
            state=nxt
            episode_return+=reward
            ep_length+=1
            if step>=td["update_after"]:
                diagnostics=agent.update(replay)
                bad=diagnostics["q_abs_max"]>config["diagnostics"]["severe_q_abs"] or (
                    diagnostics["critic_loss"]>config["diagnostics"]["severe_loss"])
                severe=severe+1 if bad else 0
                log(run/"updates.jsonl",dict(step=step,**diagnostics))
                if agent.updates in (1,2):
                    names=("actor","critic","actor_target","critic_target")
                    devices={name:str(next(getattr(agent,name).parameters()).device) for name in names}
                    if not all(d.startswith("cuda") for d in devices.values()):
                        raise RuntimeError("update off CUDA")
                    log(run/"events.jsonl",dict(event="device_check",step=step,devices=devices,
                                                 actor_updated=diagnostics["actor_updated"],batch_size=td["batch_size"]))
                if severe>=config["diagnostics"]["severe_consecutive_updates"]:
                    raise FloatingPointError("severe learning condition")
            if terminal or truncated:
                reason=info["done_reason"]
                replay.finish(reason=="success",stage)
                env.record_episode_outcome(reason=="success",collision=info["collision"],done_reason=reason)
                log(run/"episodes.jsonl",dict(step=step,episode=episode_id,reward=episode_return,
                                              reason=reason,length=ep_length,stage=stage,collision=bool(info["collision"]),
                                              start_level=info["start_curriculum_episode_level"],
                                              start_source=info["start_curriculum_source"]))
                episode_id+=1
                ep_length=0
                episode_return=0.
                raw,info=env.reset()
                stage=int(info["density_stage"])
                state=encoder(raw)
            if step%500==0 or step==args.end:
                metric=dict(step=step,episodes=episode_id,updates=agent.updates,actor_updates=agent.actor_updates,
                            steps_per_second=(step-start)/(time.perf_counter()-begin),
                            cuda_peak_mib=torch.cuda.max_memory_allocated()/2**20,**replay.stats())
                log(run/"metrics.jsonl",metric)
                print(json.dumps(metric),flush=True)
            if step%args.interval==0 or step==args.end:
                target=run/f"checkpoint_{step:06d}.pt"
                if target.exists():
                    raise FileExistsError(target)
                free=shutil.disk_usage("/mnt/d").free
                if free<20*1024**3:
                    raise OSError("D: has under 20 GiB free")
                core=dict(step=step,episode_id=episode_id,episode_return=episode_return,severe=severe,
                          episode_length=ep_length)
                payload=dict(schema=contract["schema"],contract=contract,trainer=core,agent=agent.state(),
                             replay=replay,environment=capture(env),observation=state,rng=rng_state(),
                             cuda_rng=torch.cuda.get_rng_state_all(),action_rng=generator.bit_generator.state)
                atomic_save(target,payload)
                log(run/"events.jsonl",dict(event="checkpoint",step=step,file=target.name,sha256=sha(target)))
        log(run/"events.jsonl",dict(event="stopped",step=step,time=time.time()))
    except BaseException:
        log(run/"events.jsonl",dict(event="failure",step=locals().get("step",0),traceback=traceback.format_exc()))
        raise
    finally:
        env.close()


if __name__=="__main__":
    main()
