"""Run encoding verification, exact CUDA resume test, then paired 6k and 40k training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from train_directional import sha

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
MODES=("original","directional")


def save_status(root, phase, **details):
    path=root/"status.json"
    tmp=root/"status.json.tmp"
    tmp.write_text(json.dumps(dict(phase=phase,time=time.time(),**details),indent=2))
    tmp.replace(path)


def command(mode,run,end,interval=20000,resume=None,short=False):
    args=[sys.executable,"-u",str(HERE/"train_directional.py"),"--mode",mode,
          "--run-dir",str(run),"--end",str(end),"--interval",str(interval)]
    if resume is not None:
        args+=["--resume",str(run/f"checkpoint_{resume:06d}.pt")]
    if short:
        args+=["--short-check"]
    return args


def execute(root,name,args):
    with (root/f"{name}.log").open("wb") as log:
        subprocess.run(args,check=True,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)


def parallel(root,name,jobs):
    workers={}
    for mode,args in jobs.items():
        with (root/f"{name}_{mode}.log").open("wb") as log:
            workers[mode]=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
    result={mode:worker.wait() for mode,worker in workers.items()}
    if any(result.values()):
        raise RuntimeError(f"{name} failed: {result}")


def verify_job(root,mode,step):
    directory=root/mode
    metrics=[json.loads(s) for s in (directory/"metrics.jsonl").read_text().splitlines()]
    events=[json.loads(s) for s in (directory/"events.jsonl").read_text().splitlines()]
    last=metrics[-1]
    assert last["step"]==step and last["updates"]==step-4999
    assert last["actor_updates"]==last["updates"]//2 and sum(last["exposure"])==step
    assert sum(last["sampled_main"]+last["sampled_success"])==256*last["updates"]
    updates=[r for r in events if r["event"]=="device_check"]
    assert [r["actor_updated"] for r in updates]==[False,True]
    assert all(r["batch_size"]==256 and all(v.startswith("cuda") for v in r["devices"].values()) for r in updates)
    cp=directory/f"checkpoint_{step:06d}.pt"
    assert any(r["event"]=="checkpoint" and r["step"]==step and r["sha256"]==sha(cp) for r in events)
    return last


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    a=parser.parse_args()
    root=a.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    try:
        save_status(root,"encoding_check")
        execute(root,"encoding_check",[sys.executable,str(HERE/"verify_directional.py"),str(root)])
        save_status(root,"short_resume_check")
        execute(root,"continuous",command("directional",root/"continuous",128,short=True))
        execute(root,"split_first",command("directional",root/"split",64,short=True))
        execute(root,"split_resumed",command("directional",root/"split",128,resume=64,short=True))
        execute(root,"resume_equality",[sys.executable,str(HERE/"verify_directional.py"),str(root),"--check-resume"])
        save_status(root,"train_6k")
        parallel(root,"first",{m:command(m,root/m,6000) for m in MODES})
        verified={m:verify_job(root,m,6000) for m in MODES}
        (root/"gate_6000.json").write_text(json.dumps(verified,indent=2))
        save_status(root,"train_40k",checks=verified)
        parallel(root,"continued",{m:command(m,root/m,40000,resume=6000) for m in MODES})
        save_status(root,"training_complete",checks={m:verify_job(root,m,40000) for m in MODES})
    except BaseException as error:
        save_status(root,"failed",error=repr(error))
        raise


if __name__=="__main__":
    main()
