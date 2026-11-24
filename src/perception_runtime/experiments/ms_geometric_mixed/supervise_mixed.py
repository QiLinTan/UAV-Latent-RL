"""Start a new bounded 40k mixed comparison, with gated restart and frozen evaluation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import supervise_directional as base
from mixed_course import QUOTAS, assigned_stage,describe

HERE=Path(__file__).resolve().parent
SHARED=HERE.parent/"ms_geometric_directional"


def train(mode,run,end,resume=None,short=False):
    args=base.command(mode,run,end,resume=resume,short=short)
    args[2]=str(HERE/"train_mixed.py")
    return args


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def verify_schedule(root, step):
    starts={name:rows(root/name/"episode_starts.jsonl") for name in base.MODES}
    metas={name:json.loads((root/name/"metadata.json").read_text())["contract"] for name in base.MODES}
    assert metas["original"]["seed"]==metas["directional"]["seed"]
    assert metas["original"]["obstacle_course"]==metas["directional"]["obstacle_course"]
    seed=metas["original"]["seed"]
    keys=("episode","stage","map_pool_index","geometry_sha256")
    n=min(len(x) for x in starts.values())
    for a,b in zip(starts["original"],starts["directional"]):
        assert all(a[k]==b[k] for k in keys)
    details={}
    for name, mm in starts.items():
        for i, row in enumerate(mm):
            assert row["episode"]==i and row["stage"]==assigned_stage(seed,i)
        for start in range(0,len(mm)-19,20):
            assert tuple(Counter(x["stage"] for x in mm[start:start+20])[i] for i in range(4))==QUOTAS
        done=rows(root/name/"episodes.jsonl")
        assert all(done[i]["episode"]==i and done[i]["stage"]==mm[i]["stage"] for i in range(len(done)))
        last=base.verify_job(root,name,step)
        assert all(v>0 for v in last["exposure"])
        assert all(v>0 for v in last["sampled_main"])
        starts_before_cap=[r for r in mm if r["step"]<step]
        details[name]=dict(final_metrics=last,episode_starts=[sum(r["stage"]==i for r in starts_before_cap)
                                                            for i in range(4)],
                           completed_episodes_per_stage=[sum(r["stage"]==i for r in done) for i in range(4)])
    return dict(common_episode_start_prefix=n,details=details)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("output",type=Path)
    root=p.parse_args().output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    (root/"mixed_protocol.json").write_text(json.dumps(describe(),indent=2))
    try:
        base.save_status(root,"short_resume_check")
        base.execute(root,"continuous",train("directional",root/"continuous",128,short=True))
        base.execute(root,"split_first",train("directional",root/"split",64,short=True))
        base.execute(root,"split_resumed",train("directional",root/"split",128,resume=64,short=True))
        base.execute(root,"resume_equality",[sys.executable,str(SHARED/"verify_directional.py"),str(root),"--check-resume"])
        base.save_status(root,"train_6k")
        base.parallel(root,"first",{m:train(m,root/m,6000) for m in base.MODES})
        gate=verify_schedule(root,6000)
        (root/"gate_6000.json").write_text(json.dumps(gate,indent=2))
        base.save_status(root,"train_40k",checks=gate)
        base.parallel(root,"continued",{m:train(m,root/m,40000,resume=6000) for m in base.MODES})
        result=verify_schedule(root,40000)
        base.save_status(root,"training_complete",checks=result)
        base.execute(root,"evaluation",[sys.executable,"-u",str(SHARED/"evaluate_directional.py"),
                      "--run",str(root),"--step","40000","--output",str(root/"evaluation_40000")])
        base.save_status(root,"evaluation_complete",checks=result)
    except BaseException as error:
        base.save_status(root,"failed",error=repr(error))
        raise


if __name__=="__main__":
    main()
