"""Verify archived paired encodings and exact GPU resume, as requested for this run."""
import argparse
from collections import deque
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
from directional_features import ObservationEncoder
from ms_td3 import ObservationEncoder as LegacyEncoder
from experiments.rv1_range_cnn.lidar import LidarConfig

ROOT=Path(__file__).resolve().parents[2]


def scan_checks(root):
    directory=ROOT/"MSGP_RESEARCH_REPORT/e07_e10/data/paired_scans"
    baseline=ObservationEncoder("original")
    directional=ObservationEncoder("directional")
    legacy=LegacyEncoder(LidarConfig())
    contracts=[]
    for file in ("e09_far_pairs.npz","e09_rotated_far_pairs.npz"):
        with np.load(directory/file) as archive:
            ranges=archive["ranges" if "ranges" in archive else "range_m"]
            hits=archive["hit"]
            scanned=archive["scanned"] if "scanned" in archive else np.ones_like(hits)
            for i,r in enumerate(ranges):
                obs=dict(state=np.arange(18,dtype=np.float32)/20,range_m=r,hit_mask=hits[i],scanned_mask=scanned[i])
                a=baseline(obs)
                b=directional(obs)
                np.testing.assert_array_equal(a,legacy(obs))
                np.testing.assert_array_equal(a[:268],b[:268])
                np.testing.assert_array_equal(a[268:],archive["F0"][i])
                np.testing.assert_array_equal(b[268:400],archive["F1"][i])
                np.testing.assert_array_equal(b[400:],archive["F2"][i,-48:])
            contracts.append(dict(file=file,frames=len(ranges)))
    (root/"encoding_verification.json").write_text(json.dumps(dict(exact=contracts),indent=2))


def assert_exact(a,b):
    if torch.is_tensor(a):
        assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):
        np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:
            assert_exact(a[k],b[k])
    elif isinstance(a,(tuple,list,deque)):
        assert len(a)==len(b)
        for x,y in zip(a,b):
            assert_exact(x,y)
    else:
        assert a==b,(a,b)


def check_resume(root):
    cases=[torch.load(root/q/"checkpoint_000128.pt",map_location="cpu",weights_only=False)
           for q in ("continuous","split")]
    a,b=cases
    for key in ("trainer","agent","observation","action_rng","rng","cuda_rng"):
        assert_exact(a[key],b[key])
    for key in ("episode_actions","position","quat","velocity","omega","controller",
                "upper_tick","course","sensor_range","sensor_time_s"):
        assert_exact(a["environment"][key],b["environment"][key])
    a,b=a["replay"],b["replay"]
    for k in ("pending","exposure","sampled_main","sampled_success"):
        assert_exact(getattr(a,k),getattr(b,k))
    for x,y in zip([a.main,*a.success],[b.main,*b.success]):
        for k in ("size","pointer","total","states","next_states","actions","rewards","terminal","truncated","stage"):
            assert_exact(getattr(x,k),getattr(y,k))
        assert_exact(x.rng.bit_generator.state,y.rng.bit_generator.state)
    (root/"resume_verification.json").write_text(json.dumps(dict(
        exact=True,segments=[64,128],compared=["actor","critic","targets","optimizers","episode","state",
                                               "replay","course","training and CUDA RNG"]),indent=2))


if __name__=="__main__":
    torch.set_num_threads(1)
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    p.add_argument("--check-resume",action="store_true")
    a=p.parse_args()
    (check_resume if a.check_resume else scan_checks)(a.root)
