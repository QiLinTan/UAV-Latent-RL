"""Shared obstacle-stage schedule for the 40k comparison; preserve the start course."""
import copy
from functools import lru_cache
from pathlib import Path
import hashlib
import numpy as np

from experiments.direct_velocity_full_v1.contracts import ControlSpec
from experiments.direct_velocity_full_v1.task_factory import load_config
from experiments.rv1_range_cnn.environment import RangeCurriculumEnv,plant_kwargs

QUOTAS=(10,5,3,2)
SALT=20261119
RULE="permuted block of 20 episode starts"
SOURCE=Path(__file__).resolve()


@lru_cache(maxsize=256)
def stage_block(seed,block):
    plan=np.repeat(np.arange(4),QUOTAS)
    rng=np.random.default_rng(np.random.SeedSequence([seed,SALT,block]))
    return tuple(int(x) for x in rng.permutation(plan))


def assigned_stage(seed,episode):
    block,index=divmod(episode,sum(QUOTAS))
    return stage_block(seed,block)[index]


class MixedRangeCurriculum(RangeCurriculumEnv):
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            if self._pool_draw_count:
                raise ValueError("cannot reseed an active mixed curriculum")
            self._mixed_seed=int(seed)
        if not hasattr(self,"_mixed_seed"):
            raise ValueError("the first reset must provide a seed")
        self.obstacle_curriculum.stage=assigned_stage(self._mixed_seed,self._pool_draw_count)
        obs,info=super().reset(seed=seed,options=options)
        if info["density_stage"]!=self.obstacle_curriculum.stage:
            raise RuntimeError("mixed schedule was not applied to map")
        return obs,info

    def record_episode_outcome(self, success, collision=False, done_reason=None):
        self.start_curriculum.record(source=self._start_source, success=success,
                                     collision=collision or done_reason=="collision")

    def course_state(self):
        state=super().course_state()
        state["mixed_obstacles"]=dict(seed=self._mixed_seed,quotas=QUOTAS,salt=SALT,rule=RULE)
        return state

    def restore_course(self,state):
        parameters=state["mixed_obstacles"]
        if parameters!={"seed":parameters["seed"],"quotas":QUOTAS,"salt":SALT,"rule":RULE}:
            raise ValueError("mixed curriculum contract changed")
        self._mixed_seed=int(parameters["seed"])
        super().restore_course(state)


def make_mixed_env(pool):
    return MixedRangeCurriculum(pool=pool,training_course=True,split="train",
        control_spec=ControlSpec(**load_config()["control_spec"]),
        plant_kwargs=plant_kwargs(pool.spec("train",0,0)))


def describe():
    return dict(quota=QUOTAS,nominal_episode_share=[x/sum(QUOTAS) for x in QUOTAS],
                rule=RULE,seed_salt=SALT,stage_gate="disabled; start course unchanged",
                source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest())
