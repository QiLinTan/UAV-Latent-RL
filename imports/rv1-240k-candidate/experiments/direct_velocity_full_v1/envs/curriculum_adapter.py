"""Frozen ego20x10 pool/rules; privileged routes used exclusively for reset."""
import copy
import numpy as np
import pybullet as p
from envs.forest.start_curriculum import StartDistanceCurriculum
from envs.forest.obstacle_curriculum import ObstacleCurriculum
from envs.map_atlas.forest_pool import ForestPool,COUNTS
from envs.forest.rewards import default_reward_terms
from .direct_velocity_full import DirectVelocityFullEnv

BANDS=((.35,1.2),(1.2,4.),(4.,8.),(8.,float('inf')))


class CurriculumEnv(DirectVelocityFullEnv):
    def __init__(self,*,pool,training_course=True,split='train',**kwargs):
        if split not in ('train','validation') or training_course and split!='train':raise ValueError('sealed split')
        self.pool=pool;self.training_course=training_course;self.pool_split=split
        self.start_curriculum=StartDistanceCurriculum(bands=BANDS)
        self.obstacle_curriculum=ObstacleCurriculum(route_counts=COUNTS)
        self._pool_rng=np.random.default_rng();self._start_rng=np.random.default_rng()
        self._pool_index=0;self._density_episode_stage=0;self._pool_draw_count=0;self._pool_visits={}
        self._start_level=0;self._start_source='original';self._remaining_path_length=None
        kwargs['plant_kwargs']['map_spec']=self.pool.spec(split,0,0)
        super().__init__(**kwargs)
        self.base_env.CURRICULUM_SIMPLE_MIX_PROBABILITY=0.
        if not training_course:self.start_curriculum.level=4

    def _start_candidates(self,band):
        path=self.pool.path_for(self.pool_split,self._pool_index,self._density_episode_stage)
        remaining=np.r_[np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1)[::-1])[::-1],0.]
        mask=(remaining>=band[0]) & (remaining<band[1])
        mask &= np.linalg.norm(path-self.base_env.TARGET_POS,axis=1)>max(.30,self.base_env.GOAL_TOLERANCE+.10)
        candidates=np.asarray([x for x in path[mask] if self.base_env._computeNearestTreeClearance(x)>=.18],dtype=np.float32)
        if not candidates.size:raise RuntimeError('no frozen pool band candidates')
        return candidates.reshape(-1,3)

    def _place_reset(self,position):
        b=self.base_env
        if b.step_counter!=0:raise RuntimeError('relocation only permitted during reset')
        body=int(b.DRONE_IDS[0]);q=b.quat[0].copy()
        p.resetBasePositionAndOrientation(body,np.asarray(position).tolist(),q.tolist(),physicsClientId=b.CLIENT)
        p.resetBaseVelocity(body,[0,0,0],[0,0,0],physicsClientId=b.CLIENT)
        p.performCollisionDetection(physicsClientId=b.CLIENT);b._updateAndStoreKinematicInformation()
        return b._checkCollision()

    def reset(self,*,seed=None,options=None):
        if seed is not None:
            self._start_rng=np.random.default_rng(seed)
            if self.training_course:self._pool_rng=np.random.default_rng(np.random.SeedSequence([int(seed),20261013]))
        self._density_episode_stage=self.obstacle_curriculum.stage
        self._pool_index=(int(self._pool_rng.integers(len(self.pool.entries(self.pool_split))))) if self.training_course else int(seed if seed is not None else self._pool_draw_count)%len(self.pool.entries(self.pool_split))
        self._pool_draw_count+=1
        self._pool_visits[self._pool_index]=self._pool_visits.get(self._pool_index,0)+1
        b=self.base_env;b.MAP_SPEC=self.pool.spec(self.pool_split,self._pool_index,self._density_episode_stage);b.map_validation=None
        super().reset(seed=seed,options=options)
        self._start_level=self.start_curriculum.level;self._start_source='original'
        self._episode_start_position=np.asarray(b.pos[0],dtype=np.float32).copy()
        if self._start_level<4 and self._start_rng.random()>=self.start_curriculum.original_probability:
            candidates=self._start_candidates(BANDS[self._start_level])
            chosen=candidates[int(self._start_rng.integers(len(candidates)))]
            if not self._place_reset(chosen):
                self._start_source='band';self._episode_start_position=np.asarray(b.pos[0],dtype=np.float32).copy()
            else:
                self._place_reset(self._episode_start_position);self._start_source='collision_fallback'
        # Rebuild every reward/control history from the accepted relocated state.
        b._prev_pos=np.asarray(b.pos[0],dtype=np.float32).copy()
        b._prev_goal_dist=float(np.linalg.norm(b.TARGET_POS-b._prev_pos))
        b._last_collision=b._checkCollision();b._last_clearance=b._computeNearestTreeClearance(b._prev_pos)
        b._last_reward_terms=default_reward_terms();b._last_done_reason='running'
        self.initialize_control()
        path=self.pool.path_for(self.pool_split,self._pool_index,self._density_episode_stage)
        remaining=np.r_[np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1)[::-1])[::-1],0.]
        self._remaining_path_length=float(remaining[int(np.argmin(np.linalg.norm(path-b.pos[0],axis=1)))])
        if len(b._tree_ids)!=COUNTS[self._density_episode_stage]:raise RuntimeError('wrong active geometry')
        return b._computeObs(),{**b._computeInfo(),**self.course_info()}

    def course_info(self):
        entry=self.pool.entries(self.pool_split)[self._pool_index]
        stage=entry['stages'][self._density_episode_stage]
        return dict(map_pool_index=self._pool_index,map_pool_fingerprint=self.pool.fingerprint,
            geometry_sha256=stage['geometry_sha256'],map_split=self.pool_split,
            active_tree_count=COUNTS[self._density_episode_stage],density_stage=self._density_episode_stage,
            start_curriculum_episode_level=self._start_level,start_curriculum_source=self._start_source,
            start_curriculum_remaining_path_length=self._remaining_path_length,
            **self.start_curriculum.metrics(),**self.obstacle_curriculum.metrics())

    def step(self,action):
        o,r,t,u,i=super().step(action)
        return o,r,t,u,{**i,**self.course_info()}

    def record_episode_outcome(self,success,collision=False,done_reason=None):
        if not self.training_course:return
        self.start_curriculum.record(source=self._start_source,success=success,collision=collision or done_reason=='collision')
        return self.obstacle_curriculum.record(episode_stage=self._density_episode_stage,start_level=self._start_level,
            start_complete=self._start_level>=4 and self._start_source=='original',primary_difficulty=True,
            success=success,collision=collision or done_reason=='collision')

    def course_state(self):
        return copy.deepcopy(dict(pool_fingerprint=self.pool.fingerprint,pool_split=self.pool_split,
            pool_rng=self._pool_rng.bit_generator.state,start_rng=self._start_rng.bit_generator.state,
            start_scheduler=vars(self.start_curriculum),obstacle_scheduler=vars(self.obstacle_curriculum),
            draws=self._pool_draw_count,visits=self._pool_visits))

    def restore_course(self,state):
        if state['pool_fingerprint']!=self.pool.fingerprint or state['pool_split']!=self.pool_split:raise ValueError('changed pool')
        self._pool_rng.bit_generator.state=state['pool_rng'];self._start_rng.bit_generator.state=state['start_rng']
        self.start_curriculum.__dict__.update(copy.deepcopy(state['start_scheduler']))
        self.obstacle_curriculum.__dict__.update(copy.deepcopy(state['obstacle_scheduler']))
        self._pool_draw_count=state['draws'];self._pool_visits=copy.deepcopy(state['visits'])
