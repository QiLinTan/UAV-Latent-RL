"""Isolated Full z interventions on the P0 actor; never write training replay."""
import json
from pathlib import Path

import numpy as np
import torch

from envs.forest.ego_curriculum_env import EgoCurriculumFullForestEnv
from envs.preprocess import preprocess_state
from scripts.audit_stage0_feasibility import make_stage0_env
from scripts.change_td3 import _make_argparser
from scripts.run_change_td3 import build_mode_cmd


def main():
    snapshot = torch.load('checkpoints/full_scratch_ego20x10_curr_v1_seed43/training_state_step6000.pt', map_location='cpu', weights_only=False)
    actor, critic = snapshot['agent']['actor'], snapshot['agent']['critic']
    args = _make_argparser().parse_args(build_mode_cmd('full_scratch_ego20x10_curr_v1')[3:])
    args.map_id = 'ego_forest_20x10_seed43_v1'
    def factory(**kwargs):
        for key in tuple(kwargs):
            if key.startswith('kinematic_'):
                kwargs.pop(key)
        return EgoCurriculumFullForestEnv(**kwargs)
    env = make_stage0_env(args, env_class=factory)
    rows = []
    try:
        for seed in (7, 43, 80):
            for vertical in (-1., 0., 1.):
                observation, info = env.reset(seed=seed)
                state = preprocess_state(observation)
                with torch.no_grad():
                    policy = actor(torch.as_tensor(state).reshape(1, -1))[0].numpy()
                action = policy.copy()
                action[2] = vertical
                with torch.no_grad():
                    q1, q2 = critic(torch.as_tensor(state).reshape(1, -1), torch.as_tensor(action).reshape(1, 3))
                start = env.pos[0].copy()
                minimum_distance = float(info['goal_distance'])
                discounted_return = 0.
                terminated = truncated = False
                for tick in range(150):
                    observation, reward, terminated, truncated, info = env.step(action.reshape(1, 3))
                    discounted_return += args.reward_scale * reward * args.gamma**tick
                    minimum_distance = min(minimum_distance, info['goal_distance'])
                    if terminated or truncated:
                        break
                rows.append(dict(seed=seed, map_id=info['map_id'], initial_position=start.tolist(),
                                 initial_policy_action=policy.tolist(), intervention_action=action.tolist(),
                                 initial_q1=float(q1), initial_q2=float(q2),
                                 actual_upper_steps=tick+1, duration_s=env.runtime_lower_step / 120.,
                                 discounted_scaled_return=discounted_return,
                                 final_distance=info['goal_distance'], minimum_distance=minimum_distance,
                                 final_height=float(env.pos[0, 2]), success=bool(info['success']),
                                 done_reason=info['done_reason'] if terminated or truncated else 'probe_horizon',
                                 horizon_note='up to 10s, constant initial actor XY; partial returns are not complete TD3 Q values'))
    finally:
        env.close()
    report = dict(production_replay_modified=False, production_learner_modified=False, horizon_upper_steps=150,
                  interventions=rows,
                  interpretation='Compare only the same seed/reset and XY command. These bounded physical interventions test the z preference, not a complete root-cause identification or a replacement training policy.')
    Path('analysis_outputs/full_scratch_ego20x10_curr_v1_p0/vertical_interventions.json').write_text(json.dumps(report, indent=2)+'\n')
    for row in rows:
        print({k:row[k] for k in ('seed','intervention_action','initial_q1','discounted_scaled_return','minimum_distance','final_height','done_reason')})


if __name__ == '__main__':
    main()
