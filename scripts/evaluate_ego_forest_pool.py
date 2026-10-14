"""Explicit frozen-policy pool evaluation, including sealed test on request."""
import argparse
from types import SimpleNamespace

from algos.td3.td3_upper_recurrent import TD3UpperRecurrent
from envs.forest.ego_curriculum_env import EgoCurriculumFullForestEnv
from scripts.audit_stage0_feasibility import make_stage0_env
from scripts.change_td3 import _make_argparser
from scripts.run_change_td3 import build_mode_cmd
from trainers.callbacks.ego_pool_eval import EgoPoolEvalCallback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--source-step', required=True, type=int)
    parser.add_argument('--split', choices=('validation', 'test'), required=True)
    parser.add_argument('--stages', nargs='+', type=int, choices=range(4), default=[0, 3])
    parser.add_argument('--output', required=True)
    user = parser.parse_args()
    args = _make_argparser().parse_args(build_mode_cmd('full_scratch_ego20x10_curr_v1')[3:])
    # Existing factory supplies the frozen reward and control parameters;
    # this placeholder spec is replaced before any reset by the pool wrapper.
    args.map_id = 'ego_forest_20x10_seed43_v1'
    captured = {}
    def factory(**kwargs):
        for key in tuple(kwargs):
            if key.startswith('kinematic_'):
                kwargs.pop(key)
        captured.update(kwargs)
        return EgoCurriculumFullForestEnv(**kwargs, pool_split=user.split, training_course=False,
                                          allow_test=user.split == 'test')
    env = make_stage0_env(args, env_class=factory)
    try:
        env.set_curriculum_stage_override(max(user.stages))
        agent = TD3UpperRecurrent(917, 3, 1., base_state_dim=117, history_length=8,
                                 history_frame_dim=100, hidden_dim=64, project_planar_action=True)
        agent.load(user.checkpoint)
        trainer = SimpleNamespace(env=env, agent=agent, total_steps=user.source_step,
                                  selected_checkpoint_step=user.source_step, selected_checkpoint_path=user.checkpoint)
        callback = EgoPoolEvalCallback(captured, user.output, split=user.split,
                                       allow_test=user.split == 'test', stages=user.stages)
        callback.on_step(trainer, force=True, context='explicit_frozen')
    finally:
        env.close()


if __name__ == '__main__':
    main()
