"""Fixed validation pool, original starts, separate raw/selected artifacts."""
import json
from pathlib import Path

import numpy as np

from envs.forest.ego_curriculum_env import EgoCurriculumFullForestEnv
from envs.preprocess import preprocess_state
from trainers.full_scratch import fingerprint
from trainers.paired_trajectory_trace import PairedTrajectoryTrace


class EgoPoolEvalCallback:
    def __init__(self, env_kwargs, output_dir, interval=5000, split="validation", allow_test=False, stages=None):
        if split == 'test' and not allow_test:
            raise ValueError('Sealed test requires explicit evaluation request')
        self.split = split
        self.stages = stages
        self.env_kwargs = {**env_kwargs, 'pool_split': split, 'training_course': False, 'allow_test': allow_test, 'gui': False}
        self.output_dir = Path(output_dir) / split
        self.interval = int(interval)

    def on_episode_end(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass

    def on_step(self, trainer, force=False, context=None):
        if not force and trainer.total_steps % self.interval:
            return
        context = context or ('selected' if getattr(trainer, 'selected_checkpoint_path', None) else 'periodic_raw')
        actor_hash = fingerprint(trainer.agent.actor)
        stage = trainer.env.get_curriculum_stage()
        source_step = getattr(trainer, 'selected_checkpoint_step', trainer.total_steps) if context == 'selected' else trainer.total_steps
        root = self.output_dir / context / f'step{trainer.total_steps}_source{source_step}_{actor_hash[:12]}'
        summaries = {}
        for density in sorted(set(self.stages) if self.stages is not None else {0, stage}):
            env = EgoCurriculumFullForestEnv(**self.env_kwargs)
            env.set_curriculum_stage_override(density)
            rows = []
            try:
                for index in range(len(env.pool.entries(self.split))):
                    observation, info = env.reset(seed=index)
                    trace = PairedTrajectoryTrace(env, index, trace_id=f'map{index:03d}')
                    trace.install()
                    total = 0.
                    while True:
                        state = preprocess_state(observation)
                        action = trainer.agent.project_action(trainer.agent.select_action(state))
                        observation, reward, terminated, truncated, info = env.step(np.asarray(action).reshape(1, 3))
                        total += reward
                        trace.record_upper(action, action, reward, info, pre_observation=state)
                        if terminated or truncated:
                            break
                    trace.uninstall()
                    trace.save(root / f'density{density}', done_reason=info['done_reason'], success=info['success'],
                               final_goal_distance=info['goal_distance'])
                    lower = trace.lower
                    positions = np.asarray([r['position'] for r in lower])
                    refs = np.asarray([r['reference_position'] for r in lower])
                    velocities = np.asarray([r['velocity'] for r in lower])
                    ref_velocities = np.asarray([r['reference_velocity'] for r in lower])
                    rows.append({**env._pool_info(), **env._start_info(), 'return': total, 'length': len(trace.upper),
                                 'success': bool(info['success']), 'collision': bool(info['collision']),
                                 'done_reason': info['done_reason'], 'final_goal_distance': float(info['goal_distance']),
                                 'guidance_entered': min([np.linalg.norm(trace.initial_position-trace.target)] +
                                                         [r['goal_distance'] for r in lower]) < 1.2,
                                 'minimum_tree_clearance': min(r['clearance'] for r in lower),
                                 'tracking_position_rmse': float(np.sqrt(np.mean(np.sum((positions-refs)**2, axis=1)))),
                                 'tracking_velocity_rmse': float(np.sqrt(np.mean(np.sum((velocities-ref_velocities)**2, axis=1)))),
                                 'mean_vertical_velocity': float(velocities[:, 2].mean())})
            finally:
                env.close()
            prefix = 'eval/'
            metrics = {prefix + k: float(np.mean([r[k] for r in rows])) for k in
                       ('return', 'length', 'success', 'collision', 'final_goal_distance', 'tracking_position_rmse', 'tracking_velocity_rmse')}
            metrics['eval/success_rate'] = metrics.pop('eval/success')
            metrics['eval/collision_rate'] = metrics.pop('eval/collision')
            metrics['eval/curriculum_stage'] = density
            metrics['eval/active_tree_count'] = (10, 20, 30, 40)[density]
            metrics['eval/guidance_entered_episode_fraction'] = float(np.mean([r['guidance_entered'] for r in rows]))
            for reason in ('success', 'collision', 'timeout', 'height_bound', 'xy_bound', 'attitude_bound'):
                metrics[f'eval/done_reason_{reason}_rate'] = float(np.mean([r['done_reason'] == reason for r in rows]))
            for group, subset in (('all', rows), ('success', [r for r in rows if r['success']])):
                if subset:
                    for label, q in zip(('min', 'p05', 'p50', 'p95'), np.quantile([r['minimum_tree_clearance'] for r in subset], [0, .05, .5, .95])):
                        metrics[f'eval/{group}_episode_minimum_clearance_{label}'] = float(q)
            root.mkdir(parents=True, exist_ok=True)
            payload = dict(step=trainer.total_steps, source_step=source_step, context=context,
                           actor_sha256=actor_hash, pool_fingerprint=trainer.env.pool.fingerprint,
                           density_stage=density, summary=metrics, episodes=rows)
            (root / f'density{density}_metrics.json').write_text(json.dumps(payload, indent=2) + '\n')
            history = self.output_dir / context / 'evaluation_metrics.jsonl'
            with history.open('a') as f:
                f.write(json.dumps(payload) + '\n')
            summaries[density] = metrics
            print(f"[Pool Eval {context} @ {trainer.total_steps}] density={density} trees={(10,20,30,40)[density]} SR={metrics['eval/success_rate']:.3f} CR={metrics['eval/collision_rate']:.3f}")
        main = summaries.get(stage, summaries[max(summaries)])
        trainer.last_eval_info = dict(main)
        if 0 in summaries:
            trainer.last_eval_info.update({'eval/stage0_success_rate': summaries[0]['eval/success_rate'],
                                          'eval/stage0_collision_rate': summaries[0]['eval/collision_rate']})
        trainer.last_eval_return = trainer.last_eval_info['eval/return']
        trainer.last_eval_step = trainer.total_steps
