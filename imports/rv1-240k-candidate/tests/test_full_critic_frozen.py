"""Frozen-policy critic learning and distinct value/guidance acceptance."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from algos.td3.td3_upper_recurrent import TD3UpperRecurrent
from data.replay_buffer import ReplayBuffer
from scripts.run_full_critic_frozen import freeze_actor, check_learner, build_command, FrozenCriticTrainer
from trainers.callbacks.checkpoint import BestCheckpointCallback
from scripts.run_change_td3 import build_mode_cmd
from scripts.audit_full_critic_frozen import calibration, ranking, gradient_quality, verdict, project, build_cases, noise_control_summary


def test_actor_frozen_but_both_critic_grus_and_delayed_target_critic_update():
    torch.manual_seed(17)
    agent = TD3UpperRecurrent(11, 3, 1., base_state_dim=5, history_length=2,
                             history_frame_dim=3, hidden_dim=8, project_planar_action=True,
                             device=torch.device('cpu'))
    actor = copy.deepcopy(agent.actor.state_dict())
    critic = copy.deepcopy(agent.critic.state_dict())
    target = copy.deepcopy(agent.critic_target.state_dict())
    expected = dict(gamma=agent.discount,tau=agent.tau,policy_freq=2,policy_noise=.2,noise_clip=.5)
    freeze_actor(agent)
    replay = ReplayBuffer(11,3,max_size=32)
    rng = np.random.default_rng(17)
    for _ in range(32):
        replay.push(rng.normal(size=11),rng.uniform(-.5,.5,size=3),1.,rng.normal(size=11),False)
    first = agent.train(replay,batch_size=16,success_replay_buffer=replay,success_replay_fraction=.2)
    assert not first['actor_updated'] and first['actor_frozen'] == 1.
    assert all(torch.equal(v,target[k]) for k,v in agent.critic_target.state_dict().items())
    second = agent.train(replay,batch_size=16,success_replay_buffer=replay,success_replay_fraction=.2)
    check_learner(agent,actor,expected)
    assert not second['actor_updated'] and second['success_replay_fraction'] > 0
    for key in ['q1.encoder.gru.weight_ih_l0','q2.encoder.gru.weight_ih_l0']:
        assert not torch.equal(critic[key],agent.critic.state_dict()[key])
    for key,value in agent.critic_target.state_dict().items():
        torch.testing.assert_close(value,agent.tau*agent.critic.state_dict()[key]+(1-agent.tau)*target[key],rtol=0,atol=0)
    with pytest.raises(RuntimeError,match='forbids'):
        agent.actor_optimizer.step()
    agent.critic_optimizer.param_groups[0]['lr']=1e-4
    with pytest.raises(RuntimeError,match='3e-4'):
        check_learner(agent,actor,expected)


def test_20k_budget_and_original_exploration_control_reward_protocol():
    cli=SimpleNamespace(total_steps=20000,output_root=__import__('pathlib').Path('/tmp/critic-test'),
                        checkpoint_dir='/tmp/critic-test/checkpoints',log_dir='/tmp/critic-test/runs')
    cmd=build_command(cli);base=build_mode_cmd('full_ft_v1')[3:]
    value=lambda c,k:c[c.index(k)+1]
    assert value(cmd,'--total_steps')=='20000'
    assert value(cmd,'--full_ft_actor_freeze_steps')=='20000'
    assert value(cmd,'--upper_critic_lr')=='3e-4'
    for key in ['--noise_decay_steps','--expl_noise_start','--expl_noise_end','--success_replay_fraction',
                '--reward_scale','--reward_variant','--high_level_interval','--upper_execution_model',
                '--map_id','--semantic_history_length','--upper_project_planar_action']:
        assert value(cmd,key)==value(base,key)
    assert value(cmd,'--noise_decay_steps')=='40000'


def test_best_critic_selection_is_disabled_before_baseline_and_final_restore():
    trainer=FrozenCriticTrainer.__new__(FrozenCriticTrainer)
    trainer.callbacks=[]
    best=BestCheckpointCallback.__new__(BestCheckpointCallback)
    trainer.add_callback(best)
    # A functioning selector would access unset attributes or trainer.agent.
    best.on_step(SimpleNamespace(total_steps=0))
    best.on_train_end(SimpleNamespace(total_steps=20000))
    assert trainer.callbacks==[best]


def records(mode):
    rows=[]
    for i in range(60):
        for name,a in [('source',0.)]+[(f'gradient_{sign}_{step:.2f}',step*(1 if sign=='plus' else -1))
                                     for step in (.05,.15,.30) for sign in ('plus','minus')]:
            g=10+2*a if mode!='wrong_gradient' else 10-2*a
            if mode=='ties': g=10+a*.01
            q=10+2*a-(8 if mode=='offset' else 0)
            rows.append(dict(case_id=str(i),stage=0 if i%2==0 else 3,map_seed=80+i%10,
                             candidate=name,action=[a,0,0],Q=[q,q,q],G=g))
    return rows


def decide(rows):
    s=calibration([r for r in rows if r['candidate']=='source'])
    return verdict(s,calibration(rows),ranking(rows),gradient_quality(rows),True)


def test_accurate_values_do_not_imply_correct_local_gradient_or_ranking():
    d=decide(records('wrong_gradient'))
    assert d['level1_value_pass']
    assert d['ranking_evidence_sufficient'] and d['gradient_evidence_sufficient']
    assert not d['level2_guidance_pass'] and d['level2_status']=='fail'


def test_constant_value_offset_can_fail_calibration_but_preserve_guidance():
    d=decide(records('offset'))
    assert not d['level1_value_pass']
    assert d['level2_guidance_pass'] and d['level2_status']=='pass'


def test_near_ties_and_duplicate_actions_cannot_fake_sufficient_evidence():
    rows=records('ties');d=decide(rows)
    assert d['level2_status']=='insufficient_evidence'
    normal=records('offset');a=ranking(normal)
    b=ranking(normal+[dict(r,candidate=r['candidate']+'_alias') for r in normal])
    assert a['distinguishable_pairs']==b['distinguishable_pairs']


def test_case_bank_includes_failure_states_and_only_validation_seeds80_89():
    cases=build_cases()
    assert len(cases)==78
    assert {c['map_seed'] for c in cases}==set(range(80,90))
    assert {c['stage'] for c in cases}=={0,3}
    assert any(c['baseline_outcome']=='collision' for c in cases)
    for action in ([2,2,2],[-2,-2,-2]):
        a=project(action)
        assert np.linalg.norm(a[:2])<=1.+1e-6 and abs(a[2])<=1


def noise_records():
    return [dict(step=20000,case_id='one',candidate=name,replica=replica,
                 Q=[10.,10.,10.],G=float(replica+1))
            for name in ('source','q1_best_grid','gradient_plus_0.15') for replica in range(4)]


def test_duplicate_noise_logs_do_not_reweight_replicas_or_hide_missing_evidence():
    rows=noise_records()
    repeated=rows+[copy.deepcopy(rows[0]) for _ in range(7)]
    result=noise_control_summary(repeated,{'one'},20000)
    assert result['complete'] and result['unique_records']==12
    assert result['duplicate_records_ignored']==7
    assert result['source_calibration']['q1']['mean_G']==2.5
    source=next(r for r in result['per_candidate'] if r['candidate']=='source')
    assert source['replicas']==4
    assert source['G_std']==pytest.approx(np.std([1.,2.,3.,4.],ddof=1))
    # Twelve raw lines still lack a distinct replica if one was replaced by a repeat.
    missing=noise_control_summary(rows[:-1]+[rows[0]],{'one'},20000)
    assert not missing['complete'] and missing['missing_records']==1


def test_conflicting_noise_repeats_and_mixed_checkpoint_records_are_rejected():
    rows=noise_records()
    with pytest.raises(RuntimeError,match='Conflicting duplicate'):
        noise_control_summary(rows+[dict(rows[0],G=99.)],{'one'},20000)
    with pytest.raises(RuntimeError,match='different checkpoint'):
        noise_control_summary([dict(rows[0],step=10000)],{'one'},20000)
