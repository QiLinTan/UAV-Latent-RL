"""D3 controls: real A batch equivalence and frozen Actor target cadence."""
import json

import numpy as np
import pytest
import torch

from scripts.audit_ego20x10_value import DEFAULT, load, learner, rng_restore, fingerprint
from scripts.run_ego20x10_d3 import ScheduledBuffer
from scripts.evaluate_ego20x10_d3 import rank_pairs


def test_scheduled_replay_rejects_future_transition():
    buffer=ScheduledBuffer({'state':np.arange(10)})
    with pytest.raises(AssertionError):buffer.set_batch([5],5)
    buffer.set_batch([4],5)
    np.testing.assert_array_equal(buffer.indices,[4])


def test_first_critic_lead_update_matches_A_target_and_preserves_actor():
    five,six=load(5000),load(6000);agent=learner(five)
    schedule=np.load(DEFAULT/'batch_schedule.npz')
    main=ScheduledBuffer(six['replay']['arrays']);main.set_batch(schedule['main_indices'][0],5001)
    success=ScheduledBuffer(five['success']['strata'][10]['arrays']);success.set_batch(schedule['success_indices'][0],297)
    actor=fingerprint(agent.actor);actor_target=fingerprint(agent.actor_target)
    target_before={k:v.clone() for k,v in agent.critic_target.state_dict().items()}
    agent.actor_frozen=True;rng_restore(five)
    info=agent.train(main,256,success,.2)
    baseline=json.loads((DEFAULT/'update_audit.json').read_text())[0]
    assert info['critic_loss']==baseline['critic_loss']
    assert info['q_target_mean']==baseline['q_target_mean']
    assert agent.total_it==2 and not info['actor_updated']
    assert fingerprint(agent.actor)==actor and fingerprint(agent.actor_target)==actor_target
    assert len(agent.actor_optimizer.state)==0
    for k,value in agent.critic_target.state_dict().items():
        expected=agent.tau*agent.critic.state_dict()[k]+(1-agent.tau)*target_before[k]
        torch.testing.assert_close(value,expected,atol=0,rtol=0)
    target_even={k:v.clone() for k,v in agent.critic_target.state_dict().items()}
    main.set_batch(schedule['main_indices'][1],5002);success.set_batch(schedule['success_indices'][1],297)
    agent.train(main,256,success,.2)
    assert agent.total_it==3
    for k,value in agent.critic_target.state_dict().items():torch.testing.assert_close(value,target_even[k],atol=0,rtol=0)


def test_value_concordance_excludes_tied_returns():
    assert rank_pairs([1,2,3,4,5],[1,2,3,4,5])['concordant']==10
    assert rank_pairs([1,2,3,4,5],[5,4,3,2,1])['discordant']==10
    tied=rank_pairs([1,2,3,4,5],[0,0,0,0,0])
    assert tied['ties']==10 and tied['spearman'] is None
