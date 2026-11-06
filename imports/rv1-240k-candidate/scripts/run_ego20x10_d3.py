"""Isolated shared-Replay D3. Uses the unmodified production TD3.train method."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from algos.td3.td3_upper_recurrent import _project_planar_action_tensor
from scripts.audit_ego20x10_value import (
    SOURCE, DEFAULT as BASELINE, load, learner, rng_restore, describe, probe, fingerprint, write,
)

ROOT=Path('analysis_outputs/full_scratch_ego20x10_d3_critic_lead_200')
FIELDS=('state','action','next_state','reward','not_done')
NETWORKS=('actor','critic','actor_target','critic_target')
SPECIAL={5000,5043,5125,5200,5243,5299,5325,5399,5600,6000}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ScheduledBuffer:
    """A prefix of A's actual Replay, sampled with A's recorded indices."""
    def __init__(self, arrays):
        self.arrays=arrays;self.indices=None;self.size=0

    def set_batch(self, indices, size):
        self.indices=np.asarray(indices);self.size=int(size)
        assert self.indices.min()>=0 and self.indices.max()<self.size

    def sample(self,count):
        assert count==len(self.indices)
        return tuple(torch.as_tensor(self.arrays[k][self.indices]) for k in FIELDS)


class TrainingHooks:
    def __init__(self,agent):
        self.agent=agent;self.active=False;self.output_gradient=None
        def actor(module,args,output):
            if self.active and output.requires_grad:
                output.register_hook(lambda gradient:setattr(self,'output_gradient',gradient.detach().clone()))
        def target_actor(module,args,output):
            if self.active:self.target_actor=output.detach().clone()
        def target_critic(module,args,output):
            if self.active:
                self.target_action=args[1].detach().clone();self.target_qs=tuple(q.detach().clone() for q in output)
        def critic(module,args,output):
            if self.active:self.current_qs=tuple(q.detach().clone() for q in output)
        self.handles=[agent.actor.register_forward_hook(actor),agent.actor_target.register_forward_hook(target_actor),
                      agent.critic_target.register_forward_hook(target_critic),agent.critic.register_forward_hook(critic)]

    def close(self):
        for handle in self.handles:handle.remove()


def policy_probe(agent,states,previous=None):
    result,actions=probe(agent,states,previous,gradients=False)
    result.update(z_positive_saturation=float(np.mean(actions[:,2]>=.999)),
                  z_negative_saturation=float(np.mean(actions[:,2]<=-.999)),
                  axis_saturation_fraction=np.mean(np.abs(actions)>=.999,axis=0).tolist())
    tensor=torch.as_tensor(states)
    with torch.no_grad():
        logits=agent.actor.policy(agent.actor.encoder(tensor)).numpy()
    result['logits']={axis:describe(logits[:,i]) for i,axis in enumerate(('x','y','z'))}
    result['tanh_z_derivative_mean']=float(np.mean(1-np.tanh(logits[:,2])**2))
    candidate=torch.as_tensor(actions).requires_grad_()
    q1,q2=agent.critic(tensor,candidate)
    for label,value in (('q1',q1),('q2',q2),('q_min',torch.minimum(q1,q2))):
        gradient=torch.autograd.grad(value.sum(),candidate,retain_graph=label!='q_min')[0].detach().numpy()
        result[label+'_action_gradient_mean']=gradient.mean(axis=0).tolist()
        result[label+'_positive_gradient_fraction']=np.mean(gradient>0,axis=0).tolist()
    return result,actions


def validate_training(hooks,main,success,noise,info):
    state,action,next_state,reward,mask=[torch.cat((main.sample(205)[i],success.sample(51)[i])) for i in range(5)]
    agent=hooks.agent
    target_action=(hooks.target_actor+noise).clamp(-agent.max_action,agent.max_action)
    if agent.project_planar_action:target_action=_project_planar_action_tensor(target_action,agent.max_action)
    torch.testing.assert_close(target_action,hooks.target_action,atol=0,rtol=0)
    target=reward+mask*agent.discount*torch.minimum(*hooks.target_qs)
    loss=((hooks.current_qs[0]-target)**2).mean()+((hooks.current_qs[1]-target)**2).mean()
    assert abs(float(loss)-info['critic_loss'])<1e-6
    record=dict(loss_reconstruction_error=abs(float(loss)-info['critic_loss']),target=describe(target.numpy()),
                main_count=205,success_count=51,terminal_count=int((mask==0).sum()),
                target_noise_sha256=hashlib.sha256(noise.numpy().tobytes()).hexdigest())
    if hooks.output_gradient is not None:
        record.update(actual_batch_q1_action_gradient_mean=(-hooks.output_gradient.numpy()*256).mean(axis=0).tolist(),
                      actor_output_loss_gradient_mean=hooks.output_gradient.mean(dim=0).numpy().tolist(),
                      actor_head_bias_gradient=agent.actor.policy[-1].bias.grad.numpy().tolist())
    return record,dict(state=state.numpy(),action=action.numpy(),next_state=next_state.numpy(),reward=reward.numpy(),not_done=mask.numpy(),
                       target_actor=hooks.target_actor.numpy(),noise=noise.numpy(),target_action=hooks.target_action.numpy(),
                       target_q1=hooks.target_qs[0].numpy(),target_q2=hooks.target_qs[1].numpy(),td_target=target.numpy())


def save_checkpoint(agent,root,step,actor_updates,source):
    torch.save(dict(networks={k:getattr(agent,k).state_dict() for k in NETWORKS},
                    optimizers={k:getattr(agent,k).state_dict() for k in ('actor_optimizer','critic_optimizer')},
                    total_it=agent.total_it,actor_updates=actor_updates,step=step,actor_frozen=agent.actor_frozen,
                    torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),
                    source_5k=str(SOURCE/'training_state_step5000.pt'),protocol=source['protocol'],
                    source_course=dict(start=source['start'],obstacle=source['obstacle'],start_rng=source['start_rng'],
                                       map_rng=source['map_rng'],pool_sampler=source['pool_sampler']),
                    shared_replay=dict(main_source=str(SOURCE/'training_state_step6000.pt'),prefix_size=step,
                                       success_source=str(SOURCE/'training_state_step5000.pt'),schedule=str(BASELINE/'batch_schedule.npz'))),
               root/'checkpoints'/f'step{step}.pt')


def checkpoint(root,step):
    saved=torch.load(root/'checkpoints'/f'step{step}.pt',map_location='cpu',weights_only=False)
    agent=learner(load(5000))
    for k in NETWORKS:getattr(agent,k).load_state_dict(saved['networks'][k])
    for k in ('actor_optimizer','critic_optimizer'):getattr(agent,k).load_state_dict(saved['optimizers'][k])
    agent.total_it=saved['total_it'];agent.actor_frozen=saved['actor_frozen']
    return agent


def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    protected=list(SOURCE.iterdir())+list(BASELINE.rglob('*'))
    production=json.loads((BASELINE/'production_code_guard.json').read_text())
    protected += [Path(k) for k in production]
    write(root/'protected_inputs.json',{str(p):sha(p) for p in protected if p.is_file()})
    source=load(5000);agent=learner(source)
    assert agent.total_it==1 and len(agent.actor_optimizer.state)==0
    assert fingerprint(agent.actor)==fingerprint(agent.actor_target)
    assert agent._policy_anchor_actor is None  # nominal 0.1 weight is inactive in this scratch state
    protocol=dict(name=root.name,kind='shared actual A behavior Replay, off-policy mechanism experiment',start=5000,end=6000,
                  critic_lead_updates=200,actor_initial_updates=0,critic_initial_updates=1,policy_freq=2,tau=.005,
                  target_protocol='Critic target Polyak at total_it % 2 == 0 during lead; Actor and target Actor fixed and equal. Standard TD3 afterwards.',
                  data_protocol='At each A time append its real transition, use exactly its sampled main/success indices and clipped target noise. Each branch computes its own TD target.',
                  common_counterfactual_xy='Fixed initial XY from original D2 A6k Actor at each representative state; all axes free to change after first action.',
                  common_suffix='frozen original 5k Actor, same for every group; own suffix uses that checkpoint Actor',
                  branch_C_condition='B6k L0 >=9/20 and net paired gain over A6k >=5/20, with fixed-state vertical saturation <50% at B actor63 and B6k.',
                  online_condition='B satisfies preservation criteria above; run C control first, then independent fresh-episode online 5k->6k, extend to8k only if stable.',
                  validation='same20 Stage0 maps, fixed L0 QA remaining0.8m and original16m; no test',
                  judgement='No global calibration or RL learning claim from 12 states/20 maps; 5k Actor is untrained.',
                  baseline_contract=source['protocol'],source_5k_sha256=sha(SOURCE/'training_state_step5000.pt'),
                  A_replication=json.loads((BASELINE/'replication.json').read_text()),
                  schedule_sha256=sha(BASELINE/'batch_schedule.npz'),nominal_anchor_weight=agent.policy_anchor_weight,active_anchor=False)
    write(root/'protocol.json',protocol);write(root/'protocol_fingerprint.json',dict(sha256=sha(root/'protocol.json')))


def train(root,branch='B'):
    destination=root/branch;destination.mkdir(exist_ok=False)
    (destination/'checkpoints').mkdir();(destination/'td_batches').mkdir()
    five,six=load(5000),load(6000);agent=learner(five)
    fixed=np.load(BASELINE/'fixed_states.npz')['states'];assert len(fixed)==188
    schedule=np.load(BASELINE/'batch_schedule.npz')
    # At every recorded A update main indices must reference only its then-current prefix.
    assert np.all(schedule['main_indices'].max(axis=1)<schedule['steps'])
    for k in FIELDS:np.testing.assert_array_equal(five['replay']['arrays'][k],six['replay']['arrays'][k][:5000])
    for k in FIELDS:np.testing.assert_array_equal(five['success']['strata'][10]['arrays'][k],six['success']['strata'][10]['arrays'][k])
    main=ScheduledBuffer(six['replay']['arrays']);success=ScheduledBuffer(five['success']['strata'][10]['arrays'])
    initial_actor=fingerprint(agent.actor);initial_target=fingerprint(agent.actor_target)
    hooks=TrainingHooks(agent);rng_restore(five)
    actor_updates=0;previous=None;records=[];snapshots=[]
    def snapshot(step):
        values,actions=policy_probe(agent,fixed)
        values.update(step=step,actor_updates=actor_updates,critic_updates=agent.total_it,
                      actor_sha256=fingerprint(agent.actor),critic_sha256=fingerprint(agent.critic),
                      actor_target_sha256=fingerprint(agent.actor_target))
        snapshots.append(values);save_checkpoint(agent,destination,step,actor_updates,five)
        write(destination/'checkpoint_probes.json',snapshots)
        print(f'[{branch}] step={step} actor={actor_updates} critic={agent.total_it} az={values["action_mean"][2]:+.5f} Sz={values["z_saturation_fraction"]:.1%}',flush=True)
    snapshot(5000)
    try:
        for n,step in enumerate(schedule['steps']):
            step=int(step);lead=n<200
            main.set_batch(schedule['main_indices'][n],step);success.set_batch(schedule['success_indices'][n],297)
            # C consumes the same random smoothing stream during its no-update period.
            before_rng=torch.get_rng_state();generator=torch.Generator().set_state(before_rng)
            noise=(torch.randn((256,3),generator=generator)*agent.policy_noise).clamp(-agent.noise_clip,agent.noise_clip)
            np.testing.assert_array_equal(noise.numpy(),schedule['target_noise'][n])
            if branch=='C' and lead:
                torch.set_rng_state(generator.get_state());info={'actor_updated':False};validation={}
            else:
                agent.actor_frozen=branch=='B' and lead
                hooks.output_gradient=None;hooks.active=True
                info=agent.train(main,batch_size=256,success_replay_buffer=success,success_replay_fraction=.2)
                hooks.active=False
                validation,batch=validate_training(hooks,main,success,noise,info)
                if step in SPECIAL:np.savez_compressed(destination/'td_batches'/f'step{step}.npz',**batch)
                actor_updates+=int(info['actor_updated'])
            if lead:
                assert fingerprint(agent.actor)==initial_actor and fingerprint(agent.actor_target)==initial_target
                assert len(agent.actor_optimizer.state)==0
            light,actions=probe(agent,fixed,previous,gradients=False)
            light.update(z_positive_saturation=float(np.mean(actions[:,2]>=.999)),z_negative_saturation=float(np.mean(actions[:,2]<=-.999)),
                         axis_saturation_fraction=np.mean(np.abs(actions)>=.999,axis=0).tolist())
            previous=actions.copy()
            # First 100 Actor updates and the lead phase: three-axis local Q gradients/logits.
            if (lead and (n%20==0 or step in SPECIAL)) or (info['actor_updated'] and actor_updates<=100) or step in SPECIAL:
                adjacent_rms=light['adjacent_policy_rms']
                light,_=policy_probe(agent,fixed,previous=None)
                light['adjacent_policy_rms']=adjacent_rms
            record=dict(step=step,critic_updates=agent.total_it,actor_updates=actor_updates,critic_lead=lead,
                        actor_updated=bool(info['actor_updated']),training=info,td_audit=validation,policy=light,actions=actions)
            records.append(record)
            if (step%50==0 or step in SPECIAL):snapshot(step)
        expected_actor=400 if branch in ('B','C') else 500
        assert actor_updates==expected_actor
        assert agent.total_it==(801 if branch=='C' else 1001)
        write(destination/'updates.json',[{k:v for k,v in r.items() if k!='actions'} for r in records])
        np.savez_compressed(destination/'fixed_policy_actions.npz',steps=schedule['steps'],actions=np.stack([r['actions'] for r in records]))
        proof=dict(shared_A_main_transitions=1000,shared_sampling_indices=True,shared_target_noise=True,
                   own_td_targets=True,maximum_loss_reconstruction_error=max(r['td_audit'].get('loss_reconstruction_error',0) for r in records),
                   final_actor_updates=actor_updates,final_critic_updates=agent.total_it,lead_actor_and_target_fixed=True,
                   lead_actor_optimizer_unchanged=True,source_course_and_rng_preserved=True,
                   note='Offline learner snapshots reference immutable A Replay prefixes and original course; no physical episodes executed.')
        write(destination/'training_proof.json',proof)
    finally:hooks.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('part',choices=('train','train_C'))
    parser.add_argument('--output',type=Path,default=ROOT);args=parser.parse_args()
    if args.part=='train':prepare(args.output);train(args.output)
    else:train(args.output,'C')


if __name__=='__main__':main()
