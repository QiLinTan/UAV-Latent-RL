"""Regression checks for the K1 Actor -> fresh Full TD3 finetune contract."""

from types import SimpleNamespace

import numpy as np
import torch

from algos.td3.td3_upper_recurrent import TD3UpperRecurrent
from data.replay_buffer import ReplayBuffer
from scripts.run_change_td3 import build_mode_cmd
from scripts.audit_stage0_feasibility import make_stage0_env
from scripts.diagnose_terminal_reward_rollouts import preset_args
from trainers.callbacks.eval_callback import FullFTTripleEvalCallback
from trainers.full_ft_stage_mixer import FullFTStageMixer


def _option(command, name):
    return command[command.index(name) + 1]


def _agent():
    return TD3UpperRecurrent(
        state_dim=11, action_dim=3, max_action=1.0,
        base_state_dim=5, history_length=2, history_frame_dim=3,
        hidden_dim=8, project_planar_action=True, device=torch.device("cpu"),
    )


def _same(module_a, module_b):
    return all(torch.equal(a, b) for a, b in zip(module_a.parameters(), module_b.parameters()))


def test_full_ft_preset_keeps_network_reward_and_control_contract():
    full = build_mode_cmd("full_ft_v1")
    base = build_mode_cmd("upper_dslpid_recurrent_multitree_terminal_v2_noroute_h8_k1")
    for option in (
        "--reward_variant", "--route_progress_reward_weight",
        "--upper_recurrent_hidden_dim", "--semantic_history_length",
        "--high_level_interval", "--upper_project_planar_action",
        "--success_replay_fraction", "--map_id", "--reward_scale",
    ):
        assert _option(full, option) == _option(base, option)
    assert _option(full, "--upper_execution_model") == "full"
    assert _option(full, "--full_ft_v1") == "true"
    assert _option(full, "--start_timesteps") == "0"
    assert _option(full, "--curriculum_simple_mix_probability") == "0.0"
    assert _option(full, "--full_ft_actor_freeze_steps") == "4000"
    assert _option(full, "--update_after") == "512"
    assert _option(full, "--eval_interval") == "5000"


def test_actor_only_load_preserves_fresh_critic_and_optimizer(tmp_path):
    source = _agent()
    dest = _agent()
    critic_before = [p.detach().clone() for p in dest.critic.parameters()]
    critic_target_before = [p.detach().clone() for p in dest.critic_target.parameters()]
    prefix = str(tmp_path / "source")
    source.save(prefix)
    dest.load_actor_only(prefix)
    assert _same(dest.actor, source.actor)
    assert _same(dest.actor, dest.actor_target)
    assert all(torch.equal(p, before) for p, before in zip(dest.critic.parameters(), critic_before))
    assert all(torch.equal(p, before) for p, before in zip(dest.critic_target.parameters(), critic_target_before))
    assert not dest.actor_optimizer.state
    assert not dest.critic_optimizer.state
    assert dest.total_it == 0


def test_critic_warms_while_actor_and_actor_target_remain_frozen():
    rng = np.random.default_rng(123)
    replay = ReplayBuffer(11, 3, max_size=32)
    for _ in range(32):
        state = rng.normal(size=11).astype(np.float32)
        next_state = rng.normal(size=11).astype(np.float32)
        action = rng.uniform(-0.5, 0.5, size=3).astype(np.float32)
        replay.push(state, action, rng.normal(), next_state, False)
    agent = _agent()
    actor_before = [p.detach().clone() for p in agent.actor.parameters()]
    target_before = [p.detach().clone() for p in agent.actor_target.parameters()]
    critic_before = [p.detach().clone() for p in agent.critic.parameters()]
    agent.actor_frozen = True
    first = agent.train(replay, batch_size=8)
    second = agent.train(replay, batch_size=8)
    assert not first["actor_updated"] and not second["actor_updated"]
    assert all(torch.equal(p, before) for p, before in zip(agent.actor.parameters(), actor_before))
    assert all(torch.equal(p, before) for p, before in zip(agent.actor_target.parameters(), target_before))
    assert any(not torch.equal(p, before) for p, before in zip(agent.critic.parameters(), critic_before))
    assert not _same(agent.critic, agent.critic_target)
    agent.actor_frozen = False
    agent.train(replay, batch_size=8)
    unfrozen = agent.train(replay, batch_size=8)
    assert unfrozen["actor_updated"]
    assert any(not torch.equal(p, before) for p, before in zip(agent.actor.parameters(), actor_before))


def test_stage_mixer_exact_30_70_per_ten_episodes():
    mixer = FullFTStageMixer(43)
    stages = [mixer.next_stage() for _ in range(30)]
    for i in range(0, 30, 10):
        assert stages[i:i + 10].count(0) == 3
        assert stages[i:i + 10].count(3) == 7
    repeat = FullFTStageMixer(43)
    assert stages == [repeat.next_stage() for _ in range(30)]
    assert mixer.stage_counts == {0: 9, 3: 21}


def test_atlas_override_can_be_disabled_for_exact_stage0_stage3_resets():
    args = preset_args("upper_dslpid_recurrent_multitree_terminal_v2_noroute_h8_k1")
    args.map_id = "procedural_forest_multitree_train_v2"
    env = make_stage0_env(args)
    try:
        assert env.base_env.CURRICULUM_SIMPLE_MIX_PROBABILITY == 0.2
        env.base_env.CURRICULUM_SIMPLE_MIX_PROBABILITY = 0.0
        for stage, expected_count in ((0, 1), (3, 5)):
            env.set_curriculum_stage_override(stage)
            for seed in range(4):
                observation, info = env.reset(seed=seed)
                assert observation.size == 917
                assert info["curriculum_stage"] == stage
                assert info["route_blocking_tree_count"] == expected_count
                assert info["goal_distance"] == 7.0
    finally:
        env.close()


def test_triple_validation_keeps_full_stage3_primary_and_retention():
    class FakeEval:
        interval = 5000
        def __init__(self, success, collision):
            self.success, self.collision = success, collision
            self.calls = 0
        def on_step(self, trainer, *, force=False):
            self.calls += 1
            trainer.last_eval_info = {
                "eval/success_rate": self.success,
                "eval/collision_rate": self.collision,
            }
    callbacks = (FakeEval(.7, .2), FakeEval(.9, .05), FakeEval(.8, .1))
    trainer = SimpleNamespace(total_steps=5000, last_eval_info={})
    FullFTTripleEvalCallback(*callbacks).on_step(trainer)
    assert all(cb.calls == 1 for cb in callbacks)
    assert trainer.last_eval_info["eval/success_rate"] == .7
    assert trainer.last_eval_info["eval/stage0_success_rate"] == .9
    assert trainer.last_eval_info["eval/k1_stage3_success_rate"] == .8
