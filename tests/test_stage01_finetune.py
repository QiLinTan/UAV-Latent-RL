"""Frozen Stage 0 policy adaptation must keep both evaluation tasks visible."""

from collections import Counter
from types import SimpleNamespace

from scripts.audit_stage0_feasibility import make_stage0_env
from scripts.diagnose_terminal_reward_rollouts import preset_args
from scripts.run_change_td3 import build_mode_cmd
from trainers.callbacks.checkpoint import BestCheckpointCallback
from trainers.callbacks.eval_callback import Stage01PairedEvalCallback


MODE = "upper_dslpid_recurrent_multitree_terminal_v2_noroute_stage01_ft_v1_h8_k1"
BASE = "upper_dslpid_recurrent_multitree_terminal_v2_noroute_h8_k1"


def _option(command, name):
    return command[command.index(name) + 1]


def test_warmstart_preserves_task_and_uses_pure_validation_stages():
    command = build_mode_cmd(MODE)
    base = build_mode_cmd(BASE)
    for option in (
        "--reward_variant", "--route_progress_reward_weight",
        "--upper_execution_model", "--upper_recurrent_hidden_dim",
        "--semantic_history_length", "--high_level_interval",
        "--upper_project_planar_action", "--success_replay_fraction",
    ):
        assert _option(command, option) == _option(base, option)
    assert _option(command, "--stage01_finetune") == "true"
    assert _option(command, "--stage01_mix_probability") == "0.5"
    assert _option(command, "--curriculum_stage_override") == "1"
    assert _option(command, "--start_timesteps") == "0"
    assert _option(command, "--update_after") == "5000"
    assert _option(command, "--eval_episodes") == "20"
    assert _option(command, "--eval_map_id") == "procedural_forest_multitree_validation_random_v1"


def test_real_resets_sample_both_difficulties_at_original_start():
    args = preset_args(BASE)
    args.map_id = "procedural_forest_multitree_train_v2"
    env = make_stage0_env(args)
    try:
        env.set_curriculum_stage_override(1)
        env.base_env.CURRICULUM_SIMPLE_MIX_PROBABILITY = 0.5
        counts = Counter()
        for seed in range(16):
            observation, info = env.reset(seed=seed)
            assert observation.size == 917
            assert info["curriculum_stage"] == 1
            assert info["goal_distance"] == 7.0
            counts[info["route_blocking_tree_count"]] += 1
        assert set(counts) == {1, 2}
    finally:
        env.close()


def test_paired_eval_keeps_stage1_primary_and_exposes_retention():
    class FakeEval:
        interval = 4

        def __init__(self, success):
            self.success = success
            self.called = 0

        def on_step(self, trainer, *, force=False):
            self.called += 1
            trainer.last_eval_info = {
                "eval/success_rate": self.success,
                "eval/collision_rate": 1.0 - self.success,
            }

    stage0 = FakeEval(1.0)
    stage1 = FakeEval(0.85)
    paired = Stage01PairedEvalCallback(stage0, stage1)
    trainer = SimpleNamespace(total_steps=4, last_eval_info=None)
    paired.on_step(trainer)
    assert stage0.called == stage1.called == 1
    assert trainer.last_eval_info["eval/success_rate"] == 0.85
    assert abs(trainer.last_eval_info["eval/collision_rate"] - 0.15) < 1e-12
    assert trainer.last_eval_info["eval/stage0_success_rate"] == 1.0
    assert trainer.last_eval_info["eval/stage0_collision_rate"] == 0.0


def test_selected_checkpoint_rejects_stage0_forgetting_and_falls_back(tmp_path):
    class Agent:
        def __init__(self):
            self.loaded = None

        def save(self, path):
            pass

        def load(self, path):
            self.loaded = path

    fallback = str(tmp_path / "frozen" / "model_final")
    callback = BestCheckpointCallback(
        str(tmp_path / "finetuned"), window_size=1,
        minimum_success_rate=0.8, maximum_collision_rate=0.2,
        minimum_retention_success_rate=1.0, fallback_checkpoint=fallback,
    )
    trainer = SimpleNamespace(
        agent=Agent(), total_steps=4000, last_eval_step=4000,
        last_eval_info={
            "eval/curriculum_stage": 1,
            "eval/success_rate": 0.9,
            "eval/collision_rate": 0.1,
            "eval/stage0_success_rate": 0.95,
            "eval/return": 100.0,
            "eval/final_goal_distance": 0.2,
            "eval/length": 150.0,
            "eval/tracking_position_rmse": 0.1,
            "eval/done_reason_height_bound_rate": 0.0,
            "eval/done_reason_attitude_bound_rate": 0.0,
        },
    )
    callback.on_step(trainer)
    assert callback.peak_step is None and callback.best_step is None
    callback.on_train_end(trainer)
    assert trainer.agent.loaded == fallback
    assert trainer.selected_checkpoint_step == 0
