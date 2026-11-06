"""The two training curricula must gate on the same full-route obstacle task."""

from types import SimpleNamespace
from pathlib import Path

import numpy as np

from envs.forest.obstacle_curriculum import ObstacleCurriculum
from envs.forest.start_curriculum_env import StartObstacleCurriculumKinematicUpperForestEnv
from envs.map_atlas import MapCatalog
from scripts.audit_stage0_feasibility import make_stage0_env
from scripts.diagnose_terminal_reward_rollouts import preset_args
from scripts.run_change_td3 import build_mode_cmd
from trainers.callbacks.checkpoint import BestCheckpointCallback


MODE = "upper_dslpid_recurrent_multitree_terminal_v2_noroute_startobscurr_v1_h8_k1"


def test_obstacle_gate_ignores_short_starts_and_easier_mixed_maps():
    schedule = ObstacleCurriculum(window=3, minimum_episodes=3,
                                  success_thresholds=(2 / 3,) * 3,
                                  max_collision_rates=(1 / 3,) * 3)
    for _ in range(4):
        assert not schedule.record(episode_stage=0, start_level=3,
                                   primary_difficulty=True, success=True, collision=False)
    assert schedule.stage == 0 and schedule.stage_episodes == 0
    for success, collision in ((True, False), (True, False), (False, True)):
        advanced = schedule.record(episode_stage=0, start_level=4,
                                   primary_difficulty=True, success=success,
                                   collision=collision)
    assert advanced and schedule.stage == 1
    for _ in range(4):
        assert not schedule.record(episode_stage=1, start_level=4,
                                   primary_difficulty=False, success=True, collision=False)
    assert schedule.stage_episodes == 0
    for _ in range(3):
        advanced = schedule.record(episode_stage=1, start_level=4,
                                   primary_difficulty=True, success=True, collision=False)
    assert advanced and schedule.stage == 2


def test_physical_reset_promotes_from_one_to_two_route_trees():
    args = preset_args()
    args.map_id = "procedural_forest_multitree_train_v2"
    args.curriculum_success_gated = False
    env = make_stage0_env(args, env_class=StartObstacleCurriculumKinematicUpperForestEnv)
    env.obstacle_curriculum.minimum_episodes = 1
    env.start_curriculum.level = 4
    try:
        _, first = env.reset(seed=0)
        assert first["start_curriculum_source"] == "original"
        assert first["route_blocking_tree_count"] == 1
        assert first["obstacle_curriculum_episode_stage"] == 0
        assert np.isclose(first["start_curriculum_initial_distance"], 7.0)
        assert env.record_episode_outcome(success=True)
        _, second = env.reset(seed=0)
        assert second["curriculum_stage"] == 1
        assert second["route_blocking_tree_count"] in (1, 2)
        assert second["curriculum_primary_route_blocking_tree_count"] == 2
        assert env.get_curriculum_stage() == 1
    finally:
        env.close()


def test_new_validation_family_matches_random_obstacles_without_mix():
    command = build_mode_cmd(MODE)
    assert command[command.index("--eval_map_id") + 1] == \
        "procedural_forest_multitree_validation_random_v1"
    catalog = MapCatalog(Path("data/map_atlas/catalog_v1.json"))
    train = catalog.get("procedural_forest_multitree_train_v2")
    validation = catalog.get("procedural_forest_multitree_validation_random_v1")
    assert validation.split == "validation"
    for key in ("num_trees", "route_tree_layout", "route_count_curriculum_counts"):
        assert validation.environment_overrides[key] == train.environment_overrides[key]
    assert validation.environment_overrides["curriculum_simple_mix_probability"] == 0.0


class DummyAgent:
    def save(self, path):
        self.last_saved = path

    def enter_conservative_phase(self):
        pass

    def set_policy_anchor_from_current(self):
        pass


def test_checkpoint_selection_prefers_stable_higher_stage(tmp_path):
    callback = BestCheckpointCallback(tmp_path, window_size=1,
        minimum_success_rate=0.6, maximum_collision_rate=0.2,
        prefer_higher_curriculum_stage=True)
    agent = DummyAgent()
    trainer = SimpleNamespace(agent=agent, total_steps=100,
                              last_eval_step=None, last_eval_info=None)
    for step, stage, success in ((100, 0, 1.0), (200, 1, 0.8)):
        trainer.last_eval_step = step
        trainer.last_eval_info = {
            "eval/curriculum_stage": stage,
            "eval/success_rate": success,
            "eval/collision_rate": 0.1,
            "eval/return": 100.0,
            "eval/final_goal_distance": 0.5,
            "eval/length": 150.0,
            "eval/tracking_position_rmse": 0.1,
            "eval/done_reason_height_bound_rate": 0.0,
            "eval/done_reason_attitude_bound_rate": 0.0,
        }
        callback.on_step(trainer)
    assert callback.best_step == 200
    assert callback.best_score[0] == 1


def test_checkpoint_window_cannot_mix_stages(tmp_path):
    callback = BestCheckpointCallback(tmp_path, window_size=2,
        minimum_success_rate=0.6, maximum_collision_rate=0.2,
        prefer_higher_curriculum_stage=True)
    trainer = SimpleNamespace(agent=DummyAgent(), total_steps=100,
                              last_eval_step=None, last_eval_info=None)
    for step, stage in ((100, 0), (200, 1)):
        trainer.last_eval_step = step
        trainer.last_eval_info = {
            "eval/curriculum_stage": stage, "eval/success_rate": 1.0,
            "eval/collision_rate": 0.0, "eval/return": 100.0,
            "eval/final_goal_distance": 0.2, "eval/length": 150.0,
            "eval/tracking_position_rmse": 0.1,
            "eval/done_reason_height_bound_rate": 0.0,
            "eval/done_reason_attitude_bound_rate": 0.0,
        }
        callback.on_step(trainer)
    assert callback.best_step is None
    assert [row["step"] for row in callback.eval_window] == [200]
