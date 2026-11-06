import argparse
import hashlib
import json
import pathlib
import shlex
import subprocess
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def build_base_cmd() -> list[str]:
    return [
        sys.executable,
        "-m",
        "scripts.change_td3",
        "--seed",
        "42",
        "--total_steps",
        "500000",
        "--gui",
        "false",
        "--eval_gui",
        "false",
        "--eval_stepsleep",
        "false",
        "--action_scale",
        "1.0",
    ]


def _replace_scalar_option(cmd: list[str], option: str, value: str) -> None:
    """Replace a single-value option in a fully materialized launcher command."""
    index = cmd.index(option)
    cmd[index + 1] = str(value)


def build_mode_cmd(mode: str) -> list[str]:
    cmd = build_base_cmd()

    if mode == "baseline":
        cmd += [
            "--use_latent",
            "true",
            "--use_v1trust",
            "false",
            "--actor_updates_encoder",
            "false",
            "--critic_updates_encoder",
            "false",
            "--critic_encoder_grad_scale",
            "0.05",
            "--actor_encoder_grad_scale",
            "0.0",
            "--latent_input_scale",
            "0.1",
            "--log_dir",
            "runs/baseline",
            "--ckpt_dir",
            "checkpoints/baseline",
        ]
    elif mode == "latent_detach":
        cmd += [
            "--use_latent",
            "true",
            "--use_v1trust",
            "false",
            "--latent_only",
            "false",
            "--actor_updates_encoder",
            "false",
            "--critic_updates_encoder",
            "false",
            "--critic_encoder_grad_scale",
            "0.0",
            "--actor_encoder_grad_scale",
            "0.0",
            "--latent_input_scale",
            "0.1",
            "--log_dir",
            "runs/latent_detach",
            "--ckpt_dir",
            "checkpoints/latent_detach",
        ]
    elif mode == "v1trust":
        cmd += [
            "--use_latent",
            "true",
            "--use_v1trust",
            "true",
            "--actor_updates_encoder",
            "false",
            "--critic_updates_encoder",
            "false",
            "--critic_encoder_grad_scale",
            "0.05",
            "--actor_encoder_grad_scale",
            "0.0",
            "--latent_input_scale",
            "0.1",
            "--trust_alpha",
            "0.5",
            "--trust_beta",
            "0.5",
            "--trust_q_min",
            "0.5",
            "--trust_q_max",
            "1.0",
            "--trust_ema_momentum",
            "0.99",
            "--trust_warmup_steps",
            "10000",
            "--log_dir",
            "runs/v1trust",
            "--ckpt_dir",
            "checkpoints/v1trust",
        ]
    elif mode == "v1trust_detach_trust1":
        cmd += [
            "--use_latent",
            "true",
            "--use_v1trust",
            "true",
            "--actor_updates_encoder",
            "false",
            "--critic_updates_encoder",
            "false",
            "--critic_encoder_grad_scale",
            "0.0",
            "--actor_encoder_grad_scale",
            "0.0",
            "--latent_input_scale",
            "0.1",
            "--trust_alpha",
            "0.5",
            "--trust_beta",
            "0.5",
            "--trust_q_min",
            "1.0",
            "--trust_q_max",
            "1.0",
            "--trust_ema_momentum",
            "0.99",
            "--trust_warmup_steps",
            "0",
            "--log_dir",
            "runs/v1trust_detach_trust1",
            "--ckpt_dir",
            "checkpoints/v1trust_detach_trust1",
        ]
    elif mode == "latent_only":
        cmd += [
            "--use_latent",
            "true",
            "--use_v1trust",
            "false",
            "--latent_only",
            "true",
            "--latent_only_variant",
            "base",
            "--actor_updates_encoder",
            "false",
            "--critic_updates_encoder",
            "false",
            "--critic_encoder_grad_scale",
            "0.05",
            "--critic_encoder_grad_schedule",
            "0:0.0,50000:0.03,150000:0.05",
            "--actor_encoder_grad_scale",
            "0.0",
            "--latent_input_scale",
            "1.0",
            "--progress_loss_weight",
            "0.0",
            "--log_dir",
            "runs/latent_only_route_tree_base",
            "--ckpt_dir",
            "checkpoints/latent_only_route_tree_base",
        ]
    elif mode == "latent_only_affordance":
        cmd += [
            "--use_latent",
            "true",
            "--use_v1trust",
            "false",
            "--latent_only",
            "true",
            "--latent_only_variant",
            "affordance",
            "--actor_updates_encoder",
            "false",
            "--critic_updates_encoder",
            "false",
            "--critic_encoder_grad_scale",
            "0.05",
            "--critic_encoder_grad_schedule",
            "0:0.0,50000:0.03,150000:0.05",
            "--actor_encoder_grad_scale",
            "0.0",
            "--latent_input_scale",
            "1.0",
            "--progress_loss_weight",
            "0.0",
            "--affordance_loss_weight",
            "0.005",
            "--affordance_start_step",
            "100000",
            "--affordance_gamma",
            "0.95",
            "--affordance_bootstrap_warmup_steps",
            "50000",
            "--affordance_danger_range",
            "0.20",
            "--affordance_goal_tolerance",
            "0.20",
            "--log_dir",
            "runs/latent_only_affordance_dim16",
            "--ckpt_dir",
            "checkpoints/latent_only_affordance_dim16",
        ]
    elif mode == "hierarchical_async":
        cmd += [
            "--use_latent",
            "true",
            "--hierarchical_async",
            "true",
            "--latent_only",
            "false",
            "--use_v1trust",
            "false",
            "--latent_dim",
            "16",
            "--latent_input_scale",
            "1.0",
            "--reference_sequence_length",
            "15",
            "--high_level_interval",
            "8",
            "--reference_valid_steps",
            "120",
            "--reference_loss_weight",
            "0.05",
            "--reference_smoothness_weight",
            "0.01",
            "--high_level_actor_grad_scale",
            "0.0",
            "--hierarchical_action_l2_weight",
            "0.10",
            "--hierarchical_motor_balance_weight",
            "0.10",
            "--hierarchical_action_delta_weight",
            "0.10",
            "--hierarchical_normalize_actor_q",
            "true",
            "--motor_collective_fraction",
            "0.60",
            "--motor_differential_fraction",
            "0.25",
            "--curriculum_success_gated",
            "true",
            "--eval_episodes",
            "3",
            "--reference_lookahead_points",
            "3",
            "--action_scale",
            "0.75",
            "--log_dir",
            "runs/hierarchical_async_v2",
            "--ckpt_dir",
            "checkpoints/hierarchical_async_v2",
        ]
    elif mode == "reference_tracking":
        cmd += [
            "--use_latent",
            "false",
            "--reference_tracking",
            "true",
            "--hierarchical_async",
            "false",
            "--latent_only",
            "false",
            "--use_v1trust",
            "false",
            "--num_trees",
            "0",
            "--route_blocking_tree",
            "false",
            "--curriculum",
            "false",
            "--rule_reference_mode",
            "line",
            "--reference_sequence_length",
            "15",
            "--reference_horizon_seconds",
            "1.0",
            "--high_level_interval",
            "8",
            "--reference_lookahead_points",
            "3",
            "--max_reference_speed",
            "0.6",
            "--max_reference_acceleration",
            "1.5",
            "--max_reference_vertical_speed",
            "0.4",
            "--teacher_timesteps",
            "5000",
            "--teacher_supervision_timesteps",
            "15000",
            "--tracking_actor_rl_start_step",
            "15000",
            "--teacher_exploration_noise",
            "0.03",
            "--teacher_bc_updates_per_step",
            "2",
            "--motor_action_mode",
            "asymmetric_rpm",
            "--start_timesteps",
            "5000",
            "--update_after",
            "1000",
            "--expl_noise_start",
            "0.15",
            "--expl_noise_end",
            "0.05",
            "--noise_decay_steps",
            "50000",
            "--reward_scale",
            "0.1",
            "--action_scale",
            "1.0",
            "--eval_episodes",
            "5",
            "--log_dir",
            "runs/reference_tracking_v1",
            "--ckpt_dir",
            "checkpoints/reference_tracking_v1",
        ]
    elif mode == "upper_dslpid":
        cmd += [
            "--use_latent",
            "false",
            "--learned_upper_dslpid",
            "true",
            "--hierarchical_async",
            "false",
            "--reference_tracking",
            "false",
            "--latent_only",
            "false",
            "--use_v1trust",
            "false",
            "--num_trees",
            "24",
            "--route_blocking_tree",
            "false",
            "--curriculum",
            "false",
            "--curriculum_stage_override",
            "0",
            "--reference_sequence_length",
            "15",
            "--reference_horizon_seconds",
            "1.0",
            "--high_level_interval",
            "8",
            "--max_reference_speed",
            "0.8",
            "--max_reference_acceleration",
            "1.0",
            "--max_reference_vertical_speed",
            "0.3",
            "--max_reference_vertical_acceleration",
            "0.5",
            "--start_timesteps",
            "2000",
            "--update_after",
            "1000",
            "--expl_noise_start",
            "0.20",
            "--expl_noise_end",
            "0.05",
            "--noise_decay_steps",
            "50000",
            "--reward_scale",
            "0.1",
            "--eval_episodes",
            "5",
            "--action_scale",
            "1.0",
            "--log_dir",
            "runs/upper_dslpid_velocity_v2",
            "--ckpt_dir",
            "checkpoints/upper_dslpid_velocity_v2",
        ]
    elif mode == "upper_dslpid_recurrent":
        cmd += [
            "--use_latent", "false",
            "--learned_upper_dslpid", "true",
            "--upper_recurrent", "true",
            "--upper_recurrent_hidden_dim", "32",
            "--semantic_history_length", "8",
            "--hierarchical_async", "false",
            "--reference_tracking", "false",
            "--latent_only", "false",
            "--use_v1trust", "false",
            "--num_trees", "24",
            "--route_blocking_tree", "false",
            "--curriculum", "false",
            "--curriculum_stage_override", "0",
            "--reference_sequence_length", "15",
            "--reference_horizon_seconds", "1.0",
            "--high_level_interval", "8",
            "--max_reference_speed", "0.8",
            "--max_reference_acceleration", "1.0",
            "--max_reference_vertical_speed", "0.3",
            "--max_reference_vertical_acceleration", "0.5",
            "--start_timesteps", "2000",
            "--update_after", "1000",
            "--expl_noise_start", "0.20",
            "--expl_noise_end", "0.05",
            "--noise_decay_steps", "50000",
            "--reward_scale", "0.1",
            "--eval_episodes", "5",
            "--action_scale", "1.0",
            "--log_dir", "runs/upper_dslpid_recurrent_td3_v1",
            "--ckpt_dir", "checkpoints/upper_dslpid_recurrent_td3_v1",
            "--trajectory_output_dir", "analysis_outputs/upper_dslpid_recurrent_trajectory",
        ]
    elif mode == "upper_dslpid_recurrent_safe_v2":
        cmd += [
            "--use_latent", "false",
            "--learned_upper_dslpid", "true",
            "--upper_recurrent", "true",
            "--upper_recurrent_hidden_dim", "32",
            "--semantic_history_length", "8",
            "--hierarchical_async", "false",
            "--reference_tracking", "false",
            "--latent_only", "false",
            "--use_v1trust", "false",
            "--num_trees", "24",
            "--route_blocking_tree", "false",
            "--curriculum", "false",
            "--curriculum_stage_override", "0",
            "--reference_sequence_length", "15",
            "--reference_horizon_seconds", "1.0",
            "--high_level_interval", "8",
            "--max_reference_speed", "0.8",
            "--max_reference_acceleration", "1.0",
            "--max_reference_vertical_speed", "0.3",
            "--max_reference_vertical_acceleration", "0.5",
            "--start_timesteps", "5000",
            "--update_after", "5000",
            "--expl_noise_start", "0.20",
            "--expl_noise_end", "0.05",
            "--noise_decay_steps", "100000",
            "--reward_scale", "0.1",
            "--reward_variant", "safe_terminal_v2",
            "--time_penalty_weight", "0.01",
            "--proximity_penalty_weight", "5.0",
            "--near_goal_radius", "0.8",
            "--near_goal_speed_penalty_weight", "1.5",
            "--near_goal_away_penalty_weight", "3.0",
            "--route_overshoot_penalty_weight", "1.0",
            "--eval_episodes", "5",
            "--action_scale", "1.0",
            "--log_dir", "runs/upper_dslpid_recurrent_safe_v2",
            "--ckpt_dir", "checkpoints/upper_dslpid_recurrent_safe_v2",
            "--trajectory_output_dir", "analysis_outputs/upper_dslpid_recurrent_safe_v2_trajectory",
        ]
    elif mode == "upper_dslpid_recurrent_stable_v3":
        cmd += [
            "--use_latent", "false",
            "--learned_upper_dslpid", "true",
            "--upper_recurrent", "true",
            "--upper_recurrent_hidden_dim", "32",
            "--semantic_history_length", "8",
            "--hierarchical_async", "false",
            "--reference_tracking", "false",
            "--latent_only", "false",
            "--use_v1trust", "false",
            "--num_trees", "24",
            "--route_blocking_tree", "false",
            "--curriculum", "false",
            "--curriculum_stage_override", "0",
            "--reference_sequence_length", "15",
            "--reference_horizon_seconds", "1.0",
            "--high_level_interval", "8",
            "--max_reference_speed", "0.8",
            "--max_reference_acceleration", "1.0",
            "--max_reference_vertical_speed", "0.3",
            "--max_reference_vertical_acceleration", "0.5",
            "--start_timesteps", "5000",
            "--update_after", "5000",
            "--expl_noise_start", "0.20",
            "--expl_noise_end", "0.05",
            "--noise_decay_steps", "100000",
            "--policy_noise", "0.20",
            "--policy_freq", "2",
            "--upper_actor_lr", "0.0003",
            "--upper_critic_lr", "0.0003",
            "--upper_conservative_actor_lr", "0.0001",
            "--upper_conservative_critic_lr", "0.0001",
            "--upper_conservative_policy_noise", "0.10",
            "--upper_conservative_policy_freq", "3",
            "--success_replay_fraction", "0.35",
            "--success_replay_size", "50000",
            "--policy_anchor_weight", "0.10",
            "--convergence_guard", "true",
            "--guard_min_success_rate", "0.80",
            "--guard_max_collision_rate", "0.20",
            "--guard_patience_evals", "3",
            "--reward_scale", "0.1",
            "--reward_variant", "safe_terminal_v2",
            "--time_penalty_weight", "0.01",
            "--proximity_penalty_weight", "5.0",
            "--near_goal_radius", "0.8",
            "--near_goal_speed_penalty_weight", "1.5",
            "--near_goal_away_penalty_weight", "3.0",
            "--route_overshoot_penalty_weight", "1.0",
            "--eval_episodes", "5",
            "--action_scale", "1.0",
            "--log_dir", "runs/upper_dslpid_recurrent_stable_v3",
            "--ckpt_dir", "checkpoints/upper_dslpid_recurrent_stable_v3",
            "--trajectory_output_dir", "analysis_outputs/upper_dslpid_recurrent_stable_v3_trajectory",
        ]
    elif mode == "upper_dslpid_recurrent_multitree_v4":
        cmd += [
            "--use_latent", "false",
            "--learned_upper_dslpid", "true",
            "--upper_recurrent", "true",
            "--upper_recurrent_hidden_dim", "64",
            "--semantic_history_length", "4",
            "--hierarchical_async", "false",
            "--reference_tracking", "false",
            "--latent_only", "false",
            "--use_v1trust", "false",
            "--map_id", "procedural_forest_multitree_train_v2",
            "--eval_map_id", "procedural_forest_validation_v1",
            "--range_azimuth_count", "32",
            "--range_elevation_degrees", "-30", "0", "30",
            "--curriculum", "true",
            "--curriculum_milestones", "100", "250", "450",
            "--curriculum_success_gated", "false",
            "--route_count_curriculum", "true",
            "--route_count_curriculum_counts", "1", "2", "3", "5",
            "--curriculum_simple_mix_probability", "0.20",
            "--reference_sequence_length", "15",
            "--reference_horizon_seconds", "1.0",
            "--high_level_interval", "8",
            "--max_reference_speed", "0.8",
            "--max_reference_acceleration", "1.0",
            "--max_reference_vertical_speed", "0.3",
            "--max_reference_vertical_acceleration", "0.5",
            "--buffer_size", "150000",
            "--start_timesteps", "5000",
            "--update_after", "5000",
            "--expl_noise_start", "0.20",
            "--expl_noise_end", "0.05",
            "--noise_decay_steps", "100000",
            "--policy_noise", "0.20",
            "--policy_freq", "2",
            "--upper_actor_lr", "0.0003",
            "--upper_critic_lr", "0.0003",
            "--upper_conservative_actor_lr", "0.0001",
            "--upper_conservative_critic_lr", "0.0001",
            "--upper_conservative_policy_noise", "0.10",
            "--upper_conservative_policy_freq", "3",
            "--success_replay_fraction", "0.35",
            "--success_replay_size", "40000",
            "--success_replay_stratified", "true",
            "--success_replay_difficulties", "1", "2", "3", "5",
            "--policy_anchor_weight", "0.10",
            "--convergence_guard", "true",
            "--guard_min_success_rate", "0.60",
            "--guard_max_collision_rate", "0.40",
            "--guard_patience_evals", "4",
            "--reward_scale", "0.1",
            "--reward_variant", "outcome_separated_v3",
            "--distance_reward_weight", "0.0",
            "--time_penalty_weight", "0.01",
            "--proximity_penalty_weight", "5.0",
            "--near_goal_radius", "0.8",
            "--near_goal_speed_penalty_weight", "1.5",
            "--near_goal_away_penalty_weight", "3.0",
            "--route_overshoot_penalty_weight", "1.0",
            "--success_terminal_reward", "150.0",
            "--collision_terminal_penalty", "120.0",
            "--bound_terminal_penalty", "100.0",
            "--timeout_terminal_penalty", "30.0",
            "--eval_interval", "2000",
            "--eval_episodes", "10",
            "--best_window_size", "2",
            "--best_min_success_rate", "0.80",
            "--best_max_collision_rate", "0.20",
            "--action_scale", "1.0",
            "--log_dir", "runs/upper_dslpid_recurrent_multitree_v4",
            "--ckpt_dir", "checkpoints/upper_dslpid_recurrent_multitree_v4",
            "--trajectory_output_dir", "analysis_outputs/upper_dslpid_recurrent_multitree_v4_trajectory",
        ]
    elif mode == "upper_dslpid_recurrent_multitree_v4_1":
        cmd += [
            "--use_latent", "false",
            "--learned_upper_dslpid", "true",
            "--upper_recurrent", "true",
            "--upper_recurrent_hidden_dim", "64",
            "--semantic_history_length", "4",
            "--hierarchical_async", "false",
            "--reference_tracking", "false",
            "--latent_only", "false",
            "--use_v1trust", "false",
            "--map_id", "procedural_forest_multitree_train_v2",
            "--eval_map_id", "procedural_forest_validation_v1",
            "--range_azimuth_count", "32",
            "--range_elevation_degrees", "-30", "0", "30",
            "--curriculum", "true",
            "--curriculum_milestones", "100", "250", "450",
            "--curriculum_success_gated", "true",
            "--curriculum_success_window", "100",
            "--curriculum_minimum_stage_episodes", "100",
            "--curriculum_success_thresholds", "0.60", "0.50", "0.35",
            "--curriculum_max_collision_rates", "0.30", "0.35", "0.40",
            "--route_count_curriculum", "true",
            "--route_count_curriculum_counts", "1", "2", "3", "5",
            "--curriculum_simple_mix_probability", "0.20",
            "--reference_sequence_length", "15",
            "--reference_horizon_seconds", "1.0",
            "--high_level_interval", "8",
            "--max_reference_speed", "0.8",
            "--max_reference_acceleration", "1.0",
            "--max_reference_vertical_speed", "0.3",
            "--max_reference_vertical_acceleration", "0.5",
            "--buffer_size", "60000",
            "--start_timesteps", "5000",
            "--update_after", "5000",
            "--expl_noise_start", "0.20",
            "--expl_noise_end", "0.05",
            "--noise_decay_steps", "100000",
            "--policy_noise", "0.20",
            "--policy_freq", "2",
            "--upper_actor_lr", "0.0003",
            "--upper_critic_lr", "0.0003",
            "--upper_conservative_actor_lr", "0.0001",
            "--upper_conservative_critic_lr", "0.0001",
            "--upper_conservative_policy_noise", "0.10",
            "--upper_conservative_policy_freq", "3",
            "--success_replay_fraction", "0.20",
            "--success_replay_size", "20000",
            "--success_replay_stratified", "true",
            "--success_replay_difficulties", "1", "2", "3", "5",
            "--policy_anchor_weight", "0.10",
            "--convergence_guard", "true",
            "--guard_min_success_rate", "0.60",
            "--guard_max_collision_rate", "0.40",
            "--guard_patience_evals", "4",
            "--reward_scale", "0.1",
            "--reward_variant", "outcome_separated_v3",
            "--distance_reward_weight", "0.0",
            "--time_penalty_weight", "0.01",
            "--proximity_penalty_weight", "5.0",
            "--near_goal_radius", "0.8",
            "--near_goal_speed_penalty_weight", "1.5",
            "--near_goal_away_penalty_weight", "3.0",
            "--route_overshoot_penalty_weight", "1.0",
            "--success_terminal_reward", "150.0",
            "--collision_terminal_penalty", "120.0",
            "--bound_terminal_penalty", "100.0",
            "--timeout_terminal_penalty", "30.0",
            "--eval_interval", "2000",
            "--eval_episodes", "10",
            "--best_window_size", "2",
            "--best_min_success_rate", "0.80",
            "--best_max_collision_rate", "0.20",
            "--action_scale", "1.0",
            "--log_dir", "runs/upper_dslpid_recurrent_multitree_v4_1",
            "--ckpt_dir", "checkpoints/upper_dslpid_recurrent_multitree_v4_1",
            "--trajectory_output_dir", "analysis_outputs/upper_dslpid_recurrent_multitree_v4_1_trajectory",
        ]
    elif mode in {
        "upper_dslpid_recurrent_multitree_v4_2_h4",
        "upper_dslpid_recurrent_multitree_v4_2_h8",
        "upper_dslpid_recurrent_multitree_v4_2_h16",
    }:
        # V4.2 is a strict history-window ablation over V4.1.  Reusing the
        # materialized command prevents an accidental change to task, replay,
        # curriculum, GRU width, or evaluation settings.
        history_length = mode.rsplit("h", 1)[1]
        suffix = f"upper_dslpid_recurrent_multitree_v4_2_h{history_length}"
        cmd = build_mode_cmd("upper_dslpid_recurrent_multitree_v4_1")
        _replace_scalar_option(cmd, "--semantic_history_length", history_length)
        _replace_scalar_option(cmd, "--log_dir", f"runs/{suffix}")
        _replace_scalar_option(cmd, "--ckpt_dir", f"checkpoints/{suffix}")
        _replace_scalar_option(
            cmd,
            "--trajectory_output_dir",
            f"analysis_outputs/{suffix}_trajectory",
        )
    elif mode in {
        "upper_dslpid_recurrent_multitree_v4_3_h4",
        "upper_dslpid_recurrent_multitree_v4_3_h8",
    }:
        # V4.3 keeps the V4.2 h4/h8 architecture and curriculum fixed while
        # repairing reward time units and outcome ordering.  Separate output
        # directories preserve the V4.2 history-window ablation unchanged.
        history_length = mode.rsplit("h", 1)[1]
        suffix = f"upper_dslpid_recurrent_multitree_v4_3_h{history_length}"
        cmd = build_mode_cmd("upper_dslpid_recurrent_multitree_v4_1")
        _replace_scalar_option(cmd, "--semantic_history_length", history_length)
        _replace_scalar_option(cmd, "--reward_variant", "rate_integrated_v4")
        _replace_scalar_option(cmd, "--time_penalty_weight", "0.25")
        _replace_scalar_option(cmd, "--proximity_penalty_weight", "30.0")
        _replace_scalar_option(cmd, "--near_goal_speed_penalty_weight", "4.0")
        _replace_scalar_option(cmd, "--near_goal_away_penalty_weight", "8.0")
        _replace_scalar_option(cmd, "--route_overshoot_penalty_weight", "5.0")
        _replace_scalar_option(cmd, "--success_terminal_reward", "250.0")
        _replace_scalar_option(cmd, "--collision_terminal_penalty", "300.0")
        _replace_scalar_option(cmd, "--bound_terminal_penalty", "250.0")
        _replace_scalar_option(cmd, "--timeout_terminal_penalty", "180.0")
        cmd += [
            "--route_progress_reward_weight", "12.0",
            "--goal_progress_reward_weight", "8.0",
            "--lateral_penalty_scale", "0.0",
        ]
        _replace_scalar_option(cmd, "--log_dir", f"runs/{suffix}")
        _replace_scalar_option(cmd, "--ckpt_dir", f"checkpoints/{suffix}")
        _replace_scalar_option(
            cmd,
            "--trajectory_output_dir",
            f"analysis_outputs/{suffix}_trajectory",
        )
    elif mode in {
        "upper_dslpid_recurrent_multitree_v4_3_h8_k0",
        "upper_dslpid_recurrent_multitree_v4_3_h8_k1",
    }:
        # Upper-only control-variable benchmark.  This intentionally derives
        # from the exact h8 V4.3 command and changes only action->state
        # transition plus output directories.
        transition = mode.rsplit("_", 1)[1]
        suffix = f"upper_dslpid_recurrent_multitree_v4_3_h8_{transition}"
        cmd = build_mode_cmd("upper_dslpid_recurrent_multitree_v4_3_h8")
        cmd += [
            "--upper_execution_model",
            transition,
            "--log_dir",
            f"runs/{suffix}",
            "--ckpt_dir",
            f"checkpoints/{suffix}",
            "--trajectory_output_dir",
            f"analysis_outputs/{suffix}_trajectory",
            "--collision_diagnostic_output_dir",
            f"analysis_outputs/{suffix}_collision_windows",
        ]
    elif mode == "upper_dslpid_recurrent_multitree_v4_3_h8_k1_rayfix":
        # Sensor-only ablation: the command is exactly K1 V4.3 (including its
        # reward), while the environment uses the corrected self-hit-free 3-D
        # ray implementation. Separate directories preserve the old K1 run.
        suffix = "upper_dslpid_recurrent_multitree_v4_3_h8_k1_rayfix"
        cmd = build_mode_cmd("upper_dslpid_recurrent_multitree_v4_3_h8_k1")
        _replace_scalar_option(cmd, "--log_dir", f"runs/{suffix}")
        _replace_scalar_option(cmd, "--ckpt_dir", f"checkpoints/{suffix}")
        _replace_scalar_option(
            cmd,
            "--trajectory_output_dir",
            f"analysis_outputs/{suffix}_trajectory",
        )
        _replace_scalar_option(
            cmd,
            "--collision_diagnostic_output_dir",
            f"analysis_outputs/{suffix}_collision_windows",
        )
    elif mode in {
        "upper_dslpid_recurrent_multitree_rayfix_v1_h8_full",
        "upper_dslpid_recurrent_multitree_rayfix_v1_h8_k1",
    }:
        # Clean post-rayfix baselines. They intentionally preserve the entire
        # V4.3 task/reward contract and differ only in action-to-state execution.
        execution = mode.rsplit("_", 1)[1]
        suffix = f"upper_dslpid_recurrent_multitree_rayfix_v1_h8_{execution}"
        # Both arms derive from the same single-output Full-h8 command.  Do
        # not derive K1 from the legacy K1 launcher: it contains a second set
        # of historical output options and would make the clean comparison
        # command ambiguous even though argparse uses the final occurrence.
        cmd = build_mode_cmd("upper_dslpid_recurrent_multitree_v4_3_h8")
        cmd += ["--upper_execution_model", execution]
        _replace_scalar_option(cmd, "--log_dir", f"runs/{suffix}")
        _replace_scalar_option(cmd, "--ckpt_dir", f"checkpoints/{suffix}")
        _replace_scalar_option(
            cmd, "--trajectory_output_dir", f"analysis_outputs/{suffix}_trajectory"
        )
        if "--collision_diagnostic_output_dir" in cmd:
            _replace_scalar_option(
                cmd,
                "--collision_diagnostic_output_dir",
                f"analysis_outputs/{suffix}_collision_windows",
            )
        else:
            cmd += [
                "--collision_diagnostic_output_dir",
                f"analysis_outputs/{suffix}_collision_windows",
            ]
        cmd += [
            "--timeout_diagnostic_output_dir",
            f"analysis_outputs/{suffix}_timeout_diagnostics",
        ]
    elif mode == "upper_dslpid_recurrent_multitree_terminal_v1_h8_k1":
        # Evidence-driven terminal branch. The V4.3 reward and all task
        # settings stay frozen; only the planar action representation is made
        # identical for actor, exploration, replay, target policy, and plant.
        suffix = "upper_dslpid_recurrent_multitree_terminal_v1_h8_k1"
        cmd = build_mode_cmd(
            "upper_dslpid_recurrent_multitree_rayfix_v1_h8_k1"
        )
        cmd += ["--upper_project_planar_action", "true"]
        _replace_scalar_option(cmd, "--log_dir", f"runs/{suffix}")
        _replace_scalar_option(cmd, "--ckpt_dir", f"checkpoints/{suffix}")
        _replace_scalar_option(
            cmd, "--trajectory_output_dir", f"analysis_outputs/{suffix}_trajectory"
        )
        _replace_scalar_option(
            cmd,
            "--collision_diagnostic_output_dir",
            f"analysis_outputs/{suffix}_collision_windows",
        )
        _replace_scalar_option(
            cmd,
            "--timeout_diagnostic_output_dir",
            f"analysis_outputs/{suffix}_timeout_diagnostics",
        )
    elif mode == "upper_dslpid_recurrent_multitree_terminal_v2_h8_k1":
        # Controlled reward ablation over terminal-v1.  The action projection,
        # observation, K1 dynamics, maps, seeds, curriculum and task terminal
        # conditions are unchanged.  The scalar near-goal speed costs are
        # replaced by a directed 3-D terminal velocity target.
        suffix = "upper_dslpid_recurrent_multitree_terminal_v2_h8_k1"
        cmd = build_mode_cmd(
            "upper_dslpid_recurrent_multitree_terminal_v1_h8_k1"
        )
        _replace_scalar_option(cmd, "--reward_variant", "terminal_guidance_v5")
        _replace_scalar_option(cmd, "--near_goal_speed_penalty_weight", "0.0")
        _replace_scalar_option(cmd, "--near_goal_away_penalty_weight", "0.0")
        cmd += [
            "--near_goal_guidance_radius", "1.2",
            "--near_goal_guidance_ramp_width", "0.2",
            "--near_goal_capture_speed", "0.1",
            "--near_goal_cruise_speed", "0.8",
            "--near_goal_velocity_tracking_weight", "160.0",
        ]
        _replace_scalar_option(cmd, "--log_dir", f"runs/{suffix}")
        _replace_scalar_option(cmd, "--ckpt_dir", f"checkpoints/{suffix}")
        _replace_scalar_option(
            cmd, "--trajectory_output_dir", f"analysis_outputs/{suffix}_trajectory"
        )
        _replace_scalar_option(
            cmd,
            "--collision_diagnostic_output_dir",
            f"analysis_outputs/{suffix}_collision_windows",
        )
        _replace_scalar_option(
            cmd,
            "--timeout_diagnostic_output_dir",
            f"analysis_outputs/{suffix}_timeout_diagnostics",
        )
    elif mode == "upper_dslpid_recurrent_multitree_terminal_v2_noroute_h8_k1":
        # One learning-parameter ablation: retain 3-D goal progress and terminal
        # guidance, remove the auxiliary horizontal route progress reward.
        suffix = mode
        cmd = build_mode_cmd("upper_dslpid_recurrent_multitree_terminal_v2_h8_k1")
        _replace_scalar_option(cmd, "--route_progress_reward_weight", "0.0")
        for option, value in (
            ("--log_dir", f"runs/{suffix}"),
            ("--ckpt_dir", f"checkpoints/{suffix}"),
            ("--trajectory_output_dir", f"analysis_outputs/{suffix}_trajectory"),
            ("--collision_diagnostic_output_dir", f"analysis_outputs/{suffix}_collision_windows"),
            ("--timeout_diagnostic_output_dir", f"analysis_outputs/{suffix}_timeout_diagnostics"),
        ):
            _replace_scalar_option(cmd, option, value)
        cmd += ["--eval_terminal_coverage_output_dir",
                f"analysis_outputs/{suffix}_terminal_coverage"]
    elif mode == "upper_dslpid_recurrent_multitree_terminal_v2_noroute_startcurr_v1_h8_k1":
        # Training-only start distribution ablation over the unchanged no_route mode.
        suffix = mode
        cmd = build_mode_cmd("upper_dslpid_recurrent_multitree_terminal_v2_noroute_h8_k1")
        cmd += ["--start_distance_curriculum", "true", "--curriculum_stage_override", "0"]
        _replace_scalar_option(
            cmd, "--eval_terminal_coverage_output_dir",
            f"analysis_outputs/{suffix}_terminal_coverage",
        )
        for option, value in (
            ("--log_dir", f"runs/{suffix}"),
            ("--ckpt_dir", f"checkpoints/{suffix}"),
            ("--trajectory_output_dir", f"analysis_outputs/{suffix}_trajectory"),
            ("--collision_diagnostic_output_dir", f"analysis_outputs/{suffix}_collision_windows"),
            ("--timeout_diagnostic_output_dir", f"analysis_outputs/{suffix}_timeout_diagnostics"),
        ):
            _replace_scalar_option(cmd, option, value)
    elif mode == "upper_dslpid_recurrent_multitree_terminal_v2_noroute_startobscurr_v1_h8_k1":
        # Preserve the successful reverse-start course and all TD3/reward/action
        # settings; unlock randomized obstacle stages only after full starts.
        suffix = mode
        cmd = build_mode_cmd(
            "upper_dslpid_recurrent_multitree_terminal_v2_noroute_startcurr_v1_h8_k1"
        )
        cmd += ["--obstacle_curriculum_after_start", "true"]
        for option, value in (
            ("--curriculum_success_gated", "false"),
            ("--eval_map_id", "procedural_forest_multitree_validation_random_v1"),
            ("--eval_episodes", "20"),
            ("--eval_interval", "4000"),
            ("--best_min_success_rate", "0.60"),
            ("--convergence_guard", "false"),
            ("--total_steps", "240000"),
            ("--log_dir", f"runs/{suffix}"),
            ("--ckpt_dir", f"checkpoints/{suffix}"),
            ("--trajectory_output_dir", f"analysis_outputs/{suffix}_trajectory"),
            ("--collision_diagnostic_output_dir", f"analysis_outputs/{suffix}_collision_windows"),
            ("--timeout_diagnostic_output_dir", f"analysis_outputs/{suffix}_timeout_diagnostics"),
            ("--eval_terminal_coverage_output_dir", f"analysis_outputs/{suffix}_terminal_coverage"),
        ):
            _replace_scalar_option(cmd, option, value)
    elif mode == "full_scratch_curr_v1":
        cmd = build_mode_cmd(
            "upper_dslpid_recurrent_multitree_terminal_v2_noroute_startobscurr_v1_h8_k1"
        )
        cmd += ["--full_scratch_curr_v1", "true", "--ckpt_interval", "5000"]
        for option, value in (
            ("--upper_execution_model", "full"),
            ("--curriculum_simple_mix_probability", "0.0"),
            ("--eval_interval", "5000"),
            ("--log_dir", f"runs/{mode}"),
            ("--ckpt_dir", f"checkpoints/{mode}"),
            ("--trajectory_output_dir", f"analysis_outputs/{mode}_trajectory"),
            ("--collision_diagnostic_output_dir", f"analysis_outputs/{mode}_collision_windows"),
            ("--timeout_diagnostic_output_dir", f"analysis_outputs/{mode}_timeout_diagnostics"),
            ("--eval_terminal_coverage_output_dir", f"analysis_outputs/{mode}_terminal_coverage"),
        ):
            _replace_scalar_option(cmd, option, value)
    elif mode == "full_scratch_ego20x10_curr_v1":
        cmd = build_mode_cmd("full_scratch_curr_v1")
        _replace_scalar_option(cmd, "--full_scratch_curr_v1", "false")
        cmd += ["--full_scratch_ego20x10_curr_v1", "true"]
        for option, value in (
            ("--curriculum", "false"), ("--route_count_curriculum", "false"),
            ("--eval_episodes", "20"),
            ("--log_dir", f"runs/{mode}"), ("--ckpt_dir", f"checkpoints/{mode}"),
            ("--trajectory_output_dir", f"analysis_outputs/{mode}_trajectory"),
            ("--collision_diagnostic_output_dir", f"analysis_outputs/{mode}_collision_windows"),
            ("--timeout_diagnostic_output_dir", f"analysis_outputs/{mode}_timeout_diagnostics"),
            ("--eval_terminal_coverage_output_dir", f"analysis_outputs/{mode}_terminal_coverage"),
        ):
            _replace_scalar_option(cmd, option, value)
        for option in ("--map_id", "--eval_map_id"):
            if option in cmd:
                index = cmd.index(option)
                del cmd[index:index + 2]
        index = cmd.index("--success_replay_difficulties")
        cmd[index + 1:index + 5] = ["10", "20", "30", "40"]
    elif mode == "full_ft_v1":
        # Actor-only K1 -> Full adaptation. Stage mixing and validation are
        # handled by the existing TD3 trainer and evaluation callback.
        suffix = mode
        cmd = build_mode_cmd("upper_dslpid_recurrent_multitree_terminal_v2_noroute_h8_k1")
        cmd += [
            "--full_ft_v1", "true",
            "--full_ft_actor_source", "checkpoints/startobscurr_v1_selected_20261010/seed43/model_final",
            "--full_ft_actor_freeze_steps", "4000",
            "--full_ft_trace_dir", "analysis_outputs/full_ft_v1_seed43/training_traces",
            "--ckpt_interval", "5000",
        ]
        for option, value in (
            ("--upper_execution_model", "full"),
            ("--start_timesteps", "0"),
            ("--update_after", "512"),
            ("--curriculum_simple_mix_probability", "0.0"),
            ("--eval_map_id", "procedural_forest_multitree_validation_random_v1"),
            ("--eval_interval", "5000"),
            ("--eval_episodes", "10"),
            ("--total_steps", "40000"),
            ("--best_window_size", "2"),
            ("--best_min_success_rate", "0.40"),
            ("--best_max_collision_rate", "0.60"),
            ("--policy_anchor_weight", "0.0"),
            ("--expl_noise_start", "0.15"),
            ("--expl_noise_end", "0.05"),
            ("--noise_decay_steps", "40000"),
            ("--convergence_guard", "false"),
            ("--log_dir", f"runs/{suffix}"),
            ("--ckpt_dir", f"checkpoints/{suffix}"),
            ("--trajectory_output_dir", f"analysis_outputs/{suffix}_trajectory"),
            ("--collision_diagnostic_output_dir", f"analysis_outputs/{suffix}_collision_windows"),
            ("--timeout_diagnostic_output_dir", f"analysis_outputs/{suffix}_timeout_diagnostics"),
            ("--eval_terminal_coverage_output_dir", f"analysis_outputs/{suffix}_terminal_coverage"),
        ):
            _replace_scalar_option(cmd, option, value)
    elif mode == "upper_dslpid_recurrent_multitree_terminal_v2_noroute_stage01_ft_v1_h8_k1":
        # Warm-start from the four frozen successful policies. Train on a
        # 50/50 Stage 0/1 mixture; evaluate each stage separately.
        suffix = mode
        cmd = build_mode_cmd("upper_dslpid_recurrent_multitree_terminal_v2_noroute_h8_k1")
        cmd += [
            "--stage01_finetune", "true",
            "--stage01_mix_probability", "0.5",
            "--curriculum_stage_override", "1",
            "--load_checkpoint", "checkpoints/startcurr_v1_frozen_20261009/seed42/model_final",
        ]
        for option, value in (
            ("--start_timesteps", "0"),
            ("--eval_map_id", "procedural_forest_multitree_validation_random_v1"),
            ("--eval_episodes", "20"),
            ("--eval_interval", "4000"),
            ("--best_window_size", "2"),
            ("--total_steps", "40000"),
            ("--log_dir", f"runs/{suffix}"),
            ("--ckpt_dir", f"checkpoints/{suffix}"),
            ("--trajectory_output_dir", f"analysis_outputs/{suffix}_trajectory"),
            ("--collision_diagnostic_output_dir", f"analysis_outputs/{suffix}_collision_windows"),
            ("--timeout_diagnostic_output_dir", f"analysis_outputs/{suffix}_timeout_diagnostics"),
            ("--eval_terminal_coverage_output_dir", f"analysis_outputs/{suffix}_terminal_coverage"),
        ):
            _replace_scalar_option(cmd, option, value)
    elif mode == "upper_dslpid_recurrent_multitree_occupancy_v1_h8_k1":
        # Independent spatial-memory ablation over the clean rayfix K1
        # baseline. Reward and executable action semantics remain unchanged.
        suffix = "upper_dslpid_recurrent_multitree_occupancy_v1_h8_k1"
        cmd = build_mode_cmd(
            "upper_dslpid_recurrent_multitree_rayfix_v1_h8_k1"
        )
        cmd += [
            "--use_local_occupancy", "true",
            "--local_occupancy_voxel_size", "0.4",
            "--local_occupancy_shape", "12", "12", "6",
        ]
        _replace_scalar_option(cmd, "--log_dir", f"runs/{suffix}")
        _replace_scalar_option(cmd, "--ckpt_dir", f"checkpoints/{suffix}")
        _replace_scalar_option(
            cmd, "--trajectory_output_dir", f"analysis_outputs/{suffix}_trajectory"
        )
        _replace_scalar_option(
            cmd,
            "--collision_diagnostic_output_dir",
            f"analysis_outputs/{suffix}_collision_windows",
        )
        _replace_scalar_option(
            cmd,
            "--timeout_diagnostic_output_dir",
            f"analysis_outputs/{suffix}_timeout_diagnostics",
        )
    elif mode == "upper_dslpid_recurrent_mavrl_lite_v1":
        cmd += [
            "--use_latent", "false",
            "--learned_upper_dslpid", "true",
            "--upper_recurrent", "true",
            "--upper_recurrent_hidden_dim", "32",
            "--semantic_history_length", "8",
            "--upper_planar_only", "true",
            "--upper_height_hold_gain", "1.0",
            "--hierarchical_async", "false",
            "--reference_tracking", "false",
            "--latent_only", "false",
            "--use_v1trust", "false",
            "--num_trees", "24",
            "--route_blocking_tree", "false",
            "--curriculum", "false",
            "--curriculum_stage_override", "0",
            "--reference_sequence_length", "15",
            "--reference_horizon_seconds", "1.0",
            "--high_level_interval", "8",
            "--max_reference_speed", "0.8",
            "--max_reference_acceleration", "1.0",
            "--max_reference_vertical_speed", "0.3",
            "--max_reference_vertical_acceleration", "0.5",
            "--start_timesteps", "5000",
            "--update_after", "5000",
            "--expl_noise_start", "0.20",
            "--expl_noise_end", "0.05",
            "--noise_decay_steps", "100000",
            "--policy_noise", "0.20",
            "--policy_freq", "2",
            "--upper_actor_lr", "0.0003",
            "--upper_critic_lr", "0.0003",
            "--upper_conservative_actor_lr", "0.0001",
            "--upper_conservative_critic_lr", "0.0001",
            "--upper_conservative_policy_noise", "0.10",
            "--upper_conservative_policy_freq", "3",
            "--success_replay_fraction", "0.35",
            "--success_replay_size", "50000",
            "--policy_anchor_weight", "0.10",
            "--convergence_guard", "true",
            "--guard_min_success_rate", "0.80",
            "--guard_max_collision_rate", "0.20",
            "--guard_patience_evals", "3",
            "--reward_scale", "0.1",
            "--reward_variant", "mavrl_safe_speed_v1",
            "--speed_penalty_weight", "0.0",
            "--time_penalty_weight", "0.01",
            "--proximity_penalty_weight", "5.0",
            "--near_goal_radius", "0.8",
            "--near_goal_speed_penalty_weight", "1.5",
            "--near_goal_away_penalty_weight", "3.0",
            "--route_overshoot_penalty_weight", "1.0",
            "--safe_speed_min", "0.0",
            "--safe_speed_brake_clearance", "0.35",
            "--safe_speed_open_clearance", "1.20",
            "--unsafe_speed_penalty_weight", "3.0",
            "--eval_episodes", "5",
            "--action_scale", "1.0",
            "--log_dir", "runs/upper_dslpid_recurrent_mavrl_lite_v1",
            "--ckpt_dir", "checkpoints/upper_dslpid_recurrent_mavrl_lite_v1",
            "--trajectory_output_dir", "analysis_outputs/upper_dslpid_recurrent_mavrl_lite_v1_trajectory",
        ]
    elif mode == "upper_dslpid_one_tree":
        cmd += [
            "--use_latent",
            "false",
            "--learned_upper_dslpid",
            "true",
            "--hierarchical_async",
            "false",
            "--reference_tracking",
            "false",
            "--latent_only",
            "false",
            "--use_v1trust",
            "false",
            "--num_trees",
            "24",
            "--route_blocking_tree",
            "true",
            "--route_tree_fraction",
            "0.5",
            "--curriculum",
            "false",
            "--curriculum_stage_override",
            "0",
            "--reference_sequence_length",
            "15",
            "--reference_horizon_seconds",
            "1.0",
            "--high_level_interval",
            "8",
            "--max_reference_speed",
            "0.8",
            "--max_reference_acceleration",
            "1.0",
            "--max_reference_vertical_speed",
            "0.3",
            "--max_reference_vertical_acceleration",
            "0.5",
            "--start_timesteps",
            "0",
            "--update_after",
            "1000",
            "--expl_noise_start",
            "0.20",
            "--expl_noise_end",
            "0.05",
            "--noise_decay_steps",
            "50000",
            "--reward_scale",
            "0.1",
            "--eval_episodes",
            "5",
            "--action_scale",
            "1.0",
            "--load_checkpoint",
            "checkpoints/upper_dslpid_velocity_v2/baseline_clear_route/model_best",
            "--log_dir",
            "runs/upper_dslpid_one_tree_v1",
            "--ckpt_dir",
            "checkpoints/upper_dslpid_one_tree_v1",
        ]
    elif mode == "upper_dslpid_five_trees":
        cmd += [
            "--use_latent", "false",
            "--learned_upper_dslpid", "true",
            "--hierarchical_async", "false",
            "--reference_tracking", "false",
            "--latent_only", "false",
            "--use_v1trust", "false",
            "--num_trees", "0",
            "--route_blocking_tree", "true",
            "--route_blocking_tree_count", "5",
            "--route_tree_lateral_range", "0.55",
            "--route_tree_layout", "fixed_safe_five",
            "--curriculum", "false",
            "--curriculum_stage_override", "0",
            "--reference_sequence_length", "15",
            "--reference_horizon_seconds", "1.0",
            "--high_level_interval", "8",
            "--max_reference_speed", "0.8",
            "--max_reference_acceleration", "1.0",
            "--max_reference_vertical_speed", "0.3",
            "--max_reference_vertical_acceleration", "0.5",
            "--start_timesteps", "0",
            "--update_after", "1000",
            "--expl_noise_start", "0.20",
            "--expl_noise_end", "0.05",
            "--noise_decay_steps", "50000",
            "--reward_scale", "0.1",
            "--distance_reward_weight", "0.0",
            "--time_penalty_weight", "0.01",
            "--eval_episodes", "10",
            "--action_scale", "1.0",
            "--load_checkpoint", "checkpoints/upper_dslpid_one_tree_v1/model_final",
            "--log_dir", "runs/upper_dslpid_five_trees_fixed_v2",
            "--ckpt_dir", "checkpoints/upper_dslpid_five_trees_fixed_v2",
        ]
    elif mode == "upper_dslpid_five_trees_forest24":
        cmd += [
            "--use_latent", "false",
            "--learned_upper_dslpid", "true",
            "--hierarchical_async", "false",
            "--reference_tracking", "false",
            "--latent_only", "false",
            "--use_v1trust", "false",
            "--num_trees", "24",
            "--route_blocking_tree", "true",
            "--route_blocking_tree_count", "5",
            "--route_tree_layout", "fixed_safe_five",
            "--curriculum", "false",
            "--curriculum_stage_override", "0",
            "--reference_sequence_length", "15",
            "--reference_horizon_seconds", "1.0",
            "--high_level_interval", "8",
            "--max_reference_speed", "0.8",
            "--max_reference_acceleration", "1.0",
            "--max_reference_vertical_speed", "0.3",
            "--max_reference_vertical_acceleration", "0.5",
            "--start_timesteps", "0",
            "--update_after", "1000",
            "--expl_noise_start", "0.15",
            "--expl_noise_end", "0.05",
            "--noise_decay_steps", "50000",
            "--reward_scale", "0.1",
            "--distance_reward_weight", "0.0",
            "--time_penalty_weight", "0.01",
            "--eval_episodes", "10",
            "--action_scale", "1.0",
            "--load_checkpoint", "checkpoints/upper_dslpid_five_trees_fixed_v2/model_best",
            "--log_dir", "runs/upper_dslpid_five_trees_forest24_v1",
            "--ckpt_dir", "checkpoints/upper_dslpid_five_trees_forest24_v1",
        ]
    elif mode == "upper_dslpid_semantic_latent":
        cmd += [
            "--use_latent", "false",
            "--learned_upper_dslpid", "true",
            "--upper_semantic_latent", "true",
            "--semantic_history_length", "4",
            "--latent_dim", "16",
            "--semantic_residual_scale", "0.25",
            "--semantic_loss_weight", "0.2",
            "--semantic_danger_range", "0.20",
            "--base_policy_checkpoint",
            "checkpoints/upper_dslpid_five_trees_forest24_v1/baseline_pre_semantic_latent/model_best",
            "--hierarchical_async", "false",
            "--reference_tracking", "false",
            "--latent_only", "false",
            "--use_v1trust", "false",
            "--num_trees", "24",
            "--route_blocking_tree", "true",
            "--route_blocking_tree_count", "5",
            "--route_tree_layout", "fixed_safe_five",
            "--curriculum", "false",
            "--curriculum_stage_override", "0",
            "--reference_sequence_length", "15",
            "--reference_horizon_seconds", "1.0",
            "--high_level_interval", "8",
            "--max_reference_speed", "0.8",
            "--max_reference_acceleration", "1.0",
            "--max_reference_vertical_speed", "0.3",
            "--max_reference_vertical_acceleration", "0.5",
            "--start_timesteps", "0",
            "--update_after", "1000",
            "--expl_noise_start", "0.10",
            "--expl_noise_end", "0.03",
            "--noise_decay_steps", "50000",
            "--reward_scale", "0.1",
            "--distance_reward_weight", "0.0",
            "--time_penalty_weight", "0.01",
            "--eval_episodes", "10",
            "--action_scale", "1.0",
            "--log_dir", "runs/upper_dslpid_semantic_latent_v1",
            "--ckpt_dir", "checkpoints/upper_dslpid_semantic_latent_v1",
        ]
    elif mode in ("nolatent", "nolantent"):
        cmd += [
            "--use_latent",
            "false",
            "--use_v1trust",
            "false",
            "--log_dir",
            "runs/nolatent",
            "--ckpt_dir",
            "checkpoints/nolatent",
        ]
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return cmd


def main():
    parser = argparse.ArgumentParser(description="Quick launcher for change_td3 experiment modes.")
    parser.add_argument(
        "mode",
        choices=[
            "baseline",
            "latent_detach",
            "v1trust",
            "v1trust_detach_trust1",
            "latent_only",
            "latent_only_affordance",
            "hierarchical_async",
            "reference_tracking",
            "upper_dslpid",
            "upper_dslpid_recurrent",
            "upper_dslpid_recurrent_safe_v2",
            "upper_dslpid_recurrent_stable_v3",
            "upper_dslpid_recurrent_multitree_v4",
            "upper_dslpid_recurrent_multitree_v4_1",
            "upper_dslpid_recurrent_multitree_v4_2_h4",
            "upper_dslpid_recurrent_multitree_v4_2_h8",
            "upper_dslpid_recurrent_multitree_v4_2_h16",
            "upper_dslpid_recurrent_multitree_v4_3_h4",
            "upper_dslpid_recurrent_multitree_v4_3_h8",
            "upper_dslpid_recurrent_multitree_v4_3_h8_k0",
            "upper_dslpid_recurrent_multitree_v4_3_h8_k1",
            "upper_dslpid_recurrent_multitree_v4_3_h8_k1_rayfix",
            "upper_dslpid_recurrent_multitree_rayfix_v1_h8_full",
            "upper_dslpid_recurrent_multitree_rayfix_v1_h8_k1",
            "upper_dslpid_recurrent_multitree_terminal_v1_h8_k1",
            "upper_dslpid_recurrent_multitree_terminal_v2_h8_k1",
            "upper_dslpid_recurrent_multitree_terminal_v2_noroute_h8_k1",
            "upper_dslpid_recurrent_multitree_terminal_v2_noroute_startcurr_v1_h8_k1",
            "upper_dslpid_recurrent_multitree_terminal_v2_noroute_startobscurr_v1_h8_k1",
            "upper_dslpid_recurrent_multitree_terminal_v2_noroute_stage01_ft_v1_h8_k1",
            "full_ft_v1",
            "full_scratch_curr_v1",
            "full_scratch_ego20x10_curr_v1",
            "upper_dslpid_recurrent_multitree_occupancy_v1_h8_k1",
            "upper_dslpid_recurrent_mavrl_lite_v1",
            "upper_dslpid_one_tree",
            "upper_dslpid_five_trees",
            "upper_dslpid_five_trees_forest24",
            "upper_dslpid_semantic_latent",
            "nolatent",
            "nolantent",
        ],
        help="Experiment preset to launch.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--gui", choices=["true", "false"], default=None)
    parser.add_argument("--eval_gui", choices=["true", "false"], default=None)
    args, extra = parser.parse_known_args()

    cmd = build_mode_cmd(args.mode)

    if args.seed is not None:
        cmd[cmd.index("--seed") + 1] = str(args.seed)
    if args.mode in {
        "upper_dslpid_recurrent_multitree_terminal_v2_noroute_startcurr_v1_h8_k1",
        "upper_dslpid_recurrent_multitree_terminal_v2_noroute_startobscurr_v1_h8_k1",
        "upper_dslpid_recurrent_multitree_terminal_v2_noroute_stage01_ft_v1_h8_k1",
        "full_ft_v1",
        "full_scratch_curr_v1",
        "full_scratch_ego20x10_curr_v1",
    }:
        suffix = f"{args.mode}_seed{cmd[cmd.index('--seed') + 1]}"
        for option, path in (
            ("--log_dir", f"runs/{suffix}"),
            ("--ckpt_dir", f"checkpoints/{suffix}"),
            ("--trajectory_output_dir", f"analysis_outputs/{suffix}_trajectory"),
            ("--collision_diagnostic_output_dir", f"analysis_outputs/{suffix}_collision_windows"),
            ("--timeout_diagnostic_output_dir", f"analysis_outputs/{suffix}_timeout_diagnostics"),
            ("--eval_terminal_coverage_output_dir", f"analysis_outputs/{suffix}_terminal_coverage"),
        ):
            _replace_scalar_option(cmd, option, path)
    if args.mode == "upper_dslpid_recurrent_multitree_terminal_v2_noroute_stage01_ft_v1_h8_k1":
        seed = int(cmd[cmd.index("--seed") + 1])
        manifest_path = REPO_ROOT / "checkpoints/startcurr_v1_frozen_20261009/manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if str(seed) not in manifest["seeds"]:
            parser.error("Stage 0/1 finetuning supports frozen seeds 42, 43, 44, 45 only")
        prefix = REPO_ROOT / f"checkpoints/startcurr_v1_frozen_20261009/seed{seed}/model_final"
        for name, metadata in manifest["seeds"][str(seed)]["files"].items():
            digest = hashlib.sha256((prefix.parent / name).read_bytes()).hexdigest()
            if digest != metadata["sha256"]:
                parser.error(f"Frozen checkpoint checksum mismatch: {name}")
        _replace_scalar_option(cmd, "--load_checkpoint", str(prefix))
    if args.mode == "full_ft_v1":
        seed = int(cmd[cmd.index("--seed") + 1])
        manifest_path = REPO_ROOT / "checkpoints/startobscurr_v1_selected_20261010/manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if str(seed) not in manifest["seeds"]:
            parser.error("Full FT v1 supports frozen seeds 42, 43, 44, 45 only")
        prefix = REPO_ROOT / f"checkpoints/startobscurr_v1_selected_20261010/seed{seed}/model_final"
        actor_file = pathlib.Path(str(prefix) + "_actor")
        expected = manifest["seeds"][str(seed)]["files"]["model_final_actor"]["sha256"]
        if hashlib.sha256(actor_file.read_bytes()).hexdigest() != expected:
            parser.error("Frozen source Actor checksum mismatch")
        _replace_scalar_option(cmd, "--full_ft_actor_source", str(prefix))
        _replace_scalar_option(
            cmd, "--full_ft_trace_dir", f"analysis_outputs/full_ft_v1_seed{seed}/training_traces"
        )
    if args.total_steps is not None:
        cmd[cmd.index("--total_steps") + 1] = str(args.total_steps)
    if args.gui is not None:
        cmd[cmd.index("--gui") + 1] = args.gui
    if args.eval_gui is not None:
        cmd[cmd.index("--eval_gui") + 1] = args.eval_gui

    if extra:
        if extra and extra[0] == "--extra":
            extra = extra[1:]
        if extra and extra[0] == "--":
            extra = extra[1:]
        cmd.extend(extra)

    print("Launching:")
    print(" ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
