import argparse
import pathlib
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.gym_pybullet_compat import ensure_gym_pybullet_envs_compat

ensure_gym_pybullet_envs_compat()

from envs.ForestAviary import CustomForestAviary
from envs.forest.rewards import (
    BaselineForestReward,
    MAVRLSafeSpeedForestReward,
    OutcomeSeparatedForestReward,
    RateIntegratedOutcomeForestReward,
    SafeTerminalForestReward,
    TerminalGuidanceRateIntegratedReward,
)
from envs.learned_upper_dslpid import (
    KinematicUpperForestEnv,
    LearnedUpperDSLPIDForestEnv,
)
from envs.map_atlas import MapCatalog
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

from algos.td3 import (
    TD3,
    TD3HierarchicalAsync,
    TD3LatentAffordance,
    TD3LatentOnly,
    TD3Plain,
    TD3ReferenceTracking,
    TD3UpperRecurrent,
    TD3UpperSemanticLatent,
    TD3V1Trust,
)
from trainers.td3_trainer import TD3Trainer
from trainers.callbacks.checkpoint import (
    BestCheckpointCallback,
    CheckpointCallback,
    ConvergenceGuardCallback,
)
from trainers.callbacks.eval_callback import EvalCallback
from trainers.callbacks.latent_viz import LatentEmbeddingCallback
from trainers.callbacks.logger import LoggerCallback
from trainers.callbacks.monitor import MonitorCallback
from trainers.recurrent_diagnostics import (
    recurrent_observation_contract,
    replay_buffer_memory_bytes,
)


def _make_argparser():
    parser = argparse.ArgumentParser(description="TD3 / TD3-latent training for forest UAV navigation.")

    def str2bool(v):
        if isinstance(v, bool):
            return v
        v_str = str(v).strip().lower()
        if v_str in ("1", "true", "t", "yes", "y"):
            return True
        if v_str in ("0", "false", "f", "no", "n"):
            return False
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {v!r}")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gui", type=str2bool, default=False)

    parser.add_argument("--pyb_freq", type=int, default=240)
    parser.add_argument("--ctrl_freq", type=int, default=120)
    parser.add_argument("--num_trees", type=int, default=24)
    parser.add_argument(
        "--map_id",
        type=str,
        default=None,
        help=(
            "Optional v2-lite atlas map ID. A selected map owns its start/goal, "
            "collision geometry, sensor range, and registered train/validation/test split."
        ),
    )
    parser.add_argument(
        "--map_catalog",
        type=str,
        default=str(REPO_ROOT / "data" / "map_atlas" / "catalog_v1.json"),
        help="Path to the JSON map catalog used with --map_id.",
    )
    parser.add_argument(
        "--eval_map_id",
        type=str,
        default=None,
        help=(
            "Optional separate atlas map used only by EvalCallback. If omitted, "
            "evaluation uses --map_id (or the legacy forest configuration)."
        ),
    )
    parser.add_argument(
        "--route_blocking_tree",
        type=str2bool,
        default=True,
        help="Add one fixed tree on the route between the start and target points.",
    )
    parser.add_argument(
        "--route_tree_fraction",
        type=float,
        default=0.5,
        help="Position of the fixed route tree along start->target; 0.5 means midpoint.",
    )
    parser.add_argument("--route_blocking_tree_count", type=int, default=1)
    parser.add_argument("--range_azimuth_count", type=int, default=8)
    parser.add_argument(
        "--range_elevation_degrees", type=float, nargs="+", default=(0.0,)
    )
    parser.add_argument("--route_tree_lateral_range", type=float, default=0.55)
    parser.add_argument(
        "--route_tree_layout",
        choices=("random", "fixed_safe_five", "fixed_safe_five_mirror"),
        default="random",
    )
    parser.add_argument("--curriculum", type=str2bool, default=True)
    parser.add_argument(
        "--curriculum_stage_override",
        type=int,
        choices=(0, 1, 2, 3),
        default=None,
        help="Pin the forest layout to one curriculum stage; stage 0 keeps a wide obstacle-free route corridor.",
    )
    parser.add_argument("--curriculum_milestones", type=int, nargs=3, default=(800, 2500, 6000))
    parser.add_argument("--curriculum_success_gated", type=str2bool, default=False)
    parser.add_argument("--curriculum_success_window", type=int, default=100)
    parser.add_argument(
        "--curriculum_success_thresholds",
        type=float,
        nargs=3,
        default=(0.10, 0.20, 0.30),
    )
    parser.add_argument(
        "--curriculum_max_collision_rates",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
    )
    parser.add_argument("--curriculum_minimum_stage_episodes", type=int, default=50)
    parser.add_argument("--route_count_curriculum", type=str2bool, default=False)
    parser.add_argument(
        "--route_count_curriculum_counts", type=int, nargs=4, default=(1, 2, 3, 5)
    )
    parser.add_argument(
        "--curriculum_simple_mix_probability", type=float, default=0.20
    )

    parser.add_argument("--buffer_size", type=int, default=1_000_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--total_steps", type=int, default=500_000)
    parser.add_argument("--start_timesteps", type=int, default=10_000)
    parser.add_argument("--update_after", type=int, default=10_000)
    parser.add_argument("--train_every", type=int, default=1)
    parser.add_argument(
        "--success_replay_fraction",
        type=float,
        default=0.0,
        help="Fraction of each recurrent-TD3 batch retained from completed successful episodes.",
    )
    parser.add_argument("--success_replay_size", type=int, default=50_000)
    parser.add_argument("--success_replay_stratified", type=str2bool, default=False)
    parser.add_argument(
        "--success_replay_difficulties", type=int, nargs="+", default=(1, 2, 3, 5)
    )

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy_noise", type=float, default=0.2)
    parser.add_argument("--noise_clip", type=float, default=0.5)
    parser.add_argument("--policy_freq", type=int, default=2)
    parser.add_argument("--upper_actor_lr", type=float, default=3e-4)
    parser.add_argument("--upper_critic_lr", type=float, default=3e-4)
    parser.add_argument("--upper_conservative_actor_lr", type=float, default=None)
    parser.add_argument("--upper_conservative_critic_lr", type=float, default=None)
    parser.add_argument("--upper_conservative_policy_noise", type=float, default=None)
    parser.add_argument("--upper_conservative_policy_freq", type=int, default=None)
    parser.add_argument("--policy_anchor_weight", type=float, default=0.0)
    parser.add_argument(
        "--convergence_guard",
        type=str2bool,
        default=False,
        help="Restore model_best and stop after persistent post-convergence degradation.",
    )
    parser.add_argument("--guard_min_success_rate", type=float, default=0.8)
    parser.add_argument("--guard_max_collision_rate", type=float, default=0.2)
    parser.add_argument("--guard_patience_evals", type=int, default=3)
    parser.add_argument("--best_window_size", type=int, default=3)
    parser.add_argument("--best_min_success_rate", type=float, default=0.9)
    parser.add_argument("--best_max_collision_rate", type=float, default=0.0)
    parser.add_argument("--use_latent", type=str2bool, default=True, help="Enable latent/world-model branch; disable for plain TD3.")
    parser.add_argument(
        "--latent_only",
        type=str2bool,
        default=False,
        help="Use z-only actor/critic inputs for latent ablation. Ignored when --use_latent false.",
    )
    parser.add_argument(
        "--latent_only_variant",
        choices=("base", "affordance"),
        default="base",
        help="Latent-only implementation to run. 'base' keeps the original TD3LatentOnly baseline.",
    )
    parser.add_argument("--use_v1trust", type=str2bool, default=False, help="Use the V1 trust-gated latent variant.")
    parser.add_argument(
        "--hierarchical_async",
        type=str2bool,
        default=False,
        help="Use the two-timescale latent/reference-sequence controller.",
    )
    parser.add_argument(
        "--learned_upper_dslpid",
        type=str2bool,
        default=False,
        help=(
            "Train a low-frequency TD3 upper policy whose local reference "
            "packets are executed by a frozen 120 Hz DSLPID lower controller."
        ),
    )
    parser.add_argument("--upper_semantic_latent", type=str2bool, default=False)
    parser.add_argument(
        "--upper_recurrent",
        type=str2bool,
        default=False,
        help=(
            "Use a GRU actor and two independent GRU critics for the "
            "learned_upper_dslpid ReferencePacket policy."
        ),
    )
    parser.add_argument("--upper_recurrent_hidden_dim", type=int, default=32)
    parser.add_argument(
        "--upper_project_planar_action",
        type=str2bool,
        default=False,
        help=(
            "Project recurrent Upper actor, noisy target actions, exploration "
            "actions, and replay actions onto the executable XY unit disk."
        ),
    )
    parser.add_argument("--semantic_history_length", type=int, default=0)
    parser.add_argument(
        "--use_local_occupancy",
        type=str2bool,
        default=False,
        help="Append a persistent single-scale tri-state local occupancy grid.",
    )
    parser.add_argument("--local_occupancy_voxel_size", type=float, default=0.4)
    parser.add_argument(
        "--local_occupancy_shape",
        type=int,
        nargs=3,
        default=(12, 12, 6),
        metavar=("NX", "NY", "NZ"),
    )
    parser.add_argument("--semantic_residual_scale", type=float, default=0.25)
    parser.add_argument("--semantic_loss_weight", type=float, default=0.2)
    parser.add_argument("--semantic_danger_range", type=float, default=0.20)
    parser.add_argument("--base_policy_checkpoint", type=str, default=None)
    parser.add_argument(
        "--reference_tracking",
        type=str2bool,
        default=False,
        help="Train only the low-level motor actor against deterministic feasible references.",
    )
    parser.add_argument(
        "--motor_action_mode",
        choices=("asymmetric_rpm", "asymmetric_thrust", "legacy_projected"),
        default="asymmetric_rpm",
        help="Direct-motor action encoding; legacy_projected is retained only as a negative control.",
    )
    parser.add_argument("--reference_sequence_length", type=int, default=15)
    parser.add_argument("--reference_horizon_seconds", type=float, default=1.0)
    parser.add_argument(
        "--rule_reference_mode",
        choices=("line", "hover"),
        default="line",
    )
    parser.add_argument("--max_reference_speed", type=float, default=0.8)
    parser.add_argument("--max_reference_acceleration", type=float, default=2.0)
    parser.add_argument("--max_reference_vertical_speed", type=float, default=0.5)
    parser.add_argument(
        "--max_reference_vertical_acceleration",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--upper_planar_only",
        type=str2bool,
        default=False,
        help=(
            "Expose only vx/vy reference actions to learned_upper_dslpid; "
            "a deterministic height hold generates vz."
        ),
    )
    parser.add_argument(
        "--upper_height_hold_gain",
        type=float,
        default=1.0,
        help="Proportional gain for deterministic target-height velocity hold.",
    )
    parser.add_argument(
        "--upper_execution_model",
        choices=("full", "k0", "k1"),
        default="full",
        help=(
            "Upper action-to-state transition. 'full' executes ReferencePacket "
            "through DSLPID; 'k0' executes commanded velocity ideally; 'k1' "
            "uses the same velocity action with an acceleration limit."
        ),
    )
    parser.add_argument(
        "--kinematic_planar_acceleration",
        type=float,
        default=None,
        help=(
            "K1 horizontal acceleration limit [m/s^2]. Defaults to "
            "--max_reference_acceleration so the reference interface is unchanged."
        ),
    )
    parser.add_argument(
        "--kinematic_vertical_acceleration",
        type=float,
        default=None,
        help=(
            "K1 vertical acceleration limit [m/s^2]. Defaults to "
            "--max_reference_vertical_acceleration."
        ),
    )
    parser.add_argument(
        "--tracking_environment_reward_weight",
        type=float,
        default=0.0,
        help="Optional navigation-reward contribution in low-level tracking mode.",
    )
    parser.add_argument(
        "--teacher_timesteps",
        type=int,
        default=0,
        help="Warm-start steps using a PID/geometric teacher in reference-tracking mode.",
    )
    parser.add_argument(
        "--teacher_supervision_timesteps",
        type=int,
        default=0,
        help="Continue online teacher labels after control is handed to the actor; 0 uses teacher_timesteps.",
    )
    parser.add_argument(
        "--tracking_actor_rl_start_step",
        type=int,
        default=None,
        help="First step after which TD3 may update the tracking actor; defaults to teacher_timesteps.",
    )
    parser.add_argument(
        "--teacher_exploration_noise",
        type=float,
        default=0.0,
        help="Small motor noise executed during teacher control to collect recovery states.",
    )
    parser.add_argument(
        "--teacher_bc_updates_per_step",
        type=int,
        default=1,
        help="Behavior-cloning minibatch updates for each newly labelled teacher state.",
    )
    parser.add_argument(
        "--tracking_actor_structure",
        choices=("plain", "residual", "structured"),
        default="plain",
    )
    parser.add_argument(
        "--high_level_interval",
        type=int,
        default=8,
        help="Refresh the high-level latent/reference cache every N motor-control steps.",
    )
    parser.add_argument(
        "--reference_valid_steps",
        type=int,
        default=120,
        help="Maximum age of one cached high-level result in low-level control steps.",
    )
    parser.add_argument("--reference_loss_weight", type=float, default=0.05)
    parser.add_argument("--reference_smoothness_weight", type=float, default=0.01)
    parser.add_argument(
        "--high_level_actor_grad_scale",
        type=float,
        default=0.0,
        help="Actor-gradient scale applied to the high-level encoder and reference head.",
    )
    parser.add_argument("--hierarchical_action_l2_weight", type=float, default=0.10)
    parser.add_argument("--hierarchical_motor_balance_weight", type=float, default=0.10)
    parser.add_argument("--hierarchical_action_delta_weight", type=float, default=0.10)
    parser.add_argument("--hierarchical_normalize_actor_q", type=str2bool, default=True)
    parser.add_argument("--motor_collective_fraction", type=float, default=0.60)
    parser.add_argument("--motor_differential_fraction", type=float, default=0.25)
    parser.add_argument("--reference_lookahead_points", type=int, default=3)
    parser.add_argument(
        "--actor_updates_encoder",
        type=str2bool,
        default=False,
        help="Legacy switch: if true and no actor_encoder_grad_scale is set, use full actor gradient.",
    )
    parser.add_argument(
        "--critic_updates_encoder",
        type=str2bool,
        default=False,
        help="Legacy switch: if true and no critic_encoder_grad_scale is set, use full critic gradient.",
    )
    parser.add_argument(
        "--critic_encoder_grad_scale",
        type=float,
        default=None,
        help="Scale critic RL gradients into the encoder. Default is 0.05; use 0.0 for hard detach.",
    )
    parser.add_argument(
        "--critic_encoder_grad_schedule",
        type=str,
        default=None,
        help="Optional env-step schedule such as '0:0.0,50000:0.03,150000:0.05'.",
    )
    parser.add_argument(
        "--actor_encoder_grad_scale",
        type=float,
        default=None,
        help="Scale actor RL gradients into the encoder. Default is 0.0; use 1.0 for the legacy full update.",
    )
    parser.add_argument("--latent_dim", type=int, default=16, help="Latent dimension for latent-based agents.")
    parser.add_argument(
        "--latent_input_scale",
        type=float,
        default=None,
        help="Scale applied to z before policy/critic input. Defaults to 0.1 for state+z and 1.0 for z-only.",
    )
    parser.add_argument(
        "--latent_viz_interval",
        type=int,
        default=10000,
        help="Log latent embeddings to TensorBoard every N steps; 0 disables.",
    )
    parser.add_argument("--latent_viz_samples", type=int, default=1024)
    parser.add_argument(
        "--latent_viz_start_after",
        type=int,
        default=10000,
        help="Do not log latent embeddings before this many environment steps.",
    )
    parser.add_argument(
        "--posture_separation_weight",
        type=float,
        default=0.0,
        help="Optional contrastive-style loss weight that separates hover and dive latent centers.",
    )
    parser.add_argument(
        "--posture_separation_margin",
        type=float,
        default=1.0,
        help="Minimum normalized latent-center distance between hover and dive samples.",
    )
    parser.add_argument(
        "--progress_loss_weight",
        type=float,
        default=0.0,
        help="Auxiliary delta-goal progress loss weight for latent-only agents.",
    )
    parser.add_argument(
        "--affordance_loss_weight",
        type=float,
        default=0.005,
        help="Weak future-affordance auxiliary loss weight for the affordance latent-only variant.",
    )
    parser.add_argument(
        "--affordance_start_step",
        type=int,
        default=100000,
        help="Do not apply future-affordance auxiliary loss before this env step.",
    )
    parser.add_argument("--affordance_gamma", type=float, default=0.95)
    parser.add_argument(
        "--affordance_bootstrap_warmup_steps",
        type=int,
        default=50000,
        help="Linearly ramp temporal bootstrapping after affordance_start_step.",
    )
    parser.add_argument(
        "--affordance_danger_range",
        type=float,
        default=0.20,
        help="Range-sensor threshold treated as future danger by the affordance head.",
    )
    parser.add_argument(
        "--affordance_goal_tolerance",
        type=float,
        default=0.20,
        help="Distance threshold used for the near-goal affordance target.",
    )
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--trust_alpha", type=float, default=0.5)
    parser.add_argument("--trust_beta", type=float, default=0.5, help="Kept for checkpoint/CLI compatibility; input gate uses recon-only trust.")
    parser.add_argument("--trust_q_min", type=float, default=0.5)
    parser.add_argument("--trust_q_max", type=float, default=1.0)
    parser.add_argument("--trust_ema_momentum", type=float, default=0.99)
    parser.add_argument("--trust_warmup_steps", type=int, default=10000)
    parser.add_argument("--reward_scale", type=float, default=0.01)
    parser.add_argument(
        "--action_scale",
        type=float,
        default=1.0,
        help="Scale the normalized RPM action range. 1.0 keeps [-1, 1], 0.75 limits actions to [-0.75, 0.75].",
    )
    parser.add_argument("--speed_penalty_weight", type=float, default=0.003)
    parser.add_argument("--safety_boundary_penalty_weight", type=float, default=25.0)
    parser.add_argument("--distance_reward_weight", type=float, default=0.5)
    parser.add_argument("--time_penalty_weight", type=float, default=0.0)
    parser.add_argument(
        "--reward_variant",
        choices=(
            "legacy",
            "safe_terminal_v2",
            "outcome_separated_v3",
            "rate_integrated_v4",
            "terminal_guidance_v5",
            "mavrl_safe_speed_v1",
        ),
        default="legacy",
        help="Navigation reward contract; legacy remains the reproducible pre-v2 baseline.",
    )
    parser.add_argument("--proximity_penalty_weight", type=float, default=5.0)
    parser.add_argument("--near_goal_radius", type=float, default=0.8)
    parser.add_argument("--near_goal_speed_penalty_weight", type=float, default=1.5)
    parser.add_argument("--near_goal_away_penalty_weight", type=float, default=3.0)
    parser.add_argument("--near_goal_guidance_radius", type=float, default=1.2)
    parser.add_argument("--near_goal_guidance_ramp_width", type=float, default=0.2)
    parser.add_argument("--near_goal_capture_speed", type=float, default=0.1)
    parser.add_argument("--near_goal_cruise_speed", type=float, default=0.8)
    parser.add_argument(
        "--near_goal_velocity_tracking_weight", type=float, default=160.0
    )
    parser.add_argument("--route_overshoot_penalty_weight", type=float, default=1.0)
    parser.add_argument("--success_terminal_reward", type=float, default=150.0)
    parser.add_argument("--collision_terminal_penalty", type=float, default=120.0)
    parser.add_argument("--bound_terminal_penalty", type=float, default=100.0)
    parser.add_argument("--timeout_terminal_penalty", type=float, default=30.0)
    parser.add_argument("--route_progress_reward_weight", type=float, default=12.0)
    parser.add_argument("--goal_progress_reward_weight", type=float, default=8.0)
    parser.add_argument("--lateral_penalty_scale", type=float, default=0.0)
    parser.add_argument("--safe_speed_min", type=float, default=0.0)
    parser.add_argument(
        "--safe_speed_brake_clearance",
        type=float,
        default=0.35,
        help="Clearance at or below which the MAVRL-style planar safe speed is minimal.",
    )
    parser.add_argument(
        "--safe_speed_open_clearance",
        type=float,
        default=1.20,
        help="Clearance at or above which the MAVRL-style planar safe speed is maximal.",
    )
    parser.add_argument(
        "--unsafe_speed_penalty_weight",
        type=float,
        default=3.0,
        help="Quadratic penalty weight for planar speed above the clearance-conditioned limit.",
    )

    parser.add_argument("--expl_noise_start", type=float, default=0.5)
    parser.add_argument("--expl_noise_end", type=float, default=0.1)
    parser.add_argument("--noise_decay_steps", type=float, default=100_000)

    parser.add_argument("--eval_interval", type=int, default=10_000)
    parser.add_argument("--eval_episodes", type=int, default=1)
    parser.add_argument(
        "--evaluate_only",
        type=str2bool,
        default=False,
        help="Load a checkpoint and run exactly one evaluation without collecting or updating training data.",
    )
    parser.add_argument(
        "--eval_p_height_ablation",
        type=str2bool,
        default=False,
        help=(
            "Evaluation-only ablation: preserve the learned xy action and replace "
            "only the z action with a deterministic P height controller."
        ),
    )
    parser.add_argument(
        "--eval_p_height_gain",
        type=float,
        default=1.0,
        help="P gain [1/s] for --eval_p_height_ablation.",
    )
    parser.add_argument(
        "--eval_z_q_scan_output_dir",
        type=str,
        default=None,
        help=(
            "Optional evaluation-only output directory for a five-point z-action "
            "scan of both TD3 critics."
        ),
    )
    parser.add_argument(
        "--eval_action_interface_audit_output_dir",
        type=str,
        default=None,
        help=(
            "Evaluation-only output directory for raw actor action, mapped velocity "
            "command, realized velocity, and goal-distance auditing on T1 episodes."
        ),
    )
    parser.add_argument(
        "--eval_terminal_xy_q_scan_output_dir",
        type=str,
        default=None,
        help=(
            "Evaluation-only output directory for near-goal XY speed/direction scans "
            "using min(Q1, Q2) on states from T1 episodes."
        ),
    )
    parser.add_argument(
        "--eval_terminal_coverage_output_dir",
        type=str,
        default=None,
        help=(
            "Output directory for per-seed entry, dwell, motion, "
            "and success-transition coverage in the near-goal guidance region."
        ),
    )
    parser.add_argument("--eval_gui", type=str2bool, default=True)
    parser.add_argument("--eval_stepsleep", type=str2bool, default=True)
    parser.add_argument("--eval_quiet", type=str2bool, default=True)
    parser.add_argument(
        "--monitor_interval",
        type=int,
        default=None,
        help="Print the verbose Physics Monitor every N steps. Defaults to eval_interval; 0 disables it.",
    )
    parser.add_argument(
        "--train_log_interval",
        type=int,
        default=500,
        help="Log core train scalars every N env steps. Use 1 for per-step logging; 0 disables core train scalars.",
    )
    parser.add_argument(
        "--debug_log_interval",
        type=int,
        default=1000,
        help="Log verbose/debug train scalars every N env steps. Use 0 to disable debug scalars.",
    )

    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        default=None,
        help="Checkpoint prefix to load before training, for example checkpoints/run/model_best.",
    )
    parser.add_argument("--ckpt_interval", type=int, default=50_000)
    parser.add_argument(
        "--trajectory_output_dir",
        type=str,
        default=None,
        help=(
            "Optional directory for the latest seed-0 evaluation trace plus XY/3D plots. "
            "Intended for learned_upper_dslpid evaluation."
        ),
    )
    parser.add_argument(
        "--collision_diagnostic_output_dir",
        type=str,
        default=None,
        help=(
            "Optional directory for collision_windows_latest.json and JSONL history. "
            "Each collision contains the final 2 s of upper commands, realized "
            "velocity, clearance, delta-v, position, and goal error."
        ),
    )
    parser.add_argument(
        "--timeout_diagnostic_output_dir",
        type=str,
        default=None,
        help=(
            "Optional directory for timeout T1–T4 diagnostic records during evaluation."
        ),
    )
    return parser


def main():
    args = _make_argparser().parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    catalog = MapCatalog(args.map_catalog) if (args.map_id or args.eval_map_id) else None

    def resolve_map(map_id, *, role: str):
        if map_id is None:
            return None
        assert catalog is not None
        spec = catalog.get(map_id)
        validation = spec.validate_route()
        if validation.reachable is False:
            raise ValueError(
                f"Selected {role} map {spec.map_id!r} is invalid: {validation.message}"
            )
        print(
            f"[INFO] {role}_map_id={spec.map_id}, source={spec.source}, "
            f"scenario={spec.scenario}, split={spec.split}, "
            f"route_validation={validation.message}"
        )
        if role == "eval" and spec.split == "test":
            print("[WARNING] eval_map_id belongs to the sealed test split; do not use it for model selection.")
        return spec

    map_spec = resolve_map(args.map_id, role="train")
    eval_map_spec = resolve_map(
        args.eval_map_id if args.eval_map_id is not None else args.map_id,
        role="eval",
    )

    init_xyzs = np.array(
        [[-3.5, 0.0, 1.0] if map_spec is None else map_spec.start_pos],
        dtype=np.float32,
    )
    init_rpys = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    action_scale = float(np.clip(args.action_scale, 0.0, 1.0))

    reward_integration_dt = 1.0 / float(args.ctrl_freq)
    if args.learned_upper_dslpid and args.upper_execution_model != "full":
        reward_integration_dt = float(args.high_level_interval) / float(args.ctrl_freq)

    if args.reward_variant == "safe_terminal_v2":
        reward_model = SafeTerminalForestReward(
            speed_penalty_weight=args.speed_penalty_weight,
            safety_boundary_penalty_weight=args.safety_boundary_penalty_weight,
            distance_reward_weight=args.distance_reward_weight,
            time_penalty_weight=args.time_penalty_weight,
            proximity_penalty_weight=args.proximity_penalty_weight,
            near_goal_radius=args.near_goal_radius,
            near_goal_speed_penalty_weight=args.near_goal_speed_penalty_weight,
            near_goal_away_penalty_weight=args.near_goal_away_penalty_weight,
            route_overshoot_penalty_weight=args.route_overshoot_penalty_weight,
        )
    elif args.reward_variant == "outcome_separated_v3":
        reward_model = OutcomeSeparatedForestReward(
            speed_penalty_weight=args.speed_penalty_weight,
            safety_boundary_penalty_weight=args.safety_boundary_penalty_weight,
            time_penalty_weight=args.time_penalty_weight,
            proximity_penalty_weight=args.proximity_penalty_weight,
            near_goal_radius=args.near_goal_radius,
            near_goal_speed_penalty_weight=args.near_goal_speed_penalty_weight,
            near_goal_away_penalty_weight=args.near_goal_away_penalty_weight,
            route_overshoot_penalty_weight=args.route_overshoot_penalty_weight,
            success_terminal_reward=args.success_terminal_reward,
            collision_terminal_penalty=args.collision_terminal_penalty,
            bound_terminal_penalty=args.bound_terminal_penalty,
            timeout_terminal_penalty=args.timeout_terminal_penalty,
        )
    elif args.reward_variant in ("rate_integrated_v4", "terminal_guidance_v5"):
        reward_class = (
            TerminalGuidanceRateIntegratedReward
            if args.reward_variant == "terminal_guidance_v5"
            else RateIntegratedOutcomeForestReward
        )
        reward_kwargs = {}
        if args.reward_variant == "terminal_guidance_v5":
            reward_kwargs.update(
                near_goal_guidance_radius=args.near_goal_guidance_radius,
                near_goal_guidance_ramp_width=args.near_goal_guidance_ramp_width,
                near_goal_capture_speed=args.near_goal_capture_speed,
                near_goal_cruise_speed=args.near_goal_cruise_speed,
                near_goal_velocity_tracking_weight=(
                    args.near_goal_velocity_tracking_weight
                ),
            )
        reward_model = reward_class(
            integration_dt=reward_integration_dt,
            route_progress_reward_weight=args.route_progress_reward_weight,
            goal_progress_reward_weight=args.goal_progress_reward_weight,
            lateral_penalty_scale=args.lateral_penalty_scale,
            speed_penalty_weight=args.speed_penalty_weight,
            safety_boundary_penalty_weight=args.safety_boundary_penalty_weight,
            time_penalty_weight=args.time_penalty_weight,
            proximity_penalty_weight=args.proximity_penalty_weight,
            near_goal_radius=args.near_goal_radius,
            near_goal_speed_penalty_weight=args.near_goal_speed_penalty_weight,
            near_goal_away_penalty_weight=args.near_goal_away_penalty_weight,
            route_overshoot_penalty_weight=args.route_overshoot_penalty_weight,
            success_terminal_reward=args.success_terminal_reward,
            collision_terminal_penalty=args.collision_terminal_penalty,
            bound_terminal_penalty=args.bound_terminal_penalty,
            timeout_terminal_penalty=args.timeout_terminal_penalty,
            **reward_kwargs,
        )
    elif args.reward_variant == "mavrl_safe_speed_v1":
        reward_model = MAVRLSafeSpeedForestReward(
            safety_boundary_penalty_weight=args.safety_boundary_penalty_weight,
            distance_reward_weight=args.distance_reward_weight,
            time_penalty_weight=args.time_penalty_weight,
            proximity_penalty_weight=args.proximity_penalty_weight,
            near_goal_radius=args.near_goal_radius,
            near_goal_speed_penalty_weight=args.near_goal_speed_penalty_weight,
            near_goal_away_penalty_weight=args.near_goal_away_penalty_weight,
            route_overshoot_penalty_weight=args.route_overshoot_penalty_weight,
            safe_speed_min=args.safe_speed_min,
            safe_speed_max=args.max_reference_speed,
            safe_speed_brake_clearance=args.safe_speed_brake_clearance,
            safe_speed_open_clearance=args.safe_speed_open_clearance,
            unsafe_speed_penalty_weight=args.unsafe_speed_penalty_weight,
        )
    else:
        reward_model = BaselineForestReward(
            speed_penalty_weight=args.speed_penalty_weight,
            safety_boundary_penalty_weight=args.safety_boundary_penalty_weight,
            distance_reward_weight=args.distance_reward_weight,
            time_penalty_weight=args.time_penalty_weight,
        )
    print(
        f"[INFO] reward_variant={args.reward_variant}, "
        f"time_penalty={args.time_penalty_weight:.3f}, "
        f"safe_distance=0.350, integration_dt={reward_integration_dt:.5f}"
    )
    if args.reward_variant == "mavrl_safe_speed_v1":
        print(
            "[INFO] mavrl_safe_speed="
            f"[{args.safe_speed_min:.3f}, {args.max_reference_speed:.3f}], "
            f"clearance=[{args.safe_speed_brake_clearance:.3f}, "
            f"{args.safe_speed_open_clearance:.3f}], "
            f"weight={args.unsafe_speed_penalty_weight:.3f}"
        )
    elif args.reward_variant == "terminal_guidance_v5":
        print(
            "[INFO] terminal_guidance="
            f"radius={args.near_goal_guidance_radius:.3f}, "
            f"ramp={args.near_goal_guidance_ramp_width:.3f}, "
            f"speed=[{args.near_goal_capture_speed:.3f}, "
            f"{args.near_goal_cruise_speed:.3f}], "
            f"weight={args.near_goal_velocity_tracking_weight:.3f}"
        )
    if not 0.0 <= args.success_replay_fraction < 1.0:
        raise ValueError("--success_replay_fraction must lie in [0, 1).")
    if args.success_replay_size <= 0:
        raise ValueError("--success_replay_size must be positive.")

    forest_env_kwargs = dict(
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        initial_xyzs=init_xyzs,
        initial_rpys=init_rpys,
        pyb_freq=args.pyb_freq,
        ctrl_freq=args.ctrl_freq,
        gui=args.gui,
        curriculum=args.curriculum,
        curriculum_milestones=args.curriculum_milestones,
        curriculum_success_gated=args.curriculum_success_gated,
        curriculum_success_window=args.curriculum_success_window,
        curriculum_success_thresholds=args.curriculum_success_thresholds,
        curriculum_max_collision_rates=args.curriculum_max_collision_rates,
        curriculum_minimum_stage_episodes=args.curriculum_minimum_stage_episodes,
        route_count_curriculum=args.route_count_curriculum,
        route_count_curriculum_counts=args.route_count_curriculum_counts,
        curriculum_simple_mix_probability=args.curriculum_simple_mix_probability,

        num_trees=args.num_trees,  # 避障任务
        route_blocking_tree=args.route_blocking_tree,
        route_tree_fraction=args.route_tree_fraction,
        route_blocking_tree_count=args.route_blocking_tree_count,
        route_tree_lateral_range=args.route_tree_lateral_range,
        route_tree_layout=args.route_tree_layout,
        range_azimuth_count=args.range_azimuth_count,
        range_elevation_degrees=args.range_elevation_degrees,
        target_pos=[3.5, 0.0, 1.0] if map_spec is None else map_spec.target_pos,
        map_spec=map_spec,
        reward_model=reward_model,
    )
    upper_wrapper_kwargs = dict(
        upper_control_interval=args.high_level_interval,
        reference_sequence_length=args.reference_sequence_length,
        reference_horizon_seconds=args.reference_horizon_seconds,
        max_reference_speed=args.max_reference_speed,
        max_reference_acceleration=args.max_reference_acceleration,
        max_reference_vertical_speed=args.max_reference_vertical_speed,
        max_reference_vertical_acceleration=args.max_reference_vertical_acceleration,
        semantic_history_length=args.semantic_history_length,
        planar_only=args.upper_planar_only,
        height_hold_gain=args.upper_height_hold_gain,
        use_local_occupancy=args.use_local_occupancy,
        local_occupancy_voxel_size=args.local_occupancy_voxel_size,
        local_occupancy_shape=args.local_occupancy_shape,
    )
    kinematic_wrapper_kwargs = dict(
        kinematic_model=args.upper_execution_model,
        kinematic_planar_acceleration=args.kinematic_planar_acceleration,
        kinematic_vertical_acceleration=args.kinematic_vertical_acceleration,
    )
    if args.learned_upper_dslpid:
        if args.upper_execution_model == "full":
            env = LearnedUpperDSLPIDForestEnv(
                **forest_env_kwargs,
                **upper_wrapper_kwargs,
            )
        else:
            env = KinematicUpperForestEnv(
                **forest_env_kwargs,
                **upper_wrapper_kwargs,
                **kinematic_wrapper_kwargs,
            )
    else:
        if args.upper_execution_model != "full":
            raise ValueError("--upper_execution_model k0/k1 requires --learned_upper_dslpid true.")
        env = CustomForestAviary(**forest_env_kwargs)
    if args.curriculum_stage_override is not None:
        env.set_curriculum_stage_override(args.curriculum_stage_override)
        print(
            f"[INFO] curriculum_stage_override={args.curriculum_stage_override}, "
            f"protected_corridor={args.curriculum_stage_override <= 2}"
        )

    eval_env_kwargs = dict(forest_env_kwargs)
    eval_env_kwargs["map_spec"] = eval_map_spec
    eval_env_kwargs.pop("gui", None)
    if args.learned_upper_dslpid:
        eval_env_kwargs.update(upper_wrapper_kwargs)
        if args.upper_execution_model != "full":
            eval_env_kwargs.update(kinematic_wrapper_kwargs)

    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(env.action_space.shape[-1])
    max_action = float(env.action_space.high.flatten()[0]) * action_scale
    print(
        f"[INFO] action_scale={action_scale:.2f}, max_action={max_action:.3f}, "
        f"speed_penalty_weight={args.speed_penalty_weight:.4f}"
    )

    latent_input_scale = args.latent_input_scale
    if latent_input_scale is None:
        latent_input_scale = 1.0 if args.latent_only else 0.1
    critic_encoder_grad_scale = args.critic_encoder_grad_scale
    if critic_encoder_grad_scale is None:
        critic_encoder_grad_scale = 1.0 if args.critic_updates_encoder else 0.05
    actor_encoder_grad_scale = args.actor_encoder_grad_scale
    if actor_encoder_grad_scale is None:
        actor_encoder_grad_scale = 1.0 if args.actor_updates_encoder else 0.0
    if args.learned_upper_dslpid:
        if args.use_latent or args.hierarchical_async or args.reference_tracking:
            raise ValueError(
                "--learned_upper_dslpid is an isolated upper-policy mode; "
                "disable latent, direct-motor hierarchical, and reference-tracking modes."
            )
        if args.upper_recurrent and args.upper_semantic_latent:
            raise ValueError(
                "--upper_recurrent and --upper_semantic_latent are separate upper-policy variants."
            )
        if args.upper_recurrent and args.semantic_history_length <= 0:
            raise ValueError(
                "--upper_recurrent requires --semantic_history_length greater than zero."
            )
        if args.upper_planar_only and args.upper_height_hold_gain <= 0.0:
            raise ValueError("--upper_height_hold_gain must be positive with --upper_planar_only.")
        print(
            "[INFO] learned_upper_dslpid=true, "
            f"upper_frequency={args.ctrl_freq / args.high_level_interval:.1f}Hz, "
            f"lower_frequency={args.ctrl_freq}Hz, "
            f"upper_observation_dim={state_dim}, upper_action_dim={action_dim}"
            f", planar_only={args.upper_planar_only}, "
            f"upper_execution_model={args.upper_execution_model}"
        )
        if args.upper_execution_model == "k1":
            print(
                "[INFO] k1_acceleration_limits="
                f"planar={env.kinematic_planar_acceleration:.3f}, "
                f"vertical={env.kinematic_vertical_acceleration:.3f} m/s^2"
            )
        if args.upper_semantic_latent:
            print(
                "[INFO] upper_semantic_latent=true, "
                f"history_length={args.semantic_history_length}, latent_dim={args.latent_dim}, "
                f"residual_scale={args.semantic_residual_scale:.3f}"
            )
        elif args.upper_recurrent:
            contract = recurrent_observation_contract(
                base_state_dim=env.base_upper_obs_dim,
                history_frame_dim=env.semantic_frame_dim,
                history_length=args.semantic_history_length,
                state_dim=state_dim,
            )
            if int(getattr(env, "range_dim", -1)) == 96:
                expected_96_ray_dims = {4: 517, 8: 917, 16: 1717}
                expected_base_state_dim = 117 + int(
                    getattr(env, "local_occupancy_dim", 0)
                )
                if (
                    contract["base_state_dim"] != expected_base_state_dim
                    or contract["history_frame_dim"] != 100
                ):
                    raise ValueError(
                        "96-ray recurrent observation contract must use "
                        f"base_state_dim={expected_base_state_dim} and "
                        "history_frame_dim=100."
                    )
                expected_dim = expected_96_ray_dims.get(contract["history_length"])
                if expected_dim is not None:
                    expected_dim += int(getattr(env, "local_occupancy_dim", 0))
                if expected_dim is not None and contract["state_dim"] != expected_dim:
                    raise ValueError(
                        f"96-ray H{contract['history_length']} must have "
                        f"state_dim={expected_dim}, got {contract['state_dim']}."
                    )
            main_replay_bytes = replay_buffer_memory_bytes(
                state_dim=state_dim,
                action_dim=action_dim,
                capacity=args.buffer_size,
            )
            success_replay_bytes = replay_buffer_memory_bytes(
                state_dim=state_dim,
                action_dim=action_dim,
                capacity=args.success_replay_size,
            )
            print(
                "[INFO] upper_recurrent=true, double_critic=true, "
                f"base_state_dim={contract['base_state_dim']}, "
                f"history_frame_dim={contract['history_frame_dim']}, "
                f"history_length={contract['history_length']}, "
                f"state_dim={contract['state_dim']}, "
                f"hidden_dim={args.upper_recurrent_hidden_dim}, "
                f"project_planar_action={args.upper_project_planar_action}, "
                f"local_occupancy_dim={getattr(env, 'local_occupancy_dim', 0)}"
            )
            print(
                "[INFO] replay_memory_estimate="
                f"main={main_replay_bytes / 2**20:.1f}MiB "
                f"({args.buffer_size} transitions), "
                f"success={success_replay_bytes / 2**20:.1f}MiB "
                f"({args.success_replay_size} transitions), "
                f"total={(main_replay_bytes + success_replay_bytes) / 2**20:.1f}MiB"
            )
    elif args.reference_tracking:
        print(
            "[INFO] reference_tracking=true, "
            f"mode={args.rule_reference_mode}, horizon={args.reference_horizon_seconds:.2f}s, "
            f"sequence_length={args.reference_sequence_length}, "
            f"high_level_interval={args.high_level_interval}, "
            f"teacher_timesteps={args.teacher_timesteps}"
        )
    elif args.use_latent:
        latent_msg = (
            f"[INFO] latent_only={args.latent_only}, latent_dim={args.latent_dim}, "
            f"latent_input_scale={latent_input_scale:.3f}, "
            f"critic_encoder_grad_scale={critic_encoder_grad_scale:.3f}, "
            f"actor_encoder_grad_scale={actor_encoder_grad_scale:.3f}, "
            f"critic_encoder_grad_schedule={args.critic_encoder_grad_schedule or 'none'}"
        )
        if args.hierarchical_async:
            latent_msg += (
                f", hierarchical_async=true, reference_sequence_length={args.reference_sequence_length}, "
                f"high_level_interval={args.high_level_interval}, "
                f"reference_valid_steps={args.reference_valid_steps}"
            )
        elif args.latent_only:
            latent_msg += f", latent_only_variant={args.latent_only_variant}"
            if args.latent_only_variant == "affordance":
                latent_msg += (
                    f", affordance_loss_weight={args.affordance_loss_weight:.4f}, "
                    f"affordance_start_step={args.affordance_start_step}"
                )
        print(latent_msg)

    if args.learned_upper_dslpid:
        if args.upper_recurrent:
            agent = TD3UpperRecurrent(
                state_dim=state_dim,
                action_dim=action_dim,
                max_action=max_action,
                base_state_dim=env.base_upper_obs_dim,
                history_length=args.semantic_history_length,
                history_frame_dim=env.semantic_frame_dim,
                hidden_dim=args.upper_recurrent_hidden_dim,
                discount=args.gamma,
                tau=args.tau,
                policy_noise=args.policy_noise,
                noise_clip=args.noise_clip,
                policy_freq=args.policy_freq,
                grad_clip_norm=args.grad_clip_norm,
                actor_lr=args.upper_actor_lr,
                critic_lr=args.upper_critic_lr,
                conservative_actor_lr=args.upper_conservative_actor_lr,
                conservative_critic_lr=args.upper_conservative_critic_lr,
                conservative_policy_noise=args.upper_conservative_policy_noise,
                conservative_policy_freq=args.upper_conservative_policy_freq,
                policy_anchor_weight=args.policy_anchor_weight,
                project_planar_action=args.upper_project_planar_action,
            )
            encoder_parameter_ids = {
                agent.actor.encoder.gru.weight_ih_l0.data_ptr(),
                agent.critic.q1.encoder.gru.weight_ih_l0.data_ptr(),
                agent.critic.q2.encoder.gru.weight_ih_l0.data_ptr(),
            }
            if len(encoder_parameter_ids) != 3:
                raise RuntimeError(
                    "Upper recurrent architecture must retain independent "
                    "Actor/Q1/Q2 temporal encoders."
                )
            if int(getattr(env, "range_dim", -1)) == 96 and agent.hidden_dim != 64:
                raise RuntimeError("96-ray upper recurrent presets require hidden_dim=64.")
            print(
                "[INFO] recurrent_encoders=actor,q1,q2 (independent), "
                f"hidden_dim={agent.hidden_dim}"
            )
        elif args.upper_semantic_latent:
            if not args.base_policy_checkpoint:
                raise ValueError("--base_policy_checkpoint is required for upper semantic latent mode.")
            agent = TD3UpperSemanticLatent(
                state_dim=state_dim,
                action_dim=action_dim,
                max_action=max_action,
                base_policy_checkpoint=args.base_policy_checkpoint,
                latent_dim=args.latent_dim,
                residual_scale=args.semantic_residual_scale,
                semantic_loss_weight=args.semantic_loss_weight,
                danger_range=args.semantic_danger_range,
                discount=args.gamma,
                tau=args.tau,
                policy_noise=args.policy_noise,
                noise_clip=args.noise_clip,
                policy_freq=args.policy_freq,
                grad_clip_norm=args.grad_clip_norm,
            )
        else:
            agent = TD3Plain(
                state_dim=state_dim,
                action_dim=action_dim,
                max_action=max_action,
                discount=args.gamma,
                tau=args.tau,
                policy_noise=args.policy_noise,
                noise_clip=args.noise_clip,
                policy_freq=args.policy_freq,
                grad_clip_norm=args.grad_clip_norm,
            )
    elif args.reference_tracking:
        if args.hierarchical_async or args.use_v1trust or args.latent_only:
            raise ValueError(
                "--reference_tracking is an isolated low-level mode; disable other controller modes."
            )
        tracking_actor_rl_start_step = args.tracking_actor_rl_start_step
        if tracking_actor_rl_start_step is None:
            tracking_actor_rl_start_step = args.teacher_timesteps
        agent = TD3ReferenceTracking(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            target_position=[3.5, 0.0, 1.0],
            ctrl_freq=args.ctrl_freq,
            sequence_length=args.reference_sequence_length,
            reference_horizon_seconds=args.reference_horizon_seconds,
            high_level_interval=args.high_level_interval,
            reference_mode=args.rule_reference_mode,
            max_reference_speed=args.max_reference_speed,
            max_reference_acceleration=args.max_reference_acceleration,
            max_reference_vertical_speed=args.max_reference_vertical_speed,
            lookahead_points=args.reference_lookahead_points,
            discount=args.gamma,
            tau=args.tau,
            policy_noise=args.policy_noise,
            noise_clip=args.noise_clip,
            policy_freq=args.policy_freq,
            grad_clip_norm=args.grad_clip_norm,
            action_l2_weight=args.hierarchical_action_l2_weight,
            motor_balance_weight=args.hierarchical_motor_balance_weight,
            action_delta_weight=args.hierarchical_action_delta_weight,
            normalize_actor_q=args.hierarchical_normalize_actor_q,
            motor_collective_fraction=args.motor_collective_fraction,
            motor_differential_fraction=args.motor_differential_fraction,
            environment_reward_weight=args.tracking_environment_reward_weight,
            actor_rl_start_step=tracking_actor_rl_start_step,
            actor_structure=args.tracking_actor_structure,
            motor_action_mode=args.motor_action_mode,
        )
    elif args.use_latent:
        if args.hierarchical_async:
            if args.use_v1trust or args.latent_only:
                raise ValueError(
                    "--hierarchical_async is a separate controller; disable --use_v1trust and --latent_only."
                )
            agent = TD3HierarchicalAsync(
                state_dim=state_dim,
                action_dim=action_dim,
                max_action=max_action,
                discount=args.gamma,
                tau=args.tau,
                policy_noise=args.policy_noise,
                noise_clip=args.noise_clip,
                policy_freq=args.policy_freq,
                latent_dim=args.latent_dim,
                latent_input_scale=latent_input_scale,
                grad_clip_norm=args.grad_clip_norm,
                sequence_length=args.reference_sequence_length,
                high_level_interval=args.high_level_interval,
                reference_valid_steps=args.reference_valid_steps,
                reference_loss_weight=args.reference_loss_weight,
                reference_smoothness_weight=args.reference_smoothness_weight,
                high_level_actor_grad_scale=args.high_level_actor_grad_scale,
                action_l2_weight=args.hierarchical_action_l2_weight,
                motor_balance_weight=args.hierarchical_motor_balance_weight,
                action_delta_weight=args.hierarchical_action_delta_weight,
                normalize_actor_q=args.hierarchical_normalize_actor_q,
                motor_collective_fraction=args.motor_collective_fraction,
                motor_differential_fraction=args.motor_differential_fraction,
                lookahead_points=args.reference_lookahead_points,
            )
        elif args.use_v1trust:
            if args.latent_only:
                raise ValueError("--latent_only and --use_v1trust are separate ablations; choose one for this launcher.")
            agent = TD3V1Trust(
                state_dim=state_dim,
                action_dim=action_dim,
                max_action=max_action,
                discount=args.gamma,
                tau=args.tau,
                policy_noise=args.policy_noise,
                noise_clip=args.noise_clip,
                policy_freq=args.policy_freq,
                actor_updates_encoder=args.actor_updates_encoder,
                critic_updates_encoder=args.critic_updates_encoder,
                actor_encoder_grad_scale=actor_encoder_grad_scale,
                critic_encoder_grad_scale=critic_encoder_grad_scale,
                critic_encoder_grad_schedule=args.critic_encoder_grad_schedule,
                latent_input_scale=latent_input_scale,
                grad_clip_norm=args.grad_clip_norm,
                latent_dim=args.latent_dim,
                posture_separation_weight=args.posture_separation_weight,
                posture_separation_margin=args.posture_separation_margin,
                trust_alpha=args.trust_alpha,
                trust_beta=args.trust_beta,
                trust_q_min=args.trust_q_min,
                trust_q_max=args.trust_q_max,
                trust_ema_momentum=args.trust_ema_momentum,
                trust_warmup_steps=args.trust_warmup_steps,
            )
        elif args.latent_only:
            latent_only_kwargs = dict(
                state_dim=state_dim,
                action_dim=action_dim,
                max_action=max_action,
                discount=args.gamma,
                tau=args.tau,
                policy_noise=args.policy_noise,
                noise_clip=args.noise_clip,
                policy_freq=args.policy_freq,
                actor_updates_encoder=args.actor_updates_encoder,
                critic_updates_encoder=args.critic_updates_encoder,
                actor_encoder_grad_scale=actor_encoder_grad_scale,
                critic_encoder_grad_scale=critic_encoder_grad_scale,
                critic_encoder_grad_schedule=args.critic_encoder_grad_schedule,
                latent_input_scale=latent_input_scale,
                grad_clip_norm=args.grad_clip_norm,
                latent_dim=args.latent_dim,
                posture_separation_weight=args.posture_separation_weight,
                posture_separation_margin=args.posture_separation_margin,
                progress_loss_weight=args.progress_loss_weight,
                target_pos=[3.5, 0.0, 1.0],
            )
            if args.latent_only_variant == "affordance":
                agent = TD3LatentAffordance(
                    **latent_only_kwargs,
                    affordance_loss_weight=args.affordance_loss_weight,
                    affordance_start_step=args.affordance_start_step,
                    affordance_gamma=args.affordance_gamma,
                    affordance_bootstrap_warmup_steps=args.affordance_bootstrap_warmup_steps,
                    affordance_danger_range=args.affordance_danger_range,
                    affordance_goal_tolerance=args.affordance_goal_tolerance,
                    start_pos=[-3.5, 0.0, 1.0],
                )
            else:
                agent = TD3LatentOnly(**latent_only_kwargs)
        else:
            agent = TD3(
                state_dim=state_dim,
                action_dim=action_dim,
                max_action=max_action,
                discount=args.gamma,
                tau=args.tau,
                policy_noise=args.policy_noise,
                noise_clip=args.noise_clip,
                policy_freq=args.policy_freq,
                actor_updates_encoder=args.actor_updates_encoder,
                critic_updates_encoder=args.critic_updates_encoder,
                actor_encoder_grad_scale=actor_encoder_grad_scale,
                critic_encoder_grad_scale=critic_encoder_grad_scale,
                critic_encoder_grad_schedule=args.critic_encoder_grad_schedule,
                latent_input_scale=latent_input_scale,
                grad_clip_norm=args.grad_clip_norm,
                latent_dim=args.latent_dim,
                posture_separation_weight=args.posture_separation_weight,
                posture_separation_margin=args.posture_separation_margin,
            )
    else:
        agent = TD3Plain(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            discount=args.gamma,
            tau=args.tau,
            policy_noise=args.policy_noise,
            noise_clip=args.noise_clip,
            policy_freq=args.policy_freq,
            grad_clip_norm=args.grad_clip_norm,
        )

    if args.load_checkpoint:
        agent.load(args.load_checkpoint)
        print(f"[INFO] loaded_checkpoint={args.load_checkpoint}")

    if (
        args.eval_p_height_ablation
        or args.eval_z_q_scan_output_dir
        or args.eval_action_interface_audit_output_dir
        or args.eval_terminal_xy_q_scan_output_dir
    ) and not args.evaluate_only:
        raise ValueError(
            "Evaluation diagnostics are restricted to --evaluate_only true."
        )

    eval_callback = EvalCallback(
        type(env),
        eval_env_kwargs,
        interval=args.eval_interval,
        episodes=args.eval_episodes,
        step_sleep=args.eval_stepsleep,
        eval_gui=args.eval_gui,
        quiet=args.eval_quiet,
        output_dir=args.log_dir,
        trajectory_output_dir=args.trajectory_output_dir,
        collision_diagnostic_output_dir=args.collision_diagnostic_output_dir,
        timeout_diagnostic_output_dir=args.timeout_diagnostic_output_dir,
        p_height_ablation=args.eval_p_height_ablation,
        p_height_gain=args.eval_p_height_gain,
        z_q_scan_output_dir=args.eval_z_q_scan_output_dir,
        action_interface_audit_output_dir=(
            args.eval_action_interface_audit_output_dir
        ),
        terminal_xy_q_scan_output_dir=args.eval_terminal_xy_q_scan_output_dir,
        terminal_coverage_output_dir=args.eval_terminal_coverage_output_dir,
    )
    if args.evaluate_only:
        if args.eval_interval <= 0:
            raise ValueError("--eval_interval must be positive with --evaluate_only.")
        evaluation_context = SimpleNamespace(
            env=env,
            agent=agent,
            total_steps=args.eval_interval,
            last_eval_info=None,
            last_eval_step=None,
        )
        eval_callback.on_step(evaluation_context)
        env.close()
        return

    trainer = TD3Trainer(env, agent, args)

    trainer.add_callback(
        LoggerCallback(
            args.log_dir,
            train_interval=args.train_log_interval,
            debug_interval=args.debug_log_interval,
        )
    )
    if args.use_latent and not args.reference_tracking and args.latent_viz_interval > 0:
        trainer.add_callback(
            LatentEmbeddingCallback(
                args.log_dir,
                interval=args.latent_viz_interval,
                num_samples=args.latent_viz_samples,
                start_after=args.latent_viz_start_after,
                seed=args.seed,
            )
        )
    trainer.add_callback(eval_callback)
    best_checkpoint_callback = BestCheckpointCallback(
        args.ckpt_dir,
        window_size=args.best_window_size,
        minimum_success_rate=args.best_min_success_rate,
        maximum_collision_rate=args.best_max_collision_rate,
        final_eval_callback=eval_callback,
    )
    trainer.add_callback(best_checkpoint_callback)
    trainer.add_callback(
        ConvergenceGuardCallback(
            best_checkpoint_callback,
            enabled=args.convergence_guard,
            min_success_rate=args.guard_min_success_rate,
            max_collision_rate=args.guard_max_collision_rate,
            patience_evals=args.guard_patience_evals,
        )
    )
    # Keep this after best selection: on_train_end the selected peak/stable
    # policy is restored first, then written as model_final.
    trainer.add_callback(CheckpointCallback(args.ckpt_dir, interval=args.ckpt_interval))
    monitor_interval = args.eval_interval if args.monitor_interval is None else args.monitor_interval
    if monitor_interval > 0:
        trainer.add_callback(MonitorCallback(interval=monitor_interval))
    trainer.run()
    env.close()


if __name__ == "__main__":
    main()
