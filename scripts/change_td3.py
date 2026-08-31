import argparse
import pathlib
import random
import sys

import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.gym_pybullet_compat import ensure_gym_pybullet_envs_compat

ensure_gym_pybullet_envs_compat()

from envs.ForestAviary import CustomForestAviary
from envs.forest.rewards import BaselineForestReward
from envs.learned_upper_dslpid import LearnedUpperDSLPIDForestEnv
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

from algos.td3 import (
    TD3,
    TD3HierarchicalAsync,
    TD3LatentAffordance,
    TD3LatentOnly,
    TD3Plain,
    TD3ReferenceTracking,
    TD3UpperSemanticLatent,
    TD3V1Trust,
)
from trainers.td3_trainer import TD3Trainer
from trainers.callbacks.checkpoint import BestCheckpointCallback, CheckpointCallback
from trainers.callbacks.eval_callback import EvalCallback
from trainers.callbacks.latent_viz import LatentEmbeddingCallback
from trainers.callbacks.logger import LoggerCallback
from trainers.callbacks.monitor import MonitorCallback


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
    parser.add_argument("--route_tree_lateral_range", type=float, default=0.55)
    parser.add_argument(
        "--route_tree_layout",
        choices=("random", "fixed_safe_five"),
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
    parser.add_argument("--curriculum_minimum_stage_episodes", type=int, default=50)

    parser.add_argument("--buffer_size", type=int, default=1_000_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--total_steps", type=int, default=500_000)
    parser.add_argument("--start_timesteps", type=int, default=10_000)
    parser.add_argument("--update_after", type=int, default=10_000)
    parser.add_argument("--train_every", type=int, default=1)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy_noise", type=float, default=0.2)
    parser.add_argument("--noise_clip", type=float, default=0.5)
    parser.add_argument("--policy_freq", type=int, default=2)
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
    parser.add_argument("--semantic_history_length", type=int, default=0)
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

    parser.add_argument("--expl_noise_start", type=float, default=0.5)
    parser.add_argument("--expl_noise_end", type=float, default=0.1)
    parser.add_argument("--noise_decay_steps", type=float, default=100_000)

    parser.add_argument("--eval_interval", type=int, default=10_000)
    parser.add_argument("--eval_episodes", type=int, default=1)
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
    return parser


def main():
    args = _make_argparser().parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    init_xyzs = np.array([[-3.5, 0.0, 1.0]], dtype=np.float32)
    init_rpys = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    action_scale = float(np.clip(args.action_scale, 0.0, 1.0))

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
        curriculum_minimum_stage_episodes=args.curriculum_minimum_stage_episodes,

        num_trees=args.num_trees,  # 避障任务
        route_blocking_tree=args.route_blocking_tree,
        route_tree_fraction=args.route_tree_fraction,
        route_blocking_tree_count=args.route_blocking_tree_count,
        route_tree_lateral_range=args.route_tree_lateral_range,
        route_tree_layout=args.route_tree_layout,
        target_pos=[3.5, 0.0, 1.0],
        reward_model=BaselineForestReward(
            speed_penalty_weight=args.speed_penalty_weight,
            safety_boundary_penalty_weight=args.safety_boundary_penalty_weight,
            distance_reward_weight=args.distance_reward_weight,
            time_penalty_weight=args.time_penalty_weight,
        ),
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
    )
    if args.learned_upper_dslpid:
        env = LearnedUpperDSLPIDForestEnv(
            **forest_env_kwargs,
            **upper_wrapper_kwargs,
        )
    else:
        env = CustomForestAviary(**forest_env_kwargs)
    if args.curriculum_stage_override is not None:
        env.set_curriculum_stage_override(args.curriculum_stage_override)
        print(
            f"[INFO] curriculum_stage_override={args.curriculum_stage_override}, "
            f"protected_corridor={args.curriculum_stage_override <= 2}"
        )

    eval_env_kwargs = dict(forest_env_kwargs)
    eval_env_kwargs.pop("gui", None)
    if args.learned_upper_dslpid:
        eval_env_kwargs.update(upper_wrapper_kwargs)

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
        print(
            "[INFO] learned_upper_dslpid=true, "
            f"upper_frequency={args.ctrl_freq / args.high_level_interval:.1f}Hz, "
            f"lower_frequency={args.ctrl_freq}Hz, "
            f"upper_observation_dim={state_dim}, upper_action_dim={action_dim}"
        )
        if args.upper_semantic_latent:
            print(
                "[INFO] upper_semantic_latent=true, "
                f"history_length={args.semantic_history_length}, latent_dim={args.latent_dim}, "
                f"residual_scale={args.semantic_residual_scale:.3f}"
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
        if args.upper_semantic_latent:
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
    trainer.add_callback(CheckpointCallback(args.ckpt_dir, interval=args.ckpt_interval))
    trainer.add_callback(
        EvalCallback(
            LearnedUpperDSLPIDForestEnv
            if args.learned_upper_dslpid
            else CustomForestAviary,
            eval_env_kwargs,
            interval=args.eval_interval,
            episodes=args.eval_episodes,
            step_sleep=args.eval_stepsleep,
            eval_gui=args.eval_gui,
            quiet=args.eval_quiet,
        )
    )
    trainer.add_callback(BestCheckpointCallback(args.ckpt_dir))
    monitor_interval = args.eval_interval if args.monitor_interval is None else args.monitor_interval
    if monitor_interval > 0:
        trainer.add_callback(MonitorCallback(interval=monitor_interval))
    trainer.run()
    env.close()


if __name__ == "__main__":
    main()
