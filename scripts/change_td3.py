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
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

from algos.td3 import TD3, TD3Plain, TD3V1Trust
from trainers.td3_trainer import TD3Trainer
from trainers.callbacks.checkpoint import CheckpointCallback
from trainers.callbacks.eval_callback import EvalCallback
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
    parser.add_argument("--curriculum", type=str2bool, default=True)
    parser.add_argument("--curriculum_milestones", type=int, nargs=3, default=(800, 2500, 6000))

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
    parser.add_argument("--use_v1trust", type=str2bool, default=False, help="Use the V1 trust-gated latent variant.")
    parser.add_argument(
        "--actor_updates_encoder",
        type=str2bool,
        default=False,
        help="Whether the actor update also backpropagates into the encoder. Critic updates still train the encoder.",
    )
    parser.add_argument(
        "--critic_updates_encoder",
        type=str2bool,
        default=True,
        help="Whether critic loss backpropagates into the encoder in V1 trust mode. Disable for z.detach ablations.",
    )
    parser.add_argument("--latent_input_scale", type=float, default=0.1)
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

    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
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

    env = CustomForestAviary(
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        initial_xyzs=init_xyzs,
        initial_rpys=init_rpys,
        pyb_freq=args.pyb_freq,
        ctrl_freq=args.ctrl_freq,
        gui=args.gui,
        curriculum=args.curriculum,
        curriculum_milestones=args.curriculum_milestones,

        num_trees=args.num_trees,  # 避障任务
        target_pos=[3.5, 0.0, 1.0],
        reward_model=BaselineForestReward(speed_penalty_weight=args.speed_penalty_weight),
    )

    eval_env_kwargs = dict(
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        initial_xyzs=init_xyzs,
        initial_rpys=init_rpys,
        pyb_freq=args.pyb_freq,
        ctrl_freq=args.ctrl_freq,
        curriculum=args.curriculum,
        curriculum_milestones=args.curriculum_milestones,
        num_trees=args.num_trees,
        target_pos=[3.5, 0.0, 1.0],
        reward_model=BaselineForestReward(speed_penalty_weight=args.speed_penalty_weight),
    )

    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(env.action_space.shape[-1])
    max_action = float(env.action_space.high.flatten()[0]) * action_scale
    print(
        f"[INFO] action_scale={action_scale:.2f}, max_action={max_action:.3f}, "
        f"speed_penalty_weight={args.speed_penalty_weight:.4f}"
    )

    if args.use_latent:
        if args.use_v1trust:
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
                latent_input_scale=args.latent_input_scale,
                grad_clip_norm=args.grad_clip_norm,
                trust_alpha=args.trust_alpha,
                trust_beta=args.trust_beta,
                trust_q_min=args.trust_q_min,
                trust_q_max=args.trust_q_max,
                trust_ema_momentum=args.trust_ema_momentum,
                trust_warmup_steps=args.trust_warmup_steps,
            )
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
                latent_input_scale=args.latent_input_scale,
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

    trainer = TD3Trainer(env, agent, args)
    trainer.add_callback(LoggerCallback(args.log_dir))
    trainer.add_callback(CheckpointCallback(args.ckpt_dir, interval=args.ckpt_interval))
    trainer.add_callback(
        EvalCallback(
            CustomForestAviary,
            eval_env_kwargs,
            interval=args.eval_interval,
            episodes=args.eval_episodes,
            step_sleep=args.eval_stepsleep,
            eval_gui=args.eval_gui,
            quiet=args.eval_quiet,
        )
    )
    monitor_interval = args.eval_interval if args.monitor_interval is None else args.monitor_interval
    if monitor_interval > 0:
        trainer.add_callback(MonitorCallback(interval=monitor_interval))
    trainer.run()
    env.close()


if __name__ == "__main__":
    main()
