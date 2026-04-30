import argparse
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.gym_pybullet_compat import ensure_gym_pybullet_envs_compat

ensure_gym_pybullet_envs_compat()

from envs.ForestAviary import CustomForestAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

from algos.td3 import TD3, TD3Plain
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
    parser.add_argument(
        "--actor_updates_encoder",
        type=str2bool,
        default=False,
        help="Kept for backward compatibility. V1 trust mode keeps encoder detached from actor/critic gradients.",
    )
    parser.add_argument("--latent_input_scale", type=float, default=0.1)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--trust_alpha", type=float, default=0.5)
    parser.add_argument("--trust_beta", type=float, default=0.5)
    parser.add_argument("--trust_q_min", type=float, default=0.05)
    parser.add_argument("--trust_q_max", type=float, default=1.0)
    parser.add_argument("--trust_ema_momentum", type=float, default=0.99)
    parser.add_argument("--trust_warmup_steps", type=int, default=10000)
    parser.add_argument("--reward_scale", type=float, default=0.01)

    parser.add_argument("--expl_noise_start", type=float, default=0.5)
    parser.add_argument("--expl_noise_end", type=float, default=0.1)
    parser.add_argument("--noise_decay_steps", type=float, default=100_000)

    parser.add_argument("--eval_interval", type=int, default=10_000)
    parser.add_argument("--eval_episodes", type=int, default=1)
    parser.add_argument("--eval_gui", type=str2bool, default=True)
    parser.add_argument("--eval_stepsleep", type=str2bool, default=True)

    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--ckpt_interval", type=int, default=50_000)
    return parser


def main():
    args = _make_argparser().parse_args()

    np.random.seed(args.seed)

    init_xyzs = np.array([[-3.5, 0.0, 1.0]], dtype=np.float32)
    init_rpys = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

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
    )

    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(env.action_space.shape[-1])
    max_action = float(env.action_space.high.flatten()[0])

    if args.use_latent:
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
            trust_alpha=args.trust_alpha,
            trust_beta=args.trust_beta,
            trust_q_min=args.trust_q_min,
            trust_q_max=args.trust_q_max,
            trust_ema_momentum=args.trust_ema_momentum,
            trust_warmup_steps=args.trust_warmup_steps,
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
        )
    )
    trainer.add_callback(MonitorCallback(interval=args.eval_interval))
    trainer.run()
    env.close()


if __name__ == "__main__":
    main()
