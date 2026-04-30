import numpy as np
import time

from envs.preprocess import preprocess_state
from gym_pybullet_drones.utils.utils import sync

class EvalCallback:
    def __init__(self, env_cls, env_kwargs, interval=10000, episodes=1, step_sleep=True, eval_gui=True):
        self.env_cls = env_cls
        self.env_kwargs = dict(env_kwargs)
        self.interval = interval
        self.episodes = episodes
        self.step_sleep = step_sleep
        self.eval_gui = bool(eval_gui)

    def on_step(self, trainer):
        if trainer.total_steps % self.interval != 0:
            return

        returns = []

        print(f"[Eval @ {trainer.total_steps}] Starting evaluation...")

        eval_env = self.env_cls(**self.env_kwargs, gui=self.eval_gui)
        try:
            if hasattr(eval_env, "set_curriculum_stage_override") and hasattr(trainer.env, "get_curriculum_stage"):
                eval_env.set_curriculum_stage_override(trainer.env.get_curriculum_stage())
            for ep in range(self.episodes):
                obs, _ = eval_env.reset(seed=ep)
                state = preprocess_state(obs.reshape(-1))
                done = False
                ep_ret = 0.0
                start_time = time.time()
                step_idx = 0

                while not done:
                    action = trainer.agent.select_action(state)
                    obs, reward, terminated, truncated, _ = eval_env.step(action.reshape(1, -1))
                    state = preprocess_state(obs.reshape(-1))
                    done = terminated or truncated
                    ep_ret += float(reward)
                    step_idx += 1
                    if self.step_sleep:
                        sync(step_idx, start_time, eval_env.CTRL_TIMESTEP)

                returns.append(ep_ret)
        finally:
            eval_env.close()

        eval_return = float(np.mean(returns))
        trainer.last_eval_return = eval_return
        print(f"[Eval @ {trainer.total_steps}] return={eval_return:.2f}")

    def on_episode_end(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass
