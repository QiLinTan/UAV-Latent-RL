import os

class CheckpointCallback:
    def __init__(self, save_dir, interval=50000):
        self.save_dir = save_dir
        self.interval = interval
        os.makedirs(save_dir, exist_ok=True)

    def on_step(self, trainer):
        if trainer.total_steps % self.interval == 0:
            path = os.path.join(self.save_dir, f"model_{trainer.total_steps}")
            trainer.agent.save(path)
            print(f"[Save] {path}")

    def on_episode_end(self, trainer):
        pass

    def on_train_end(self, trainer):
        path = os.path.join(self.save_dir, "model_final")
        trainer.agent.save(path)
        print(f"[Save] {path}")