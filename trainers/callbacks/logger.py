from torch.utils.tensorboard import SummaryWriter

class LoggerCallback:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def on_step(self, trainer):
        """记录训练信息（损失值等）"""
        if hasattr(trainer, "last_train_info") and trainer.last_train_info:
            for k, v in trainer.last_train_info.items():
                if v is not None:
                    self.writer.add_scalar(k, v, trainer.total_steps)

    def on_episode_end(self, trainer):
        """记录 episode 级别的指标"""
        # 记录返回值
        self.writer.add_scalar("episode/return", trainer.episode_return, trainer.total_steps)
        self.writer.add_scalar("episode/length", trainer.episode_step, trainer.total_steps)
        self.writer.add_scalar("episode/avg_reward", 
                              trainer.episode_return / max(1, trainer.episode_step), 
                              trainer.total_steps)
        
        # 记录环境信息（如果有）
        if hasattr(trainer, "last_info") and trainer.last_info:
            for k, v in trainer.last_info.items():
                if isinstance(v, (int, float, bool)):
                    # 布尔值转为 0/1 方便 TensorBoard 显示
                    val = float(v) if isinstance(v, (int, float, bool)) else v
                    self.writer.add_scalar(f"env/{k}", val, trainer.total_steps)

    def on_train_end(self, trainer):
        self.writer.close()