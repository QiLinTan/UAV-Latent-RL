import os
import json
import math
from collections import deque

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


class BestCheckpointCallback:
    """Select stable policies from a rolling window of navigation evaluations."""

    def __init__(self, save_dir, window_size=3, minimum_success_rate=0.9):
        self.save_dir = save_dir
        self.window_size = int(window_size)
        self.minimum_success_rate = float(minimum_success_rate)
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        self.best_score = None
        self.last_eval_step = None
        self.eval_window = deque(maxlen=self.window_size)
        os.makedirs(save_dir, exist_ok=True)

    @staticmethod
    def _finite(value, fallback):
        value = float(value)
        return value if math.isfinite(value) else float(fallback)

    def on_step(self, trainer):
        eval_step = getattr(trainer, "last_eval_step", None)
        eval_info = getattr(trainer, "last_eval_info", None)
        if eval_step is None or not eval_info or eval_step == self.last_eval_step:
            return
        self.last_eval_step = eval_step

        success = self._finite(eval_info.get("eval/success_rate", 0.0), 0.0)
        collision = self._finite(eval_info.get("eval/collision_rate", 1.0), 1.0)
        episode_return = self._finite(eval_info.get("eval/return", -float("inf")), -float("inf"))
        self.eval_window.append(
            {
                "step": int(eval_step),
                "success": success,
                "collision": collision,
                "return": episode_return,
                "length": self._finite(eval_info.get("eval/length", float("inf")), float("inf")),
                "tracking_rmse": self._finite(
                    eval_info.get("eval/tracking_position_rmse", float("inf")), float("inf")
                ),
                "height_bound": self._finite(
                    eval_info.get("eval/done_reason_height_bound_rate", 1.0), 1.0
                ),
                "attitude_bound": self._finite(
                    eval_info.get("eval/done_reason_attitude_bound_rate", 1.0), 1.0
                ),
            }
        )
        if len(self.eval_window) < self.window_size:
            return

        if any(
            item["success"] < self.minimum_success_rate
            or item["collision"] > 0.0
            or item["height_bound"] > 0.0
            or item["attitude_bound"] > 0.0
            for item in self.eval_window
        ):
            return

        window_mean = {
            key: sum(item[key] for item in self.eval_window) / self.window_size
            for key in ("success", "collision", "return", "length", "tracking_rmse")
        }
        score = (
            window_mean["success"],
            -window_mean["collision"],
            window_mean["return"],
            -window_mean["length"],
            -window_mean["tracking_rmse"],
        )
        if self.best_score is not None and score <= self.best_score:
            return

        self.best_score = score
        path = os.path.join(self.save_dir, "model_best")
        trainer.agent.save(path)
        metadata = {
            "step": int(eval_step),
            "score": list(score),
            "selection_window_size": self.window_size,
            "selection_minimum_success_rate": self.minimum_success_rate,
            "selection_window_steps": [item["step"] for item in self.eval_window],
            **{f"selection_window_mean/{key}": float(value) for key, value in window_mean.items()},
            **{key: float(value) for key, value in eval_info.items()},
        }
        with open(os.path.join(self.save_dir, "model_best_metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
        print(
            f"[Save best] {path} @ step={eval_step} "
            f"window_success={window_mean['success']:.3f} "
            f"window_collision={window_mean['collision']:.3f} "
            f"window_return={window_mean['return']:.2f}"
        )

    def on_episode_end(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass
