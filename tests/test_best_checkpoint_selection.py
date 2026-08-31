import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from trainers.callbacks.checkpoint import BestCheckpointCallback


class _SavingAgent:
    def __init__(self):
        self.paths = []

    def save(self, path):
        self.paths.append(path)


def _evaluation(success=1.0, collision=0.0, episode_return=500.0, length=150.0, tracking=0.02):
    return {
        "eval/success_rate": success,
        "eval/collision_rate": collision,
        "eval/return": episode_return,
        "eval/length": length,
        "eval/tracking_position_rmse": tracking,
        "eval/done_reason_height_bound_rate": 0.0,
        "eval/done_reason_attitude_bound_rate": 0.0,
    }


class BestCheckpointSelectionTest(unittest.TestCase):
    def test_requires_three_consecutive_safe_evaluations(self):
        with tempfile.TemporaryDirectory() as tmp:
            callback = BestCheckpointCallback(tmp)
            trainer = SimpleNamespace(agent=_SavingAgent())
            for step, success in ((1000, 1.0), (2000, 0.8), (3000, 1.0), (4000, 1.0)):
                trainer.last_eval_step = step
                trainer.last_eval_info = _evaluation(success=success)
                callback.on_step(trainer)
            self.assertEqual(trainer.agent.paths, [])

            trainer.last_eval_step = 5000
            trainer.last_eval_info = _evaluation()
            callback.on_step(trainer)
            self.assertEqual(len(trainer.agent.paths), 1)
            metadata = json.loads(Path(tmp, "model_best_metrics.json").read_text())
            self.assertEqual(metadata["selection_window_steps"], [3000, 4000, 5000])

    def test_window_mean_return_breaks_tie_without_terminal_distance(self):
        with tempfile.TemporaryDirectory() as tmp:
            callback = BestCheckpointCallback(tmp)
            trainer = SimpleNamespace(agent=_SavingAgent())
            for step, episode_return in ((1000, 400.0), (2000, 400.0), (3000, 400.0), (4000, 500.0)):
                trainer.last_eval_step = step
                trainer.last_eval_info = _evaluation(episode_return=episode_return)
                callback.on_step(trainer)
            self.assertEqual(len(trainer.agent.paths), 2)


if __name__ == "__main__":
    unittest.main()
