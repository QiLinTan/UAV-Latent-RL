import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from trainers.callbacks.checkpoint import BestCheckpointCallback, ConvergenceGuardCallback


class _SavingAgent:
    def __init__(self):
        self.paths = []

    def save(self, path):
        self.paths.append(path)

    def load(self, path):
        self.loaded_path = path


class _FinalEval:
    def __init__(self):
        self.calls = []

    def on_step(self, trainer, *, force=False):
        self.calls.append((trainer.total_steps, force))
        trainer.last_eval_info = _evaluation()


def _evaluation(
    success=1.0,
    collision=0.0,
    episode_return=500.0,
    length=150.0,
    tracking=0.02,
    final_distance=0.1,
):
    return {
        "eval/success_rate": success,
        "eval/collision_rate": collision,
        "eval/return": episode_return,
        "eval/length": length,
        "eval/tracking_position_rmse": tracking,
        "eval/final_goal_distance": final_distance,
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
            self.assertEqual(
                [path for path in trainer.agent.paths if path.endswith("model_best")],
                [],
            )

            trainer.last_eval_step = 5000
            trainer.last_eval_info = _evaluation()
            callback.on_step(trainer)
            self.assertEqual(
                len([path for path in trainer.agent.paths if path.endswith("model_best")]),
                1,
            )
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
            self.assertEqual(
                len([path for path in trainer.agent.paths if path.endswith("model_best")]),
                2,
            )

    def test_single_evaluation_peak_is_not_lost_after_later_degradation(self):
        with tempfile.TemporaryDirectory() as tmp:
            callback = BestCheckpointCallback(tmp)
            trainer = SimpleNamespace(agent=_SavingAgent())
            trainer.last_eval_step = 94000
            trainer.last_eval_info = _evaluation(
                success=0.8, collision=0.1, final_distance=0.18
            )
            callback.on_step(trainer)
            trainer.last_eval_step = 96000
            trainer.last_eval_info = _evaluation(
                success=0.7, collision=0.2, final_distance=0.10
            )
            callback.on_step(trainer)

            peak_paths = [
                path
                for path in trainer.agent.paths
                if path.endswith("model_peak_validation")
            ]
            self.assertEqual(len(peak_paths), 1)
            self.assertEqual(callback.peak_step, 94000)
            metadata = json.loads(
                Path(tmp, "model_peak_validation_metrics.json").read_text()
            )
            self.assertEqual(metadata["step"], 94000)

    def test_optional_curriculum_thresholds_are_serialized_as_null(self):
        with tempfile.TemporaryDirectory() as tmp:
            callback = BestCheckpointCallback(tmp, window_size=3)
            trainer = SimpleNamespace(agent=_SavingAgent())
            for step in (1000, 2000, 3000):
                trainer.last_eval_step = step
                trainer.last_eval_info = _evaluation()
                trainer.last_eval_info.update(
                    {
                        "eval/curriculum_success_threshold": None,
                        "eval/curriculum_max_collision_rate": None,
                    }
                )
                callback.on_step(trainer)

            peak_metadata = json.loads(
                Path(tmp, "model_peak_validation_metrics.json").read_text()
            )
            best_metadata = json.loads(Path(tmp, "model_best_metrics.json").read_text())
            self.assertIsNone(peak_metadata["eval/curriculum_success_threshold"])
            self.assertIsNone(best_metadata["eval/curriculum_max_collision_rate"])

    def test_train_end_restores_peak_and_forces_matching_final_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            final_eval = _FinalEval()
            callback = BestCheckpointCallback(
                tmp, window_size=3, final_eval_callback=final_eval
            )
            trainer = SimpleNamespace(agent=_SavingAgent(), total_steps=120000)
            trainer.last_eval_step = 94000
            trainer.last_eval_info = _evaluation(success=0.8)
            callback.on_step(trainer)
            callback.on_train_end(trainer)

            self.assertTrue(trainer.agent.loaded_path.endswith("model_peak_validation"))
            self.assertEqual(final_eval.calls, [(120000, True)])
            metadata = json.loads(Path(tmp, "model_final_selection.json").read_text())
            self.assertEqual(metadata["selected_checkpoint_step"], 94000)

    def test_convergence_guard_restores_best_after_three_bad_evaluations(self):
        with tempfile.TemporaryDirectory() as tmp:
            best = BestCheckpointCallback(tmp)
            guard = ConvergenceGuardCallback(best, enabled=True, patience_evals=3)
            trainer = SimpleNamespace(agent=_SavingAgent(), request_stop=False, stop_reason=None)
            for step in (1000, 2000, 3000):
                trainer.last_eval_step = step
                trainer.last_eval_info = _evaluation()
                best.on_step(trainer)
                guard.on_step(trainer)
            self.assertEqual(best.best_step, 3000)

            for step in (4000, 5000, 6000):
                trainer.last_eval_step = step
                trainer.last_eval_info = _evaluation(success=0.0, collision=1.0)
                best.on_step(trainer)
                guard.on_step(trainer)
            self.assertTrue(trainer.request_stop)
            self.assertIn("model_best", trainer.agent.loaded_path)


if __name__ == "__main__":
    unittest.main()
