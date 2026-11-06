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

    def __init__(
        self,
        save_dir,
        window_size=3,
        minimum_success_rate=0.9,
        maximum_collision_rate=0.0,
        final_eval_callback=None,
        prefer_higher_curriculum_stage=False,
        minimum_retention_success_rate=None,
        fallback_checkpoint=None,
        enable_conservative_phase=True,
    ):
        self.save_dir = save_dir
        self.window_size = int(window_size)
        self.minimum_success_rate = float(minimum_success_rate)
        self.maximum_collision_rate = float(maximum_collision_rate)
        self.final_eval_callback = final_eval_callback
        self.prefer_higher_curriculum_stage = bool(prefer_higher_curriculum_stage)
        self.minimum_retention_success_rate = minimum_retention_success_rate
        self.fallback_checkpoint = fallback_checkpoint
        self.enable_conservative_phase = bool(enable_conservative_phase)
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if not 0.0 <= self.minimum_success_rate <= 1.0:
            raise ValueError("minimum_success_rate must lie in [0, 1].")
        if not 0.0 <= self.maximum_collision_rate <= 1.0:
            raise ValueError("maximum_collision_rate must lie in [0, 1].")
        if minimum_retention_success_rate is not None and not 0.0 <= minimum_retention_success_rate <= 1.0:
            raise ValueError("minimum_retention_success_rate must lie in [0, 1].")
        self.best_score = None
        self.best_step = None
        self.best_stage = None
        self.peak_score = None
        self.peak_step = None
        self.peak_stage = None
        self.last_eval_step = None
        self.eval_window = deque(maxlen=self.window_size)
        self._window_stage = None
        os.makedirs(save_dir, exist_ok=True)

    @staticmethod
    def _finite(value, fallback):
        value = float(value)
        return value if math.isfinite(value) else float(fallback)

    @staticmethod
    def _serializable_eval_info(eval_info):
        """Preserve optional telemetry (e.g. terminal curriculum thresholds)."""
        return {
            key: None if value is None else float(value)
            for key, value in eval_info.items()
        }

    def on_step(self, trainer):
        eval_step = getattr(trainer, "last_eval_step", None)
        eval_info = getattr(trainer, "last_eval_info", None)
        if eval_step is None or not eval_info or eval_step == self.last_eval_step:
            return
        self.last_eval_step = eval_step

        stage = int(self._finite(eval_info.get("eval/curriculum_stage", 0), 0))
        if self.prefer_higher_curriculum_stage and stage != self._window_stage:
            self.eval_window.clear()
            self._window_stage = stage

        success = self._finite(eval_info.get("eval/success_rate", 0.0), 0.0)
        collision = self._finite(eval_info.get("eval/collision_rate", 1.0), 1.0)
        if self.minimum_retention_success_rate is not None:
            retained = self._finite(eval_info.get("eval/stage0_success_rate", 0.0), 0.0)
            if retained < self.minimum_retention_success_rate:
                self.eval_window.clear()
                return
        episode_return = self._finite(eval_info.get("eval/return", -float("inf")), -float("inf"))
        final_goal_distance = self._finite(
            eval_info.get("eval/final_goal_distance", float("inf")), float("inf")
        )
        length = self._finite(eval_info.get("eval/length", float("inf")), float("inf"))
        tracking_rmse = self._finite(
            eval_info.get("eval/tracking_position_rmse", float("inf")), float("inf")
        )
        peak_score = (
            success,
            -collision,
            -final_goal_distance,
            episode_return,
            -length,
            -tracking_rmse,
        )
        if self.prefer_higher_curriculum_stage:
            peak_score = (stage, *peak_score)
        if self.peak_score is None or peak_score > self.peak_score:
            self.peak_score = peak_score
            self.peak_step = int(eval_step)
            self.peak_stage = stage
            peak_path = os.path.join(self.save_dir, "model_peak_validation")
            trainer.agent.save(peak_path)
            peak_metadata = {
                "step": int(eval_step),
                "score": list(peak_score),
                "selection": "single_evaluation_lexicographic_peak",
                **self._serializable_eval_info(eval_info),
            }
            with open(
                os.path.join(self.save_dir, "model_peak_validation_metrics.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(peak_metadata, handle, ensure_ascii=False, indent=2)
            print(
                f"[Save peak] {peak_path} @ step={eval_step} "
                f"success={success:.3f} collision={collision:.3f} "
                f"final_dist={final_goal_distance:.3f}"
            )
        self.eval_window.append(
            {
                "step": int(eval_step),
                "success": success,
                "collision": collision,
                "return": episode_return,
                "length": length,
                "tracking_rmse": tracking_rmse,
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
            or item["collision"] > self.maximum_collision_rate
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
        if self.prefer_higher_curriculum_stage:
            score = (stage, *score)
        if self.best_score is not None and score <= self.best_score:
            return

        self.best_score = score
        self.best_stage = stage
        path = os.path.join(self.save_dir, "model_best")
        # The first stable checkpoint starts the conservative phase before its
        # optimizer state is saved. The current actor is then frozen as a
        # retention target for successful replay states.
        if self.enable_conservative_phase and hasattr(trainer.agent, "enter_conservative_phase"):
            trainer.agent.enter_conservative_phase()
        trainer.agent.save(path)
        self.best_step = int(eval_step)
        if hasattr(trainer.agent, "set_policy_anchor_from_current"):
            trainer.agent.set_policy_anchor_from_current()
        metadata = {
            "step": int(eval_step),
            "score": list(score),
            "selection_window_size": self.window_size,
            "selection_minimum_success_rate": self.minimum_success_rate,
            "selection_maximum_collision_rate": self.maximum_collision_rate,
            "selection_window_steps": [item["step"] for item in self.eval_window],
            **{f"selection_window_mean/{key}": float(value) for key, value in window_mean.items()},
            **self._serializable_eval_info(eval_info),
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
        selected_path = None
        selected_step = None
        if self.best_step is not None:
            selected_path = os.path.join(self.save_dir, "model_best")
            selected_step = self.best_step
            selected_stage = self.best_stage
        elif self.peak_step is not None:
            selected_path = os.path.join(self.save_dir, "model_peak_validation")
            selected_step = self.peak_step
            selected_stage = self.peak_stage
        if selected_path is None:
            if self.fallback_checkpoint is None:
                return
            selected_path = self.fallback_checkpoint
            selected_step = 0
            selected_stage = 0
        if not hasattr(trainer.agent, "load"):
            raise RuntimeError("Cannot restore the selected policy: agent has no load().")
        trainer.agent.load(selected_path)
        trainer.selected_checkpoint_path = selected_path
        trainer.selected_checkpoint_step = int(selected_step)
        print(f"[Restore selected] {selected_path} @ step={selected_step}")
        if self.final_eval_callback is not None:
            print("[Final selected evaluation] evaluating restored policy")
            self.final_eval_callback.on_step(trainer, force=True)
        final_metadata = {
            "selected_checkpoint": os.path.basename(selected_path),
            "selected_checkpoint_source_path": selected_path,
            "selected_checkpoint_step": int(selected_step),
            "selected_checkpoint_curriculum_stage": int(selected_stage),
            "model_final_is_restored_selection": True,
            **self._serializable_eval_info(
                getattr(trainer, "last_eval_info", {}) or {}
            ),
        }
        with open(
            os.path.join(self.save_dir, "model_final_selection.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(final_metadata, handle, ensure_ascii=False, indent=2)


class ConvergenceGuardCallback:
    """Stop and restore the best policy if a stable policy persistently collapses.

    This is intentionally opt-in. It contains late off-policy TD3 degradation
    without hiding it: the stop reason is stored on the trainer and printed,
    while ``model_final`` is written from the restored ``model_best`` weights.
    """

    def __init__(
        self,
        best_callback,
        *,
        enabled=False,
        min_success_rate=0.8,
        max_collision_rate=0.2,
        patience_evals=3,
    ):
        self.best_callback = best_callback
        self.enabled = bool(enabled)
        self.min_success_rate = float(min_success_rate)
        self.max_collision_rate = float(max_collision_rate)
        self.patience_evals = int(patience_evals)
        if not 0.0 <= self.min_success_rate <= 1.0:
            raise ValueError("min_success_rate must lie in [0, 1].")
        if not 0.0 <= self.max_collision_rate <= 1.0:
            raise ValueError("max_collision_rate must lie in [0, 1].")
        if self.patience_evals <= 0:
            raise ValueError("patience_evals must be positive.")
        self._last_eval_step = None
        self._best_step_seen = None
        self._consecutive_degraded = 0

    def on_step(self, trainer):
        if not self.enabled:
            return
        eval_step = getattr(trainer, "last_eval_step", None)
        eval_info = getattr(trainer, "last_eval_info", None)
        best_step = getattr(self.best_callback, "best_step", None)
        if eval_step is None or not eval_info or eval_step == self._last_eval_step:
            return
        self._last_eval_step = eval_step
        if best_step is None:
            return
        if best_step != self._best_step_seen:
            self._best_step_seen = best_step
            self._consecutive_degraded = 0
        if int(eval_step) <= int(best_step):
            return

        success = float(eval_info.get("eval/success_rate", 0.0))
        collision = float(eval_info.get("eval/collision_rate", 1.0))
        degraded = success < self.min_success_rate or collision > self.max_collision_rate
        self._consecutive_degraded = self._consecutive_degraded + 1 if degraded else 0
        if self._consecutive_degraded < self.patience_evals:
            return

        best_path = os.path.join(self.best_callback.save_dir, "model_best")
        if not hasattr(trainer.agent, "load"):
            raise RuntimeError("Convergence guard cannot restore an agent without load().")
        trainer.agent.load(best_path)
        trainer.request_stop = True
        trainer.stop_reason = (
            "convergence_guard: restored model_best from step "
            f"{best_step} after {self._consecutive_degraded} degraded evaluations "
            f"(success={success:.3f}, collision={collision:.3f})"
        )
        print(f"[Convergence guard] {trainer.stop_reason}")

    def on_episode_end(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass
