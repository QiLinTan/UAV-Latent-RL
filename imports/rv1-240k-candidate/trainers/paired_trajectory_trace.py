"""Read-only 15 Hz / 120 Hz traces for targeted K1--Full evaluations."""

import json
from pathlib import Path

import numpy as np


class PairedTrajectoryTrace:
    def __init__(self, env, environment_seed, *, trace_id=None):
        self.env = env
        self.environment_seed = None if environment_seed is None else int(environment_seed)
        self.trace_id = (
            f"trace_seed{self.environment_seed}" if trace_id is None else str(trace_id)
        )
        self.upper = []
        self.lower = []
        self.initial_position = np.asarray(env.pos[0], dtype=np.float64).copy()
        self.initial_velocity = np.asarray(env.vel[0], dtype=np.float64).copy()
        self.target = np.asarray(env.TARGET_POS, dtype=np.float64).copy()
        self._base_step = None

    def install(self):
        # Only Full calls base_env.step. Wrapping the instance method records
        # its output without changing the action, reward, or transition.
        self._base_step = self.env.base_env.step

        def observed_step(action):
            result = self._base_step(action)
            self.record_lower(action, result[1], result[4])
            return result

        self.env.base_env.step = observed_step

    def uninstall(self):
        if self._base_step is not None:
            self.env.base_env.step = self._base_step
            self._base_step = None

    def record_lower(self, motor_action, reward, info):
        env = self.env
        sample = env.last_reference_sample
        if sample is None:
            raise RuntimeError("Full lower step had no ReferencePacket sample")
        raw_action = np.asarray(motor_action, dtype=np.float64).reshape(4)
        requested_rpm = env.motor_codec.normalized_action_to_rpm(raw_action)
        actual_rpm = np.asarray(env.base_env._last_motor_rpm[0], dtype=np.float64)
        self.lower.append({
            "time_seconds": (env.runtime_lower_step + 1) * env.lower_control_dt,
            "position": np.asarray(env.pos[0], dtype=np.float64).copy(),
            "velocity": np.asarray(env.vel[0], dtype=np.float64).copy(),
            "rpy": np.asarray(env.rpy[0], dtype=np.float64).copy(),
            "angular_velocity": np.asarray(env.ang_v[0], dtype=np.float64).copy(),
            "reference_position": np.asarray(sample["current_position"], dtype=np.float64).copy(),
            "reference_velocity": np.asarray(sample["current_velocity"], dtype=np.float64).copy(),
            "command_velocity": np.asarray(env.command_velocity, dtype=np.float64).copy(),
            "motor_action": raw_action.copy(),
            "commanded_rpm_clipped": requested_rpm.copy(),
            "executed_rpm": actual_rpm.copy(),
            "packet_version": int(sample["version"]),
            "packet_age_seconds": float(sample["age_seconds"]),
            "packet_valid": bool(sample["valid"]),
            "clearance": float(info.get("min_tree_clearance", np.nan)),
            "goal_distance": float(info.get("goal_distance", np.nan)),
            "reward": float(reward),
            "collision": bool(info.get("collision", False)),
        })

    def record_upper(self, action, policy_action, reward, info, *, pre_observation=None):
        env = self.env
        goal_xy = self.target[:2] - np.asarray(env.pos[0], dtype=np.float64)[:2]
        goal_xy /= max(float(np.linalg.norm(goal_xy)), 1e-9)
        goal_lateral = np.array([-goal_xy[1], goal_xy[0]], dtype=np.float64)
        lateral_command = float(np.dot(np.asarray(env.command_velocity)[:2], goal_lateral))
        turn_sign = int(np.sign(lateral_command)) if abs(lateral_command) >= 0.05 else 0
        packet = env.last_reference_packet
        endpoint = (
            np.asarray(packet.to_world(packet.positions[-1]), dtype=np.float64)
            if packet is not None else np.full(3, np.nan)
        )
        self.upper.append({
            "time_seconds": env.runtime_lower_step * env.lower_control_dt,
            "pre_observation": np.asarray(pre_observation, dtype=np.float32).copy()
            if pre_observation is not None else np.empty(0, dtype=np.float32),
            "position": np.asarray(env.pos[0], dtype=np.float64).copy(),
            "velocity": np.asarray(env.vel[0], dtype=np.float64).copy(),
            "rpy": np.asarray(env.rpy[0], dtype=np.float64).copy(),
            "action": np.asarray(action, dtype=np.float64).reshape(-1).copy(),
            "policy_action": np.asarray(policy_action, dtype=np.float64).reshape(-1).copy(),
            "command_velocity": np.asarray(env.command_velocity, dtype=np.float64).copy(),
            "goal_lateral_command_mps": lateral_command,
            "goal_lateral_turn_sign": turn_sign,
            "reference_position": np.asarray(env.reference_position, dtype=np.float64).copy(),
            "reference_velocity": np.asarray(env.reference_velocity, dtype=np.float64).copy(),
            "packet_endpoint": endpoint,
            "packet_version": int(packet.version) if packet is not None else -1,
            "packet_t_gen": float(packet.t_gen) if packet is not None else np.nan,
            "clearance": float(info.get("min_tree_clearance", np.nan)),
            "goal_distance": float(info.get("goal_distance", np.nan)),
            "reward": float(reward),
            "collision": bool(info.get("collision", False)),
        })

    def tree_specs(self):
        spec = getattr(self.env.base_env, "MAP_SPEC", None)
        if spec is not None and spec.obstacles:
            return [{"xy": o.position[:2], "radius": o.radius, "height": o.height}
                    for o in spec.obstacles if o.kind == "cylinder"]
        return getattr(self.env, "_tree_specs", ())

    def save(self, output_dir, *, done_reason, success, final_goal_distance):
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        stem = self.trace_id
        arrays = {}
        for prefix, rows in (("upper", self.upper), ("lower", self.lower)):
            if not rows:
                continue
            for key in rows[0]:
                arrays[f"{prefix}_{key}"] = np.asarray([row[key] for row in rows])
        np.savez_compressed(root / f"{stem}.npz", **arrays)
        metadata = {
            "schema_version": 1,
            "environment_seed": self.environment_seed,
            "trace_id": self.trace_id,
            "mode": "full" if self.lower else "k1",
            "upper_rate_hz": float(self.env.CTRL_FREQ),
            "lower_rate_hz": float(self.env.lower_ctrl_freq) if self.lower else None,
            "initial_position": self.initial_position.tolist(),
            "initial_velocity": self.initial_velocity.tolist(),
            "target_position": self.target.tolist(),
            "upper_steps": len(self.upper),
            "lower_steps": len(self.lower),
            "done_reason": str(done_reason),
            "success": bool(success),
            "final_goal_distance": float(final_goal_distance),
            "tree_specs": [
                {
                    "xy": np.asarray(spec["xy"], dtype=float).tolist(),
                    "radius": float(spec["radius"]),
                    "height": float(spec["height"]),
                }
                for spec in self.tree_specs()
            ],
        }
        if hasattr(self.env, "_pool_info"):
            metadata.update(self.env._pool_info())
        samples = self.lower or self.upper
        if samples:
            clearances = np.asarray([row["clearance"] for row in samples])
            finite = clearances[np.isfinite(clearances)]
            metadata["clearance_quantiles_m"] = (
                dict(zip(("min", "p05", "p50", "p95"),
                         np.quantile(finite, [0, .05, .5, .95]).tolist()))
                if finite.size else None
            )
            metadata["entered_guidance_1p2"] = bool(
                np.linalg.norm(self.initial_position - self.target) < 1.2
                or any(row["goal_distance"] < 1.2 for row in samples)
            )
        (root / f"{stem}.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
