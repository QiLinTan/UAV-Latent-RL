"""Audit frozen Encoder -> Actor -> native motor RPM, without a planner/PID."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from utils.gym_pybullet_compat import ensure_gym_pybullet_envs_compat
ensure_gym_pybullet_envs_compat()
from gym_pybullet_drones.utils.enums import ActionType, ObservationType
from algos.td3.networks import Actor
from benchmarks.map_contract import from_environment
from envs.ForestAviary import CustomForestAviary
from envs.preprocess import preprocess_state
from models.encoder import Encoder
from models.motor_action_codec import MotorActionCodec, MotorPhysicalLimits, ASYMMETRIC_RPM
from scripts.freeze_logical_async import file_record


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def array_stats(values):
    values = np.asarray(values)
    return {"shape": list(values.shape), "min": float(values.min()), "max": float(values.max()),
            "mean": float(values.mean()), "std": float(values.std()), "first_values": values.reshape(-1)[:12].tolist()}


def load_networks(prefix):
    encoder, actor = Encoder(263, 16), Actor(16, 4, 1.0)
    records = {}
    for name, module in (("encoder", encoder), ("actor", actor)):
        path = Path(str(prefix) + "_" + name)
        state = torch.load(path, map_location="cpu", weights_only=True)
        result = module.load_state_dict(state, strict=True)
        module.eval().requires_grad_(False)
        records[name] = {**file_record(path), "missing_keys": result.missing_keys,
                         "unexpected_keys": result.unexpected_keys, "eval": not module.training,
                         "architecture": str(module), "parameter_count": sum(p.numel() for p in module.parameters())}
    return encoder, actor, records


def historical_env_class(commit, output):
    """Load repository-owned historical modules under an isolated namespace.

    Does not replace the working environment or silently patch its semantics.
    External simulator sources are current and explicitly hashed in the audit.
    """
    package = "native_audit_historical_forest"
    root_module = types.ModuleType(package)
    root_module.__path__ = []
    sys.modules[package] = root_module
    records = {}
    for name in ("geometry", "curriculum", "layout", "rewards", "core"):
        relative = f"envs/forest/{name}.py"
        source = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        path = output / "historical_source" / f"{name}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source)
        module = types.ModuleType(package + "." + name)
        module.__package__ = package
        module.__file__ = str(path)
        sys.modules[module.__name__] = module
        exec(compile(source, str(path), "exec"), module.__dict__)
        records[relative] = file_record(path)
    return sys.modules[package + ".core"].CustomForestAviary, records


def historical_evidence(log_dir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = accumulator.Tags()["scalars"]
    output = {"event_files": [file_record(path) for path in sorted(Path(log_dir).glob("events.*"))], "series": {}}
    for tag in ("eval/success_rate", "eval/collision_rate", "eval/final_goal_distance", "eval/return", "eval/length", "env/curriculum_stage"):
        if tag in tags:
            events = accumulator.Scalars(tag)
            output["series"][tag] = [{"step": e.step, "value": e.value} for e in events]
    successes = output["series"].get("eval/success_rate", [])
    output["maximum_logged_eval_success_rate"] = max((e["value"] for e in successes), default=None)
    output["last_logged_eval_success_rate"] = successes[-1]["value"] if successes else None
    return output


def checkpoint_eval_step(checkpoint_prefix):
    prefix = Path(checkpoint_prefix)
    if prefix.name == "model_best":
        metrics = prefix.parent / "model_best_metrics.json"
        if metrics.exists():
            return int(json.loads(metrics.read_text())["step"])
    if prefix.name.startswith("model_") and prefix.name.split("_", 1)[1].isdigit():
        return int(prefix.name.split("_", 1)[1])
    return None


def scalar_at_step(history, tag, step=None):
    events = history["series"].get(tag, [])
    if not events:
        return None
    if step is None:
        return events[-1]["value"]
    matches = [event for event in events if event["step"] == step]
    return matches[-1]["value"] if matches else None


def terminal_reason(env, info, terminated, truncated):
    if info.get("done_reason") not in (None, "running", "unknown"):
        return str(info["done_reason"])
    if info.get("success", False):
        return "success"
    if info.get("collision", False):
        return "collision"
    if truncated:
        position, rpy = env.pos[0], env.rpy[0]
        xy_limit = max(env.FOREST_HALF_EXTENT + .75, np.abs(env.START_POS[:2]).max() + .5, np.abs(env.TARGET_POS[:2]).max() + .5)
        if np.abs(position[:2]).max() > xy_limit:
            return "xy_bound"
        if position[2] < .08 or position[2] > 2.75:
            return "height_bound"
        if np.abs(rpy[:2]).max() > .65:
            return "attitude_bound"
        return "timeout"
    return "terminated_unknown" if terminated else "running"


def rollout(env, encoder, actor, *, seed, profile, output, codec_mode="native_symmetric_5pct"):
    observation, reset_info = env.reset(seed=seed)
    raw = np.asarray(observation, dtype=np.float32).reshape(-1)
    if raw.shape != (263,):
        raise ValueError(f"Checkpoint requires 263D, observed {raw.shape}")
    reset_hash = hashlib.sha256(raw.tobytes()).hexdigest()
    reset_stats = {"raw": array_stats(raw), "preprocessed": array_stats(preprocess_state(raw))}
    for name, section in (("kin12", slice(0,12)), ("action_history240", slice(12,252)), ("relative_goal3", slice(252,255)), ("ranges8", slice(255,263))):
        reset_stats[name] = array_stats(raw[section])
    if codec_mode == ASYMMETRIC_RPM:
        env.set_motor_action_codec(MotorActionCodec(MotorPhysicalLimits(0, env.HOVER_RPM, env.MAX_RPM, env.KF), ASYMMETRIC_RPM))
    captured = {}
    original_decoder = env._preprocessAction

    def capture(action):
        captured["env_action"] = np.asarray(action).copy().reshape(-1)
        rpm = original_decoder(action)
        captured["rpm"] = np.asarray(rpm).copy().reshape(-1)
        return rpm

    env._preprocessAction = capture
    names = ("time", "raw_observation", "preprocessed_state", "latent", "actor_action", "clipped_action", "env_action", "rpm", "position", "velocity", "yaw", "ranges", "goal_distance", "minimum_clearance", "next_position", "next_velocity", "reward", "done_reason")
    trace = {name: [] for name in names}
    map_record = from_environment(env, family=profile, seed=seed)
    step, reward_sum = 0, 0.0
    terminated = truncated = False
    with torch.inference_mode():
        while not (terminated or truncated):
            state = preprocess_state(raw)
            z = encoder(torch.from_numpy(state).unsqueeze(0))
            action = actor(z)[0].numpy().copy()
            clipped = np.clip(action, -1, 1)
            before_position, before_velocity = env.pos[0].copy(), env.vel[0].copy()
            before_yaw = float(env.rpy[0][2])
            clearance = float(env._computeNearestTreeClearance(before_position))
            observation, reward, terminated, truncated, info = env.step(clipped.reshape(1,4))
            reason = terminal_reason(env, info, terminated, truncated)
            row = dict(time=step*env.CTRL_TIMESTEP, raw_observation=raw.copy(), preprocessed_state=state,
                       latent=z[0].numpy().copy(), actor_action=action, clipped_action=clipped,
                       env_action=captured["env_action"], rpm=captured["rpm"], position=before_position,
                       velocity=before_velocity, yaw=before_yaw, ranges=raw[-8:].copy(),
                       goal_distance=np.linalg.norm(env.TARGET_POS-before_position), minimum_clearance=clearance,
                       next_position=env.pos[0].copy(), next_velocity=env.vel[0].copy(), reward=float(reward), done_reason=reason)
            for name in names:
                trace[name].append(row[name])
            raw = np.asarray(observation, dtype=np.float32).reshape(-1)
            reward_sum += float(reward)
            step += 1
    output.mkdir(parents=True, exist_ok=False)
    trace = {name: np.asarray(value) for name, value in trace.items()}
    np.savez_compressed(output / "trace.npz", **trace)
    final_distance = float(np.linalg.norm(env.TARGET_POS-env.pos[0]))
    expected = env.HOVER_RPM * (1 + .05*trace["clipped_action"])
    summary = {
        "profile": profile, "seed": seed, "map": map_record, "map_id": map_record["map_id"],
        "reset_observation_sha256": reset_hash, "reset_statistics": reset_stats,
        "steps": step, "reward": reward_sum, "success": bool(info.get("success",False)),
        "collision": bool(info.get("collision",False)), "done_reason": reason,
        "final_distance": final_distance, "minimum_goal_distance": float(min(trace["goal_distance"].min(), final_distance)),
        "final_position": env.pos[0].copy(), "final_velocity": env.vel[0].copy(),
        "minimum_clearance": float(min(trace["minimum_clearance"].min(), env._computeNearestTreeClearance(env.pos[0]))),
        "codec_mode": codec_mode, "latent_input_scale": 1.0,
        "native_rpm_formula_max_error": float(np.abs(trace["rpm"] - expected).max()) if codec_mode == "native_symmetric_5pct" else None,
        "action_clipping_max_change": float(np.abs(trace["actor_action"]-trace["clipped_action"]).max()),
        "action_statistics": array_stats(trace["actor_action"]), "rpm_statistics": array_stats(trace["rpm"]),
        "preprocessed_statistics": array_stats(trace["preprocessed_state"]),
        "environment": {"pyb_freq": env.PYB_FREQ, "ctrl_freq": env.CTRL_FREQ, "episode_len_sec": env.EPISODE_LEN_SEC,
                        "collision_radius": env.COLLISION_R, "collision_height": env.COLLISION_H,
                        "goal_tolerance": env.GOAL_TOLERANCE, "curriculum_stage": env.get_curriculum_stage(),
                        "sensor_range": env.OBSTACLE_SENSOR_RANGE, "forest_half_extent": env.FOREST_HALF_EXTENT,
                        "hover_rpm": env.HOVER_RPM, "max_rpm": env.MAX_RPM,
                        "reward_parameters": vars(env._reward_model)},
        "trace": file_record(output / "trace.npz"),
        "trace_alignment": "All rows are pre-action state at t; RPM/action execute t->t+dt, next_position/velocity are post-action. Includes every action and terminal reason.",
    }
    write_json(output / "episode_summary.json", summary)
    print(f"{profile} seed={seed}: {reason}, distance={final_distance:.4f}, return={reward_sum:.3f}", flush=True)
    return clean_json(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/latent_only_affordance_dim16/model_final")
    parser.add_argument(
        "--log-dir",
        default="runs/latent_only_affordance_dim16",
        help="TensorBoard run directory that corresponds to --checkpoint.",
    )
    parser.add_argument("--output", default="analysis_outputs/native_actor_audit_v1")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    parser.add_argument("--historical-commit", default="5a4bfe3")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    encoder, actor, networks = load_networks(args.checkpoint)
    history = historical_evidence(ROOT / args.log_dir)
    write_json(output / "historical_metrics.json", history)
    old_class, historical_sources = historical_env_class(args.historical_commit, output)
    profiles = [
        ("historical_stage2_native", old_class, 24, True, 2, "native_symmetric_5pct"),
        ("current_stage2_native", CustomForestAviary, 24, True, 2, "native_symmetric_5pct"),
        ("current_clear_native", CustomForestAviary, 0, False, 3, "native_symmetric_5pct"),
        ("current_forest_native", CustomForestAviary, 24, True, 3, "native_symmetric_5pct"),
        ("current_clear_asymmetric_negative_control", CustomForestAviary, 0, False, 3, ASYMMETRIC_RPM),
    ]
    summaries = {}
    for profile, env_class, trees, blocking, stage, codec in profiles:
        results = []
        for seed in args.seeds:
            with contextlib.redirect_stdout(io.StringIO()):
                env = env_class(obs=ObservationType.KIN, act=ActionType.RPM, pyb_freq=240, ctrl_freq=120,
                                gui=False, curriculum=True, num_trees=trees, route_blocking_tree=blocking)
            try:
                env.set_curriculum_stage_override(stage)
                results.append(rollout(env, encoder, actor, seed=seed, profile=profile, output=output/profile/f"seed_{seed}", codec_mode=codec))
            finally:
                env.close()
        summaries[profile] = {"episodes":results, "success_rate": float(np.mean([r["success"] for r in results])),
                              "collision_rate":float(np.mean([r["collision"] for r in results])),
                              "mean_final_distance":float(np.mean([r["final_distance"] for r in results]))}
    from gym_pybullet_drones.envs import BaseRLAviary as base_module
    # Evaluate the archived claim against its actual logged seed-0 evaluation.
    original = summaries["historical_stage2_native"]["episodes"]
    seed_zero = next((r for r in original if r["seed"] == 0), None)
    comparison_step = checkpoint_eval_step(args.checkpoint)
    comparisons = {}
    if seed_zero:
        for tag,key,tolerance in (("eval/return","reward",1e-3),("eval/final_goal_distance","final_distance",1e-5),("eval/length","steps",0)):
            value = scalar_at_step(history, tag, comparison_step)
            if value is not None:
                comparisons[tag] = {"historical":value,"reproduced":seed_zero[key],"absolute_error":abs(value-seed_zero[key]),"tolerance":tolerance,"passed":abs(value-seed_zero[key])<=tolerance}
    reproducible = bool(comparisons) and all(c["passed"] for c in comparisons.values())
    paired_traces = {}
    for seed in args.seeds:
        with np.load(output/"historical_stage2_native"/f"seed_{seed}"/"trace.npz") as old, np.load(output/"current_stage2_native"/f"seed_{seed}"/"trace.npz") as new:
            paired_traces[str(seed)] = {key:bool(np.array_equal(old[key],new[key])) for key in ("raw_observation","latent","actor_action","rpm","position")}
    report = {"networks": networks, "historical_sources": historical_sources,
              "historical_commit": args.historical_commit, "historical_metrics": "historical_metrics.json",
              "checkpoint_prefix": args.checkpoint, "log_dir": args.log_dir,
              "comparison_eval_step": comparison_step,
              "external_base_rl": file_record(Path(base_module.__file__)), "profiles": summaries,
              "historical_metric_comparison":comparisons, "historical_result_reproduced":reproducible,
              "current_vs_historical_trace_equality":paired_traces,
              "classification": "historical_failure_reproduced" if reproducible and history["last_logged_eval_success_rate"]==0 else "requires_review",
              "teacher_gate": "failed_no_successful_native_episodes" if all(s["success_rate"]==0 for s in summaries.values()) else "requires_successful_teacher_review", "a_star_used": False,
              "audit_round_limit": "Five predefined profiles; no checkpoint search, retraining or reward changes."}
    write_json(output / "audit.json", report)


if __name__ == "__main__":
    main()
