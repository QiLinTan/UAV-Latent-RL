"""Summarize P0 observation divergence/interventions and B terminal timing."""

import json
from pathlib import Path

import numpy as np
import torch

from algos.td3.td3_upper_recurrent import RecurrentUpperActor

ROOT = Path("analysis_outputs/full_preimprovement_diagnostics_20261010")
P0_CASES = ((42, 34, 22), (45, 20, 5), (42, 32, 5))
B_CASES = (25, 30)
SCALE = np.array([0.8, 0.8, 0.3])


def trace(tag, seed, map_seed, mode):
    stem = ROOT / tag / f"seed{seed}_map{map_seed}_{mode}" / "traces" / f"trace_seed{map_seed}"
    return np.load(stem.with_suffix(".npz")), json.loads(stem.with_suffix(".json").read_text())


def actor(seed):
    model = RecurrentUpperActor(
        917, 3, 1.0, base_state_dim=117, history_length=8,
        history_frame_dim=100, hidden_dim=64, project_planar_action=True,
    )
    checkpoint = f"checkpoints/startobscurr_v1_selected_20261010/seed{seed}/model_final_actor"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    return model


def command(model, observation):
    with torch.no_grad():
        action = model(torch.as_tensor(observation[None], dtype=torch.float32)).numpy()[0]
    return action * SCALE, action


def group_indices():
    current_rays = np.arange(15, 111)
    history_rays = np.concatenate([np.arange(117 + j * 100, 117 + j * 100 + 96) for j in range(8)])
    history_command = np.concatenate([np.arange(117 + j * 100 + 96, 117 + (j + 1) * 100) for j in range(8)])
    return {
        "kinematics": np.arange(0, 12), "goal": np.arange(12, 15),
        "current_rays": current_rays, "reference_state": np.arange(111, 117),
        "history_rays": history_rays, "history_command_error": history_command,
        "all_rays": np.r_[current_rays, history_rays],
    }


def p0():
    output = []
    for seed, map_seed, step in P0_CASES:
        k1, meta = trace("p0_baseline", seed, map_seed, "k1")
        full, _ = trace("p0_baseline", seed, map_seed, "full")
        tag = f"p0_intervention_step{step:03d}_len08"
        intervention, _ = trace(tag, seed, map_seed, "full")
        K, F = k1["upper_pre_observation"][step], full["upper_pre_observation"][step]
        assert K.shape == F.shape == (917,)
        model = actor(seed)
        ck, ak = command(model, K)
        cf, af = command(model, F)
        np.testing.assert_allclose(ak, k1["upper_policy_action"][step], atol=1e-6)
        np.testing.assert_allclose(af, full["upper_policy_action"][step], atol=1e-6)
        groups = {}
        for name, indices in group_indices().items():
            hybrid = F.copy()
            hybrid[indices] = K[indices]
            shifted, _ = command(model, hybrid)
            groups[name] = {
                "input_l2": float(np.linalg.norm(K[indices] - F[indices])),
                "hybrid_command": shifted.tolist(),
                "command_shift_from_full_mps": float(np.linalg.norm(shifted - cf)),
                "residual_to_k1_command_mps": float(np.linalg.norm(shifted - ck)),
            }
        pos_k = np.asarray(meta["initial_position"] if step == 0 else k1["upper_position"][step - 1])
        pos_f = np.asarray(meta["initial_position"] if step == 0 else full["upper_position"][step - 1])
        vel_k = np.asarray(meta["initial_velocity"] if step == 0 else k1["upper_velocity"][step - 1])
        vel_f = np.asarray(meta["initial_velocity"] if step == 0 else full["upper_velocity"][step - 1])
        goal = np.asarray(meta["target_position"])
        g = goal[:2] - pos_f[:2]
        g /= np.linalg.norm(g)
        lateral = np.array([-g[1], g[0]])
        raydiff = np.abs(K[15:111] - F[15:111])
        maximum_ray = int(np.argmax(raydiff))
        angle = float(np.degrees(np.arccos(np.clip(
            np.dot(ck[:2], cf[:2]) / max(np.linalg.norm(ck[:2]) * np.linalg.norm(cf[:2]), 1e-9), -1, 1,
        ))))
        n = min(len(k1["upper_action"]), len(full["upper_action"]))
        command_differences = np.linalg.norm(
            k1["upper_command_velocity"][:n] - full["upper_command_velocity"][:n], axis=1,
        )
        first_above = next((j for j, value in enumerate(command_differences) if value > 0.2), None)
        first_sustained = next((j for j in range(n - 2) if np.all(command_differences[j:j + 3] > 0.2)), None)
        assert first_sustained == step
        np.testing.assert_array_equal(
            full["upper_pre_observation"][:step + 1], intervention["upper_pre_observation"][:step + 1]
        )
        np.testing.assert_array_equal(
            full["lower_position"][:step * 8], intervention["lower_position"][:step * 8]
        )
        np.testing.assert_array_equal(
            intervention["upper_command_velocity"][step:step + 8],
            k1["upper_command_velocity"][step:step + 8]
        )
        windows = {}
        for length in (8, 15):
            end = min(step + length, len(full["upper_clearance"]), len(intervention["upper_clearance"]))
            windows[f"{length}_steps"] = {
                "baseline_min_clearance_m": float(np.min(full["upper_clearance"][step:end])),
                "intervention_min_clearance_m": float(np.min(intervention["upper_clearance"][step:end])),
                "position_separation_at_common_end_m": float(np.linalg.norm(
                    full["upper_position"][end - 1] - intervention["upper_position"][end - 1]
                )),
            }
        intervention_row = json.loads((ROOT / tag / f"seed{seed}_map{map_seed}_full" / "evaluation_metrics.jsonl").read_text().splitlines()[0])["episodes"][0]
        baseline_row = json.loads((ROOT / "p0_baseline" / f"seed{seed}_map{map_seed}_full" / "evaluation_metrics.jsonl").read_text().splitlines()[0])["episodes"][0]
        output.append({
            "training_seed": seed, "map_seed": map_seed, "first_single_command_diff_step": first_above,
            "first_three_step_command_diff_step": step, "branch_time_s": step / 15.0,
            "pre_position_difference_m": float(np.linalg.norm(pos_k - pos_f)),
            "pre_velocity_difference_mps": float(np.linalg.norm(vel_k - vel_f)),
            "current_ray_mean_abs_difference": float(np.mean(raydiff)),
            "current_ray_max_abs_difference": float(np.max(raydiff)),
            "current_rays_different_over_0_05": int(np.sum(raydiff > 0.05)),
            "largest_ray_difference": {"index": maximum_ray, "elevation_degrees": (-30, 0, 30)[maximum_ray // 32],
                                       "azimuth_degrees": 360 * (maximum_ray % 32) / 32},
            "k1_command_mps": ck.tolist(), "full_command_mps": cf.tolist(),
            "planar_heading_difference_degrees": angle,
            "k1_planar_speed_mps": float(np.linalg.norm(ck[:2])),
            "full_planar_speed_mps": float(np.linalg.norm(cf[:2])),
            "k1_goal_lateral_command_mps": float(np.dot(ck[:2], lateral)),
            "full_goal_lateral_command_mps": float(np.dot(cf[:2], lateral)),
            "feature_swap": groups,
            "intervention": {"start_step": step, "steps": 8, "prefix_exact": True,
                             "tape_command_exact": True, "windows": windows,
                             "baseline_done_reason": baseline_row["done_reason"],
                             "intervention_done_reason": intervention_row["done_reason"],
                             "baseline_final_goal_distance_m": baseline_row["final_goal_distance"],
                             "intervention_final_goal_distance_m": intervention_row["final_goal_distance"]},
        })
    return output


def b():
    output = []
    for map_seed in B_CASES:
        traces = {seconds: trace(f"b_{seconds}s", 45, map_seed, "full") for seconds in (12, 14, 16)}
        baseline, meta = traces[12]
        goal = np.asarray(meta["target_position"])
        case = {"training_seed": 45, "map_seed": map_seed, "runs": {}}
        for seconds, (data, _) in traces.items():
            t = data["lower_time_seconds"]
            p = data["lower_position"]
            v = data["lower_velocity"]
            q = goal - p
            distance = np.linalg.norm(q, axis=1)
            radial = np.sum(v * q, axis=1) / np.maximum(distance, 1e-9)
            tangential = np.sqrt(np.maximum(0, np.sum(v * v, axis=1) - radial * radial))
            closest = int(np.argmin(distance))
            entry = np.flatnonzero(distance <= 1.2)
            success = np.flatnonzero(distance <= 0.2)
            n = len(baseline["lower_position"])
            np.testing.assert_array_equal(baseline["lower_position"], p[:n])
            np.testing.assert_array_equal(baseline["lower_velocity"], v[:n])
            row = json.loads((ROOT / f"b_{seconds}s" / f"seed45_map{map_seed}_full" / "evaluation_metrics.jsonl").read_text().splitlines()[0])["episodes"][0]
            case["runs"][str(seconds)] = {
                "done_reason": row["done_reason"], "end_time_s": float(t[-1]),
                "first_guidance_entry_time_s": float(t[entry[0]]) if len(entry) else None,
                "first_success_ball_time_s": float(t[success[0]]) if len(success) else None,
                "minimum_distance_m": float(distance[closest]),
                "minimum_distance_time_s": float(t[closest]),
                "radial_velocity_at_min_mps": float(radial[closest]),
                "tangential_velocity_at_min_mps": float(tangential[closest]),
                "height_error_at_min_m": float(q[closest, 2]),
                "baseline_12s_prefix_exact": True,
            }
        output.append(case)
    return output


def main():
    summary = {"p0": p0(), "b": b()}
    (ROOT / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("wrote", ROOT / "analysis_summary.json")


if __name__ == "__main__":
    main()
