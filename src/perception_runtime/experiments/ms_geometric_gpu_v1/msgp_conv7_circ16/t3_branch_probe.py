"""Export T3 branch probes and review the 240k Stage0 boundary failures."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path

import numpy as np
import torch

from agent import Actor, Critic, project
from evaluate import make_validation_env, realtime_decisions
from perception import unpack
from t0_eval import follow, initial_info
from t2_diag_traces import xy_bounds
from t2_eval import read_json, read_lines, write_json
import t3_eval as evaluator
from train import sha

NODES = (160000, 200000, 240000)


def distribution(near, far):
    a, b = float(near.mean()), float(far.mean())
    return dict(near_mean_l2=a, far_mean_l2=b, near_over_far=a / b if b else None,
                near_median_l2=float(np.median(near)), far_median_l2=float(np.median(far)))


def branch_norms(vectors):
    return torch.stack([vectors[:, :32].norm(dim=1), vectors[:, 32:64].norm(dim=1)], dim=1)


def training_probes(run, step):
    rows = [r for r in read_lines(run / "probes_002000.jsonl") if step - 10000 < r["step"] <= step]
    if [r["step"] for r in rows] != list(range(step - 8000, step + 1, 2000)):
        raise RuntimeError("missing trailing training probe rows")
    result = dict(last_2k=rows[-1]["branches"], last_10k={})
    for name in ("actor", "critic"):
        data = [p["branches"][name]["averaged"] for p in rows]
        means = {k: float(np.mean([r[k] for r in data])) for k in
                 ("near_feature_l2", "far_feature_l2", "near_gradient_l2", "far_gradient_l2")}
        means["near_over_far_feature_l2"] = means["near_feature_l2"] / means["far_feature_l2"]
        means["near_over_far_gradient_l2"] = means["near_gradient_l2"] / means["far_gradient_l2"]
        result["last_10k"][name] = means
    return result


def load_networks(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not checkpoint["contract"]["use_far_branch"]:
        raise RuntimeError("checkpoint does not contain both branches")
    actor, critic = Actor(True), Critic(True)
    actor.load_state_dict(checkpoint["agent"]["actor"])
    critic.load_state_dict(checkpoint["agent"]["critic"])
    return actor.eval().requires_grad_(False), critic.eval().requires_grad_(False)


def fixed_panel_probes(actor, critic, panel, reference_actions, stages):
    data = {}
    scale = torch.tensor([.8, .8, .3])
    for features, reference in zip(panel.split(128), reference_actions.split(128)):
        inputs = unpack(features, True)
        with torch.no_grad():
            actor_z = actor.encoder(*inputs)
            critic_z = critic.encoder(*inputs)
        actor_z.requires_grad_()
        critic_z.requires_grad_()
        action = project(actor.policy(actor_z))
        velocity = action * scale
        action_jacobian = torch.stack([
            torch.autograd.grad(velocity[:, axis].sum(), actor_z, retain_graph=True)[0]
            for axis in range(3)
        ])
        actor_loss = -critic.q1(torch.cat((critic_z.detach(), action), dim=1))
        actor_objective_grad = torch.autograd.grad(actor_loss.sum(), actor_z)[0]
        critic_inputs = torch.cat((critic_z, reference), dim=1)
        dq1 = torch.autograd.grad(critic.q1(critic_inputs).sum(), critic_z, retain_graph=True)[0]
        dq2 = torch.autograd.grad(critic.q2(critic_inputs).sum(), critic_z)[0]
        batch = dict(
            actor_feature=branch_norms(actor_z.detach()),
            critic_feature=branch_norms(critic_z.detach()),
            actor_policy_objective_gradient=branch_norms(actor_objective_grad),
            actor_velocity_jacobian=(torch.stack([branch_norms(axis) for axis in action_jacobian])
                                     .square().sum(dim=0).sqrt()),
            critic_twin_q_sensitivity=((branch_norms(dq1).square() + branch_norms(dq2).square()) / 2).sqrt(),
        )
        for key, value in batch.items():
            data.setdefault(key, []).append(value.numpy())
    cohorts = dict(all=np.ones(len(panel), dtype=bool), **{f"stage{s}": stages == s for s in range(4)})
    return {key: {name: distribution(vec[mask, 0], vec[mask, 1]) for name, mask in cohorts.items()}
            for key, batches in data.items() for vec in [np.concatenate(batches, axis=0)]}


def height_trace(step, checkpoint, fingerprint, expected):
    evaluator.init_worker("t3", checkpoint, fingerprint)
    env = make_validation_env(evaluator.contract)
    env.obstacle_curriculum.stage = 0
    try:
        obs, info = env.reset(seed=expected["case"])
        if initial_info(env, info) != expected["init"]:
            raise RuntimeError("Stage0 height trace scene mismatch")
        commands, altitudes, clearances = [], [float(env.base_env.pos[0, 2])], []
        original = env.step

        def tracked(action):
            outcome = original(action)
            meta = outcome[-1]
            commands.append(float(meta["accepted_world_command"][2]))
            clearances.append(float(meta["min_tree_clearance"]))
            altitudes.append(float(env.base_env.pos[0, 2]))
            return outcome

        env.step = tracked
        with realtime_decisions():
            row, _ = follow(env, evaluator.agent, evaluator.builder, obs, info)
        if (evaluator.step != step or row["reason"] != expected["reason"]
                or row["upper_steps"] != expected["upper_steps"]
                or not np.allclose(env.base_env.pos[0], expected["final_position_m"], atol=1e-6, rtol=0)):
            raise RuntimeError("Stage0 height trace did not reproduce the saved outcome")
        return dict(step=step, case=expected["case"], reason=row["reason"], altitude_m=altitudes,
                    accepted_vertical_command_mps=commands, tree_clearance_m=clearances,
                    end_altitude_m=altitudes[-1], last_15_mean_command_vz=float(np.mean(commands[-15:])),
                    last_15_min_clearance_m=min(clearances[-15:]), xy_threshold_m=xy_bounds(env.base_env).tolist())
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("results_160k", type=Path)
    parser.add_argument("results_240k", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root, results = args.run, args.results_240k
    provenance = read_json(root / "fixed_actor_panel.json")
    panel_path = root / "fixed_actor_panel.npy"
    if provenance["sha256"] != sha(panel_path):
        raise RuntimeError("fixed panel digest mismatch")
    panel = torch.from_numpy(np.load(panel_path, allow_pickle=False))
    stages = np.array([row["stage"] for row in provenance["cases"] for _ in range(row["observations"])])
    if panel.shape != (2048, 784) or len(stages) != len(panel):
        raise RuntimeError("unexpected fixed panel shape")
    sources = {}
    for step in NODES:
        folder = args.results_160k if step == 160000 else results
        source = read_json(folder / f"t3_{step:06d}.json")
        checkpoint = root / "train" / f"checkpoint_{step:06d}.pt"
        if sha(checkpoint) != source["checkpoint_sha256"]:
            raise RuntimeError(f"mismatched checkpoint at {step}")
        sources[step] = checkpoint, source
    a160, _ = load_networks(sources[160000][0])
    with torch.no_grad():
        reference_actions = torch.cat([project(a160(unpack(batch, True))) for batch in panel.split(128)])
    probes = {}
    for step, (checkpoint, source) in sources.items():
        actor, critic = load_networks(checkpoint)
        probes[step] = dict(checkpoint_sha256=source["checkpoint_sha256"],
                            training=training_probes(root, step),
                            fixed_inputs=fixed_panel_probes(actor, critic, panel, reference_actions, stages))
    final_rows = sources[240000][1]["rows"]
    failed = [r for r in final_rows if r["stage"] == 0 and not r["success"]]
    height_cases = {r["case"] for r in failed if r["reason"] == "height_bound"}
    pool_hash = read_json(root / "train/metadata.json")["contract"]["pool_fingerprint"]
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
        jobs = [pool.submit(height_trace, step, checkpoint, pool_hash, row)
                for step, (checkpoint, doc) in sources.items()
                for row in doc["rows"] if row["stage"] == 0 and row["case"] in height_cases]
        traces = [job.result() for job in jobs]
    write_json(results / "branch_and_stage0_diagnostics.json", dict(
        fixed_panel_sha256=provenance["sha256"], fixed_panel_n=len(panel),
        fixed_panel_stage_counts=np.bincount(stages).tolist(), branch_probes=probes,
        stage0_failures_240k=failed, stage0_height_case_traces=traces,
        definitions=dict(
            training_gradient="Post-clip parameter-gradient L2 on near_net and far_conv as logged; far_head excluded.",
            fixed_output="Mean per-sample L2 of each 32D latent over the archived 2048-observation fixed panel.",
            actor_policy_objective_gradient="Mean per-sample L2 of direct latent derivative of -Q1(s, project(actor(s))) using each checkpoint's critic.",
            actor_velocity_jacobian="Mean per-sample Frobenius norm of projected physical XYZ command derivatives in the yaw frame with respect to each latent.",
            critic_twin_q_sensitivity="Mean per-sample root-mean-square of the Q1 and Q2 latent-gradient norms, holding action fixed to the 160k actor for all checkpoints.",
        )))
    print(f"Verified and exported {len(failed)} boundary failures, {len(traces)} height replays, and {len(probes)} checkpoint comparisons.", flush=True)


if __name__ == "__main__":
    main()
