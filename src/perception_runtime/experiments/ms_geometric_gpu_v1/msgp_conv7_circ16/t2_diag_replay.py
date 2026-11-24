"""Reconstruct exact replay membership and action/update trends from T2 archives."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from agent import Actor, project
from perception import CELLS, GRID, unpack
from t2_eval import read_lines, write_json
from train import sha

NODES = tuple(range(40000, 120001, 10000))
REASONS = ("nonterminal", "success", "collision", "xy_bound", "height_bound", "timeout", "attitude_bound")
DISTANCES = ("<1.2m", "1.2–4m", "4–8m", ">8m")
VELOCITY_SCALE = np.array([.8, .8, .3])


def reconstruct(episodes, metrics, end):
    terminal = np.zeros(end + 1, dtype=np.int8)
    levels = np.full(end + 1, -1, dtype=np.int8)
    successful = [[] for _ in range(4)]
    episode_stage = np.full(end + 1, -1, dtype=np.int8)
    previous = 0
    for index, row in enumerate(episodes):
        step = row["step"]
        if row["episode_id"] != index or step != previous + row["episode_length"]:
            raise RuntimeError("episode log cannot be mapped to replay transition steps")
        steps = slice(previous + 1, step + 1)
        terminal[step] = REASONS.index(row["reason"])
        levels[steps] = row["course"]["start_curriculum_episode_level"]
        episode_stage[steps] = row["stage"]
        if row["reason"] == "success":
            successful[row["stage"]].extend(range(previous + 1, step + 1))
        previous = step
    levels[previous + 1:] = metrics["course"]["start_curriculum_episode_level"]
    episode_stage[previous + 1:] = metrics["course"]["density_stage"]
    if np.any(levels[1:] < 0):
        raise RuntimeError("missing episode level for a live transition")
    return terminal, levels, episode_stage, [np.asarray(x, dtype=np.int64) for x in successful]


def ordered_indices(ring):
    return (ring.pointer - ring.size + np.arange(ring.size, dtype=np.int64)) % ring.capacity


def velocity_stats(actions):
    velocity = actions * VELOCITY_SCALE
    speed = np.linalg.norm(velocity[:, :2], axis=1)
    return dict(n=len(actions), velocity_mean=velocity.mean(axis=0).tolist(),
                velocity_std=velocity.std(axis=0).tolist(),
                abs_vy_mean=float(np.abs(velocity[:, 1]).mean()),
                horizontal_speed_mean=float(speed.mean()),
                horizontal_saturated_fraction=float((speed >= .99 * .8).mean()),
                vertical_saturated_fraction=float((np.abs(velocity[:, 2]) >= .99 * .3).mean()))


def describe(steps, states, terminal, levels, stage):
    if not len(steps):
        return dict(n=0)
    d = states[:, -15] * 20
    occupied_z = (states[:, :CELLS].reshape(-1, GRID, GRID, GRID) > 0).any(axis=(1, 2))
    altitude = .08 + states[:, -2] * 2.67
    lower_voxel_face = -1.4 + np.arange(GRID) * .4
    above_floor = (occupied_z & (altitude[:, None] + lower_voxel_face > .01)).any(axis=1)
    groups = np.select([d < 1.2, d < 4., d <= 8.], [0, 1, 2], default=3)
    n = len(steps)
    counts = dict(terminal_reason=dict(zip(REASONS, np.bincount(terminal[steps], minlength=len(REASONS)).tolist())),
                  episode_start_level={f"Level{i}": int((levels[steps] == i).sum()) for i in range(5)},
                  distance=dict(zip(DISTANCES, np.bincount(groups, minlength=4).tolist())),
                  stage={str(i): int((stage[steps] == i).sum()) for i in range(4)})
    fractions = {key: {name: value / n for name, value in values.items()} for key, values in counts.items()}
    return dict(n=n, first_step=int(steps[0]), last_step=int(steps[-1]), counts=counts, fractions=fractions,
                range_m_quantiles=np.quantile(d, [.05, .5, .95]).tolist(),
                near_any_occupied_fraction=float(occupied_z.any(axis=1).mean()),
                near_occupied_above_floor_proxy_fraction=float(above_floor.mean()))


def main_segment(ring, terminal, stage):
    indices = ordered_indices(ring)
    steps = np.arange(ring.total - ring.size + 1, ring.total + 1, dtype=np.int64)
    flags = (ring.terminal[indices] | ring.truncated[indices]).reshape(-1)
    if not (np.array_equal(flags, terminal[steps] > 0) and np.array_equal(ring.stage[indices], stage[steps])):
        raise RuntimeError("main replay order does not match logged endings and stages")
    return steps, indices


def describe_success(ring, step_ids, steps, main_indices, main, terminal, levels, stage, node):
    indices = ordered_indices(ring)
    last_completed = np.flatnonzero(terminal[:node + 1])[-1]
    slot_steps = step_ids[:np.searchsorted(step_ids, last_completed, side="right")]
    if len(slot_steps) != ring.total:
        raise RuntimeError("success replay totals do not match completed successful episodes")
    if not ring.size:
        return dict(n=0, capacity=ring.capacity, total=ring.total)
    ids = slot_steps[-ring.size:]
    flags = (ring.terminal[indices] | ring.truncated[indices]).reshape(-1)
    if not np.array_equal(flags, terminal[ids] > 0):
        raise RuntimeError("success replay order disagrees with logged endings")
    common = ids >= steps[0]
    reference = main_indices[ids[common] - steps[0]]
    if not np.array_equal(ring.states[indices[common]], main.states[reference]):
        raise RuntimeError("success and main replay observations for the same steps differ")
    return dict(capacity=ring.capacity, total=ring.total,
                **describe(ids, ring.states[indices], terminal, levels, stage))


def snapshot(pack, steps, indices, terminal, levels, stage, successful):
    replay = pack["replay"]
    main = replay.main
    node = pack["trainer"]["step"]
    groups = dict(main=dict(capacity=main.capacity, total=main.total,
                            **describe(steps, main.states[indices], terminal, levels, stage)))
    groups["success"] = {str(i): describe_success(ring, successful[i], steps, indices, main,
                                                 terminal, levels, stage, node)
                         for i, ring in enumerate(replay.success)}
    if list(replay.exposure) != [node, 0, 0, 0]:
        raise RuntimeError("unexpected obstacle-stage exposure")
    splits = {80000: {"evicted_by_120k": (20001, 60000), "common": (60001, 80000)},
              120000: {"common": (60001, 80000), "added_after_80k": (80001, 120000)}}[node]
    groups["main_segments"] = {}
    for name, (low, high) in splits.items():
        take = (steps >= low) & (steps <= high)
        groups["main_segments"][name] = describe(steps[take], main.states[indices[take]], terminal, levels, stage)
    return groups


@torch.no_grad()
def panel_velocity(pack, panel):
    actor = Actor(False)
    actor.load_state_dict(pack["agent"]["actor"])
    actor.eval()
    commands = np.concatenate([project(actor(unpack(batch, False))).numpy()
                               for batch in torch.from_numpy(panel).split(128)])
    distance = panel[:, -15] * 20
    return dict(all=velocity_stats(commands), near_or_mid=velocity_stats(commands[distance <= 8.]),
                far=velocity_stats(commands[distance > 8.]))


def curves(episodes, updates, velocities):
    result = []
    for end in range(40000, 120001, 2000):
        completed = [r for r in episodes if end - 2000 < r["step"] <= end]
        loss = [u for u in updates if end - 2000 < u["step"] <= end]
        counts = Counter(r["reason"] for r in completed)
        result.append(dict(step=end, episodes=len(completed), reason=dict(counts),
                           success_rate=counts["success"] / len(completed) if completed else None,
                           collision_rate=counts["collision"] / len(completed) if completed else None,
                           xy_bound_rate=counts["xy_bound"] / len(completed) if completed else None,
                           critic_loss_mean=float(np.mean([u["critic_loss"] for u in loss])),
                           critic_loss_median=float(np.median([u["critic_loss"] for u in loss])),
                           q_mean=float(np.mean([(u["q1_mean"] + u["q2_mean"]) / 2 for u in loss])),
                           q_absolute_batch_max_p99=float(np.quantile([u["q_abs_max"] for u in loss], .99)),
                           behavior_velocity=velocities[end]))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run = args.run / "train"
    episodes = read_lines(run / "episodes.jsonl")
    metrics = read_lines(run / "metrics.jsonl")[-1]
    updates = read_lines(run / "updates.jsonl")
    terminal, levels, stage, successful = reconstruct(episodes, metrics, 120000)
    checksums = {r["step"]: r["sha256"] for r in read_lines(run / "events.jsonl") if r["event"] == "checkpoint"}
    promotions = []
    level_before = 0
    for row in episodes:
        level = row["course"]["start_curriculum_level"]
        if level != level_before:
            promotions.append(dict(step=row["step"], old_level=level_before, new_level=level))
            level_before = level
    action_bins, actor_panel, snapshots = {}, {}, {}
    panel = None
    for node in (80000, *(s for s in NODES if s != 80000)):
        path = run / f"checkpoint_{node:06d}.pt"
        if sha(path) != checksums[node]:
            raise RuntimeError(f"source checkpoint digest mismatch at {node}")
        pack = torch.load(path, map_location="cpu", weights_only=False)
        main = pack["replay"].main
        if pack["trainer"]["step"] != main.total or main.total != node or pack["contract"]["use_far_branch"]:
            raise RuntimeError("incompatible replay checkpoint")
        steps, indices = main_segment(main, terminal, stage)
        if panel is None:
            selected = np.random.default_rng(43).choice(indices, size=2048, replace=False)
            panel = main.states[selected].copy()
        for end in range(max(40000, node - 8000), node + 1, 2000):
            subset = (steps > end - 2000) & (steps <= end)
            if subset.sum() != 2000:
                raise RuntimeError("missing executed training actions")
            action_bins[end] = velocity_stats(main.actions[indices[subset]])
        actor_panel[node] = panel_velocity(pack, panel)
        if node in (80000, 120000):
            snapshots[node] = snapshot(pack, steps, indices, terminal, levels, stage, successful)
        print(f"Audited replay and Actor {node}", flush=True)
        del pack, main
    data = dict(hashes={step: checksums[step] for step in NODES}, promotions=promotions, replay=snapshots,
                timeline_2k=curves(episodes, updates, action_bins), fixed_actor_panel=actor_panel,
                fixed_actor_panel_source="2,048 fixed 80k main replay states, seed 43",
                training_action_source="Executed exploration-affected projected body-frame replay actions",
                q_max_metric="99th percentile of per-update max absolute Q over both heads",
                distance_from="State18[3] * 20m",
                ground_proxy="Occupied voxels whose lowest face is over 1 cm above ground, using State18 altitude",
                sample_mix_when_success_available={"main": 205 / 256, "success": 51 / 256})
    write_json(args.output / "replay_and_timeline.json", data)


if __name__ == "__main__":
    main()
