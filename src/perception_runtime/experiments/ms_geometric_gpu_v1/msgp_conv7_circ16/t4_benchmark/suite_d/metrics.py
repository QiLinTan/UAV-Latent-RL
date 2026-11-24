"""Suite D topology milestones and evidence-backed failure categories."""
import numpy as np

from geometry import RADIUS, blocked, extents

FAILURE_BUCKETS = ("F1", "F2", "F3", "F4", "F5", "unclassified")


def gate_crossings(positions, plane):
    before, after = positions[:-1], positions[1:]
    steps = np.flatnonzero((before[:, 0] < plane) & (after[:, 0] >= plane))
    return [(int(step + 1), (before[step] + (after[step] - before[step])
                            * ((plane - before[step, 0]) / (after[step, 0] - before[step, 0]))).tolist())
            for step in steps]


def mine_labels(case, positions):
    details = case["details"]
    x, y, _ = positions.T
    crossbar = y >= details["junction_south"]
    gate = details["arm_gate_x"]
    return np.select((crossbar & (x < -gate), crossbar & (x > gate), crossbar),
                     ("left", "right", "junction"), default="stem")


def changes(labels):
    return [str(label) for index, label in enumerate(labels) if not index or labels[index - 1] != label]


def milestones(case, positions, progress, outcome):
    details = case["details"]
    topology = case["topology"]
    success = outcome == "success"
    if topology == "D1":
        crossings = gate_crossings(positions, details["rear_x"] + RADIUS)
        if outcome == "collision":
            crossings = [(tick, point) for tick, point in crossings if tick < len(positions) - 1]
        min_z = details["z"] - details["thickness"] / 2
        max_z = details["z"] + details["thickness"] / 2
        above = any(point[2] > max_z for _, point in crossings)
        below = any(point[2] < min_z for _, point in crossings)
        return dict(beam_plane_cleared=above or below, above=above, below=below,
                    intended_route=(above if details["intended_pass"] == "above" else below),
                    first_event_tick=crossings[0][0] if crossings else None)
    if topology == "D2":
        crossings = gate_crossings(positions, details["rear_x"] + RADIUS)
        lower_y, upper_y = details["y_window"]
        lower_z, upper_z = details["z_window"]
        def inside(point):
            return lower_y < point[1] < upper_y and lower_z < point[2] < upper_z

        front = gate_crossings(positions, details["front_x"])
        passed = any(inside(point) and any(entry < tick and inside(start)
                                           and (positions[entry:tick, 0] >= details["front_x"]).all()
                                           for entry, start in front)
                     for tick, point in crossings if outcome != "collision" or tick < len(positions) - 1)
        outside = any(abs(point[1]) > details["y_wall_half"] for _, point in crossings)
        return dict(through_window=passed, through_then_success=passed and success,
                    through_then_failure=passed and not success, outside_edge_then_success=outside and success and not passed,
                    first_event_tick=crossings[0][0] if crossings else None)
    if topology == "D3":
        crossings = gate_crossings(positions, details["first_bend_x"])
        peak = float(np.max(progress) / case["l_geo_m"])
        return dict(first_bend=bool(crossings), reached_50_pct=peak >= .5, reached_80_pct=peak >= .8,
                    first_event_tick=crossings[0][0] if crossings else None)
    labels = mine_labels(case, positions)
    target = details["target_arm"]
    wrong = "left" if target == "right" else "right"
    transitions = changes(labels)
    entries = np.flatnonzero(labels == "junction")
    return dict(entered_junction=bool(len(entries)), entered_target_arm=target in transitions,
                first_arm_correct=next((name for name in transitions if name in ("left", "right")), None) == target,
                wrong_arm_then_success=wrong in transitions and success and target in transitions
                and transitions.index(wrong) < transitions.index(target),
                first_event_tick=int(entries[0]) if len(entries) else None,
                wrong_arm_entries=transitions.count(wrong), transitions=transitions)


def clearance(record, positions):
    lower, upper = extents(record)
    output = np.empty(len(positions))
    for start in range(0, len(positions), 256):
        part = positions[start:start + 256, None, :]
        offset = np.maximum(lower - part, part - upper)
        signed = np.linalg.norm(np.maximum(offset, 0), axis=2) + np.minimum(offset.max(axis=2), 0)
        output[start:start + 256] = signed.min(axis=1)
    if record["scenario"] in ("D1", "D2"):
        output = np.minimum(output, positions[:, 2])
    return output - RADIUS


def obstacle_match(scan_history, contact):
    mark = np.asarray(contact["point"])
    frames = []
    for scan in scan_history[-3:]:
        direction = mark - scan["origin"]
        norm = np.linalg.norm(direction)
        close = np.linalg.norm(scan["xyz"] - mark, axis=1) <= .30
        cone = scan["directions"] @ (direction / max(norm, 1e-9)) >= np.cos(np.deg2rad(5.0))
        frames.append(bool(np.any((scan["body"] == contact["id"]) & close & cone)))
    return frames


def terminal_failure(case, outcome, positions, velocities, heading, progress, contacts, evidence, hz):
    details = case["details"]
    topology = case["topology"]
    flags = []
    if outcome == "collision":
        for contact, matched in zip(contacts, evidence):
            name = contact["name"]
            _, y, z = contact["point"]
            nx, _, _ = contact["normal"]
            seen = any(matched)
            beam = name == "thin_beam"
            window = name.startswith("window_")
            edge = False
            vertical = beam and abs(nx) > .65 or name in ("floor", "roof", "window_sill", "window_lintel")
            goal_z = float(case["map"]["target_pos"][2])
            if beam:
                bottom, top = details["z"] - details["thickness"] / 2, details["z"] + details["thickness"] / 2
                edge = min(abs(y + case["map"]["xy_half_extents"][1]),
                           abs(y - case["map"]["xy_half_extents"][1]),
                           min(abs(z - bottom), abs(z - top)) if abs(nx) > .65 else .061) <= .06
                goal_z = top + RADIUS if details["intended_pass"] == "above" else bottom - RADIUS
                if len(matched) == 3 and not seen:
                    flags.append("F1")
            if window:
                edges_y, edges_z = details["y_window"], details["z_window"]
                closest_y = np.clip(y, *edges_y)
                closest_z = np.clip(z, *edges_z)
                edge = min(*(np.hypot(y - yc, z - closest_z) for yc in edges_y),
                           *(np.hypot(y - closest_y, z - zc) for zc in edges_z)) <= .06
                goal_z = float(np.mean(edges_z))
            if topology in ("D3", "D4") and name not in ("floor", "roof", "unknown", "ground_plane"):
                edge = abs(nx) < .65 and abs(contact["normal"][2]) < .35
            if edge and seen:
                flags.append("F2")
            speed = np.linalg.norm(velocities[-int(1.5 * hz):, :2], axis=1)
            z_start = abs(positions[max(0, len(positions) - int(1.5 * hz)), 2] - goal_z)
            z_end = abs(positions[-1, 2] - goal_z)
            if vertical and (np.mean(speed > .6) >= .5 or (z_end >= .15 and z_start - z_end < .08)):
                flags.append("F3")
    if outcome == "timeout" and topology in ("D3", "D4") and len(positions) >= hz * 4 + 1:
        if topology == "D4":
            labels = mine_labels(case, positions)
            transitions = changes(labels)
            wrong = "left" if details["target_arm"] == "right" else "right"
            late_transitions = changes(labels[-8 * hz:])
            if transitions.count(wrong) >= 2 or (late_transitions.count("junction") >= 3 and len(late_transitions) >= 5):
                flags.append("F5")
        if progress[-1] - progress[-1 - hz * 4] < .12:
            stride = hz // 5
            sample = positions[-1 - 4 * hz::stride]
            lower, upper = extents(case["map"])
            target = np.asarray(case["map"]["target_pos"])
            occluded = blocked(sample, target, lower, upper)
            toward = target[:2] - sample[:, :2]
            yaw = np.asarray(heading)[-1 - 4 * hz::stride]
            forward = np.column_stack((np.cos(yaw), np.sin(yaw)))
            facing = np.sum(toward * forward, axis=1) >= .5 * np.linalg.norm(toward, axis=1)
            span = np.max(np.linalg.norm(sample - sample[-1], axis=1))
            if np.mean(occluded & facing) >= .8 and span < .55:
                flags.append("F4")
    priority = ("F1", "F2", "F3", "F5", "F4")
    return next((bucket for bucket in priority if bucket in flags), "unclassified"), sorted(set(flags))


def stats(case, positions, velocities, clearances, progress, okay, outcome, hz):
    differences = np.diff(positions, axis=0)
    length = float(np.linalg.norm(differences, axis=1).sum())
    duration = (len(positions) - 1) / hz
    milestones_ = milestones(case, positions, progress, outcome)
    window_end = milestones_["first_event_tick"] or len(positions) - 1
    window_start = max(0, window_end - hz * 2)
    horizontal = np.linalg.norm(velocities[:, :2], axis=1)
    heights = abs(velocities[:, 2])
    stride = hz * 4
    window = slice(window_start, window_end + 1)
    stall_values = progress[stride:] - progress[:-stride]
    return dict(
        geodesic_progress_ratio=float(np.clip(progress[-1] / case["l_geo_m"], 0., 1.)),
        max_geodesic_progress_ratio=float(np.clip(progress.max() / case["l_geo_m"], 0., 1.)),
        invalid_progress_frames=int((~okay).sum()), milestone=milestones_, path_length_m=length,
        duration_s=duration, successful_path_ratio=length / case["l_geo_m"] if outcome == "success" else None,
        successful_effective_speed_mps=case["l_geo_m"] / max(duration, 1e-9) if outcome == "success" else None,
        successful_actual_mean_speed_mps=length / max(duration, 1e-9) if outcome == "success" else None,
        clearance_min_m=float(clearances.min()), clearance_mean_m=float(clearances.mean()),
        clearance_p10_m=float(np.percentile(clearances, 10)),
        stall_windows=len(stall_values), stalled_windows=int(np.count_nonzero(stall_values < .12)),
        horizontal_saturated_fraction=float(np.mean(horizontal >= .95 * .8)),
        vertical_usage_mean=float(np.mean(heights / .3)),
        z_saturated_fraction=float(np.mean(heights >= .95 * .3)),
        approach=dict(start_tick=window_start, end_tick=window_end,
                      latest_window_without_milestone=milestones_["first_event_tick"] is None,
                      horizontal_mean_mps=float(np.mean(horizontal[window])),
                      horizontal_saturated_fraction=float(np.mean(horizontal[window] >= .95 * .8)),
                      vertical_usage_mean=float(np.mean(heights[window] / .3))))
