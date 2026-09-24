"""Physical 3-D candidate actions for fixed-observation critic diagnostics."""

import numpy as np


def spatial_candidates(actor_action, position, target, *, xy_limit=0.8, z_limit=0.3):
    if xy_limit <= 0 or z_limit <= 0:
        raise ValueError("Physical velocity limits must be positive.")
    actor = np.asarray(actor_action, dtype=np.float32).reshape(3)
    delta = np.asarray(target)-np.asarray(position)
    goal = delta[:2]/max(np.linalg.norm(delta[:2]), 1e-9)
    if np.linalg.norm(goal) < 1e-6:
        goal = np.array([1.,0.])
    direction = actor[:2]/max(np.linalg.norm(actor[:2]),1e-9)
    if np.linalg.norm(direction) < 1e-6:
        direction = goal
    actions, metadata = [], []

    def add(name, action, family):
        action = np.asarray(action,dtype=np.float32)
        if np.linalg.norm(action[:2])>1+1e-6 or abs(action[2])>1+1e-6:
            raise ValueError("Candidate outside executable action disk.")
        actions.append(action)
        metadata.append(dict(name=name, family=family,
            xy_speed_mps=float(np.linalg.norm(action[:2])*xy_limit),
            z_command_mps=float(action[2]*z_limit)))

    add("actor",actor,"actor")
    for z in (-1.,-.5,0.,.5,1.):
        add(f"actor_xy_z{z:+.1f}",np.r_[actor[:2],z],"vertical_only")
        add(f"stop_xy_z{z:+.1f}",[0.,0.,z],"stop_xy")
    for name, vec in (("goal",goal),("actor_direction",direction),
                      ("away",-goal),("tangent_left",[-goal[1],goal[0]]),
                      ("tangent_right",[goal[1],-goal[0]])):
        for speed in (.1,.2,.4,.6,.8):
            if speed > xy_limit+1e-6:
                continue
            for z in (-1.,-.5,0.,.5,1.):
                add(f"{name}_speed{speed:.1f}_z{z:+.1f}",
                    np.r_[np.asarray(vec)*speed/xy_limit,z],name)
    return np.asarray(actions), metadata


def calibration_indices(q1, q2, metadata):
    """Fixed probes plus both critic optima, deduplicated by candidate index."""
    names = [m["name"] for m in metadata]
    chosen = [names.index(name) for name in (
        "actor", "actor_xy_z-1.0", "actor_xy_z+0.0", "actor_xy_z+1.0",
        "stop_xy_z+0.0", "goal_speed0.4_z-1.0",
    )]
    chosen += [int(np.argmax(q1)), int(np.argmax(np.minimum(q1,q2)))]
    return list(dict.fromkeys(chosen))


def q_summary(records):
    if not records:
        return {"state_count":0}
    result = {"state_count":len(records)}
    for key in ("q1","q2","q_min"):
        spans, gaps = [], []
        for record in records:
            values = np.array([c[key] for c in record["candidates"]])
            spans.append(float(np.ptp(values)))
            gaps.append(float(values.max()-values[0]))
        result[key] = dict(mean_within_state_range=float(np.mean(spans)),
            mean_best_minus_actor=float(np.mean(gaps)),
            actor_within_0_01_of_best_count=sum(g<=.01 for g in gaps))
    return result
