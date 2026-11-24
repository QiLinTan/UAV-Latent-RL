"""Frozen v2.2 modulation of an already collected full LiDAR scan."""

import numpy as np

RESULT_FIELDS = ("u_kin", "u_clear", "u_corridor", "u_goal", "alpha_t", "s_t")
BODY = .062
MARGIN = .10
SAFE = .162
TAU = .0784
BRAKE = .221289
NEAR = 1.4
FAR = 8.0
GOAL_DECAY = 1.2
KAPPA_NEAR = 2.0
KAPPA_FAR = 1.5
GROUND_EPS = .05
SOFT = .05
ANGLE = np.deg2rad(30)
RADIAL = (1., 2.4)


def smoothstep(value, start, stop):
    t = np.clip((value - start) / (stop - start), 0., 1.)
    return t * t * (3. - 2. * t)


def motion_risk(distance, approach):
    required = TAU * approach + BRAKE * (approach / .8) ** 2
    return np.clip(required / (np.maximum(distance - SAFE, 0.) + SOFT), 0., 1.)


def ground_risk(height, vz):
    return motion_risk(np.asarray(height), np.maximum(-np.asarray(vz), 0.))


def scores(scan, rotation, velocity, goal):
    rays = scan.directions_world.reshape(-1, 3) @ rotation
    ranges = scan.range_m.ravel().astype(float)
    hit = scan.hit_mask.ravel()
    if len(hit) != 4096 or not scan.scanned_mask.all():
        raise ValueError("v2.2 requires a complete 4,096-ray scan")
    visible = hit & (scan.raw_hit_world.reshape(-1, 3)[:, 2] > GROUND_EPS)
    obstacles = ranges[visible]
    approach = np.maximum(rays[visible] @ velocity, 0.)
    risk = motion_risk(obstacles, approach)
    obs_kin = float(np.sort(risk)[-5:].sum() / 5) if len(risk) < 5 else float(np.partition(risk, -5)[-5:].mean())
    altitude = float(scan.origin_world[2] - scan.config.origin_height_m)
    floor_kin = float(ground_risk(altitude, velocity[2]))
    kin = max(obs_kin, floor_kin)
    obstacle_fifth = float(np.partition(obstacles, 4)[4]) if len(obstacles) >= 5 else FAR
    clear = float(np.clip((NEAR - obstacle_fifth) / (NEAR - SAFE), 0., 1.))
    near = max(kin, clear)
    goal_dist = float(np.linalg.norm(goal))
    look = min(FAR, goal_dist)
    corridor = 0.
    if look > NEAR:
        theta = np.arccos(np.clip(rays @ (goal / goal_dist), -1., 1.))
        angular = np.cos(.5 * np.pi * np.minimum(theta / ANGLE, 1.)) ** 2
        radial = smoothstep(ranges[hit], *RADIAL)
        span = np.clip((look - np.maximum(ranges[hit], NEAR)) / (look - NEAR), 0., 1.)
        corridor = float(smoothstep(look, NEAR, NEAR + .4)
                         * np.sum(angular[hit] * radial * span) / max(1., float(angular.sum())))
    goal_term = float(np.exp(-goal_dist / GOAL_DECAY) * (1. - near))
    alpha = float(.5 + .25 * np.tanh(KAPPA_NEAR * near - KAPPA_FAR * corridor))
    scale = 1. - .5 * goal_term
    return dict(zip(RESULT_FIELDS, (kin, clear, corridor, goal_term, alpha, scale)),
                u_kin_obs=obs_kin, u_kin_ground=floor_kin, d_min=obstacle_fifth)


def state_scores(scan, rotation, state):
    velocity = state[4:7].astype(float) * [.8, .8, .3]
    goal = state[:3].astype(float) * (float(state[3]) * 20.)
    return scores(scan, rotation, velocity, goal)
