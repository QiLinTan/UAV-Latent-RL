"""Frozen public calibration tapes and separate complete acceptance cases."""
from copy import deepcopy
import numpy as np


def case(name, family, duration, phases, **kwargs):
    return dict(name=name, family=family, duration_s=duration, phases=phases,
                initial_rpy=[0., 0., 0.], initial_position=[0., 0., 1.],
                initial_velocity=[0., 0., 0.], disturbances=[], **kwargs)


def vector_axis(axis, magnitude):
    value = [0., 0., 0.]
    value[axis] = magnitude
    return value


def calibration_cases():
    return [case('cal_hover', 'hover', 8., [(0, [0, 0, 0])]),
            case('cal_x', 'step', 10., [(0, [0, 0, 0]), (2, [.4, 0, 0]), (6, [0, 0, 0])]),
            case('cal_z', 'step', 10., [(0, [0, 0, 0]), (2, [0, 0, .15]), (4, [0, 0, 0]),
                                            (6, [0, 0, -.15]), (8, [0, 0, 0])]),
            case('cal_circle_response', 'circle', 16., [(0, [0, 0, 0])], waveform='circle'),
            case('cal_reverse_response', 'turn', 13., [(0, [0, 0, 0]), (1, [.7, 0, 0]),
                                                       (5, [-.7, 0, 0]), (9, [0, 0, 0])])]


def acceptance_cases():
    cases = [case('hover20', 'hover', 20., [(0, [0, 0, 0])])]
    for axis in range(3):
        for level in ((.4, .8) if axis < 2 else (.15, .3)):
            if axis < 2:
                phases = [(0, [0, 0, 0]), (2, vector_axis(axis, level)), (6, [0, 0, 0]),
                          (10, vector_axis(axis, -level)), (14, [0, 0, 0])]
                duration = 18.
            else:
                phases = [(0, [0, 0, 0]), (2, vector_axis(axis, level)), (4, [0, 0, 0]),
                          (7, vector_axis(axis, -level)), (9, [0, 0, 0])]
                duration = 12.
            cases.append(case(f'step_{"xyz"[axis]}_{level:g}', 'step', duration, phases))
    d = .8 / np.sqrt(2)
    cases += [case('diagonal_xyz', 'diagonal', 14., [(0, [0, 0, 0]), (2, [d, d, .3]),
                        (4, [d, d, 0]), (7, [-d, -d, -.3]), (9, [-d, -d, 0]), (11, [0, 0, 0])]),
              case('circle_xyz', 'circle', 20., [(0, [0, 0, 0])], waveform='circle'),
              case('turn90', 'turn', 22., [(0, [0, 0, 0]), (2, [.8, 0, 0]), (6, [0, .8, 0]),
                                         (10, [-.8, 0, 0]), (14, [0, -.8, 0]), (18, [0, 0, 0])]),
              case('reverse180', 'turn', 18., [(0, [0, 0, 0]), (2, [.8, 0, 0]), (6, [-.8, 0, 0]),
                                             (10, [.8, 0, 0]), (14, [0, 0, 0])])]
    for sign in (1, -1):
        disturbance = case(f'disturbance_{sign:+d}', 'disturbance', 10., [(0, [0, 0, 0])])
        disturbance['disturbances'] = [dict(start_s=4., duration_s=.15,
                            force_world_n=(sign * .027 * np.array([.3, -.2, .1]) / .15).tolist())]
        cases.append(disturbance)
        initial = case(f'initial_velocity_{sign:+d}', 'initial_velocity', 8., [(0, [0, 0, 0])])
        initial['initial_velocity'] = (sign * np.array([.3, -.2, .1])).tolist()
        cases.append(initial)
    for index, yaw in enumerate((0., np.pi/2, np.pi, -np.pi+1e-4)):
        for sign in (1, -1):
            initial = case(f'initial_attitude_yaw{index}_{sign:+d}', 'attitude', 8., [(0, [0, 0, 0])])
            initial['initial_rpy'] = [sign * .1, -sign * .1, yaw]
            cases.append(initial)
    saturated = case('saturation_release', 'saturation', 14., [(0, [0, 0, 0]), (2, [.8, 0, 0]), (6, [0, 0, 0])],
                     limit_window=dict(start_s=2., end_s=6., max_tilt_rad=.06, max_force_fraction=.5))
    saturated['disturbances'] = [dict(start_s=2., duration_s=4., force_world_n=[-.03, 0., 0.])]
    cases.append(saturated)
    return cases


def command_at(scenario, simulation_time):
    if scenario.get('waveform') == 'circle':
        if simulation_time < 2:
            return np.zeros(3)
        t = simulation_time - 2
        return np.array([.6*np.cos(.6*t), .6*np.sin(.6*t), .2*np.sin(.6*t)])
    result = np.zeros(3)
    for start, value in scenario['phases']:
        if simulation_time + 1e-9 >= start:
            result = np.array(value, dtype=float)
    return result


def sampled_tape(scenario):
    return np.array([command_at(scenario, k/15) for k in range(round(scenario['duration_s'] * 15))])
