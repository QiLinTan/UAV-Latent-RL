"""Metrics refer to the external world command, sampled after each 120Hz step."""
import numpy as np


def settling_time(times, errors, start_s, end_s, tolerance, *, sustain_s=None):
    indices = np.flatnonzero((times > start_s + 1e-8) & (times <= end_s + 1e-8))
    if not len(indices):
        return None
    inside = np.all(abs(errors[indices]) <= np.asarray(tolerance), axis=1)
    for local, index in enumerate(indices):
        if sustain_s is None:
            selected = inside[local:]
        else:
            stop = np.searchsorted(times[indices], times[index] + sustain_s - 1e-8, side='right')
            if stop - local < round(sustain_s * 120) - 1:
                continue
            selected = inside[local:stop]
        if len(selected) and np.all(selected):
            return float(times[index] - start_s)
    return None


def evaluate(data, scenario, model, spec):
    times, command, velocity = data['time_s'], data['command_world'], data['velocity_after']
    error = velocity - command
    position = data['position_after']
    phase_starts = [float(p[0]) for p in scenario['phases']]
    nominal_end = scenario['duration_s']
    masks = np.ones(len(times), dtype=bool)
    for start in phase_starts:
        masks &= ~((times > start - 1e-9) & (times < start + 1. - 1e-9))
    if scenario.get('waveform') == 'circle':
        masks = times >= 3.
    rms = lambda a: float(np.sqrt(np.mean(np.square(a)))) if len(a) else None
    metrics = dict(command_velocity_rmse_vector=rms(np.linalg.norm(error, axis=1)),
                   command_velocity_rmse_xyz=np.sqrt(np.mean(error**2, axis=0)).tolist(),
                   steady_rmse_xy=rms(np.linalg.norm(error[masks, :2], axis=1)),
                   steady_rmse_z=rms(error[masks, 2]),
                   max_abs_rpy_rad=np.max(abs(data['rpy_after']), axis=0).tolist(),
                   max_speed_mps=float(np.linalg.norm(velocity, axis=1).max()),
                   final_position=position[-1].tolist(),
                   displacement_xyz=(position[-1] - np.asarray(scenario['initial_position'])).tolist(),
                   max_abs_velocity_integral=np.max(abs(data['velocity_integral']), axis=0).tolist(),
                   max_abs_attitude_integral=np.max(abs(data['attitude_integral']), axis=0).tolist(),
                   rpm_min=float(data['rpm_executed'].min()), rpm_max=float(data['rpm_executed'].max()),
                   effort_rpm2_integral=float(np.sum(data['rpm_executed']**2) / 120),
                   pwm_or_torque_saturation_fraction=float(np.mean(data['allocation_saturated'])),
                   force_saturation_fraction=float(np.mean(data['force_saturated'])),
                   physical_saturation_fraction=float(np.mean(data['physical_rpm_saturated'])),
                   lower_ticks=len(times), physics_substeps=int(data['physics_tick'][-1]),
                   duration_sim_s=float(times[-1]),
                   max_codec_roundtrip_rpm_error=float(np.max(abs(data['rpm_requested'] - data['rpm_executed']))))
    finite = all(np.isfinite(value).all() for value in data.values() if value.dtype.kind in 'fiu')
    safety = dict(finite=finite, physical_rpm=metrics['rpm_min'] >= 0 and metrics['rpm_max'] <= model.max_rpm + 1e-6,
                  attitude=float(np.max(abs(data['rpy_after'][:, :2]))) < .6,
                  height=bool(np.all((position[:, 2] >= .08) & (position[:, 2] <= 2.75))),
                  no_unexpected_done=not bool(data['ended_early'][-1]),
                  velocity_integral=bool(np.all(np.max(abs(data['velocity_integral']), axis=0) <= np.asarray(spec.integral_limit_mps2) + 1e-10)),
                  attitude_integral=bool(np.all(np.max(abs(data['attitude_integral']), axis=0) <= np.asarray(spec.attitude_integral_limit) + 1e-10)),
                  exact_substeps=metrics['physics_substeps'] == 2 * len(times),
                  exact_duration=len(times) == round(nominal_end * 120),
                  codec_roundtrip=metrics['max_codec_roundtrip_rpm_error'] < .005)
    criteria = {}
    events = []
    for index, start in enumerate(phase_starts):
        end = phase_starts[index+1] if index+1 < len(phase_starts) else nominal_end
        if end-start < 1.5:
            continue
        # A zero baseline before the first change does not create a step.
        target = np.array(scenario['phases'][index][1])
        selected = (times > end - 1. - 1e-8) & (times <= end + 1e-8)
        mean_error = np.mean(abs(error[selected]), axis=0)
        settle = settling_time(times, error, start, end, [.05, .05, .03])
        overshoots = []
        allowed = []
        active = (times > start) & (times <= end)
        for axis in range(3):
            if abs(target[axis]) > 1e-9:
                overshoots.append(float(np.max(np.sign(target[axis]) * (velocity[active, axis]-target[axis]))))
                allowed.append(.25*abs(target[axis])+.02)
        events.append(dict(start_s=start, end_s=end, target=target.tolist(),
                           last_1s_abs_mean_error_xyz=mean_error.tolist(), settling_s=settle,
                           overshoot_mps=overshoots, overshoot_limit_mps=allowed))
    metrics['events'] = events
    if scenario['family'] == 'hover':
        last = times >= nominal_end-5.
        criteria.update(hover_mean_speed=bool(np.all(np.mean(abs(velocity[last]), axis=0) <= .02)),
                        hover_max_speed=metrics['max_speed_mps'] <= .05,
                        hover_z_drift=abs(metrics['displacement_xyz'][2]) <= .15,
                        hover_xy_drift=np.linalg.norm(metrics['displacement_xyz'][:2]) <= .25)
    if scenario['family'] in ('step', 'turn'):
        criteria.update(steady_error=all(np.all(np.array(e['last_1s_abs_mean_error_xyz']) <= [.05, .05, .03]) for e in events),
                        response_time=all(e['settling_s'] is not None and e['settling_s'] <= 2. for e in events),
                        overshoot=all(np.all(np.array(e['overshoot_mps']) <= np.array(e['overshoot_limit_mps'])) for e in events))
    if scenario['family'] in ('diagonal', 'circle'):
        criteria.update(tracking_rmse_xy=metrics['steady_rmse_xy'] <= .10, tracking_rmse_z=metrics['steady_rmse_z'] <= .06)
    if scenario['family'] in ('disturbance', 'initial_velocity', 'attitude', 'saturation'):
        recovery_start = (4.15 if scenario['family'] == 'disturbance' else 6. if scenario['family'] == 'saturation' else 0.)
        recovery = settling_time(times, error, recovery_start, nominal_end, [.05, .05, .03], sustain_s=2.)
        metrics['recovery_s'] = recovery
        criteria['velocity_recovery'] = recovery is not None and recovery <= 2.
    if scenario['family'] == 'saturation':
        criteria['constraint_exercised'] = metrics['force_saturation_fraction'] > 0
    braking = []
    for event in events:
        if event['start_s'] > 0 and np.linalg.norm(event['target']) == 0:
            index = np.searchsorted(times, event['start_s'])
            stop_time = event['start_s'] + (event['settling_s'] or (event['end_s']-event['start_s']))
            stop = min(np.searchsorted(times, stop_time), len(times)-1)
            braking.append(dict(start_s=event['start_s'], settling_s=event['settling_s'],
                                displacement_m=float(np.linalg.norm(position[stop] - position[max(0, index-1)]))))
    metrics['braking'] = braking
    return dict(safety=safety, criteria=criteria, safety_pass=all(safety.values()),
                performance_pass=all(criteria.values()), passed=all(safety.values()) and all(criteria.values()), metrics=metrics)
