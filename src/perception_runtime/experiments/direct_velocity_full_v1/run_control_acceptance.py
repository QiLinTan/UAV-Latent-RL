"""Control-only calibration / frozen Gate A / paired old Full execution."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from .contracts import ControlSpec, ModelParameters, VelocityCommand, body_to_world, force_to_attitude, collective_force_to_pwm, json_hash
from .controls.velocity_tracking import VelocityTrackingController
from .testing.control_fixture import FullControlFixture
from .testing.scenarios import calibration_cases, acceptance_cases, sampled_tape
from .testing.control_metrics import evaluate

EXPERIMENT = Path(__file__).resolve().parent
ROOT = EXPERIMENT.parents[1]


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def write_json(path, value):
    path.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def collect_metadata():
    hashes = {str(path.relative_to(EXPERIMENT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(EXPERIMENT.rglob('*.py')) if 'outputs' not in path.parts}
    old_paths = ['envs/learned_upper_dslpid.py', 'models/motor_action_codec.py', 'models/reference_packet.py', 'envs/forest/core.py']
    hashes.update({str(ROOT / p): hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in old_paths})
    package = ROOT / '.wsl2/drones-cpu/lib/python3.10/site-packages/gym_pybullet_drones'
    for p in ['control/DSLPIDControl.py', 'envs/BaseAviary.py', 'assets/cf2x.urdf']:
        hashes[str(package / p)] = hashlib.sha256((package / p).read_bytes()).hexdigest()
    return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), source_hashes=hashes,
                source_set_sha256=json_hash(hashes), git_head=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                simulation_time_base='integer lower tick / 120; physics tick / 240',
                wall_clock_not_used_for_control=True, rollout_type='scripted control only; no RL')


def original_controller_diagnostics(teacher, kwargs, previous_rpy):
    current_rotation = body_to_world(kwargs['cur_quat'])
    force = teacher.P_COEFF_FOR * (kwargs['target_pos']-kwargs['cur_pos']) + teacher.I_COEFF_FOR * teacher.integral_pos_e + teacher.D_COEFF_FOR * (kwargs['target_vel']-kwargs['cur_vel']) + [0., 0., teacher.GRAVITY]
    target, euler = force_to_attitude(force, 0.)
    import pybullet as p
    rpy = np.asarray(p.getEulerFromQuaternion(kwargs['cur_quat']))
    rates = (rpy - previous_rpy) * 120
    error_matrix = target.T @ current_rotation - current_rotation.T @ target
    error = np.array([error_matrix[2, 1], error_matrix[0, 2], error_matrix[1, 0]])
    torque = -teacher.P_COEFF_TOR * error - teacher.D_COEFF_TOR * rates + teacher.I_COEFF_TOR * teacher.integral_rpy_e
    torque_clipped = np.clip(torque, -3200, 3200)
    collective = collective_force_to_pwm(max(0., force@current_rotation[:, 2]), ModelParameters())
    pwm_raw = collective + teacher.MIXER_MATRIX @ torque_clipped
    pwm = np.clip(pwm_raw, teacher.MIN_PWM, teacher.MAX_PWM)
    return dict(force_raw_n=force, force_limited_n=force, collective_requested_n=max(0., force@current_rotation[:, 2]),
                collective_pwm=collective, velocity_integral_after=np.zeros(3),
                legacy_position_integral=teacher.integral_pos_e.copy(), attitude_integral_after=teacher.integral_rpy_e.copy(),
                target_euler_intrinsic_XYZ=euler, measured_euler_rates=rates,
                attitude_rotation_error=error, torque_raw=torque, torque_limited=torque_clipped,
                pwm_raw=pwm_raw, pwm_limited=pwm,
                attitude_integrator_blocked=np.zeros(3, dtype=bool), integrator_blocked=np.zeros(3, dtype=bool),
                force_limited=False, allocation_saturated=bool(np.any(abs(torque-torque_clipped)>1e-9) or np.any(abs(pwm_raw-pwm)>1e-9)),
                physical_rpm_saturation_mask=np.zeros(4, dtype=bool))


def run_case(scenario, seed, model, spec, *, legacy=False):
    fixture = FullControlFixture(rpy=scenario['initial_rpy'], position=scenario['initial_position'])
    rows = []
    tape = sampled_tape(scenario)
    fixture.disturbances = scenario['disturbances']
    current = {}
    controller = None
    legacy_wrapper = None
    begin_wall = time.perf_counter()
    try:
        if legacy:
            from envs.learned_upper_dslpid import LearnedUpperDSLPIDEnv
            legacy_wrapper = LearnedUpperDSLPIDEnv(fixture, upper_control_interval=8,
                reference_sequence_length=15, reference_horizon_seconds=1., max_reference_speed=.8,
                max_reference_acceleration=1., max_reference_vertical_speed=.3,
                max_reference_vertical_acceleration=.5)
            legacy_wrapper.reset(seed=seed)
            fixture.initialize_motion(scenario['initial_velocity'])
            original_compute = legacy_wrapper.teacher.computeControl
            def control_observer(**kwargs):
                previous = legacy_wrapper.teacher.last_rpy.copy()
                rpm, pe, ye = original_compute(**kwargs)
                current['diagnostic'] = original_controller_diagnostics(legacy_wrapper.teacher, kwargs, previous)
                current['rpm'] = rpm.copy()
                return rpm, pe, ye
            legacy_wrapper.teacher.computeControl = control_observer  # Disposable instance only.
        else:
            fixture.reset(seed=seed)
            fixture.initialize_motion(scenario['initial_velocity'])
            controller = VelocityTrackingController(model, spec)
            controller.reset(fixture.motion_state())

        def record(result):
            _, _, terminated, truncated, info = result
            d = current['diagnostic']
            row = dict(time_s=fixture.physics_ticks/240, physics_tick=fixture.physics_ticks,
                command_world=current['command'].copy(), velocity_before=current['velocity_before'].copy(),
                position_after=fixture.pos[0].copy(), velocity_after=fixture.vel[0].copy(),
                angular_velocity_world_after=fixture.ang_v[0].copy(), quaternion_after=fixture.quat[0].copy(),
                rpy_after=fixture.rpy[0].copy(), rpm_requested=current['rpm'].copy(),
                rpm_executed=fixture._last_motor_rpm[0].copy(),
                force_raw_n=d['force_raw_n'], force_limited_n=d['force_limited_n'],
                collective_requested_n=d['collective_requested_n'], collective_pwm=d['collective_pwm'],
                velocity_integral=d['velocity_integral_after'], attitude_integral=d['attitude_integral_after'],
                legacy_position_integral=d.get('legacy_position_integral', np.zeros(3)),
                target_euler=d['target_euler_intrinsic_XYZ'], measured_euler_rates=d['measured_euler_rates'],
                torque_raw=d['torque_raw'], torque_limited=d['torque_limited'],
                pwm_raw=d['pwm_raw'], pwm_limited=d['pwm_limited'],
                velocity_integrator_blocked=d['integrator_blocked'], attitude_integrator_blocked=d['attitude_integrator_blocked'],
                allocation_saturated=d['allocation_saturated'], force_saturated=d['force_limited'],
                physical_rpm_saturated=bool(np.any(d['physical_rpm_saturation_mask'])),
                ended_early=bool(terminated or truncated), done_reason=info['done_reason'],
                lower_tick=len(rows), upper_tick=current['upper_tick'],
                control_wall_seconds=current.get('control_wall_seconds', 0.))
            rows.append(row)
        fixture.tick_observer = record

        if legacy:
            # Capture each lower input without modifying the wrapper's control law.
            original_step = fixture.step
            def lower_step(action):
                current['velocity_before'] = fixture.vel[0].copy()
                return original_step(action)
            fixture.step = lower_step
        done = False
        for upper_index, command in enumerate(tape):
            current.update(command=command.copy(), upper_tick=upper_index)
            if legacy:
                _, _, terminated, truncated, _ = legacy_wrapper.step((command/[.8, .8, .3]).reshape(1, 3))
                done = bool(terminated or truncated)
            else:
                for inner in range(8):
                    lower_tick = controller.tick
                    time_s = lower_tick/120
                    active_spec = spec
                    window = scenario.get('limit_window')
                    if window and window['start_s'] <= time_s < window['end_s']:
                        active_spec = replace(spec, max_tilt_rad=window['max_tilt_rad'], max_force_fraction=window['max_force_fraction'])
                    controller.spec = active_spec
                    controller.pi.spec = active_spec
                    controller.adapter.spec = active_spec
                    current['velocity_before'] = fixture.vel[0].copy()
                    start = time.perf_counter()
                    rpm, diagnostic = controller.update(
                        VelocityCommand(upper_index, lower_tick, command) if inner == 0 else None,
                        fixture.motion_state())
                    current.update(diagnostic=diagnostic, rpm=rpm, control_wall_seconds=time.perf_counter()-start)
                    motor = fixture.motor_action_codec.rpm_to_normalized_action(rpm).astype(np.float32)
                    _, _, terminated, truncated, _ = fixture.step(motor.reshape(1, 4))
                    done = bool(terminated or truncated)
                    if done:
                        break
            if done:
                break
        if controller is not None:
            controller.finish_episode()
        data = {key: np.array([row[key] for row in rows]) for key in rows[0]}
        report = evaluate(data, scenario, model, spec)
        import pybullet as p
        report['physics_parameters'] = dict(
            engine=p.getPhysicsEngineParameters(physicsClientId=fixture.CLIENT),
            base_dynamics_info=p.getDynamicsInfo(int(fixture.DRONE_IDS[0]), -1, physicsClientId=fixture.CLIENT),
            drone_model=str(fixture.DRONE_MODEL), physics=str(fixture.PHYSICS),
            pyb_hz=fixture.PYB_FREQ, control_hz=fixture.CTRL_FREQ,
            mass_kg=fixture.M, kf=fixture.KF, km=fixture.KM,
            inertia=fixture.J.tolist(), collision_radius_m=fixture.COLLISION_R,
            collision_height_m=fixture.COLLISION_H, collision_z_offset_m=fixture.COLLISION_Z_OFFSET,
            hover_rpm=fixture.HOVER_RPM, max_rpm=fixture.MAX_RPM)
        report.update(name=scenario['name'], family=scenario['family'], seed=seed,
                      controller='legacy_full' if legacy else 'direct_velocity',
                      elapsed_wall_s=time.perf_counter()-begin_wall,
                      applied_disturbance_substeps=fixture.applied_disturbance_substeps,
                      tape_sha256=hashlib.sha256(tape.tobytes()).hexdigest())
        return data, report, tape
    finally:
        fixture.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('calibration', 'acceptance'), required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--legacy-comparison', action='store_true')
    parser.add_argument('--cases', nargs='*')
    args = parser.parse_args()
    configuration = json.loads(args.config.read_text())
    if args.phase == 'acceptance' and not configuration.get('frozen'):
        raise ValueError('acceptance requires a frozen config')
    output = args.output.resolve()
    if EXPERIMENT / 'outputs' not in output.parents:
        raise ValueError('output must be a new directory below experiment/outputs')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'trajectories').mkdir()
    metadata = collect_metadata()
    model = ModelParameters.from_installed()
    spec = ControlSpec(**configuration['control_spec'])
    cases = calibration_cases() if args.phase == 'calibration' else acceptance_cases()
    if args.cases:
        cases = [c for c in cases if c['name'] in args.cases]
        if not cases:
            raise ValueError('no matching cases')
    seeds = configuration.get('seeds', [0, 1, 2])
    write_json(output / 'configuration.json', configuration)
    write_json(output / 'metadata.json', metadata)
    write_json(output / 'physical_parameters.json', asdict(model))
    write_json(output / 'scenarios.json', cases)
    reports = []
    for scenario in cases:
        for seed in seeds:
            for legacy in ([False, True] if args.legacy_comparison and scenario['family'] not in ('attitude', 'saturation') else [False]):
                data, report, tape = run_case(scenario, seed, model, spec, legacy=legacy)
                basename = f'{scenario["name"]}_seed{seed}_{report["controller"]}'
                np.savez_compressed(output / 'trajectories' / (basename + '.npz'), **data)
                np.save(output / 'trajectories' / (basename + '_upper_world_tape.npy'), tape)
                write_json(output / 'trajectories' / (basename + '.json'), report)
                reports.append(report)
                write_json(output / 'results.json', reports)
                failure = [k for group in ('safety', 'criteria') for k,v in report[group].items() if not v]
                print(json.dumps(dict(case=scenario['name'], seed=seed, controller=report['controller'],
                                     passed=report['passed'], failures=failure,
                                     rmse=report['metrics']['command_velocity_rmse_vector']), ensure_ascii=False), flush=True)
    new = [r for r in reports if r['controller'] == 'direct_velocity']
    write_json(output / 'summary.json', dict(phase=args.phase, config_sha256=json_hash(configuration),
        required_new_cases=len(new), new_case_passed=sum(r['passed'] for r in new),
        new_safety_failures=sum(not r['safety_pass'] for r in new),
        new_performance_failures=sum(not r['performance_pass'] for r in new),
        all_cases_passed=all(r['passed'] for r in new), report_count=len(reports),
        complete_case_matrix=not bool(args.cases), controller_implementation='isolated DSL-form attitude with explicit state'))


if __name__ == '__main__':
    main()
