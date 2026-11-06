from copy import deepcopy

from experiments.direct_velocity_full_v1.contracts import ModelParameters, ControlSpec
from experiments.direct_velocity_full_v1.testing.control_metrics import evaluate
from experiments.direct_velocity_full_v1.testing.scenarios import case
from experiments.direct_velocity_full_v1.run_control_acceptance import run_case


def test_safe_tracking_average_cannot_override_illegal_motor_output():
    model, spec = ModelParameters(), ControlSpec()
    scenario = case('metric_safety_check', 'hover', 2., [(0, [0, 0, 0])])
    data, good, _ = run_case(scenario, 0, model, spec)
    assert good['passed']
    bad = {k:v.copy() for k,v in data.items()}
    bad['rpm_executed'][100, 0] = 1.1 * model.max_rpm
    outcome = evaluate(bad, scenario, model, spec)
    assert outcome['performance_pass']
    assert not outcome['safety_pass']
    assert not outcome['passed']
