"""Sign and censoring checks for the A2 collision-tree response audit."""

import numpy as np

from scripts.analyze_full_k1_response_budget import (
    directional_series,
    outward_run_lengths,
    turn_events,
)


def test_directional_projection_and_velocity_gap():
    trace = {
        "lower_position": np.array([[2.0, 0.0, 0.0]]),
        "lower_time_seconds": np.array([1.0]),
        "lower_clearance": np.array([0.7]),
        "lower_command_velocity": np.array([[-0.2, 0.0, 0.0]]),
        "lower_reference_velocity": np.array([[0.1, 0.0, 0.0]]),
        "lower_velocity": np.array([[-0.3, 0.0, 0.0]]),
    }
    result = directional_series(trace, center=[0.0, 0.0], tree_radius=1.0, drone_radius=0.3)
    assert np.isclose(result["clearance_to_collision_tree"][0], 0.7)
    assert np.isclose(result["inward"]["cmd"][0], 0.2)
    assert np.isclose(result["inward"]["ref"][0], -0.1)
    assert np.isclose(result["inward"]["actual"][0], 0.3)
    assert np.isclose(result["e_gen_3d"][0], 0.3)
    assert np.isclose(result["e_track_3d"][0], 0.4)
    assert np.isclose(result["e_gen_inward"][0], -0.3)
    assert np.isclose(result["e_track_inward"][0], 0.4)


def test_sustained_turn_response_and_censoring():
    count = 30
    time = np.arange(count, dtype=float) / 120
    actual = np.full(count, 0.6)
    actual[8:] = 0.45
    reference = np.full(count, 0.5)
    reference[6:] = 0.35
    command = np.full(count, 0.2)
    command[:16] = -0.2
    series = {
        "time": time,
        "clearance_to_collision_tree": np.full(count, 0.12),
        "outward": np.tile([1.0, 0.0], (count, 1)),
        "inward": {"cmd": command, "ref": reference, "actual": actual},
    }
    assert outward_run_lengths(series, collision_time=time[-1]) == [16]
    events = turn_events(series, collision_time=time[-1])
    assert len(events) == 1
    event = events[0]
    assert np.isclose(event["available_time_s"], 0.2)
    assert np.isclose(event["reference_stop_distance_optimistic_m"], 0.125)
    assert np.isclose(event["actual_stop_distance_at_reference_bound_m"], 0.18)
    assert np.isclose(event["reference_drop_0_1_s"], 6 / 120)
    assert np.isclose(event["actual_drop_0_1_s"], 8 / 120)
    assert event["actual_noninward_s"] is None
    assert event["actual_noninward_censored"]
    series["inward"]["cmd"][15] = 0.2
    assert turn_events(series, collision_time=time[-1]) == []
