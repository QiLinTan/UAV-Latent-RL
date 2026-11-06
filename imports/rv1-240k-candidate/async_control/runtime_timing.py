"""Wall-clock timing summaries for runtime asynchronous control."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


LOW_LEVEL_PERIOD_SECONDS = 1.0 / 120.0


def summarize_series(values, *, deadline_seconds: float | None = None) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        result = {
            "count": 0,
            "mean_seconds": 0.0,
            "median_seconds": 0.0,
            "p95_seconds": 0.0,
            "p99_seconds": 0.0,
            "max_seconds": 0.0,
        }
    else:
        result = {
            "count": int(values.size),
            "mean_seconds": float(np.mean(values)),
            "median_seconds": float(np.median(values)),
            "p95_seconds": float(np.percentile(values, 95.0)),
            "p99_seconds": float(np.percentile(values, 99.0)),
            "max_seconds": float(np.max(values)),
        }
    if deadline_seconds is not None:
        result["deadline_seconds"] = float(deadline_seconds)
        result["count_above_deadline"] = int(np.sum(values > float(deadline_seconds)))
    return result


@dataclass
class RuntimeTimingStats:
    cycle_start_time: list[float] = field(default_factory=list)
    cycle_end_time: list[float] = field(default_factory=list)
    cycle_duration: list[float] = field(default_factory=list)
    since_previous_start: list[float] = field(default_factory=list)
    observation_publish_time: list[float] = field(default_factory=list)
    packet_read_select_time: list[float] = field(default_factory=list)
    pid_compute_time: list[float] = field(default_factory=list)
    environment_step_time: list[float] = field(default_factory=list)
    packet_age_at_receive: list[float] = field(default_factory=list)
    packet_transport_time: list[float] = field(default_factory=list)

    def record_cycle(
        self,
        *,
        start: float,
        end: float,
        previous_start: float | None,
        observation_publish_time: float,
        packet_read_select_time: float,
        pid_compute_time: float,
        environment_step_time: float,
    ) -> None:
        self.cycle_start_time.append(float(start))
        self.cycle_end_time.append(float(end))
        self.cycle_duration.append(float(end) - float(start))
        self.since_previous_start.append(
            0.0 if previous_start is None else float(start) - float(previous_start)
        )
        self.observation_publish_time.append(float(observation_publish_time))
        self.packet_read_select_time.append(float(packet_read_select_time))
        self.pid_compute_time.append(float(pid_compute_time))
        self.environment_step_time.append(float(environment_step_time))

    def record_packet_receive(self, *, receive_wall_time: float, packet_generate_wall_time: float, packet_publish_wall_time: float) -> None:
        self.packet_age_at_receive.append(float(receive_wall_time) - float(packet_generate_wall_time))
        self.packet_transport_time.append(float(receive_wall_time) - float(packet_publish_wall_time))

    def summary(self) -> dict:
        return {
            "total_cycles": int(len(self.cycle_duration)),
            "cycle_duration": summarize_series(
                self.cycle_duration,
                deadline_seconds=LOW_LEVEL_PERIOD_SECONDS,
            ),
            "since_previous_low_level_cycle": summarize_series(
                self.since_previous_start[1:],
                deadline_seconds=LOW_LEVEL_PERIOD_SECONDS,
            ),
            "observation_publish_time": summarize_series(self.observation_publish_time),
            "packet_read_select_time": summarize_series(self.packet_read_select_time),
            "pid_compute_time": summarize_series(self.pid_compute_time),
            "environment_step_time": summarize_series(self.environment_step_time),
            "packet_age_at_receive": summarize_series(self.packet_age_at_receive),
            "packet_transport_time": summarize_series(self.packet_transport_time),
        }


def summarize_high_level_events(events: list[dict]) -> dict:
    if not events:
        return {"decision_count": 0}
    fields = (
        "preprocess_time_seconds",
        "encoder_inference_time_seconds",
        "planner_inference_time_seconds",
        "actor_adapter_time_seconds",
        "packet_generation_time_seconds",
        "artificial_delay_time_seconds",
        "total_high_level_time_seconds",
        "observation_age_at_start_seconds",
    )
    return {
        "decision_count": int(len(events)),
        **{field: summarize_series([event[field] for event in events]) for field in fields},
        "planner_loaded": bool(any(event.get("planner_loaded", False) for event in events)),
    }


def count_low_cycles_during_high_compute(low_cycle_starts, high_events: list[dict]) -> dict:
    starts = np.asarray(low_cycle_starts, dtype=np.float64)
    counts = []
    for event in high_events:
        begin = float(event["compute_start_wall_time"])
        end = float(event["compute_end_wall_time"])
        counts.append(int(np.sum((starts >= begin) & (starts <= end))))
    if not counts:
        return {"max": 0, "mean": 0.0, "counts": []}
    return {"max": int(max(counts)), "mean": float(np.mean(counts)), "counts": counts}
