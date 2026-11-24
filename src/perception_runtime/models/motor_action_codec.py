from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ASYMMETRIC_RPM = "asymmetric_rpm"
ASYMMETRIC_THRUST = "asymmetric_thrust"
SUPPORTED_MOTOR_ACTION_CODECS = (ASYMMETRIC_RPM, ASYMMETRIC_THRUST)


@dataclass(frozen=True)
class MotorPhysicalLimits:
    """Physical parameters shared by motor-action encoders and constraints."""

    min_rpm: float
    hover_rpm: float
    max_rpm: float
    kf: float

    def __post_init__(self):
        if not 0.0 <= float(self.min_rpm) < float(self.hover_rpm) < float(self.max_rpm):
            raise ValueError("Expected 0 <= min_rpm < hover_rpm < max_rpm.")
        if not np.isfinite(float(self.kf)) or float(self.kf) <= 0.0:
            raise ValueError("kf must be finite and positive.")

    @property
    def min_thrust(self) -> float:
        return float(self.kf) * float(self.min_rpm) ** 2

    @property
    def hover_thrust(self) -> float:
        return float(self.kf) * float(self.hover_rpm) ** 2

    @property
    def max_thrust(self) -> float:
        return float(self.kf) * float(self.max_rpm) ** 2


class PhysicalMotorConstraint:
    """Apply only per-motor physical RPM limits and an optional RPM slew limit."""

    def __init__(
        self,
        limits: MotorPhysicalLimits,
        *,
        max_delta_rpm: float | None = None,
    ):
        self.limits = limits
        if max_delta_rpm is not None and float(max_delta_rpm) <= 0.0:
            raise ValueError("max_delta_rpm must be positive when provided.")
        self.max_delta_rpm = None if max_delta_rpm is None else float(max_delta_rpm)

    def __call__(self, rpm, previous_rpm=None) -> np.ndarray:
        constrained = np.clip(
            np.asarray(rpm, dtype=np.float64),
            self.limits.min_rpm,
            self.limits.max_rpm,
        )
        if previous_rpm is not None and self.max_delta_rpm is not None:
            previous = np.asarray(previous_rpm, dtype=np.float64)
            constrained = np.clip(
                constrained,
                previous - self.max_delta_rpm,
                previous + self.max_delta_rpm,
            )
            constrained = np.clip(
                constrained,
                self.limits.min_rpm,
                self.limits.max_rpm,
            )
        return constrained

    def saturation_mask(self, rpm, *, atol: float = 1e-9) -> np.ndarray:
        rpm = np.asarray(rpm, dtype=np.float64)
        return np.logical_or(
            rpm <= self.limits.min_rpm + float(atol),
            rpm >= self.limits.max_rpm - float(atol),
        )


class MotorActionCodec:
    """Invertible motor-action encoding around hover.

    ``asymmetric_rpm`` is piecewise linear in RPM. ``asymmetric_thrust`` is
    piecewise linear in per-motor thrust. Both map hover to zero and the
    physical lower/upper limits to -1/+1 without imposing collective or
    differential projections.
    """

    def __init__(
        self,
        limits: MotorPhysicalLimits,
        mode: str = ASYMMETRIC_RPM,
        *,
        max_delta_rpm: float | None = None,
    ):
        if mode not in SUPPORTED_MOTOR_ACTION_CODECS:
            raise ValueError(
                f"Unsupported motor action codec {mode!r}; "
                f"expected one of {SUPPORTED_MOTOR_ACTION_CODECS}."
            )
        self.limits = limits
        self.mode = str(mode)
        self.physical_constraint = PhysicalMotorConstraint(
            limits,
            max_delta_rpm=max_delta_rpm,
        )

    @staticmethod
    def _piecewise_encode(values, lower: float, center: float, upper: float):
        values = np.asarray(values, dtype=np.float64)
        return np.where(
            values >= center,
            (values - center) / (upper - center),
            (values - center) / (center - lower),
        )

    @staticmethod
    def _piecewise_decode(actions, lower: float, center: float, upper: float):
        actions = np.clip(np.asarray(actions, dtype=np.float64), -1.0, 1.0)
        return np.where(
            actions >= 0.0,
            center + actions * (upper - center),
            center + actions * (center - lower),
        )

    def rpm_to_motor_thrust(self, rpm) -> np.ndarray:
        rpm = np.asarray(rpm, dtype=np.float64)
        return float(self.limits.kf) * np.square(rpm)

    def motor_thrust_to_rpm(self, motor_thrust) -> np.ndarray:
        thrust = np.maximum(np.asarray(motor_thrust, dtype=np.float64), 0.0)
        return np.sqrt(thrust / float(self.limits.kf))

    def rpm_to_normalized_action(self, rpm) -> np.ndarray:
        rpm = self.physical_constraint(rpm)
        if self.mode == ASYMMETRIC_RPM:
            return self._piecewise_encode(
                rpm,
                self.limits.min_rpm,
                self.limits.hover_rpm,
                self.limits.max_rpm,
            )
        return self.motor_thrust_to_normalized_action(
            self.rpm_to_motor_thrust(rpm)
        )

    def normalized_action_to_rpm(self, action, previous_rpm=None) -> np.ndarray:
        if self.mode == ASYMMETRIC_RPM:
            rpm = self._piecewise_decode(
                action,
                self.limits.min_rpm,
                self.limits.hover_rpm,
                self.limits.max_rpm,
            )
        else:
            rpm = self.motor_thrust_to_rpm(
                self.normalized_action_to_motor_thrust(action)
            )
        return self.physical_constraint(rpm, previous_rpm=previous_rpm)

    def motor_thrust_to_normalized_action(self, motor_thrust) -> np.ndarray:
        thrust = np.clip(
            np.asarray(motor_thrust, dtype=np.float64),
            self.limits.min_thrust,
            self.limits.max_thrust,
        )
        return self._piecewise_encode(
            thrust,
            self.limits.min_thrust,
            self.limits.hover_thrust,
            self.limits.max_thrust,
        )

    def normalized_action_to_motor_thrust(self, action) -> np.ndarray:
        return self._piecewise_decode(
            action,
            self.limits.min_thrust,
            self.limits.hover_thrust,
            self.limits.max_thrust,
        )

    # Explicit aliases matching the engineering-interface terminology.
    encode_rpm = rpm_to_normalized_action
    decode_rpm = normalized_action_to_rpm
    encode_motor_thrust = motor_thrust_to_normalized_action
    decode_motor_thrust = normalized_action_to_motor_thrust


class LegacyFixedScaleMotorActionCodec:
    """The historical 0.05 RPM mapping retained only for compatibility/tests."""

    def __init__(self, hover_rpm: float, scale: float = 0.05):
        if float(hover_rpm) <= 0.0 or float(scale) <= 0.0:
            raise ValueError("hover_rpm and scale must be positive.")
        self.hover_rpm = float(hover_rpm)
        self.scale = float(scale)

    def rpm_to_normalized_action(self, rpm) -> np.ndarray:
        rpm = np.asarray(rpm, dtype=np.float64)
        return (rpm / self.hover_rpm - 1.0) / self.scale

    def normalized_action_to_rpm(self, action) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64)
        return self.hover_rpm * (1.0 + self.scale * action)

    encode_rpm = rpm_to_normalized_action
    decode_rpm = normalized_action_to_rpm
