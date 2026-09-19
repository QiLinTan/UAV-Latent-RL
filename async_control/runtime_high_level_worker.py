"""Spawned high-level worker for runtime asynchronous reference generation."""

from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
from multiprocessing.context import BaseContext
import queue
import time
import traceback

import numpy as np

from .latest_channel import LatestObservationSlot, get_latest, put_latest
from .reference_generator import HighLevelDecision, ReferenceGenerator


@dataclass(frozen=True)
class RuntimeWorkerConfig:
    checkpoint_prefix: str = "checkpoints/latent_only_affordance_dim16/model_final"
    high_frequency_hz: float = 15.0
    artificial_delay_seconds: float = 0.0
    policy_mode: str = "latent_actor_adapter"
    horizon_seconds: float = 0.5
    sequence_length: int = 16
    max_speed: float = 0.6
    max_acceleration: float = 1.2
    max_vertical_speed: float = 0.35


def _load_encoder_actor(checkpoint_prefix: str):
    import sys
    from pathlib import Path

    import torch

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from algos.td3.networks import Actor
    from models.encoder import Encoder

    encoder = Encoder(263, 16)
    actor = Actor(16, 4, 1.0)
    encoder.load_state_dict(torch.load(str(checkpoint_prefix) + "_encoder", map_location="cpu", weights_only=True))
    actor.load_state_dict(torch.load(str(checkpoint_prefix) + "_actor", map_location="cpu", weights_only=True))
    encoder.eval().requires_grad_(False)
    actor.eval().requires_grad_(False)
    torch.set_num_threads(1)
    return torch, encoder, actor


def _decision_from_actor(torch_module, encoder, actor, state, snapshot, max_speed: float):
    position = np.asarray(snapshot["position"], dtype=np.float32).reshape(3)
    target = np.asarray(snapshot["target_position"], dtype=np.float32).reshape(3)
    yaw = float(snapshot["yaw"])
    with torch_module.inference_mode():
        t0 = time.perf_counter()
        latent = encoder(torch_module.from_numpy(state).reshape(1, -1))
        t1 = time.perf_counter()
        motor_action = actor(latent)
        t2 = time.perf_counter()
    latent_value = latent[0].numpy().astype(np.float32)
    motor = motor_action[0].numpy().astype(np.float32)

    delta = target - position
    planar_norm = float(np.linalg.norm(delta[:2]))
    nominal = np.zeros(3, dtype=np.float32)
    if planar_norm > 1e-6:
        nominal[:2] = delta[:2] / planar_norm * min(float(max_speed), planar_norm)
    nominal[2] = float(np.clip(delta[2], -0.25, 0.25))

    body_residual = np.array(
        [
            0.25 * (motor[0] + motor[1] - motor[2] - motor[3]),
            0.25 * (-motor[0] + motor[1] + motor[2] - motor[3]),
        ],
        dtype=np.float32,
    )
    rotation = np.array(
        [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]],
        dtype=np.float32,
    )
    residual_world = rotation @ body_residual
    desired_velocity = nominal + np.array(
        [0.08 * residual_world[0], 0.08 * residual_world[1], 0.025 * float(motor.mean())],
        dtype=np.float32,
    )
    planar = float(np.linalg.norm(desired_velocity[:2]))
    if planar > float(max_speed):
        desired_velocity[:2] *= float(max_speed) / planar
    desired_velocity[2] = np.clip(desired_velocity[2], -0.35, 0.35)
    return (
        HighLevelDecision(
            target_position=target,
            desired_velocity=desired_velocity,
            latent=latent_value,
            policy_action=motor,
        ),
        float(t1 - t0),
        float(t2 - t1),
    )


def _constant_decision(snapshot, max_speed: float) -> HighLevelDecision:
    position = np.asarray(snapshot["position"], dtype=np.float32).reshape(3)
    target = np.asarray(snapshot["target_position"], dtype=np.float32).reshape(3)
    delta = target - position
    velocity = np.zeros(3, dtype=np.float32)
    norm = float(np.linalg.norm(delta[:2]))
    if norm > 1e-6:
        velocity[:2] = delta[:2] / norm * min(float(max_speed), norm)
    velocity[2] = float(np.clip(delta[2], -0.25, 0.25))
    return HighLevelDecision(
        target_position=target,
        desired_velocity=velocity,
        latent=np.zeros(16, dtype=np.float32),
        policy_action=np.zeros(4, dtype=np.float32),
    )


def high_level_worker_main(observation_channel, packet_channel, status_channel, stop_event, config: RuntimeWorkerConfig) -> None:
    version = 1
    last_start = 0.0
    period = 1.0 / float(config.high_frequency_hz)
    torch_module = encoder = actor = None
    try:
        if config.policy_mode == "crash_after_start":
            raise RuntimeError("intentional worker crash for runtime async test")
        if config.policy_mode == "latent_actor_adapter":
            torch_module, encoder, actor = _load_encoder_actor(config.checkpoint_prefix)
        elif config.policy_mode not in ("constant", "crash_after_snapshot"):
            raise ValueError(f"Unknown policy_mode {config.policy_mode!r}")
        generator = ReferenceGenerator(
            horizon_seconds=config.horizon_seconds,
            sequence_length=config.sequence_length,
            max_speed=config.max_speed,
            max_acceleration=config.max_acceleration,
            max_vertical_speed=config.max_vertical_speed,
        )
        status_channel.put({"type": "started", "wall_time": time.perf_counter(), "policy_mode": config.policy_mode})
        last_observation_version = 0
        while not stop_event.is_set():
            now = time.perf_counter()
            if last_start and now - last_start < period:
                time.sleep(min(0.001, period - (now - last_start)))
                continue
            snapshot = observation_channel.get_latest(last_observation_version)
            if snapshot is None:
                time.sleep(0.001)
                continue
            last_observation_version = int(snapshot["version"])
            compute_start = time.perf_counter()
            last_start = compute_start
            if config.policy_mode == "crash_after_snapshot":
                raise RuntimeError("intentional worker crash after receiving snapshot")

            t0 = time.perf_counter()
            from envs.preprocess import preprocess_state

            state = preprocess_state(np.asarray(snapshot["observation"], dtype=np.float32).reshape(-1))
            t1 = time.perf_counter()
            if config.policy_mode == "latent_actor_adapter":
                decision, encoder_time, actor_time = _decision_from_actor(
                    torch_module,
                    encoder,
                    actor,
                    state,
                    snapshot,
                    config.max_speed,
                )
                t2 = time.perf_counter()
            else:
                decision = _constant_decision(snapshot, config.max_speed)
                t2 = time.perf_counter()
                encoder_time = 0.0
                actor_time = float(t2 - t1)
            if config.artificial_delay_seconds > 0.0:
                time.sleep(float(config.artificial_delay_seconds))
            t3 = time.perf_counter()
            packet = generator.generate(
                decision,
                version=version,
                t_generate=float(snapshot["wall_time"]),
                t_ready=t3,
                previous_packet=snapshot.get("previous_packet"),
                fallback_position=snapshot["fallback_position"],
                fallback_velocity=snapshot["fallback_velocity"],
            )
            version += 1
            t4 = time.perf_counter()
            event = {
                "type": "packet",
                "version": int(packet.version),
                "snapshot_sim_time": float(snapshot["sim_time"]),
                "snapshot_wall_time": float(snapshot["wall_time"]),
                "compute_start_wall_time": float(compute_start),
                "compute_end_wall_time": float(t4),
                "publish_wall_time": float(t4),
                "preprocess_time_seconds": float(t1 - t0),
                "encoder_inference_time_seconds": encoder_time,
                "planner_inference_time_seconds": 0.0,
                "actor_adapter_time_seconds": actor_time,
                "packet_generation_time_seconds": float(t4 - t3),
                "artificial_delay_time_seconds": float(config.artificial_delay_seconds),
                "total_high_level_time_seconds": float(t4 - compute_start),
                "observation_age_at_start_seconds": float(compute_start - float(snapshot["wall_time"])),
                "planner_loaded": False,
                "reference_switching_error": float(generator.last_switching_error),
            }
            put_latest(packet_channel, {"packet": packet, "worker_timing": event})
            status_channel.put(event)
    except BaseException:
        status_channel.put(
            {"type": "error", "wall_time": time.perf_counter(), "traceback": traceback.format_exc()}
        )


class RuntimeHighLevelWorker:
    """Small lifecycle wrapper around a spawned high-level process."""

    def __init__(self, config: RuntimeWorkerConfig, *, context: BaseContext | None = None):
        self.config = config
        self.context = context or mp.get_context("spawn")
        self.observation_channel = LatestObservationSlot(self.context)
        self.packet_channel = self.context.Queue(maxsize=1)
        self.status_channel = self.context.Queue()
        self.stop_event = self.context.Event()
        self.process = self.context.Process(
            target=high_level_worker_main,
            args=(self.observation_channel, self.packet_channel, self.status_channel, self.stop_event, self.config),
            daemon=False,
        )

    def start(self) -> None:
        self.process.start()

    def publish_observation(self, snapshot: dict) -> None:
        self.observation_channel.put(snapshot)

    def get_packet(self):
        return get_latest(self.packet_channel)

    def get_status(self):
        return get_latest(self.status_channel)

    def get_all_status(self) -> list[dict]:
        values = []
        while True:
            try:
                values.append(self.status_channel.get_nowait())
            except queue.Empty:
                return values

    def stop(self, timeout: float = 5.0) -> dict:
        self.stop_event.set()
        self.process.join(timeout)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout)
        return {"pid": self.process.pid, "exitcode": self.process.exitcode, "alive": self.process.is_alive()}
