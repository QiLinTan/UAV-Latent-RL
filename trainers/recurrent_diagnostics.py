"""Observation-contract and replay-allocation diagnostics for upper TD3."""

from __future__ import annotations


def recurrent_observation_contract(
    *,
    base_state_dim: int,
    history_frame_dim: int,
    history_length: int,
    state_dim: int,
) -> dict[str, int]:
    base_state_dim = int(base_state_dim)
    history_frame_dim = int(history_frame_dim)
    history_length = int(history_length)
    state_dim = int(state_dim)
    expected_state_dim = base_state_dim + history_length * history_frame_dim
    if state_dim != expected_state_dim:
        raise ValueError(
            "Recurrent observation contract mismatch: "
            f"state_dim={state_dim}, expected={expected_state_dim} "
            f"({base_state_dim} + {history_length}*{history_frame_dim})."
        )
    return {
        "base_state_dim": base_state_dim,
        "history_frame_dim": history_frame_dim,
        "history_length": history_length,
        "state_dim": state_dim,
        "expected_state_dim": expected_state_dim,
    }


def replay_buffer_memory_bytes(*, state_dim: int, action_dim: int, capacity: int) -> int:
    """Return the allocation of the float32 ReplayBuffer arrays.

    The buffer owns state, action, next_state, reward, and not_done arrays.
    """
    state_dim = int(state_dim)
    action_dim = int(action_dim)
    capacity = int(capacity)
    if min(state_dim, action_dim, capacity) <= 0:
        raise ValueError("state_dim, action_dim, and capacity must be positive")
    floats_per_transition = 2 * state_dim + action_dim + 2
    return int(capacity * floats_per_transition * 4)
