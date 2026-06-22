import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((self.max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((self.max_size, 1), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        self.push_batch(
            np.asarray(state).reshape(1, -1),
            np.asarray(action).reshape(1, -1),
            np.asarray([reward]),
            np.asarray(next_state).reshape(1, -1),
            np.asarray([done]),
        )

    def push_batch(self, state, action, reward, next_state, done):
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        reward = np.asarray(reward, dtype=np.float32).reshape(-1)
        next_state = np.asarray(next_state, dtype=np.float32)
        done = np.asarray(done, dtype=np.float32).reshape(-1)

        if state.ndim != 2 or state.shape[1] != self.state.shape[1]:
            raise ValueError(
                f"Expected state batch [N, {self.state.shape[1]}], got {state.shape}."
            )
        if next_state.shape != state.shape:
            raise ValueError("State and next_state batches must have identical shapes.")
        if action.ndim != 2 or action.shape[1] != self.action.shape[1]:
            raise ValueError(
                f"Expected action batch [N, {self.action.shape[1]}], got {action.shape}."
            )
        batch_size = state.shape[0]
        if (
            action.shape[0] != batch_size
            or reward.shape[0] != batch_size
            or done.shape[0] != batch_size
        ):
            raise ValueError("All replay batch fields must have the same leading dimension.")
        if batch_size == 0:
            return

        if batch_size > self.max_size:
            state = state[-self.max_size :]
            action = action[-self.max_size :]
            reward = reward[-self.max_size :]
            next_state = next_state[-self.max_size :]
            done = done[-self.max_size :]
            batch_size = self.max_size

        indices = (np.arange(batch_size) + self.ptr) % self.max_size
        self.state[indices] = state
        self.action[indices] = action
        self.next_state[indices] = next_state
        self.reward[indices, 0] = reward
        self.not_done[indices, 0] = 1.0 - done

        self.ptr = (self.ptr + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.state[indices]),
            torch.as_tensor(self.action[indices]),
            torch.as_tensor(self.next_state[indices]),
            torch.as_tensor(self.reward[indices]),
            torch.as_tensor(self.not_done[indices]),
        )
