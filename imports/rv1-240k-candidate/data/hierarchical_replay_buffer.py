import numpy as np
import torch


class HierarchicalReplayBuffer:
    """Replay buffer that preserves the exact asynchronous policy context."""

    def __init__(self, state_dim, action_dim, context_dim, max_size=int(1e6)):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.context = np.zeros((self.max_size, context_dim), dtype=np.float32)
        self.next_context = np.zeros((self.max_size, context_dim), dtype=np.float32)
        self.reward = np.zeros((self.max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((self.max_size, 1), dtype=np.float32)

    def push(self, state, action, reward, next_state, done, *, context, next_context):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.context[self.ptr] = context
        self.next_context[self.ptr] = next_context
        self.reward[self.ptr] = float(reward)
        self.not_done[self.ptr] = 1.0 - float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.state[indices]),
            torch.as_tensor(self.action[indices]),
            torch.as_tensor(self.next_state[indices]),
            torch.as_tensor(self.reward[indices]),
            torch.as_tensor(self.not_done[indices]),
            torch.as_tensor(self.context[indices]),
            torch.as_tensor(self.next_context[indices]),
        )
