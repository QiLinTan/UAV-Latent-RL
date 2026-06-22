import numpy as np

from data.replay_buffer import ReplayBuffer


def test_push_batch_and_wraparound():
    buffer = ReplayBuffer(state_dim=2, action_dim=1, max_size=4)
    state = np.arange(6, dtype=np.float32).reshape(3, 2)
    action = np.arange(3, dtype=np.float32).reshape(3, 1)
    reward = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    next_state = state + 1.0
    done = np.array([False, True, False])

    buffer.push_batch(state, action, reward, next_state, done)
    assert buffer.size == 3
    assert buffer.ptr == 3
    np.testing.assert_allclose(buffer.not_done[:3, 0], [1.0, 0.0, 1.0])

    buffer.push_batch(
        np.array([[10.0, 11.0], [12.0, 13.0]], dtype=np.float32),
        np.array([[4.0], [5.0]], dtype=np.float32),
        np.array([4.0, 5.0], dtype=np.float32),
        np.array([[11.0, 12.0], [13.0, 14.0]], dtype=np.float32),
        np.array([False, False]),
    )
    assert buffer.size == 4
    assert buffer.ptr == 1
    np.testing.assert_allclose(buffer.state[3], [10.0, 11.0])
    np.testing.assert_allclose(buffer.state[0], [12.0, 13.0])
