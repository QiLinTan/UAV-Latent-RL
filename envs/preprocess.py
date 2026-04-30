import numpy as np

def preprocess_state(state_1d: np.ndarray) -> np.ndarray:
    state = np.asarray(state_1d, dtype=np.float32).reshape(-1)

    obs_mean = np.array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=np.float32)
    obs_std = np.array([0.5, 0.5, 0.5, np.pi, np.pi, np.pi, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0], dtype=np.float32)

    kin = state[:12]
    kin_norm = (kin - obs_mean) / obs_std

    if state.shape[0] == 12:
        return kin_norm.astype(np.float32)

    return np.concatenate([kin_norm, state[12:]], axis=0).astype(np.float32)
