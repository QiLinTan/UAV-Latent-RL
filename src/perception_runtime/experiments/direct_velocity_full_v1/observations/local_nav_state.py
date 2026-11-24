"""dv0: 18 state scalars in the frozen Step 0 normalization."""
import numpy as np
from ..contracts import body_to_world, yaw_from_quaternion, yaw_rotation


def local_state(state, goal, accepted_world):
    state=np.asarray(state,dtype=np.float64)
    if state.shape!=(20,) or not np.isfinite(state).all():
        raise ValueError('finite 20D physical state required')
    rot=body_to_world(state[3:7]); rz=yaw_rotation(yaw_from_quaternion(state[3:7]))
    delta=np.asarray(goal)-state[:3];distance=np.linalg.norm(delta)
    return np.concatenate([rz.T@delta/distance if distance>=1e-6 else np.zeros(3),
        [np.clip(distance/20,0,4)],np.clip((rz.T@state[10:13])/[.8,.8,.3],-4,4),
        np.clip((rot.T@state[13:16])/4,-4,4),rot.T@np.array([0,0,-1]),
        (rz.T@np.asarray(accepted_world))/[.8,.8,.3],
        [np.clip((state[2]-.08)/2.67,-1,2),np.clip((2.75-state[2])/2.67,-1,2)]]).astype(np.float32)
