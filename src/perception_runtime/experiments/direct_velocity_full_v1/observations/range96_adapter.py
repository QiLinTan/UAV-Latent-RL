"""Rotate actual ray geometry in yaw; retain original range conversion."""
import numpy as np
from ..contracts import yaw_rotation,yaw_from_quaternion


def yaw_ray_geometry(plant,position):
    directions=plant._range_directions @ yaw_rotation(yaw_from_quaternion(plant.quat[0])).T
    origin=np.asarray(position,dtype=np.float32)+[0,0,.02]
    offset=plant.RANGE_RAY_ORIGIN_OFFSET
    return (origin+directions*offset,origin+directions*plant.OBSTACLE_SENSOR_RANGE,
            offset,plant.OBSTACLE_SENSOR_RANGE-offset)
