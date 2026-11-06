import unittest

import numpy as np

from envs.local_occupancy import FREE, OCCUPIED, UNKNOWN, PersistentLocalOccupancyMap


class PersistentLocalOccupancyMapTest(unittest.TestCase):
    def test_ray_marks_free_interior_occupied_hit_and_unknown_elsewhere(self):
        occupancy = PersistentLocalOccupancyMap(voxel_size=1.0, shape=(6, 6, 4))
        occupancy.update_from_rays(
            position=[0.0, 0.0, 1.0],
            directions=[[1.0, 0.0, 0.0]],
            range_fractions=[0.5],
            sensor_range=4.0,
            origin_offset=0.0,
            vertical_origin_offset=0.0,
        )
        self.assertEqual(occupancy._voxels[(0, 0, 1)], FREE)
        self.assertEqual(occupancy._voxels[(1, 0, 1)], FREE)
        self.assertEqual(occupancy._voxels[(2, 0, 1)], OCCUPIED)
        grid = occupancy.local_grid([0.0, 0.0, 1.0])
        self.assertGreater(np.count_nonzero(grid == UNKNOWN), 0)

    def test_obstacle_persists_when_later_ray_looks_elsewhere(self):
        occupancy = PersistentLocalOccupancyMap(voxel_size=1.0, shape=(8, 8, 4))
        common = dict(
            position=[0.0, 0.0, 1.0],
            sensor_range=4.0,
            origin_offset=0.0,
            vertical_origin_offset=0.0,
        )
        occupancy.update_from_rays(
            directions=[[1.0, 0.0, 0.0]], range_fractions=[0.5], **common
        )
        occupancy.update_from_rays(
            directions=[[0.0, 1.0, 0.0]], range_fractions=[1.0], **common
        )
        self.assertEqual(occupancy._voxels[(2, 0, 1)], OCCUPIED)

    def test_sliding_window_prunes_cells_that_leave_local_extent(self):
        occupancy = PersistentLocalOccupancyMap(voxel_size=1.0, shape=(6, 6, 4))
        occupancy.update_from_rays(
            position=[0.0, 0.0, 1.0],
            directions=[[1.0, 0.0, 0.0]],
            range_fractions=[0.5],
            sensor_range=4.0,
            origin_offset=0.0,
            vertical_origin_offset=0.0,
        )
        occupancy.local_grid([10.0, 0.0, 1.0])
        self.assertEqual(len(occupancy._voxels), 0)


if __name__ == "__main__":
    unittest.main()
