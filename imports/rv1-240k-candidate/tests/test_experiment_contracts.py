import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from benchmarks.map_contract import map_id, validate_partitions
from benchmarks.forest_benchmark import generate_map, grid_route, save_benchmark, load_split
from high_level_planner.coordinate_contract import future_to_local, future_to_world, rotate_z, local_goal, ranges_to_heading_bins
from analysis.analyze_planner_observability import block_standardize, cross_map_neighbors, build_probe_features
from scripts.freeze_logical_async import compare_json


class ExperimentContractsTest(unittest.TestCase):
    def test_map_identity_excludes_seed_order_and_body_ids(self):
        a={"xy":[1.,2.],"radius":.2,"height":2.,"id":123}
        b={"xy":[-1.,0.],"radius":.1,"height":1.5}
        first=map_id([-3,0,1],[3,0,1],[a,b])
        self.assertEqual(first,map_id([-3,0,1],[3,0,1],[b,{**a,"id":999}]))
        self.assertNotEqual(first,map_id([-3,0,1],[2,0,1],[a,b]))

    def test_duplicates_across_splits_rejected(self):
        record=generate_map("single",30)
        with self.assertRaisesRegex(ValueError,"leakage"):
            validate_partitions({"train":[record],"validation":[dict(record,seed=999)]})

    def test_coordinates_roundtrip_and_rotated_scene_equivariance(self):
        position=np.array([1.,2.,.8])
        future=np.array([[2.,2.3,.9],[3.,3.,1.]])
        velocity=np.array([[.4,.1,0.],[.2,-.3,.1]])
        yaw=.72
        local=future_to_local(future,velocity,position,yaw)
        p,v=future_to_world(local,position,yaw)
        np.testing.assert_allclose(p,future,atol=1e-12)
        np.testing.assert_allclose(v,velocity,atol=1e-12)
        rotation=1.2
        translation=np.array([3.,-4.,.3])
        equivalent=future_to_local(rotate_z(future,rotation)+translation,rotate_z(velocity,rotation),rotate_z(position,rotation)+translation,yaw+rotation)
        np.testing.assert_allclose(equivalent,local,atol=1e-12)
        np.testing.assert_allclose(local_goal([3,2,.8],position,np.pi/2),[0,-2,0],atol=1e-12)

    def test_radar_permutation_and_fractional_conservative_bins(self):
        r=np.arange(8)/8
        np.testing.assert_array_equal(ranges_to_heading_bins(r,np.pi/2),r[[2,3,4,5,6,7,0,1]])
        np.testing.assert_array_equal(ranges_to_heading_bins(r,np.pi/8),np.minimum(r,np.roll(r,-1)))

    def test_benchmark_test_lock_and_hash_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"maps"
            save_benchmark(path,counts=(2,2,2),seed=987)
            self.assertEqual(len(load_split(path,"train")),2)
            with self.assertRaisesRegex(ValueError,"sealed"):
                load_split(path,"test")
            train=path/"train.json"
            train.write_text(train.read_text()+" ")
            with self.assertRaisesRegex(ValueError,"changed"):
                load_split(path,"train")

    def test_grid_rejects_blocked_start(self):
        record=generate_map("clear",9)
        self.assertIsNotNone(grid_route(record))
        record["trees"]=[{"xy":record["start"][:2],"radius":1.,"height":2.}]
        self.assertIsNone(grid_route(record))

    def test_neighbors_exclude_same_map_and_have_distinct_maps(self):
        x=np.array([[0.,0.],[.001,0.],[.2,0.],[.21,0.],[.4,0.]])
        maps=np.array(["a","a","b","b","c"])
        ids,dist=cross_map_neighbors(x,maps,np.array([0]),np.arange(5),k=2)
        self.assertEqual(set(maps[ids[0]]),{"b","c"})
        self.assertTrue(np.isfinite(dist).all())

    def test_scaler_fits_training_only(self):
        x=np.array([[0.,0.],[2.,2.],[100.,100.]])
        scaled=block_standardize(x,np.array([True,True,False]),[1,1])
        np.testing.assert_allclose(scaled[:2].mean(axis=0),[0,0],atol=1e-12)
        self.assertGreater(scaled[2,0],50)

    def test_shuffled_latent_cannot_cross_map_split(self):
        data={"planner_input":np.zeros((6,27),dtype=np.float32),"raw_state":np.zeros((6,263),dtype=np.float32),"split":np.array(["train"]*3+["validation"]*3)}
        data["planner_input"][:3,:16]=np.arange(3)[:,None]
        data["planner_input"][3:,:16]=100+np.arange(3)[:,None]
        features=build_probe_features(data,7)
        self.assertTrue((features["P2_shuffled_latent"][:3,11:27]<100).all())
        self.assertTrue((features["P2_shuffled_latent"][3:,11:27]>=100).all())
        np.testing.assert_array_equal(features["P1_task_latent"][:,:11],features["P0_goal_ranges"][:,:11])

    def test_freeze_comparison_includes_startup_missing_cycles(self):
        self.assertEqual(compare_json({"missing_cycles":3},{"missing_cycles":3}),[])
        self.assertEqual(compare_json({"missing_cycles":3},{"missing_cycles":0})[0]["field"],"/missing_cycles")


if __name__=="__main__":
    unittest.main()
