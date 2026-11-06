import unittest
import numpy as np

from trainers.spatial_q_diagnostics import spatial_candidates, calibration_indices, q_summary


class SpatialQTest(unittest.TestCase):
    def test_candidates_keep_actor_and_change_z_independently(self):
        actor = np.array([.6,.8,.7], dtype=np.float32)
        actions, meta = spatial_candidates(actor,[0,0,2],[1,0,1])
        np.testing.assert_array_equal(actions[0],actor)
        vertical = [i for i,m in enumerate(meta) if m['family']=='vertical_only']
        np.testing.assert_allclose(actions[vertical,:2],np.tile(actor[:2],(5,1)))
        np.testing.assert_allclose(actions[vertical,2],[-1,-.5,0,.5,1])
        self.assertTrue(np.all(np.linalg.norm(actions[:,:2],axis=1)<=1+1e-6))
        self.assertTrue(np.all(np.abs(actions[:,2])<=1))
        goal = next(i for i,m in enumerate(meta) if m['name']=='goal_speed0.4_z-1.0')
        np.testing.assert_allclose(actions[goal]*[.8,.8,.3],[.4,0,-.3])

    def test_calibration_includes_distinct_q1_and_qmin_winners(self):
        actions, meta = spatial_candidates([1,0,0],[0,0,2],[1,0,1])
        q1,q2=np.zeros(len(actions)),np.zeros(len(actions))
        q1[-1],q2[-1]=9,-1
        q1[-2],q2[-2]=8,8
        chosen=calibration_indices(q1,q2,meta)
        self.assertIn(len(actions)-1,chosen)
        self.assertIn(len(actions)-2,chosen)
        self.assertEqual(len(chosen),len(set(chosen)))

    def test_q_gaps_are_relative_to_actor_in_same_state(self):
        records=[{'candidates':[{'q1':10.,'q2':8.,'q_min':8.},
                                {'q1':11.,'q2':6.,'q_min':6.}]},
                 {'candidates':[{'q1':-10.,'q2':-8.,'q_min':-10.},
                                {'q1':-12.,'q2':-7.,'q_min':-12.}]}]
        result=q_summary(records)
        self.assertAlmostEqual(result['q1']['mean_best_minus_actor'],.5)
        self.assertAlmostEqual(result['q_min']['mean_best_minus_actor'],0.)
        self.assertEqual(result['q1']['actor_within_0_01_of_best_count'],1)

    def test_calibration_compares_pairs_within_each_state(self):
        from scripts.summarize_spatial_q import analyze
        rows=[]
        for case,values in [('a',[(10,0),(11,1)]),('b',[(-10,0),(-9,-1)])]:
            for name,(q,g) in zip(('actor','actor_xy_z-1.0'),values):
                rows.append(dict(case_id=case,name=name,q1=q,q2=q,q_min=q,
                    actual_return=g,outcome='timeout',category='outside'))
        result=analyze([],rows,[])['calibration']['q1']
        self.assertEqual(result['nontrivial_pair_count'],2)
        self.assertEqual(result['concordant_pairs'],1)
        self.assertEqual(result['discordant_pairs'],1)


if __name__=='__main__':
    unittest.main()
