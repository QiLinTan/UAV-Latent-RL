import unittest
import numpy as np
import torch
from ms_td3 import BalancedReplay,GeometricTD3,Ring

class ReplayTests(unittest.TestCase):
    def test_mask_and_success_strata(self):
        r=BalancedReplay(32,8,.25,43);s=np.zeros(320,np.float32);a=np.zeros(3,np.float32)
        r.add(s,a,s,1.,True,False);r.finish(True,10)
        r.add(s,a,s,2.,False,True);r.finish(True,20)
        self.assertEqual((r.success[10].size,r.success[20].size),(1,1))
        self.assertEqual(float(r.success[10].sample(1)[4][0,0]),0.)
        self.assertEqual(float(r.success[20].sample(1)[4][0,0]),0.)
        self.assertEqual(tuple(x.shape for x in r.sample(16)),((16,320),(16,3),(16,320),(16,1),(16,1)))
    def test_ring_and_validation(self):
        r=Ring(2,43);s=np.zeros(320,np.float32);a=np.zeros(3,np.float32)
        for i in range(3):r.add(s,a,s,i,False,False)
        self.assertEqual((r.size,r.pointer,r.total),(2,1,3))
        with self.assertRaises(ValueError):r.add(s,np.ones(3),s,0,False,False)

if __name__=='__main__':unittest.main()
