import unittest
import numpy as np
from multiscale import Encoder
from fast_multiscale import FastEncoder

class TestRangeBoundary(unittest.TestCase):
 def test_float32_start_hit(self):
  d=np.array([[1.,0.,0.]])
  r=np.array([.08],np.float32)
  a=Encoder().voxelize(d,r,[True],[True]);b=FastEncoder().voxelize(d,r,[True],[True])
  np.testing.assert_array_equal(a,b);self.assertEqual(int((a==2).sum()),1)
 def test_true_out_of_range_rejected(self):
  for cls in (Encoder,FastEncoder):
   with self.assertRaises(ValueError):cls().voxelize([[1.,0.,0.]],[.079],[True],[True])

if __name__=='__main__':unittest.main()
