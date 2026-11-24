import time, unittest
import numpy as np
from multiscale import Encoder, paper53, REGIONS, rv1_directions

class Tests(unittest.TestCase):
    def test_paper(self):
        a=np.zeros((9,9,9),int); self.assertEqual(paper53(a).sum(),0)
        a[:]=1; np.testing.assert_array_equal(paper53(a),np.ones(53))
        a[:]=0; a[4,4,4]=1; self.assertEqual(paper53(a)[:27].sum(),1); self.assertEqual(paper53(a)[27:].sum(),0)
        a[:]=0; a[6:9,3:6,3:6]=1; self.assertEqual(paper53(a)[27+REGIONS.index((2,1,1))],1)
        b=a[::-1,:,:]; self.assertEqual(paper53(b)[27+REGIONS.index((0,1,1))],1)
    def test_occlusion_and_masks(self):
        e=Encoder(); d=np.array([[1.,0,0],[0,1,0],[0,0,1]])
        f,g=e.encode(d,[1.,8.,float('nan')],[1,0,0],[1,1,0]); self.assertEqual(f.shape,(302,))
        self.assertEqual(g[23,20,20],2); self.assertEqual(g[24,20,20],0); self.assertEqual(g[20,23,20],1); self.assertEqual(g[20,20,23],0)
        _,g=e.encode(np.array([[1.,0,0],[1.,0,0]]),[1.,8.],[1,0],[1,1]); self.assertEqual(g[23,20,20],2)
    def test_far_and_mirror(self):
        e=Encoder(); d=np.array([[1.,0,0],[-1.,0,0]])
        f,g=e.encode(d,[2.,2.],[1,0],[1,0]); a=f[250:].reshape(26,2)
        self.assertGreater(a[REGIONS.index((2,1,1)),0],0)
        f2,_=e.encode(d,[2.,2.],[0,1],[0,1]); b=f2[250:].reshape(26,2)
        self.assertAlmostEqual(a[REGIONS.index((2,1,1)),0],b[REGIONS.index((0,1,1)),0])
    def test_bad(self):
        e=Encoder()
        with self.assertRaises(ValueError): e.encode([[1,0,0]],[float('nan')],[0],[1])
        with self.assertRaises(ValueError): e.encode([[1,0,0]],[1],[1],[0])

if __name__=='__main__': unittest.main()
