"""Loss-bounded uint16 range + packed masks, no geometric resampling."""
import numpy as np

class RangeCodec:
    schema='rv1_uint16_range_packed_masks_v1'
    def __init__(self,shape,max_range_m):
        self.shape=tuple(shape);self.size=int(np.prod(shape));self.max_range_m=float(max_range_m)
        if self.size<1 or not np.isfinite(max_range_m) or max_range_m<=0:raise ValueError('invalid range codec')
        self.quantum_m=self.max_range_m/65534.
    def encode(self,range_m,hit_mask,scanned_mask):
        r=np.asarray(range_m);h=np.asarray(hit_mask,bool);s=np.asarray(scanned_mask,bool)
        if any(a.shape!=self.shape for a in (r,h,s)) or np.any(h & ~s):raise ValueError('mask/shape contract')
        if not np.isfinite(r[s]).all() or np.any((r[s]<0)|(r[s]>self.max_range_m+1e-6)):raise ValueError('range contract')
        if np.any(~h & s & (abs(r-self.max_range_m)>1e-5)):raise ValueError('scanned nohit must be max range')
        q=np.full(self.shape,65535,np.uint16);q[s]=np.rint(r[s]/self.quantum_m).astype(np.uint16)
        masks=np.stack((np.packbits(h.ravel(),bitorder='little'),np.packbits(s.ravel(),bitorder='little')))
        return q,masks
    def decode(self,q,masks):
        q=np.asarray(q);m=np.asarray(masks)
        if q.shape!=self.shape or q.dtype!=np.uint16 or m.shape!=(2,(self.size+7)//8) or m.dtype!=np.uint8:raise ValueError('compressed schema')
        h,s=[np.unpackbits(x,bitorder='little',count=self.size).reshape(self.shape).astype(bool) for x in m]
        if np.any(h & ~s) or not np.array_equal(q==65535,~s):raise ValueError('corrupt sentinel/masks')
        r=q.astype(np.float32)*self.quantum_m;r[~s]=np.nan
        return r,h,s
    def policy_image(self,q,masks):
        r,h,s=self.decode(q,masks)
        return np.stack((np.where(s,r/self.max_range_m,0),h,s)).astype(np.float32)
