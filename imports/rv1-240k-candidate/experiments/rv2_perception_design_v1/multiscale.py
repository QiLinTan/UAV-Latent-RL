"""Independent input-level radial strata. No gate/auxiliary Head/planner."""
from typing import Protocol
import torch
from torch import nn
from torch.nn import functional as F
from experiments.rv1_range_cnn.models import AngularConv

BANDS=((.08,1.5),(1.5,4.),(4.,8.))

def radial_images(ranges,hit,scanned,bands=BANDS):
    """B x H x W -> B x scales x 6 x H x W.

    Channels: band-local clipped distance, surface-in-band, scanned,
    visible free fraction, occluded fraction, unknown fraction.
    Free interval begins at the radial ray offset, not the vehicle center.
    A surface only validates its visible prefix. Behind it remains unknown.
    """
    if ranges.ndim!=3 or hit.shape!=ranges.shape or scanned.shape!=ranges.shape:raise ValueError('BHW matching sensor tensors')
    if hit.dtype!=torch.bool or scanned.dtype!=torch.bool or bool((hit&~scanned).any()):raise ValueError('boolean mask semantics')
    if bool((~torch.isfinite(ranges[scanned])).any()) or bool(((ranges[scanned]<.08)|(ranges[scanned]>8)).any()):raise ValueError('scanned range bounds')
    if bool((torch.isfinite(ranges[~scanned])).any()):raise ValueError('unscanned must have NaN range')
    if bool(((~hit&scanned)&(ranges!=8)).any()):raise ValueError('no hit means measured clear to maximum range')
    d=torch.where(scanned,ranges,torch.full_like(ranges,8.));result=[]
    for i,(low,high) in enumerate(bands):
        if high<=low:raise ValueError('ordered nonempty bands')
        local=((d-low)/(high-low)).clamp(0,1)
        surface=hit&scanned&(d>=low)&((d<=high) if i==len(bands)-1 else (d<high))
        free=local*scanned
        occluded=(1-local)*hit*scanned
        unknown=torch.where(scanned,occluded,torch.ones_like(d))
        result.append(torch.stack((local*scanned,surface.to(d.dtype),scanned.to(d.dtype),free,occluded,unknown),dim=1))
    return torch.stack(result,dim=1)

class FutureStateGate(Protocol):
    """Future implementation: normalized Bx3 weights from State18 and Bx3x64.

    Intentionally no implementation, no trainable gate and no rv1 integration.
    """
    def __call__(self,state18:torch.Tensor,scale_latents:torch.Tensor)->torch.Tensor:...

class ScaleEncoder(nn.Module):
    def __init__(self):
        super().__init__();self.convolutions=nn.Sequential(AngularConv(6,16),AngularConv(16,32),AngularConv(32,32))
        self.projection=nn.Linear(32*4*16,64)
    def forward(self,x):
        spatial=self.convolutions(x);return torch.relu(self.projection(spatial.flatten(1))),spatial

class InputMultiscalePrototype(nn.Module):
    """Concat-only research prototype. Bypass preserves full angular resolution."""
    def __init__(self):
        super().__init__();self.encoders=nn.ModuleList([ScaleEncoder() for _ in BANDS])
        self.near_detail=nn.Conv2d(6,8,1);self.concat_projection=nn.Linear(192,64)
    def forward(self,ranges,hit,scanned,state18=None):
        if tuple(ranges.shape[1:])!=(32,128):raise ValueError('32x128 prototype contract')
        if state18 is not None and tuple(state18.shape)!=(ranges.shape[0],18):raise ValueError('future State18 gate input contract')
        images=radial_images(ranges,hit,scanned);latents=[];spatial=[]
        for i,encoder in enumerate(self.encoders):
            z,s=encoder(images[:,i]);latents.append(z);spatial.append(s)
        concatenated=torch.cat(latents,dim=1)
        return dict(latent64=torch.relu(self.concat_projection(concatenated)),concat192=concatenated,
            scale_latents=torch.stack(latents,dim=1),spatial_features=torch.stack(spatial,dim=1),
            near_detail_full_resolution=self.near_detail(images[:,0]),band_images=images,
            fusion='ungated concatenation',gate_weights=None)
