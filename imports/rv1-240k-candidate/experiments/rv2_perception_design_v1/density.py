"""Static density protocols and explicitly separate legacy angular geometry."""
from dataclasses import replace,asdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from experiments.rv1_range_cnn.lidar import LidarConfig,Lidar,LAYOUTS
from experiments.rv1_range_cnn.models import RangeEncoder

ROWS=(0,6,12,19,25,31)
COLS=tuple(range(0,128,8))

def common96_mask():
    mask=np.zeros((32,128),bool);mask[np.ix_(ROWS,COLS)]=True
    assert mask.sum()==96
    return mask

class CapacityMatchedCompactEncoder(nn.Module):
    """Encoding ablation only. Resample FEATURES, never invent raw range rays.

    Identical convolution channels and Linear(2048,64) parameter capacity to
    frozen rv1. Native angular receptive fields remain a documented difference.
    """
    def __init__(self,shape,latent_dim=64):
        super().__init__();base=RangeEncoder((32,128),latent_dim)
        self.shape=tuple(shape);self.convolutions=base.convolutions;self.projection=base.projection
    def forward(self,image,return_spatial=False):
        if tuple(image.shape[1:])!=(3,*self.shape):raise ValueError('native compact sensor image shape')
        native=self.convolutions(image)
        spatial=F.interpolate(native,size=(4,16),mode='nearest')
        latent=torch.relu(self.projection(spatial.flatten(1)))
        return (latent,spatial) if return_spatial else latent

def design():
    sensor=LidarConfig();dense=Lidar(sensor);legacy=Lidar(LAYOUTS['96_3x32'])
    # Dot-product test detects the 0-degree elevation row missing from even H32.
    a=legacy.directions_local.reshape(-1,3);b=dense.directions_local.reshape(-1,3)
    nearest=np.arccos(np.clip((a@b.T).max(axis=1),-1,1))*180/np.pi
    params=sum(p.numel() for p in RangeEncoder().parameters())
    return dict(schema='rv1_controlled_density_design_v1',train_budget=240000,seeds=[43],sensor_rate_hz=15,
        sensor=asdict(sensor),common96_rows=list(ROWS),common96_columns=list(COLS),
        common96_elevations_deg=np.linspace(-30,30,32)[list(ROWS)].tolist(),
        legacy96_is_subset=False,legacy_nearest_grid_angular_error_deg=nearest.tolist(),
        primary_density_pair=[dict(id='D96',rays=96,layout='fixed common-grid 6x16 directions, sparse 32x128 image',encoder_params=params),
            dict(id='D4096',rays=4096,layout='full 32x128 grid',encoder_params=params)],
        primary_input_semantics='D96 casts only the static 96 rays; all other pixels are UNSCANNED, not free/no-hit. Same exact CNN architecture, masks and weights initialization.',
        encoding_pair=['D96 sparse full-grid CNN','E96 same exact directions packed into native 6x16, capacity-matched feature resampling head'],
        legacy_reference=dict(id='L96',geometry='3x32, elevations -30/0/+30, 11.25-degree azimuth',
            encoding='native 3x32 compact CNN with capacity-matched FEATURE resampling; floating-point legacy ray geometry parity audited separately',encoder_params=params),
        frequency_pair=dict(status='design only; outside primary density pair',geometry='same common96',encoding='same sparse full-grid CNN',rates_hz=[7.5,15],
            upper_decisions_hz=15,observation='latest completed scan held; timestamps/age logged, no State18 change',
            interpretation='isolates sensor staleness; legacy 120Hz vs rv1 15Hz is not a density-only comparison'),
        frozen_factors=['State18','64D latent / 82D Actor input','action','15/120/240 upper/control/physics','PI','Full dynamics',
            'reward/done','original Forest map splits and curriculum','TD3 hyperparameters / Batch256','240k budget','seed and initialization rule'],
        limitations=['Equal parameter count does not imply equal inductive bias or compute.',
            'Legacy96 vs common96 changes direction layout and cannot establish density causality.',
            'Single seed only validates a pipeline; multi-seed uncertainty remains.',
            'Challenge task training is held until a separately versioned boundary/state/generator contract is frozen.'])
