"""Near 0–4m / Far 4–8m, independent state-conditioned Actor/Critic."""
import torch
from torch import nn
from experiments.rv2_multiscale_range.features import validate_image
from experiments.rv2_multiscale_range.models import Expert


def two_band_images(image):
    """First-return features: hit, local depth, visible free, occluded, scanned.

    The 0–0.08m blind region is excluded from visible-free length; unscanned
    rays remain unknown. The first band is [0, 4), the second is [4, 8].
    """
    distance, hit, scanned = validate_image(image)
    distance = torch.where(scanned, distance.clamp_min(.08), torch.full_like(distance, 8.))
    out = []
    for index, (low, high) in enumerate(((0., 4.), (4., 8.))):
        surface = scanned & hit & (distance >= low) & ((distance <= high) if index else (distance < high))
        local = torch.where(surface, ((distance-low)/(high-low)).clamp(0, 1), torch.zeros_like(distance))
        free = (torch.minimum(distance, torch.full_like(distance, high))-max(low, .08)).clamp_min(0)/(high-low)
        free = free * scanned
        occluded = scanned & hit & (distance < low)
        out.append(torch.stack((surface.to(distance.dtype),local,free,occluded.to(distance.dtype),scanned.to(distance.dtype)),1))
    return torch.stack(out,1)


class Geometry(nn.Module):
    def __init__(self):
        super().__init__()
        self.near=Expert(64,near=True)
        self.far=Expert(64,near=False)
        self.near_norm=nn.LayerNorm(64)
        self.far_norm=nn.LayerNorm(64)
        self.gate=nn.Sequential(nn.Linear(18,32),nn.ReLU(),nn.Linear(32,2))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self,obs,return_details=False):
        state=obs['state']
        if state.ndim!=2 or state.shape[1]!=18 or state.shape[0]!=obs['image'].shape[0] or not bool(torch.isfinite(state).all()):
            raise ValueError('finite State18 batch required')
        bands=two_band_images(obs['image'])
        near=self.near_norm(self.near(bands[:,0])[0])
        far=self.far_norm(self.far(bands[:,1])[0])
        weights=torch.softmax(self.gate(state),dim=1)
        z=torch.cat((weights[:,0:1]*near,weights[:,1:2]*far),dim=1)
        return (z,{'near':near,'far':far,'gate':weights,'bands':bands}) if return_details else z


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=Geometry()
        self.policy=nn.Sequential(nn.Linear(146,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,3),nn.Tanh())
    def forward(self,obs):
        return self.policy(torch.cat((obs['state'],self.encoder(obs)),dim=1))


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=Geometry()
        def head():return nn.Sequential(nn.Linear(149,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,1))
        self.q1=head();self.q2=head()
    def forward(self,obs,action):
        x=torch.cat((obs['state'],self.encoder(obs),action),dim=1)
        return self.q1(x),self.q2(x)
    def Q1(self,obs,action):
        return self.q1(torch.cat((obs['state'],self.encoder(obs),action),dim=1))
