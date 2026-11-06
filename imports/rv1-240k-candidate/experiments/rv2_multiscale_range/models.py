"""Parameter-matched full-grid radial experts and optional raw geometry bypass."""
import torch
from torch import nn
from torch.nn import functional as F
from experiments.rv1_range_cnn.models import RangeEncoder
from experiments.rv1_range_cnn.lidar import LidarConfig
from .features import radial_images, NearBypass
from .contracts import PerceptionSpec

class AngularBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride)
    def forward(self, x):
        return F.relu(self.conv(F.pad(F.pad(x, (1, 1, 0, 0), mode='circular'), (0, 0, 1, 1))))

class Expert(nn.Module):
    def __init__(self, out, near=False, cin=5, channels=(12, 20, 30), hidden=None):
        super().__init__()
        self.layers = nn.ModuleList([AngularBlock(cin, channels[0], 1 if near else 2),
                                    AngularBlock(channels[0], channels[1], 2), AngularBlock(channels[1], channels[2], 2)])
        self.pool = nn.AdaptiveAvgPool2d((4, 16))
        size = channels[-1]*4*16
        self.head = nn.Linear(size, out) if hidden is None else nn.Sequential(nn.Linear(size, hidden), nn.ReLU(), nn.Linear(hidden, out))
    def forward(self, x):
        early = self.layers[0](x)
        spatial = self.layers[2](self.layers[1](early))
        latent = F.relu(self.head(self.pool(spatial).flatten(1)))
        return latent, spatial, early

class Encoder(nn.Module):
    def __init__(self, spec=PerceptionSpec()):
        super().__init__()
        self.spec = spec
        if spec.encoder == 'single':
            self.base = RangeEncoder()
        elif spec.encoder == 'multiscale':
            self.experts = nn.ModuleList([Expert(d, near=i == 0) for i, d in enumerate(spec.latent_dims)])
        elif spec.encoder == 'joint_bands':
            self.base = Expert(64, cin=15, channels=(16, 32, 32))
        else:
            self.base = Expert(64, cin=3, channels=(16, 32, 32), hidden=206)

    def forward(self, image, return_spatial=False):
        if self.spec.encoder == 'single':
            z, s = self.base(image, return_spatial=True)
            detail = dict(spatial=[s], early=[], band_images=None)
        elif self.spec.encoder == 'multiscale':
            bands = radial_images(image, self.spec)
            values = [e(bands[:, i]) for i, e in enumerate(self.experts)]
            z = torch.cat([v[0] for v in values], 1)
            detail = dict(spatial=[v[1] for v in values], early=[v[2] for v in values], band_images=bands)
        else:
            bands = radial_images(image, self.spec) if self.spec.encoder == 'joint_bands' else None
            x = bands.flatten(1, 2) if bands is not None else image
            z, s, early = self.base(x)
            detail = dict(spatial=[s], early=[early], band_images=bands)
        return (z, detail) if return_spatial else z

class ObservationFeatures(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.encoder = Encoder(spec)
        self.bypass = NearBypass(LidarConfig().directions(), spec) if spec.bypass else None
        self.input_dim = 82+spec.bypass_dim
    def forward(self, obs):
        s, image = obs['state'], obs['image']
        if s.shape != (image.shape[0], 18) or not bool(torch.isfinite(s).all()):
            raise ValueError('finite State18 batch required')
        pieces = [s, self.encoder(image)]
        if self.bypass is not None:
            pieces.append(self.bypass(image))
        return torch.cat(pieces, 1)

class Actor(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.features = ObservationFeatures(spec)
        self.policy = nn.Sequential(nn.Linear(self.features.input_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 3), nn.Tanh())
    def forward(self, obs):
        return self.policy(self.features(obs))

class Critic(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.features = ObservationFeatures(spec)
        def head():
            return nn.Sequential(nn.Linear(self.features.input_dim+3, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
        self.q1, self.q2 = head(), head()
    def forward(self, obs, action):
        x = torch.cat((self.features(obs), action), 1)
        return self.q1(x), self.q2(x)
    def Q1(self, obs, action):
        return self.q1(torch.cat((self.features(obs), action), 1))
