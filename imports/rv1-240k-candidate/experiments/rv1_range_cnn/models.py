"""Single-scale CNN, angular seam padding, task-trained latent, no auxiliary head."""
import torch
from torch import nn
from torch.nn import functional as F

class AngularConv(nn.Module):
    def __init__(self,cin,cout,circular=True):
        super().__init__();self.circular=circular;self.conv=nn.Conv2d(cin,cout,3,stride=2)
    def forward(self,x):
        x=F.pad(x,(1,1,0,0),mode='circular' if self.circular else 'constant')
        x=F.pad(x,(0,0,1,1),mode='constant',value=0.)
        return torch.relu(self.conv(x))

class RangeEncoder(nn.Module):
    def __init__(self,shape=(32,128),latent_dim=64,circular=True):
        super().__init__()
        if latent_dim not in (32,64,128):raise ValueError('supported capacity ablations: 32/64/128')
        if min(shape)<3:raise ValueError('encoder expects at least three elevation/azimuth samples')
        self.shape=tuple(shape);self.latent_dim=latent_dim
        self.convolutions=nn.Sequential(AngularConv(3,16,circular),AngularConv(16,32,circular),AngularConv(32,32,circular))
        with torch.no_grad():self.spatial_shape=tuple(self.convolutions(torch.zeros(1,3,*shape)).shape[1:])
        self.projection=nn.Linear(int(torch.tensor(self.spatial_shape).prod()),latent_dim)
    def forward(self,image,return_spatial=False):
        if image.ndim!=4 or tuple(image.shape[1:])!=(3,*self.shape):raise ValueError('B x 3 x H x W image required')
        spatial=self.convolutions(image);latent=torch.relu(self.projection(spatial.flatten(1)))
        return (latent,spatial) if return_spatial else latent

class Actor(nn.Module):
    def __init__(self,shape=(32,128),latent_dim=64,circular=True):
        super().__init__();self.encoder=RangeEncoder(shape,latent_dim,circular)
        self.policy=nn.Sequential(nn.Linear(18+latent_dim,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,3),nn.Tanh())
    def forward(self,observation):
        state,image=observation['state'],observation['image']
        if state.ndim!=2 or state.shape[1]!=18 or state.shape[0]!=image.shape[0]:raise ValueError('State18 batch contract')
        return self.policy(torch.cat((state,self.encoder(image)),dim=1))

class Critic(nn.Module):
    def __init__(self,shape=(32,128),latent_dim=64,circular=True):
        super().__init__();self.encoder=RangeEncoder(shape,latent_dim,circular)
        def head():return nn.Sequential(nn.Linear(18+latent_dim+3,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,1))
        self.q1=head();self.q2=head()
    def forward(self,observation,action):
        x=torch.cat((observation['state'],self.encoder(observation['image']),action),dim=1)
        return self.q1(x),self.q2(x)
    def Q1(self,observation,action):
        return self.q1(torch.cat((observation['state'],self.encoder(observation['image']),action),dim=1))
