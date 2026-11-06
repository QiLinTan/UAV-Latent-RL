"""State-independent, trainable Actor/Critic fusion; original NF body unchanged."""
import sys, types
from pathlib import Path
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'nf_gated_main_v2'))
from nf_main.agent import NFRangeTD3
from nf_main.models import two_band_images

class GlobalLogits(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(2))

    def forward(self):
        return self.logits.softmax(0)

def global_geometry(self, obs, return_details=False):
    state = obs['state']
    if state.ndim!=2 or state.shape[1]!=18 or state.shape[0]!=obs['image'].shape[0] or not bool(torch.isfinite(state).all()):
        raise ValueError('finite State18 batch required')
    bands = two_band_images(obs['image'])
    near = self.near_norm(self.near(bands[:,0])[0])
    far = self.far_norm(self.far(bands[:,1])[0])
    # Gate receives no observation/state argument. Only broadcast its two scalars.
    weights = self.gate().unsqueeze(0).expand(len(near), -1)
    z = torch.cat((weights[:,0:1]*near, weights[:,1:2]*far), 1)
    return (z, {'near':near, 'far':far, 'gate':weights, 'bands':bands}) if return_details else z

class StaticLearnableTD3(NFRangeTD3):
    def __init__(self, config, contract, device='cpu'):
        # Construct every original network first, preserving the common RNG order.
        super().__init__(config, contract, device)
        for owner in ('actor', 'critic', 'actor_target', 'critic_target'):
            encoder = getattr(self, owner).encoder
            encoder.gate = GlobalLogits().to(self.device)
            encoder.forward = types.MethodType(global_geometry, encoder)
        # Original optimizers reference the replaced MLP; rebuild using unchanged settings.
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config['actor_lr'])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config['critic_lr'])

    def update(self, replay):
        result = super().update(replay)
        for owner in ('actor','critic'):
            current = owner=='critic' or result['actor_updated']
            for branch in ('near','far'):
                if self.updates>2 and self.updates%100:
                    result[owner+'_'+branch+'_gradient_norm']=None
                    continue
                grads=[p.grad.detach() for p in getattr(getattr(self,owner).encoder,branch).parameters() if p.grad is not None]
                result[owner+'_'+branch+'_gradient_norm'] = float(torch.stack([g.square().sum() for g in grads]).sum().sqrt()) if current and grads else None
        return result
