import torch
from torch import nn

from agents.fabian.mlp import MLP


class QNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = MLP(obs_dim + act_dim, 1)

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))
