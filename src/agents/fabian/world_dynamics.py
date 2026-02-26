import torch
from torch import nn

from agents.fabian.mlp import MLP


# (s,a) -> (s',r)
# Learn to predict the future observation and reward given a current observation and action
class WorldDynamicsModel(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = MLP(obs_dim + act_dim, obs_dim + 1)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        out = self.net(x)
        delta_obs = out[:, :-1]
        reward = out[:, -1]
        return obs + delta_obs, reward
