import torch
from torch import nn

from agents.fabian.mlp import MLP


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, log_std_bounds=(-20, 2)):
        super().__init__()
        self.mu_net = MLP(obs_dim, act_dim)
        self.log_std_net = MLP(obs_dim, act_dim)
        self.log_std_bounds = log_std_bounds

    def forward(self, obs):
        mu = self.mu_net(obs)
        log_std = self.log_std_net(obs)
        log_std = torch.clamp(log_std, self.log_std_bounds[0], self.log_std_bounds[1])
        std = torch.exp(log_std)
        return mu, std

    def sample(self, obs):
        mu, std = self(obs)
        dist = torch.distributions.Normal(mu, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return y_t, log_prob
