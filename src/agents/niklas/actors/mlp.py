import torch
import torch.nn as nn

from agents.niklas.actors.actor_loss import calc_dpg_loss
from agents.niklas.actors.base import BaseActor


class Actor(BaseActor):
    def __init__(self, cfg, input_dim: int, act_dim: int, activation=nn.Mish, **kwargs):
        super().__init__(cfg, input_dim, act_dim, activation, **kwargs)
        hidden_dims = cfg.hidden_dims
        layers = []

        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(activation())
            input_dim = h
        layers.append(nn.Linear(input_dim, act_dim))
        self.net = nn.Sequential(*layers, nn.Tanh())

    def forward(self, x):
        return self.net(x)

    def get_action(self, state_emb, **kwargs):
        action = self.forward(state_emb)
        return action

    def get_target_action(self, state_emb, **kwargs):
        action = self.forward(state_emb)
        noise = (torch.randn_like(action) * self.cfg.policy_noise).clamp(
            -self.cfg.noise_clip, self.cfg.noise_clip
        )
        return (action + noise).clamp(-1, 1)

    def compute_loss(self, critics, state_emb, action, **kwargs):
        return calc_dpg_loss(actor=self, critics=critics, state_emb=state_emb)
