from abc import ABC, abstractmethod

import torch
import torch.nn as nn


@torch.no_grad()
def soft_update(net, target_net, tau: float):
    """
    theta_target = tau * theta + (1 - tau) * theta_target
    """
    target_params = [p for p in target_net.parameters()]
    source_params = [p for p in net.parameters()]
    torch._foreach_lerp_(target_params, source_params, tau)


class BaseActor(nn.Module, ABC):
    def __init__(self, cfg, input_dim: int, act_dim: int, activation=nn.Mish, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.action_dim = act_dim

    @abstractmethod
    def get_action(self, state_emb, **kwargs):
        """
        Get action from the actor
        !!! NOISE IS ADDED IN THE AGENT !!!
        a = μ(s)
        """
        pass

    @abstractmethod
    def get_target_action(self, state_emb, **kwargs):
        """
        Get target action from the target actor
        a' = clip( μ_target(s') + ε, a_min, a_max )
        where ε ~ clip( N(0, σ), -c, c )
        """
        pass

    @abstractmethod
    def compute_loss(self, critics, state_emb, action, **kwargs):
        """
        Compute loss for the actor
        J(φ) = -E [ Q_θ(s, μ_φ(s)) ]
        """
        pass
