import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.niklas.critics.critic_loss import (
    calc_huber_loss,
    calc_mse_loss,
    calc_physics_aux,
)
from agents.niklas.critics.critic_target import calc_quantile_target, calc_scalar_target


class EnsembleLinear(nn.Module):
    """
    Batch Critics for speed
    """

    def __init__(
        self, ensemble_size: int, in_features: int, out_features: int, bias: bool = True
    ):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(ensemble_size, in_features, out_features)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        fan_in = self.in_features
        a = math.sqrt(5)
        gain = torch.nn.init.calculate_gain("leaky_relu", a)
        std = gain / math.sqrt(fan_in)
        bound = math.sqrt(3.0) * std

        with torch.no_grad():
            self.weight.uniform_(-bound, bound)
            if self.bias is not None:
                bound_bias = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                self.bias.uniform_(-bound_bias, bound_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            out = torch.einsum("bi,eio->ebo", x, self.weight)

        else:
            out = torch.matmul(x, self.weight)

        if self.bias is not None:
            out = out + self.bias

        return out


class Critic(nn.Module):
    def __init__(self, cfg, input_dim: int, action_dim: int, **kwargs):
        super().__init__()

        self.cfg = cfg
        self.ensemble_size = cfg.n_critics
        self.n_quantiles = cfg.n_quantiles
        self.use_aux = cfg.aux_loss
        self.gamma = cfg.gamma

        # Project action space to 128 dim so it can be clearly seen by the network
        self.act_proj = nn.Linear(action_dim, 128)
        curr_dim = input_dim + 128

        layers = []
        for h_dim in cfg.hidden_dims:
            layers.append(EnsembleLinear(self.ensemble_size, curr_dim, h_dim))
            layers.append(nn.Mish())
            curr_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.q_head = EnsembleLinear(self.ensemble_size, curr_dim, self.n_quantiles)

        if self.use_aux:
            self.aux_head = EnsembleLinear(self.ensemble_size, curr_dim, cfg.obs_dim)

        if self.n_quantiles > 1:
            self.loss_fn = calc_huber_loss
            self.target_fn = calc_quantile_target
        else:
            self.loss_fn = calc_mse_loss
            self.target_fn = calc_scalar_target

    def forward(self, obs_emb, action):
        """
        Forward pass through the critic network
        Q(s, a) = f(s, a)
        """
        act_emb = F.mish(self.act_proj(action))
        x = torch.cat([obs_emb, act_emb], dim=-1)
        features = self.backbone(x)
        q_vals = self.q_head(features)
        aux_vals = self.aux_head(features) if self.use_aux else None

        return q_vals, aux_vals

    def compute_loss(self, q_pred, target_q, weights=None):
        """
        Compute the loss for the critic network
        Scalar (MSE):
        L = 1/N * Σ (Q_θ(s, a) - y)^2

        Quantile (Huber Pinball):
        L = 1/N * Σ |τ - I(u < 0)| * Huber(u)  where  u = y - Q_θ(s, a)
        """
        return self.loss_fn(
            q_pred,
            target_q,
            weights,
            quantile_huber_kappa=1.0,
        )

    def compute_target(self, r, d, next_q, top_quantiles_to_drop=0.0, **kwargs):
        """
        Compute the target for the critic network
        Scalar Target (TD3):
        y = r + γ * (1 - d) * min_i Q_i(s', a')

        Quantile Target (TQC):
        y = r + γ * (1 - d) * sort(Z_i(s', a'))_{1..N_keep}
        """
        return self.target_fn(
            r,
            d,
            next_q,
            kwargs.get("gamma", self.gamma),
            top_quantiles_to_drop=top_quantiles_to_drop,
        )

    def compute_aux_loss(self, aux_pred, s, ns, obs_dim, aux_loss_coeff=1.0):
        """
        Compute the auxiliary loss for the critic network
        L_aux = c * MSE(Δs_pred, s' - s)
        """
        if not self.use_aux or aux_pred is None:
            return torch.tensor(0.0, device=s.device)

        return calc_physics_aux(
            aux_pred,
            s,
            ns,
            obs_dim=obs_dim,
            aux_loss_coeff=aux_loss_coeff,
        )
