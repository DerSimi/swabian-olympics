import torch
import torch.nn as nn

from agents.common.mlp import MLP


class Critic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        depth: int = 1,
    ):
        super().__init__()

        self.net = MLP(
            input_dim=obs_dim + act_dim,
            output_dim=1,
            hidden_dims=depth * [hidden_dim],
            output_activation=nn.ReLU,
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        return self.net(torch.cat([obs, act], dim=-1))


class DoubleCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        depth: int = 1,
    ):
        super().__init__()

        self.critic1 = Critic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=hidden_dim,
            depth=depth,
        )

        # don't copy, want different initalization...
        self.critic2 = Critic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=hidden_dim,
            depth=depth,
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        return self.critic1(obs, act), self.critic2(obs, act)


class QuantileCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        depth: int = 1,
        n_critics: int = 5,
        n_quantiles: int = 25,
    ):
        super().__init__()
        self.n_critics = n_critics
        self.n_quantiles = n_quantiles

        self.nets = nn.ModuleList()

        for _ in range(n_critics):
            self.nets.append(
                MLP(
                    input_dim=obs_dim + act_dim,
                    output_dim=n_quantiles,
                    hidden_dims=depth * [hidden_dim],
                    output_activation=None,
                )
            )

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        inputs = torch.cat([obs, act], dim=-1)
        outputs = [net(inputs) for net in self.nets]

        # Returns (batch size, n_critics, n_quantiles)
        return torch.stack(outputs, dim=1)


# Direct copy from: https://github.com/alxlampe/tqc_pytorch/blob/master/tqc/functions.py
# IN essence this is just a quantile version for the huber loss introduced by
# https://arxiv.org/pdf/1710.10044
def quantile_huber_loss_f(quantiles, samples, dev):
    pairwise_delta = (
        samples[:, None, None, :] - quantiles[:, :, :, None]
    )  # batch x nets x quantiles x samples
    abs_pairwise_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(
        abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta**2 * 0.5
    )

    n_quantiles = quantiles.shape[2]
    tau = (
        torch.arange(n_quantiles, device=dev).float() / n_quantiles
        + 1 / 2 / n_quantiles
    )
    loss = (
        torch.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss
    ).mean()
    return loss
