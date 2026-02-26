import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from agents.common.mlp import MLP
from agents.simon.utils.noise import PinkNoise

# Additional clamping of std for numerical stability
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SacPolicy(nn.Module):
    """
    Implementation of SAC Policy
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        depth: list[int] = [1, 0],
    ):
        super().__init__()
        # first depth is for shared MLP, second for mean and std stage
        assert len(depth) == 2

        self.mlp = MLP(
            input_dim=obs_dim,
            output_dim=hidden_dim,
            hidden_dims=depth[0] * [hidden_dim],
            output_activation=nn.ReLU,
        )

        self.mlp_mean = MLP(
            input_dim=hidden_dim,
            output_dim=act_dim,
            hidden_dims=depth[1] * [hidden_dim],
        )

        self.mlp_logstd = MLP(
            input_dim=hidden_dim,
            output_dim=act_dim,
            hidden_dims=depth[1] * [hidden_dim],
        )

        # TODO: maybe remove this on the hockey env not needed.
        self.register_buffer("action_scale", torch.tensor(1.0))
        self.register_buffer("action_bias", torch.tensor(0.0))

        # Agent will override this
        self.pink_noise = None

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        This function is used only within the train loop and nowhere else.
        """
        shared = self.mlp(obs)

        mean = self.mlp_mean(shared)
        log_std = torch.clamp(self.mlp_logstd(shared), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # need this also for action probability
        dist = Normal(mean, std)
        action = dist.rsample()  # uses reparameterization trick

        tanh_action = torch.tanh(action)

        # Calculate log prob,
        # this is more than just dist.log_prob(action) as we use tanh for action clamping,
        # change of variables is required, the naive implementation from the original sac
        # paper might be numerically unstable,
        # this trick comes from OpenAI SpinningUp.
        # https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py
        # if numeric_stable:
        log_prob = dist.log_prob(action).sum(axis=-1)
        log_prob -= (
            2 * (torch.log(torch.tensor(2.0)) - action - F.softplus(-2 * action))
        ).sum(axis=-1)
        # else:
        #     # default implementation, which is the simple change of variables trick,
        #     # see SAC paper Section C. https://arxiv.org/pdf/1801.01290
        #     log_prob = dist.log_prob(action)

        #     log_prob -= torch.log(1 - tanh_action.pow(2) + 1e-6)
        #     log_prob = log_prob.sum(axis=-1)

        rescaled_action = tanh_action * self.action_scale + self.action_bias

        # action to be applied, action log probability
        return rescaled_action, log_prob.unsqueeze(1)

    def forward_exploration(self, obs: torch.Tensor, pink_noise=False) -> torch.Tensor:
        """
        This forward version is only used for environment exploration, not for
        training and not for inference.
        """
        shared = self.mlp(obs)

        mean = self.mlp_mean(shared)
        log_std = torch.clamp(self.mlp_logstd(shared), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # This is only when we sample actions in the main environment loop.
        # Meaning in this case we do not differentiate through the policy
        if pink_noise and self.pink_noise is not None:
            noise_handler: PinkNoise = self.pink_noise
            action, _ = noise_handler(mean, log_std)
            return torch.tanh(action)

        dist = Normal(mean, std)
        action = dist.rsample()

        tanh_action = torch.tanh(action)

        rescaled_action = tanh_action * self.action_scale + self.action_bias
        return rescaled_action

    # This function will be compiled and should be extremly fast...
    def forward_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """
        This function is only used for inference, we aim for speed.
        """
        shared = self.mlp(obs)

        action = self.mlp_mean(shared)
        tanh_action = torch.tanh(action)

        rescaled_action = tanh_action * self.action_scale + self.action_bias

        # action to be applied, action log probability
        return rescaled_action
