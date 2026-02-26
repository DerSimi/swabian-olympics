from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from agents.fabian.buffer import ReplayBuffer
from agents.fabian.gaussian_policy import GaussianPolicy
from agents.fabian.opponent_pool import DefaultPool
from agents.fabian.q_net import QNet
from framework.agent import AbstractAgent
from framework.argument_config import ArgumentConfig

HYPERPARAMETERS = {
    "policy_learning_rate": 3e-4,
    "qnet1_learning_rate": 3e-4,
    "qnet2_learning_rate": 3e-4,
    "alpha_learning_rate": 3e-3,
    "batch_update_frequency": 64,
    "sample_batch_size": 256,
    "gamma": 0.99,
    "tau": 0.005,
}


class SacSubAgent(AbstractAgent):
    def __init__(
        self, run_name: str, storage_path: str, config: ArgumentConfig, **kwargs: Any
    ):
        super().__init__(run_name, storage_path, config, **kwargs)

        self.device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_name)

        self.gamma = HYPERPARAMETERS["gamma"]
        self.tau = HYPERPARAMETERS["tau"]
        self.target_entropy = -float(self.action_dim)

        # store transitions in a buffer and do batch updates to every X steps
        self.update_buffer: ReplayBuffer = ReplayBuffer(
            self.obs_dim, self.action_dim, 1_000_000, self.device_name
        )
        self.steps_until_update = HYPERPARAMETERS["batch_update_frequency"]

        # gaussian policy (actor)
        self.policy = GaussianPolicy(self.obs_dim, self.action_dim).to(self.device)
        self.policy_opt = optim.Adam(
            self.policy.parameters(), lr=HYPERPARAMETERS["policy_learning_rate"]
        )

        # q networks: critic 1 and 2
        self.q1 = QNet(self.obs_dim, self.action_dim).to(self.device)
        self.q2 = QNet(self.obs_dim, self.action_dim).to(self.device)
        self.q1_opt = optim.Adam(
            self.q1.parameters(), lr=HYPERPARAMETERS["qnet1_learning_rate"]
        )
        self.q2_opt = optim.Adam(
            self.q2.parameters(), lr=HYPERPARAMETERS["qnet2_learning_rate"]
        )

        # target networks (for soft updates)
        self.q1_t = QNet(self.obs_dim, self.action_dim).to(self.device)
        self.q2_t = QNet(self.obs_dim, self.action_dim).to(self.device)
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())

        # alpha is the temperature parameter that scales the entropy term dynamically
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam(
            [self.log_alpha], lr=HYPERPARAMETERS["alpha_learning_rate"]
        )

    def act(
        self,
        obs: np.ndarray,
        inference: bool = False,
        info: dict[str, Any] | None = None,
    ):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if not inference:
                a, _ = self.policy.sample(obs)
            else:
                mu, _ = self.policy(obs)
                a = torch.tanh(mu)
        return a.cpu().numpy()[0]

    def learn(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        info: np.ndarray,
        done: np.ndarray,
    ):
        self.update_buffer.add_batch(obs, action, reward, next_obs, done)
        self.steps_until_update -= 1

        if len(self.update_buffer) < HYPERPARAMETERS["sample_batch_size"]:
            return

        # train the SAC networks, but batch the updates into groups for better performance
        if self.steps_until_update <= 0:
            self.steps_until_update = HYPERPARAMETERS["batch_update_frequency"]

            for _ in range(HYPERPARAMETERS["batch_update_frequency"]):
                transition = self.update_buffer.sample(
                    HYPERPARAMETERS["sample_batch_size"]
                )
                self._train_sac(transition)
            # finally: update q networks slowly
            self._soft_update(self.q1, self.q1_t)
            self._soft_update(self.q2, self.q2_t)

    def _train_sac(self, transition):
        obs, act, reward, next_obs, done = transition

        obs = torch.as_tensor(obs, device=self.device)
        act = torch.as_tensor(act, device=self.device)
        reward = torch.as_tensor(reward, device=self.device)
        next_obs = torch.as_tensor(next_obs, device=self.device)
        done = torch.as_tensor(done, device=self.device)

        # --- train Q networks ---
        with torch.no_grad():
            next_act, logp2 = self.policy.sample(next_obs)
            q1_next = self.q1_t(next_obs, next_act)
            q2_next = self.q2_t(next_obs, next_act)
            q_next = torch.min(q1_next, q2_next)
            target = reward + (1 - done) * self.gamma * q_next

        q1_loss = F.mse_loss(self.q1(obs, act), target)
        q2_loss = F.mse_loss(self.q2(obs, act), target)

        # update q1 network
        self.q1_opt.zero_grad(set_to_none=True)
        q1_loss.backward()
        self.q1_opt.step()
        # update q2 network
        self.q2_opt.zero_grad(set_to_none=True)
        q2_loss.backward()
        self.q2_opt.step()

        # --- train policy network ---
        a, log_p = self.policy.sample(obs)
        q_pi = torch.min(self.q1(obs, a), self.q2(obs, a))
        alpha = self.log_alpha.exp()
        policy_loss = (alpha * log_p - q_pi).mean()
        self.policy_opt.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_opt.step()

        # --- automatic entropy tuning ---
        # We minimize:
        #   L(alpha) = E[ -alpha * (log_prob + target_entropy) ]
        # to increase alpha if entropy < target, and decreases it if > target
        alpha_loss = (-self.log_alpha * (log_p + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

    def _soft_update(self, source, target):
        with torch.no_grad():
            for params, targetParams in zip(source.parameters(), target.parameters()):
                targetParams.mul_(1 - self.tau)
                targetParams.add_(self.tau * params)

    def store_dict(self, agent_descriptor: str) -> dict[str, Any]:
        return {
            "policy": self.policy.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }

    def load_dict(self, state_dict: dict[str, Any], inference: bool = False) -> None:
        self.policy.load_state_dict(state_dict["policy"])
        self.q1.load_state_dict(state_dict["q1"])
        self.q2.load_state_dict(state_dict["q2"])
        self.q1_t.load_state_dict(state_dict["q1"])
        self.q2_t.load_state_dict(state_dict["q2"])
        self.log_alpha = state_dict["log_alpha"].to(self.device).requires_grad_(True)

    def opponent_pool(
        self, opponents: list["AbstractAgent"], eval_env: gym.Env, n_env: int
    ) -> Any:
        return DefaultPool(self, self.argument_config, opponents, eval_env, n_env)
