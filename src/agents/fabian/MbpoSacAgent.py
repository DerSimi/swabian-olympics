from typing import Any

import numpy as np
import torch
import torch.optim as optim

from agents.fabian.SacSubAgent import ReplayBuffer, SacSubAgent
from agents.fabian.world_dynamics import WorldDynamicsModel
from framework.argument_config import ArgumentConfig
from framework.registry import register_agent

"""
Model-Based Policy Optimization (MBPO) + Soft-Actor-Critic (SAC)

During training, real environment transitions are used to train the dynamics model. The agent then generates 
short imagined rollouts using this model and updates the policy from these synthetic transitions. This improves 
data efficiency by allowing many policy updates per real environment step. The implementation uses short model 
rollouts to limit model error and is intended as a minimal, educational MBPO-style agent.

Potential risks: As the policy is learning while training, the model is likely to overfit on this specific behaviour 
and is easily thrown off by changing opponent behaviour. Previously undiscovered states (for example by change
of behaviour or harder to achieve states) can throw of the world model, which in turn can have strong negative
impact on the learning process. That is why the imagined transitions are newly generated each step while the real
transitions are reused to weaken buildup of errors.

Learning strategy:
- Train world model and SAC from real data first (pretraining) with a model rollout horizon of zero steps
- Enable updating the SAC Policy with imagined world data one step more into the future
- Randomized opponent pool, could later be improved by selecting with adjusting probability
- Slowly add new opponents (other agents and older checkpoints)
"""

HYPERPARAMETERS = {
    "world_rollout_horizon": 0,  # keep this short (0-2) and increase carefully and slowly
    "world_model_learning_rate": 2e-5,
    "world_model_reward_weight": 0.1,
    "world_sample_batch_size": 256,
}


@register_agent("mbpo_sac")
class MbpoSacAgent(SacSubAgent):
    def __init__(
        self, run_name: str, storage_path: str, config: ArgumentConfig, **kwargs: Any
    ):
        super().__init__(run_name, storage_path, config, **kwargs)

        # prediction-making model
        self.world_model = WorldDynamicsModel(self.obs_dim, self.action_dim).to(
            self.device
        )
        self.world_model_opt = optim.Adam(
            self.world_model.parameters(),
            lr=HYPERPARAMETERS["world_model_learning_rate"],
        )

        # buffer for real transitions
        self.reality_buffer = ReplayBuffer(
            self.obs_dim, self.action_dim, 500_000, self.device_name
        )
        # buffer for imagined transitions
        self.synthetic_buffer = ReplayBuffer(
            self.obs_dim, self.action_dim, 500_000, self.device_name
        )

    def act(
        self,
        obs: np.ndarray,
        inference: bool = False,
        info: dict[str, Any] | None = None,
    ) -> np.ndarray:
        # simply use SAC policy
        return super().act(obs, inference, info)

    def learn(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        info: np.ndarray,
        done: np.ndarray,
    ) -> None:
        # train SAC with real world data
        super().learn(obs, action, reward, next_obs, info, done)
        self.reality_buffer.add_batch(obs, action, reward, next_obs, done)

        if len(self.reality_buffer) < HYPERPARAMETERS["world_sample_batch_size"]:
            return

        self._train_world_model()
        if HYPERPARAMETERS["world_rollout_horizon"] > 0:
            self._train_sac_from_world_model()

    def _train_world_model(self):
        obs, act, rew, next_obs, _ = self.reality_buffer.sample(
            HYPERPARAMETERS["world_sample_batch_size"]
        )

        obs = torch.FloatTensor(obs).to(self.device)
        act = torch.FloatTensor(act).to(self.device)
        rew = torch.FloatTensor(rew).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)

        predicted_next_obs, predicted_rew = self.world_model(obs, act)

        # mean squared losses from observation and reward
        msl_obs = ((predicted_next_obs - next_obs) ** 2).mean()
        msl_rew = ((predicted_rew - rew) ** 2).mean()
        loss = msl_obs + HYPERPARAMETERS["world_model_reward_weight"] * msl_rew

        # use the combined loss to train the world model
        self.world_model_opt.zero_grad()
        loss.backward()
        self.world_model_opt.step()

    def _evaluate_world_model(self, batch_size=256):
        obs, act, rew, next_obs, _ = self.reality_buffer.sample(batch_size)

        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        act = torch.tensor(act, dtype=torch.float32).to(self.device)
        rew = torch.tensor(rew, dtype=torch.float32).to(self.device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            pred_next_obs, pred_reward = self.world_model(obs, act)

        # 1. Normalized root-mean-square error of one-step state prediction
        obs_pred_error = torch.sqrt(((pred_next_obs - next_obs) ** 2).mean())
        state_change = torch.sqrt(((next_obs - obs) ** 2).mean())
        nrmse = (obs_pred_error / (state_change + 1e-8)).item()

        # 2. Sign accuracy of reward prediction
        sign_accuracy = (
            (torch.sign(pred_reward) == torch.sign(rew)).float().mean().item()
        )

        # print(
        #    f"{'World model metrics':<25} "
        #    f"| NRMSE: {nrmse:.3f} "
        #    f"| sign_accuracy: {sign_accuracy:.3f} "
        # )
        return nrmse, sign_accuracy

    def _generate_model_rollouts(self):
        self.synthetic_buffer.clear()

        # take a batch from the real experience
        s, _, _, _, _ = self.reality_buffer.sample(
            HYPERPARAMETERS["world_sample_batch_size"]
        )
        s = torch.tensor(s, dtype=torch.float32).to(self.device)

        # fill the model buffer with the imagined next states
        for _ in range(HYPERPARAMETERS["world_rollout_horizon"]):
            with torch.no_grad():
                a, _ = self.policy.sample(s)
                s2, r = self.world_model(s, a)

            for i in range(len(s)):
                self.synthetic_buffer.add(
                    s[i],
                    a[i],
                    r[i],
                    s2[i],
                    False,
                )
            s = s2

    def _train_sac_from_world_model(self):
        self._generate_model_rollouts()
        obs, act, rew, next_obs, done = self.synthetic_buffer.sample(
            HYPERPARAMETERS["world_sample_batch_size"]
        )

        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        act = torch.tensor(act, dtype=torch.float32).to(self.device)
        rew = torch.tensor(rew, dtype=torch.float32).to(self.device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).to(self.device)

        # reuse SAC training function
        t = (obs, act, rew, next_obs, done)
        self._train_sac(t)

    def store_dict(self, agent_descriptor: str) -> dict[str, Any]:
        sac_dict = super().store_dict(agent_descriptor)
        return {
            **sac_dict,
            "world_model": self.world_model.state_dict(),
        }

    def load_dict(self, state_dict: dict[str, Any], inference: bool = False) -> None:
        super().load_dict(state_dict, inference)
        self.world_model.load_state_dict(state_dict["world_model"])
