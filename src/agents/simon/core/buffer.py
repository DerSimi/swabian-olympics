import numpy as np
import torch
from framework.common import AbstractBuffer


class TorchReplayBuffer(AbstractBuffer):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        dev,
        max_size: int = 1000000,
    ):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.dev = dev

        self.obs = torch.zeros((max_size, obs_dim), dtype=torch.float32).pin_memory()
        self.action = torch.zeros(
            (max_size, action_dim), dtype=torch.float32
        ).pin_memory()
        self.reward = torch.zeros((max_size, 1), dtype=torch.float32).pin_memory()
        self.next_obs = torch.zeros(
            (max_size, obs_dim), dtype=torch.float32
        ).pin_memory()
        self.done = torch.zeros((max_size, 1), dtype=torch.float32).pin_memory()

    def add_batch(self, obs, action, reward, next_obs, done):
        obs = torch.from_numpy(obs).float()
        action = torch.from_numpy(action).float()
        reward = torch.from_numpy(reward).float()
        next_obs = torch.from_numpy(next_obs).float()
        done = torch.from_numpy(done.astype(np.float32)).float()

        n = obs.shape[0]
        idx = torch.arange(self.ptr, self.ptr + n) % self.max_size

        self.obs[idx] = obs
        self.action[idx] = action
        self.reward[idx] = reward
        self.next_obs[idx] = next_obs
        self.done[idx] = done

        self.ptr = (self.ptr + n) % self.max_size
        self.size = min(self.size + n, self.max_size)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,))

        return (
            self.obs[idx].to(self.dev, non_blocking=True),
            self.action[idx].to(self.dev, non_blocking=True),
            self.reward[idx].to(self.dev, non_blocking=True),
            self.next_obs[idx].to(self.dev, non_blocking=True),
            self.done[idx].to(self.dev, non_blocking=True),
        )

    def save(self, path: str):
        state = {
            "obs": self.obs[: self.size],
            "action": self.action[: self.size],
            "reward": self.reward[: self.size],
            "next_obs": self.next_obs[: self.size],
            "done": self.done[: self.size],
            "ptr": self.ptr,
            "size": self.size,
        }
        torch.save(state, path)

    def load(self, path: str):
        state = torch.load(path, map_location=self.dev)
        size = state["size"]

        self.obs[:size] = state["obs"]
        self.action[:size] = state["action"]
        self.reward[:size] = state["reward"]
        self.next_obs[:size] = state["next_obs"]
        self.done[:size] = state["done"]

        self.ptr = state["ptr"]
        self.size = size

    def clear(self):
        self.ptr = 0
        self.size = 0
