import numpy as np
import torch

from framework.common import AbstractBuffer


class ReplayBuffer(AbstractBuffer):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        max_size: int = 1000000,
        device: str = "cpu",
    ):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = device

        self.obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

    def clear(self):
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = done
        # once full, overwrite from the start and keep using the whole buffer
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_batch(self, obs, action, reward, next_obs, done):
        n = obs.shape[0]
        ids = np.arange(self.ptr, self.ptr + n) % self.max_size
        self.obs[ids] = obs
        self.action[ids] = action
        self.reward[ids] = reward
        self.next_obs[ids] = next_obs
        self.done[ids] = done
        # once full, overwrite from the start and keep using the whole buffer
        self.ptr = (self.ptr + n) % self.max_size
        self.size = min(self.size + n, self.max_size)

    def sample(self, batch_size, beta=0.0):
        ind = np.random.randint(0, self.size, size=batch_size)
        return self._to_tensor(ind)

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass

    def _to_tensor(self, ind):
        return (
            torch.FloatTensor(self.obs[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.next_obs[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device),
        )

    def __len__(self):
        return self.size
