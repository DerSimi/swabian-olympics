from abc import ABC, abstractmethod

import numpy as np
import pink.colorednoise as cn
from pink import PinkNoiseProcess


class BaseActionNoise(ABC):
    """Abstract base class for action noise"""

    @abstractmethod
    def __call__(self, shape):
        """Get noise sample matching shape"""
        pass

    @abstractmethod
    def reset(self, dones=None):
        """Reset noise state conditionally based on dones array"""
        pass


def create_noise(action_dim, noise_type, sigma):
    if noise_type == "pink":
        return PinkNoise(action_dim, sigma)
    elif noise_type == "ou":
        return OUNoise(action_dim, sigma)
    else:
        return GaussianNoise(action_dim, sigma)


class PinkNoise(BaseActionNoise):
    """Vectorized Pink Noise"""

    def __init__(self, action_dim, sigma, seq_len=250):
        self.action_dim = action_dim
        self.sigma = sigma
        self.seq_len = seq_len
        self.noise = None

    def __call__(self, shape):
        n_envs = shape[0]
        if self.noise is None or self.noise.size[0] != n_envs:
            self.noise = PinkNoiseProcess(
                size=(n_envs, self.action_dim, self.seq_len), scale=self.sigma
            )
        return self.noise.sample()

    def reset(self, dones=None):
        if self.noise is not None and dones is not None:
            done_indices = np.where(dones)[0]
            if len(done_indices) > 0:
                fresh_buffer = cn.powerlaw_psd_gaussian(
                    exponent=self.noise.beta,
                    size=(len(done_indices), self.action_dim, self.seq_len),
                    fmin=self.noise.minimum_frequency,
                    rng=self.noise.rng,
                )
                self.noise.buffer[done_indices] = fresh_buffer
        elif dones is None or dones.all():
            self.noise = None


class GaussianNoise(BaseActionNoise):
    """Standard Gaussian Noise"""

    def __init__(self, action_dim, sigma):
        self.action_dim = action_dim
        self.sigma = sigma

    def __call__(self, shape):
        return np.random.normal(0, self.sigma, size=shape)

    def reset(self, dones=None):
        pass


class OUNoise(BaseActionNoise):
    """Vectorized Ornstein-Uhlenbeck Noise"""

    def __init__(
        self,
        action_dim,
        sigma,
        theta=0.15,
        dt=1e-2,
    ):
        self.action_dim = action_dim
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x_prev = None

    def __call__(self, shape):
        if self.x_prev is None or self.x_prev.shape != shape:
            self.x_prev = np.zeros(shape)

        noise = (
            self.x_prev
            + self.theta * (0.0 - self.x_prev) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=shape)
        )
        self.x_prev = noise
        return noise

    def reset(self, dones=None):
        if self.x_prev is not None:
            if dones is not None:
                self.x_prev[dones] = 0.0
            else:
                self.x_prev.fill(0.0)
