from functools import partial
from typing import Callable

import gymnasium as gym
import hockey.hockey_env as h_env
import numpy as np
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics
from gymnasium.wrappers.vector import (
    RecordEpisodeStatistics as VectorRecordEpisodeStatistics,
)

from framework.argument_config import ArgumentConfig


class FrameworkWrapper(gym.Wrapper):
    """
    Wrapper to inser the opponent's observation into the info dict.
    """

    def __init__(
        self, env, custom_reward: Callable[[np.ndarray, bool, bool, dict], float]
    ):
        super().__init__(env)

        self.max_steps = 250
        self.current_step = 0
        self.custom_reward = custom_reward

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.current_step = 0

        info["opponent_obs"] = self.unwrapped.obs_agent_two()
        info["time_step"] = self.current_step

        if "custom_reward" in info:
            del info["custom_reward"]

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # We need opponent observation
        info["opponent_obs"] = self.unwrapped.obs_agent_two()

        # Time step handling
        self.current_step += 1
        info["time_step"] = self.current_step

        custom_reward = self.custom_reward(reward, obs, terminated, truncated, info)
        if custom_reward is not None:
            reward = custom_reward

        return obs, reward, terminated, truncated, info


def default_reward(*args, **kwargs):
    pass


def make_env(rank, seed, mode, custom_reward):
    env = h_env.HockeyEnv(mode=mode)
    env.seed(seed + rank)
    env = FrameworkWrapper(env, custom_reward)
    return env


def build_env(config: ArgumentConfig):
    train_env = AsyncVectorEnv(
        [
            partial(make_env, i, config.seed, config.mode, default_reward)
            for i in range(config.parallel_envs)
        ]
    )
    # Train environment
    train_env = VectorRecordEpisodeStatistics(train_env, buffer_length=5000)
    # Eval environment
    eval_env = RecordEpisodeStatistics(
        make_env(0, config.seed, "NORMAL", default_reward)
    )

    return train_env, eval_env
