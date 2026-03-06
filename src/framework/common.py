import logging
import os
import warnings
from abc import abstractmethod

import coloredlogs
from rich.console import Console

# Base storage path
BASE_PATH = os.path.join(os.getcwd(), "checkpoints")

# Setup global logging
logger = logging.getLogger("rl_framework")
logger.propagate = False
coloredlogs.install(
    level="INFO",
    logger=logger,
    fmt="\033[90m[\033[0m\033[38;5;208mRunner\033[0m\033[90m]\033[0m %(levelname)s %(message)s",
)

# For pretty printouts
console = Console()

# Kill user warnings for gymnasium, on ctrl+c things get messy...
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")


class AbstractBuffer:
    """Interface for all buffers."""

    @abstractmethod
    def add_batch(self, obs, action, reward, next_obs, done):
        raise NotImplementedError

    @abstractmethod
    def sample(self, batch_size, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str):
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str):
        raise NotImplementedError

    @abstractmethod
    def clear(self):
        raise NotImplementedError


def newest_time_step(agent_path):
    """
    Gives the newest (highest checkpoint number) from an agent_path
    """
    max_time_steps = -1
    for file in os.listdir(agent_path):
        if file.startswith("agent_") and "_state_dict" in file:
            ts = int(file.split("_")[1])
            if ts > max_time_steps:
                max_time_steps = ts

    return max_time_steps
