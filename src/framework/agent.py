import os
from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from hockey.hockey_env import BasicOpponent

from framework.argument_config import ArgumentConfig
from framework.common import AbstractBuffer
from framework.common import logger as log
from framework.registry import register_agent


class AbstractAgent(ABC):
    def __init__(
        self,
        run_name: str,
        storage_path: str,
        config: ArgumentConfig,
        **kwargs: Any,
    ):
        """
        AbstractAgent

        run_name: This is the name you specify with --name when starting the trainer.
        This name is unreliable when this agents is loaded for inference only. Do not rely on it.

        storage_path: Agent location on file system, e. g. runs/test.sac/

        knowledge. The same for inference.

        config: ArgumentConfig with which the program is launched. Inference Agents will receive the same config.
        Only rely on its contents when training.

        kwargs: Additional configuration params that can be passed when starting the program, see argument_config.py
        ONLY available in training. Inference agents are required to LOAD without a user config.
        """

        # Unique name for this run
        self.run_name = run_name

        # Starting configuration
        self.argument_config = config

        # This is ONLY available in training, make sure every exported inference
        # agents works WITHOUT configuration.
        self.config = kwargs

        # You store your data here and no where else. What you store, is up to you.
        self.storage_path = storage_path

        # For fast access
        self.obs_dim = 18
        self.action_dim = 4

        self.total_steps = self.argument_config.total_steps
        self.total_env_steps = (
            self.argument_config.total_steps * self.argument_config.parallel_envs
        )
        self.num_env = self.argument_config.parallel_envs

        # Trainer will increase this time step every step
        # TODO use correct steps in my agent
        self.time_step = 0  # Trainer will override this
        self.env_step = 0  # Trainer will override this

        # Safe buffer, set this to false if you want!
        self.safe_buffer = True

        # OVERRIDE THIS VARIABLE, DON'T CHANGE THE NAME
        self.buffer: AbstractBuffer | None = None

        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reset(self):
        """
        Resets the agent's internal state (e.g. TimeKeeper, RNN states).
        Called at the beginning of a new episode in Eval/Competition.
        """
        pass

    @abstractmethod
    def act(self, obs: np.ndarray, inference: bool = False, **kwargs) -> np.ndarray:
        """
        Returns the action for a given observation.

        !!!!
        In this method, only USE STUFF AVAILABLE IN INFERENCE MODE
        !!!!

        obs: The observation from the environment (n_envs, 18)
        inference: Whether the agent is in inference mode (evaluation)
        kwargs: usfull for training to pass info dicts from the environment

        return: The action to take (n_envs, 4)

        In evaluation: n_envs = 1, but a batch axis is ALWAYS given.
        """
        raise NotImplementedError

    @abstractmethod
    def learn(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        info: np.ndarray,
        done: np.ndarray,
    ):
        """
        This is one train step, called every step in the environment.

        All inputs are of shape (n_envs, dim)

        obs: (n_envs, 18)
        action: (n_envs, 4) <- the action taken by calling act before
        reward: (n_envs,)
        next_obs: (n_envs, 18)
        done: (n_envs,)
        """

        raise NotImplementedError

    @abstractmethod
    def store_dict(self, agent_descriptor: str) -> dict[str, Any]:
        """
        Store your agent, you can handle checkpoint storing, and other things by yourself.

        agent_descriptor: This is the compact agent name if you want to load the checkpoint at some point, e. g.
        test@example_agent:123. This could be useful in your opponent pool.

        For example:
        return {
            "actor": self.actor.state_dict(),
            ...
        }

        For your information:
        Behind the scenes, the system also stores this additional information:
        state_dict = {
            "steps": self.time_step,
            "total_steps": f"{self.total_steps}",
            "config": self.config,
            **state_dict,
        }
        Note the config is in here, in inference mode, this is not available as constructor argument,
        but you can still grab it on load...

        Make sure to use 'self.storage_path', never deviate from it.
        Also ensure that your inference agent can be loaded WITHOUT any specific configuration required for training.
        """
        raise NotImplementedError

    def stored_agent(self, agent_descriptor: str):
        """
        This function is called after your agent with the given descriptor has been written on disk.
        store_dict does not allow for this.

        agent_descriptor: This is the compact agent name if you want to load the checkpoint at some point, e. g.
        test@example_agent:123. This could be useful in your opponent pool.

        Function is optional.
        """
        pass

    @abstractmethod
    def load_dict(self, state_dict: dict[str, Any], inference: bool = False) -> None:
        """
        Load important files, depending on whether in inference mode or not.
        Agents implement this all by themself. Make sure to use 'self.storage_path', never deviate from it.

        state_dict: The dict previously stored by store_dict.
        inference: Whether we are loading for inference or training.
        """
        raise NotImplementedError

    @abstractmethod
    def opponent_pool(
        self,
        opponents: list["AbstractAgent"],
        eval_env: gym.Env,
        n_env: int,
    ) -> Any:
        """
        In here, create YOUR opponent pool which inherits from OpponentPool,
        instantiate and RETURN it!
        """
        pass

    def print_report(self) -> dict[str, Any]:
        """
        For printing debug report in train_loop, implementation not mandatory.
        return: A dict of strings to print and log.
        """
        return {}

    @staticmethod
    def custom_reward(
        original_reward: np.ndarray,
        obs: np.ndarray,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> float:
        """
        This allows you to override the environment reward.

        Note: THIS WILL NOT replace the reward argument in learn! Your custom reward is always under:
        info['reward']

        This function is not mandatory

        original_reward: Original reward
        obs: (18,) <- here without batch axis!
        terminated: whether environment was terminated
        truncated: whether environment was truncated
        info: Dict with information, see environment.py
        """
        pass

    def internal_save(self):
        """
        This stores the agent to the predefined file path.

        DO NOT TOUCH
        """

        if isinstance(self, DefaultAgent):
            return

        if "agent_descriptor" not in self.config:
            raise ValueError(
                "Calling the save method on a inference agent makes no sense."
            )

        agent_descriptor = f"{self.config['agent_descriptor']}:{self.time_step}"

        state_dict = self.store_dict(agent_descriptor)

        state_dict = {
            "steps": self.time_step,
            "total_steps": f"{self.total_steps}",
            "config": self.config,
            **state_dict,
        }

        dict_path = os.path.join(
            self.storage_path, f"agent_{self.time_step}_state_dict"
        )
        torch.save(state_dict, dict_path)

        if self.safe_buffer:
            buffer_path = os.path.join(
                self.storage_path, f"agent_{self.time_step}_buffer"
            )
            if self.buffer is not None:
                self.buffer.save(buffer_path)
        else:
            log.warning(f"Saved agent {agent_descriptor} WITHOUT buffer.")

        self.stored_agent(agent_descriptor)

    def internal_load(self, path: str, inference: bool = False):
        """
        Loads an agent from a given path.

        Avoid using this method directly, this contains all the technical details.

        For usage, consider the much simpler, agent_descriptor compaptible method in opponent_pool,
        which is also available to your agent.

        Expected path format: runs/agent_name/run_name/agent_X

        DO NOT TOUCH
        """

        if isinstance(self, DefaultAgent):
            return

        buffer_path = f"{path}_buffer"
        if self.buffer is not None and os.path.exists(buffer_path):
            print("Trying to load", buffer_path)
            self.buffer.load(buffer_path)

        state_dict_path = f"{path}_state_dict"
        state_dict = torch.load(
            state_dict_path, map_location=self.dev, weights_only=False
        )
        self.load_dict(state_dict, inference=inference)


class DefaultAgent(AbstractAgent):
    """
    This is the default agent serving as baseline.
    """

    def __init__(
        self, run_name: str, storage_path: str, config: ArgumentConfig, **kwargs
    ):
        super().__init__(run_name, storage_path, config, **kwargs)
        self.opponent = BasicOpponent(weak=kwargs.get("weak", False))

    def act(self, obs: np.ndarray, inference: bool = False, **kwargs) -> np.ndarray:
        return np.array([self.opponent.act(o) for o in obs])

    def learn(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        info: np.ndarray,
        done: np.ndarray,
    ) -> None:
        pass

    def store_dict(self, agent_descriptor: str):
        pass

    def load_dict(self, state_dict, inference=False):
        pass

    def print_report(self) -> dict[str, Any]:
        return {}

    def opponent_pool(self, opponents, eval_env, n_env):
        raise ValueError("You can not train the default agent.")


@register_agent("weak")
class WeakDefaultAgent(DefaultAgent):
    def __init__(
        self, run_name: str, storage_path: str, config: ArgumentConfig, **kwargs
    ):
        super().__init__(run_name, storage_path, config, **{**kwargs, "weak": True})


@register_agent("strong")
class StrongDefaultAgent(DefaultAgent):
    def __init__(
        self, run_name: str, storage_path: str, config: ArgumentConfig, **kwargs
    ):
        super().__init__(run_name, storage_path, config, **{**kwargs, "weak": False})
