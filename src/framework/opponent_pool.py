from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np

from framework.agent import AbstractAgent
from framework.argument_config import ArgumentConfig


class OpponentPool(ABC):
    def __init__(
        self,
        agent: AbstractAgent,
        config: ArgumentConfig,
        opponents: list[AbstractAgent],
        eval_env: gym.Env,
        n_env: int,
        **kwargs,
    ):
        """
        agent: Your agent, the agent to train
        opponents: List of loaded opponents via command line argument
        eval_env: Evaluation environment
        n_env: Number of parallel environments

        """
        self.agent = agent
        self.config = config
        self.opponents = opponents
        self.eval_env = eval_env
        self.n_env = n_env

        # In your own implementation you have to specify the default, start opponents!
        # Override selected_opponents for this purpose.
        self.selected_opponents = [None] * self.n_env
        self.action_dim = 4

    @abstractmethod
    def add_opponent(self, opponent: AbstractAgent):
        """
        Add a new opponent

        opponent: The opponent
        """
        raise NotImplementedError

    @abstractmethod
    def select_agent(self, env_id: int) -> AbstractAgent:
        """
        If a vectorized envrionment ends its episode, the system asks to select a new opponents.

        env_id: The env_id of a single environment
        """
        raise NotImplementedError

    def eval(
        self, opponent: AbstractAgent, n_games: int = 100
    ) -> tuple[float, float, float]:
        """
        Useful method to measure agent performance any time you want.

        opponent: The opponent which plays against self.agent
        n_games: Amount of games
        """

        wins, losses, draws = 0, 0, 0

        for _ in range(n_games):
            self.agent.reset()
            obs, info = self.eval_env.reset()
            done = False

            while not done:
                obs = np.expand_dims(obs, axis=0)
                opponent_obs = np.expand_dims(info["opponent_obs"], axis=0)
                action = self.agent.act(obs, inference=True)

                assert action.ndim == 2, (
                    f"Expected action shape: (1, 4), got shape {action.shape}"
                )
                opponent_action = opponent.act(opponent_obs, inference=True)

                assert opponent_action.ndim == 2, (
                    f"Expected opponent action shape: (1, 4), got shape {action.shape}"
                )

                obs, _, terminated, truncated, info = self.eval_env.step(
                    np.hstack([action, opponent_action])[0]
                )
                done = terminated or truncated

            winner = info["winner"]

            if winner == 1:
                wins += 1
            elif winner == -1:
                losses += 1
            else:
                draws += 1

        return wins / n_games, losses / n_games, draws / n_games

    def load(self, agent_descriptor: str) -> AbstractAgent:
        """
        With this function you can load an agent within your opponent pool from file.

        agent_descriptor: The full agent descriptor, e. g. test@example_agent:123
        """
        from framework.registry import load_opponents

        return load_opponents([agent_descriptor], self.config)

    def internal_act(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        """
        Internal method to create actions.

        DO NOT TOUCH

        obs: (n_env, 18)
        """

        actions = []
        for env_id in range(self.n_env):
            agent = self.selected_opponents[env_id]
            if agent is None:
                raise ValueError(
                    "No opponent assigned. Read the docstring in opponent_pool."
                )

            single_obs = np.expand_dims(obs[env_id], axis=0)  # (1, obs_dim)
            if "info" in kwargs:
                single_info = {"time_step": kwargs["info"]["time_step"][env_id]}
                action = agent.act(
                    single_obs, inference=True, info=single_info
                )  # (1, act_dim)
            else:
                action = agent.act(single_obs, inference=True)  # (1, act_dim)

            assert action.ndim == 2, (
                f"Expected action shape: (n_env, 4), got shape {action.shape}"
            )

            actions.append(action[0])  # (act_dim,)

        return np.stack(actions, axis=0)  # (n_env, act_dim)

    def internal_trigger_selection(self, dones: np.ndarray):
        """
        Internal method to trigger agent selection.

        DO NOT TOUCH

        dones: (n_env, bool)
        """
        for env_id, done in enumerate(dones):
            if done[0]:
                self.selected_opponents[env_id] = self.select_agent(env_id)
