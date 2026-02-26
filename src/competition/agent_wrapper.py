import numpy as np
from comprl.client import Agent

from framework.agent import AbstractAgent
from framework.common import logger as log


class CompetitionAgentWrapper(Agent):
    def __init__(self, wrapped_agent: AbstractAgent):
        super().__init__()
        self.wrapped_agent = wrapped_agent

    def get_step(self, observation: list[float]) -> list[float]:
        act = self.wrapped_agent.act(
            np.expand_dims(np.array(observation), axis=0), inference=True
        )
        return [float(x) for x in act[0]]

    def on_start_game(self, game_id) -> None:
        if isinstance(game_id, bytes):
            try:
                game_id_str = game_id.decode()
            except UnicodeDecodeError:
                game_id_str = game_id.hex()

        log.info(f"Game started: {game_id_str}")
        self.wrapped_agent.reset()

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        self.wrapped_agent.reset()
        text_result = "won" if result else "lost"
        log.info(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )
