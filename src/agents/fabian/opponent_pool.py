import random

from framework.opponent_pool import OpponentPool

"""
An opponent pool that simply selects one of the available opponents randomly.
"""


class DefaultPool(OpponentPool):
    def __init__(self, agent, config, opponents, eval_env, n_env):
        super().__init__(agent, config, opponents, eval_env, n_env)

        for i in range(n_env):
            self.selected_opponents[i] = random.choice(self.opponents)

    def add_opponent(self, opponent):
        pass

    def select_agent(self, env_id):
        return random.choice(self.opponents)
