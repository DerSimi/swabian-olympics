import random

import numpy as np

from framework.common import logger as log
from framework.opponent_pool import OpponentPool


class RandomOpponentPool(OpponentPool):
    def __init__(self, agent, config, opponents, eval_env, n_env, **kwargs):
        super().__init__(agent, config, opponents, eval_env, n_env, **kwargs)
        log.info("Random Pool initialized")

    def select_agent(self, env_id):
        return random.choice(self.opponents)

    def add_opponent(self, opponent):
        self.opponents.append(opponent)


class SequentialOpponentPool(OpponentPool):
    def __init__(self, agent, config, opponents, eval_env, n_env, **kwargs):
        super().__init__(agent, config, opponents, eval_env, n_env, **kwargs)
        self.indices = [0] * n_env
        log.info("Sequential Pool initialized")

    def select_agent(self, env_id):
        idx = self.indices[env_id]
        opponent = self.opponents[idx]
        # Advance index cyclically
        self.indices[env_id] = (idx + 1) % len(self.opponents)
        return opponent

    def add_opponent(self, opponent):
        self.opponents.append(opponent)


class SafeWeightedOpponentPool(OpponentPool):
    """
    Prioritized Fictitious Self-Play (PFSP) with Fast Staleness and Sequential Curriculum Shocks
    (Turns out this is way too much e.g decreseing weights on win and spiking them should probaply have been enogth :)
    but why do it the easy way when you can do it the hard way ;))

    Global Staleness Growth (Every episode):
    w_k = w_k * (1 + τ)  for all opponents k

    Outcome Update (For played opponent i):
    w_i = w_i * 0.98  (if win 2% decay)
    w_i = w_i * 1.08  (if draw 8% boost)
    w_i = w_i * 1.10  (if loss 10% boost)

    Curriculum Shocks:
    Every ~50k steps, the next opponent in sequence (including base opponents)
    is spiked to maximum weight (5.0)
    """

    def __init__(self, agent, config, opponents, eval_env, n_env, **kwargs):
        super().__init__(agent, config, opponents, eval_env, n_env, **kwargs)
        log.info("Safe Weighted Opponent Pool initialized")

        self.initial_opponent_count = len(opponents)
        self.weights = np.ones(len(opponents)) * 5.0
        self.agent = agent
        self.seq_indices = [0] * n_env
        self.next_replace_idx = self.initial_opponent_count
        self.selection_counts = {opp.run_name: 0 for opp in opponents}
        self.env_to_opp_idx = [0] * n_env
        self.last_spike_step = self.agent.env_step
        self.spike_cycle_idx = 0

        for i in range(self.n_env):
            self.selected_opponents[i] = self.select_agent(i)

    def register_outcome(self, env_id, win_signal):
        # TODO pool colapses to uniform pool after some time use better stratagie
        # TODO dont overwhelm the agent with opponents
        opp_idx = self.env_to_opp_idx[env_id]
        n_env = 8
        tau = 0.01 / (max(1, len(self.opponents)) * n_env)

        mask = np.ones_like(self.weights, dtype=bool)
        mask[opp_idx] = False
        self.weights[mask] *= 1.0 + tau

        if win_signal == 1:
            self.weights[opp_idx] *= 0.98
        elif win_signal == 0:
            self.weights[opp_idx] *= 1.08
        else:
            self.weights[opp_idx] *= 1.10
        self.weights = np.clip(self.weights, 0.1, 5.0)

        current_step = self.agent.env_step
        if current_step - self.last_spike_step >= 50000:
            self.last_spike_step = current_step

            if self.spike_cycle_idx >= len(self.opponents):
                self.spike_cycle_idx = 0

            spike_idx = self.spike_cycle_idx
            self.weights[spike_idx] = 5.0
            log.info(
                f"Curriculum Shock: Sequentially spiked '{self.opponents[spike_idx].run_name}' to 5.0 weight."
            )
            self.spike_cycle_idx = (self.spike_cycle_idx + 1) % len(self.opponents)

    def add_opponent(self, opponent):
        if len(self.opponents) < self.agent.cfg.max_pool_size:
            self.opponents.append(opponent)
            self.weights = np.append(self.weights, 5.0)
            self.selection_counts[opponent.run_name] = 0
        else:
            idx = self.next_replace_idx
            old_opponent = self.opponents[idx]

            log.info(
                f"Replacing opponent {old_opponent.run_name} with {opponent.run_name}"
            )

            if old_opponent.run_name in self.selection_counts:
                del self.selection_counts[old_opponent.run_name]

            self.opponents[idx] = opponent
            self.weights[idx] = 5.0
            self.selection_counts[opponent.run_name] = 0

            self.next_replace_idx += 1
            if self.next_replace_idx >= self.agent.cfg.max_pool_size:
                self.next_replace_idx = self.initial_opponent_count

    def select_agent(self, env_id):
        # Fixed base opponent selection (as sequential subpool)
        if (
            np.random.rand() < self.agent.cfg.base_opponent_percent
            and self.initial_opponent_count > 0
        ):
            idx = self.seq_indices[env_id]
            self.seq_indices[env_id] = (idx + 1) % self.initial_opponent_count
        else:
            # Weighted opponent selection
            w_sum = self.weights.sum()
            if w_sum <= 0:
                probs = np.ones_like(self.weights) / len(self.weights)
            else:
                probs = self.weights / w_sum

            idx = np.random.choice(len(self.opponents), p=probs)

        opponent = self.opponents[idx]
        self.env_to_opp_idx[env_id] = idx

        self.selection_counts[opponent.run_name] += 1
        return opponent

    def get_selection_stats(self):
        stats = {f"pool_selections/{k}": v for k, v in self.selection_counts.items()}
        for idx, opp in enumerate(self.opponents):
            stats[f"pool_weights/{opp.run_name}"] = min(5.0, float(self.weights[idx]))
        return stats


OPPONENT_POOL_REGISTRY = {
    "random": RandomOpponentPool,
    "sequential": SequentialOpponentPool,
    "safe_weighted": SafeWeightedOpponentPool,
}
