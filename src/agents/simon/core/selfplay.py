import re
import os
import random

from agents.simon.utils.logger import agent_print
from framework.opponent_pool import OpponentPool


class Selfplay(OpponentPool):
    def __init__(
        self,
        agent,
        config,
        opponents,
        eval_env,
        n_env,
        use_selfplay: bool,
        max_agents: int,
    ):
        """
        Selfplay Strategy for SAC

        use_selfplay: If selfplay is used, if not we fall back to the specificed starting opponent,
        always the FIRST one.
        max_agents: Size of the opponent FIFO queue.
        """

        super().__init__(agent, config, opponents, eval_env, n_env)

        # If selfplay is used
        self.use_selfplay = use_selfplay
        # When selfplay is used and the winning threshold is reached, selfplay switches mode
        self.selfplay_phase = False

        # We always keep weak and strong in the queue, only removing the newest custom agent
        self.queue_size = max_agents

        self.base_pool = []

        # need to add it like this, because = opponent makes references issues
        for opponent in self.opponents:
            self.base_pool.append(opponent)

        self.selfplay_pool = []

        # to identify opponent by env_id
        self.env_map = {}
        # to track win rates
        self.win_rates = {opponent.run_name: 0.5 for opponent in self.base_pool}

        names = [getattr(agent, "run_name", str(agent)) for agent in self.base_pool]
        agent_print("Basic opponents: " + ", ".join(names))

        for i in range(n_env):
            opponent = random.choice(self.base_pool)
            self.selected_opponents[i] = opponent
            self.env_map[i] = opponent.run_name

        if use_selfplay:
            agent_print(f"Selfplay activated. Randomly sampling from base_pool.")
        else:
            agent_print(f"Selfplay deactivated. Using one of the opponents randomly.")

    def add_opponent(self, opponent, win_rate=0.5):
        # This will add the opponent to the runners internal list, allowing for concise results
        # on final evaluation, we do not clear if list gets too large!
        self.opponents.append(opponent)

        self.win_rates[opponent.run_name] = win_rate
        self.selfplay_pool.append(opponent)

        # remove oldest selfplay checkpoint
        if len(self.selfplay_pool) > self.queue_size:
            self.selfplay_pool.pop(0)

    def visualize_pool(self):
        names = [opponent.run_name for opponent in self.selfplay_pool]
        agent_print("Self-Play Pool: " + ", ".join(names))

    def select_agent(self, env_id):
        if not self.use_selfplay:
            opponent = random.choice(self.base_pool)
            self.env_map[env_id] = opponent.run_name
            return opponent

        if not self.selfplay_phase or not self.selfplay_pool:
            target_pool = self.base_pool
        # 40% command line opponents
        elif random.random() < 0.40:
            target_pool = self.base_pool
        else:  # 60% self play opponent
            target_pool = self.selfplay_pool

        # Weighting by tracked win rate
        weights = []
        for opponent in target_pool:
            win_rate = self.win_rates[opponent.run_name]
            weights.append((1.0 - win_rate) ** 2 + 0.1)

        choosen = random.choices(target_pool, weights=weights, k=1)[0]
        self.env_map[env_id] = choosen.run_name

        # agent_print(f"Selected: {choosen.run_name} Env-id: {env_id}")

        return choosen

    def update_win_rate(self, env_id, score):
        name = self.env_map[env_id]

        if score > 0:
            score = 1.0
        elif score < 0:
            score = 0.0
        else:
            score = 0.5

        weight = self.win_rates.get(name, 0.5)
        self.win_rates[name] = 0.9 * weight + 0.1 * score

    def win_rate_base_pool(self, threshold=0.8):
        base_pool_maintained = True

        for opponent in self.base_pool:
            wins, _, _ = self.eval(opponent, n_games=25)
            agent_print(f"Base {opponent.run_name}: {round(wins * 100)}%")

            if wins < threshold:
                base_pool_maintained = False

        return base_pool_maintained

    def attempt_continue_training(self, storage_path):
        # Continue Training
        files = os.listdir(storage_path)

        if not files:
            return False

        state_dict_files = []
        pattern = re.compile(r"agent_(\d+)_state_dict$")
        for f in files:
            match = pattern.match(f)
            if match:
                step = int(match.group(1))
                state_dict_files.append(step)

        state_dict_files.sort()

        if not state_dict_files:
            return False

        agent_descriptor = os.path.basename(storage_path)
        # recreate agent descriptor
        agent_descriptor = agent_descriptor.replace(".", "@")

        agent_print(
            f"Selfplay found previous checkpoints for {agent_descriptor}, attempting to recover, ..."
        )

        for checkpoint in state_dict_files:
            # TODO: you still have to load the previous win rates!!!
            self.add_opponent(self.load(f"{agent_descriptor}:{checkpoint}")[0])

        return True
