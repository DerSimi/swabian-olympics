import time
from collections import deque

import numpy as np
import torch
from rich.panel import Panel
from rich.table import Table

import wandb
from framework.agent import AbstractAgent
from framework.argument_config import ArgumentConfig
from framework.common import console
from framework.common import logger as log
from framework.environment import build_env
from framework.opponent_pool import OpponentPool
from framework.registry import discover_agents, load_agent, load_opponents


class Runner:
    def __init__(self, config: ArgumentConfig):
        self.config = config

        # Discover agents
        discover_agents()

        # Build environment
        self.env, self.eval_env = build_env(config)

        # Set seeds for libraries
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Init OWN agent
        self.agent: AbstractAgent = load_agent(config)

        self.env.unwrapped.set_attr("custom_reward", self.agent.custom_reward)

        # Init opponents (Inference mode)
        self.opponents = load_opponents(config.opponent, config)

        # Init opponent pool
        self.pool: OpponentPool = self.agent.opponent_pool(
            self.opponents, self.eval_env, config.parallel_envs
        )

        self.outcome_history = deque(maxlen=100)

        if self.config.backup_frequency == -1:
            log.warning("Runner will not store any backups.")

    def learn(self):
        start_time = time.time()
        last_print_time = start_time
        obs, infos = self.env.reset()

        if self.config.batching:
            n_batches = self.config.total_steps // self.config.parallel_envs
        else:
            n_batches = self.config.total_steps

        try:
            for steps in range(n_batches):
                opponent_obs = infos["opponent_obs"]

                # Inference is here false because the agents needs to explore.
                action = self.agent.act(obs, inference=False, info=infos)

                assert action.ndim == 2, (
                    f"Expected action shape: (n_env, 4), got shape {action.shape}"
                )

                # Vectorized Opponent Action
                opponent_action = self.pool.internal_act(opponent_obs, info=infos)

                next_obs, reward, terminated, truncated, infos = self.env.step(
                    np.hstack([action, opponent_action])
                )

                # unused rendering for local debugging
                # self.env.render()

                # Get noisy winning signal
                if "winner" in infos:
                    for i, w in enumerate(infos["winner"]):
                        if terminated[i]:
                            self.outcome_history.append(w)

                reward = np.expand_dims(reward, axis=1)
                dones = np.expand_dims(np.logical_or(terminated, truncated), axis=1)

                # Notify opponent pool about possible opponent switch
                self.pool.internal_trigger_selection(dones)

                self.agent.learn(obs, action, reward, next_obs, infos, dones)
                self.agent.time_step += 1
                self.agent.env_step += self.config.parallel_envs

                # Next observation
                obs = next_obs

                mean_reward = (
                    np.mean(self.env.return_queue)
                    if len(self.env.return_queue) > 0
                    else np.nan
                )
                mean_length = (
                    np.mean(self.env.length_queue)
                    if len(self.env.length_queue) > 0
                    else np.nan
                )

                if len(self.outcome_history) > 0:
                    wins = sum(1 for x in self.outcome_history if x == 1)
                    losses = sum(1 for x in self.outcome_history if x == -1)
                    draws = sum(1 for x in self.outcome_history if x == 0)
                    total = len(self.outcome_history)

                    win_rate = wins / total
                    loss_rate = losses / total
                    draw_rate = draws / total
                else:
                    win_rate, loss_rate, draw_rate = 0.0, 0.0, 0.0

                report_dict = {
                    "mean_eps_reward": round(mean_reward, 3),
                    "mean_eps_length": round(mean_length, 3),
                    "train_win_rate": round(win_rate, 3),
                    "train_draw_rate": round(draw_rate, 3),
                    "train_loss_rate": round(loss_rate, 3),
                    **self.agent.print_report(),
                }

                if hasattr(self.pool, "get_selection_stats"):
                    report_dict.update(self.pool.get_selection_stats())

                if self.config.batching:
                    wandb.log(report_dict, step=self.agent.env_step)
                else:
                    wandb.log(report_dict)

                if steps % 1250 == 0:
                    now = time.time()
                    updates_per_sec = (
                        1250 / (now - last_print_time) if now > last_print_time else 0.0
                    )
                    last_print_time = now

                    report_dict = {  # this is not logged to wandb
                        "env_steps": self.agent.env_step,
                        "updates": self.agent.time_step,
                        "max_updates": self.config.total_steps,
                        "updates/sec": round(updates_per_sec, 3),
                        **report_dict,
                    }

                    table = Table(show_header=False, header_style="dim", box=None)
                    table.add_column("Key", style="dim")

                    for key, value in report_dict.items():
                        table.add_row(str(key), str(value))

                    console.print(
                        Panel(
                            table,
                            title="Training Stats",
                            border_style="#DC8702",
                            expand=False,
                        )
                    )

                if (
                    self.config.backup_frequency != -1  # Disabled backup
                    and steps % self.config.backup_frequency == 0
                    and steps > 0
                ):
                    self.agent.internal_save()
        except KeyboardInterrupt:
            log.warning(
                f"Received termination signal. Saving agent at time step {self.agent.time_step} and existing..."
            )
        finally:
            self.agent.safe_buffer = True
            self.agent.internal_save()
            log.info(
                f"Training ended after {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}"
            )

        try:
            self.env.close()
            self.eval_env.close()
        except Exception:
            pass

    def eval(self):
        eval_result = []
        for opponent in self.pool.opponents:
            # This already returns rates.
            wins, losses, draws = self.pool.eval(opponent, self.config.eval_games)

            eval_result.append(
                [
                    str(opponent.run_name),
                    float(wins),
                    float(losses),
                    float(draws),
                ]
            )

        eval_table = wandb.Table(
            columns=["Opponent", "Win Rate", "Loss Rate", "Draw Rate"],
            data=eval_result,
        )
        wandb.log({"Evaluation": eval_table}, commit=True)

        table = Table(show_header=True, header_style="dim", box=None)
        table.add_column("Opponent", style="dim")
        table.add_column("Win Rate")
        table.add_column("Loss Rate")
        table.add_column("Draw Rate")

        for row in eval_result:
            table.add_row(
                str(row[0]), f"{row[1]:.3f}", f"{row[2]:.3f}", f"{row[3]:.3f}"
            )

        console.print(
            Panel(
                table, title="Evaluation Results", border_style="#DC8702", expand=False
            )
        )
