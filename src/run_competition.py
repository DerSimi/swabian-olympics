import os

import wandb
from comprl.client import Agent, launch_client
from dotenv import load_dotenv

from competition.agent_wrapper import CompetitionAgentWrapper
from competition.competition_config import parse_competition_arguments
from framework.registry import discover_agents

# disable wandb
wandb.init(entity="disabled", project="disabled", mode="disabled")

# make sure, all environment variables are set
load_dotenv()
required_env_vars = [
    "COMPRL_SERVER_URL",
    "COMPRL_SERVER_PORT",
    "COMPRL_ACCESS_TOKEN",
]

for env_var in required_env_vars:
    if env_var not in os.environ:
        raise EnvironmentError(f"Environment variable {env_var} is required")

# Discover agents
discover_agents()


def initialize_agent(agent_args: list[str]) -> Agent:
    # parse command line arguments
    config = parse_competition_arguments(agent_args)

    from framework.registry import load_opponents
    return CompetitionAgentWrapper(load_opponents([config.agent], config)[0])


if __name__ == "__main__":
    # launch the client and use the agent for playing
    launch_client(initialize_agent)
