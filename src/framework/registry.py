import importlib
import os
import traceback
from typing import TYPE_CHECKING

import wandb
from framework.argument_config import ArgumentConfig
from framework.common import BASE_PATH, newest_time_step
from framework.common import logger as log

if TYPE_CHECKING:
    from framework.agent import AbstractAgent

# Global Registry
AGENT_REGISTRY = {}


def register_agent(name: str):
    """
    Decorator to register an agent class with a specific name.
    """

    def decorator(cls):
        AGENT_REGISTRY[name.lower()] = cls
        return cls

    return decorator


def discover_agents():
    """
    Automatically discovers agents in the 'agents' directory.
    This works by walking the package structure and importing modules,
    which triggers the @register_agent decorators.
    """
    agents_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "agents")

    if not os.path.exists(agents_root):
        print("DEBUG: agents_root does not exist!")
        return

    # Recursively import
    for root, dirs, files in os.walk(agents_root):
        for file in files:
            if file.endswith(".py") and not file.startswith("__"):
                # Construct module path (e.g., agents.simon.sac)
                rel_path = os.path.relpath(
                    os.path.join(root, file), os.path.dirname(os.path.dirname(__file__))
                )
                module_name = rel_path.replace(os.sep, ".")[:-3]  # remove .py

                try:
                    importlib.import_module(module_name)
                    log.debug(f"Successfully imported: {module_name}")
                except Exception as e:
                    # ignore errors here because some agents might be broken
                    log.error(f"Could not import {module_name}: {e}")
                    pass
    print(list(AGENT_REGISTRY.keys()))


def load_agent(config: ArgumentConfig) -> "AbstractAgent":
    """
    Will load the agent from file if the checkpoint for the given run_name already exists. If not,
    the folder is created, and a new agent initalized.

    gym: The environment expected for training. This is just for reference.
    """
    run_name, agent_name = config.name.lower(), config.agent.lower()

    if agent_name not in AGENT_REGISTRY:
        raise ValueError(
            f"Agent '{agent_name}' not found. Available: {list(AGENT_REGISTRY.keys())}"
        )

    # Path handling
    agent_path = os.path.join(BASE_PATH, f"{run_name}.{agent_name}")
    exists = os.path.exists(agent_path)

    if not (agent_name == "weak" or agent_name == "strong"):
        os.makedirs(agent_path, exist_ok=True)

    config.kwargs["agent_descriptor"] = f"{run_name}@{agent_name}"

    # Init agent
    agent_instance: AbstractAgent = AGENT_REGISTRY[agent_name](
        run_name=config.name,
        storage_path=agent_path,
        config=config,
        **config.kwargs,
    )

    # Backup start parameters (actually use the wandb feature)
    wandb.config.update(config)

    if exists and os.listdir(agent_path):
        time_step = newest_time_step(agent_path)
        agent_path = os.path.join(agent_path, f"agent_{time_step}")

        log.warning(
            f"Run {run_name}@{agent_name} already exists. Loading {run_name}@{agent_name}:{time_step} from file..."
        )

        agent_instance.internal_load(agent_path, inference=False)
        agent_instance.time_step = time_step
        agent_instance.env_step = time_step * config.parallel_envs

    return agent_instance


def load_opponents(agents: list[str], config: ArgumentConfig) -> list["AbstractAgent"]:
    """
    Syntax:
    run_name@agent_name:time_steps where :time_steps is strictly optional

    The only expection: weak and strong agent work just by "weak" and "strong"
    """

    result = []

    for a in agents:
        a = a.lower()

        # Handle Default Agents
        if a == "weak" or a == "strong":
            result.append(
                AGENT_REGISTRY[a](run_name=a, storage_path=None, config=config)
            )
            log.info(f"Loading basic opponent {a}")
            continue

        # @ is required, : not
        if "@" not in a:
            log.error(f"Opponent {a} is not valid.")
            raise ValueError(f"Opponent {a} is not valid.")

        run_name = a.split("@")[0].lower()
        agent_part = a.split("@")[1]

        if ":" in agent_part:
            agent_name = agent_part.split(":")[0].lower()
            time_steps = int(agent_part.split(":")[1])
        else:
            agent_name = agent_part.lower()
            time_steps = None

        if agent_name not in AGENT_REGISTRY:
            raise ValueError(f"Opponent Agent '{agent_name}' not found in registry.")

        agent_path = os.path.join(BASE_PATH, f"{run_name}.{agent_name}")

        if not os.path.exists(agent_path):
            log.error(f"Opponent path {agent_path} does not exist.")
            raise ValueError(f"Opponent path {agent_path} does not exist.")

        if time_steps is None:
            time_steps = newest_time_step(agent_path)

            if time_steps == -1:
                log.error(f"The opponent {a} has an empty folder, nothing to load.")
                raise ValueError(
                    f"The opponent {a} has an empty folder, nothing to load."
                )

        agent_path = os.path.join(agent_path, f"agent_{time_steps}")

        log.info(f"Loading opponent {run_name}@{agent_name}:{time_steps} from file...")

        # here we insert "a" as run name, e. g. test@sac:123 to ensure a good overview later!
        agent_instance: AbstractAgent = AGENT_REGISTRY[agent_name](
            run_name=a, storage_path=agent_path, config=config
        )  # basic_env=env?? env was not defined
        try:
            agent_instance.internal_load(agent_path, inference=True)
        except FileNotFoundError:
            log.error(
                f"Loading opponent {run_name}@{agent_name}:{time_steps} does not exist!"
            )
            traceback.print_exc()
            raise
        result.append(agent_instance)
    return result
