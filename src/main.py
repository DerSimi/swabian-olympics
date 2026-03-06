import os
import sys

from dotenv import load_dotenv

import wandb
from framework.argument_config import parse_arguments
from framework.common import logger as log
from framework.runner import Runner

if __name__ == "__main__":
    log.info(f"Start arguments: {' '.join(sys.argv[1:])}")
    config = parse_arguments()

    if (
        isinstance(config.opponent, list)
        and len(config.opponent) == 1
        and isinstance(config.opponent[0], str)
    ):
        config.opponent = config.opponent[0].split()

    # make sure, all environment variables are set
    load_dotenv()
    required_env_vars = [
        "WANDB_ENTITY",
        "WANDB_PROJECT",
        "WANDB_MODE",
    ]

    for env_var in required_env_vars:
        if env_var not in os.environ:
            raise EnvironmentError(f"Environment variable {env_var} is required")

    if "--name" in sys.argv:
        # We allow in principle continue training for debugging in this case.
        wandb.init(
            entity=os.environ.get("WANDB_ENTITY"),
            project=os.environ.get("WANDB_PROJECT"),
            mode=os.environ.get("WANDB_MODE"),
            name=config.name,
            id=config.name,
            resume="allow",
        )
    else:
        # This is a cluster run, we never allow continue training and use randommized run names!
        wandb.init(
            entity=os.environ.get("WANDB_ENTITY"),
            project=os.environ.get("WANDB_PROJECT"),
            mode=os.environ.get("WANDB_MODE"),
        )

        log.info(f"No name was set, inject wandb run name: {wandb.run.name}")
        config.name = wandb.run.name

    log.info(f'Starting training run "{config.name}" with agent "{config.agent}"...')
    log.info(f"Agent specific arguments: {config.kwargs}")

    try:
        runner = Runner(config=config)
        runner.learn()

        log.info("Training finished, evaluating...")
        runner.eval()
    except KeyboardInterrupt:
        pass
