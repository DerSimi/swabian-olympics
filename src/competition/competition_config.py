import argparse

from framework.argument_config import ArgumentConfig


def parse_competition_arguments(agent_args: list[str]) -> ArgumentConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        "-a",
        type=str,
        required=True,
        help="The agent to train",
    )

    args = parser.parse_args(agent_args)

    config = ArgumentConfig(
        name="competition",
        agent=args.agent,
        mode="NORMAL",
        total_steps=0,
        backup_frequency=0,
        parallel_envs=1,
        seed=0,
        eval_games=0,
        opponent=[],
        batching=False,
        kwargs={},
    )

    return config
