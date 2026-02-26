import argparse
from typing import Any


class ArgumentConfig:
    """
    Wrap the configuration into an object for IDE support on parameters
    """

    def __init__(
        self,
        name: str,
        agent: str,
        mode: str,
        total_steps: int,
        backup_frequency: int,
        parallel_envs: int,
        seed: int,
        eval_games: int,
        opponent: list[str],
        batching: bool,
        kwargs: dict[str, Any],
    ):
        """
        Everything is self explainatory, beside:
        kwargs: This is the EXTRA CONFIGURATION YOU CAN PASS AS ARGUMENTS e. g. --learning-rate 123
        for your AGENT
        """
        self.name = name
        self.agent = agent
        self.mode = mode

        self.total_steps = total_steps
        self.backup_frequency = backup_frequency
        self.parallel_envs = parallel_envs
        self.seed = seed
        self.eval_games = eval_games
        self.opponent = opponent
        self.batching = batching

        self.kwargs = kwargs


def parse_kwargs(unknown_args):
    kwargs = {}
    key = None
    for item in unknown_args:
        if item.startswith("--"):
            key = item.lstrip("-")
            kwargs[key] = True
        elif item.startswith("-"):
            key = item.lstrip("-")
            kwargs[key] = True
        elif key:
            kwargs[key] = item
            key = None
    return kwargs


def parse_arguments() -> ArgumentConfig:
    parser = argparse.ArgumentParser()
    # select the active agent (learns)
    parser.add_argument(
        "--agent",
        "-a",
        type=str,
        required=True,
        help="The agent to train",
    )
    # select the opponent (does not learn)
    parser.add_argument(
        "--opponent",
        "-o",
        type=str,
        nargs="+",
        default=["weak"],
        help="The agents to play against, e.g. v1@SAC v2@TD3 v2@TD3:10000 where the last is optional to indicate the time step to load. Multiple opponents are possible.",
    )
    # select the mode, that the environment should reset to
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="NORMAL",
        choices=["NORMAL", "TRAIN_SHOOTING", "TRAIN_DEFENSE"],
    )
    # define how often a new checkpoint is created (use 0 to disable)
    parser.add_argument(
        "--backup-freq",
        "-b",
        type=int,
        default=200000,
        help="How often the save() method is called. Relative to the steps. -1 to disable.",
    )
    # give a name to the run to split checkpoints
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default="default",
        help="Give a name to the run to further separate checkpoints automatically",
    )
    # set the number of environments that should run in parallel
    parser.add_argument(
        "--num-parallel-envs", "-p", type=int, default=8, help="Currently unused"
    )
    # random seed for environments.
    parser.add_argument("--seed", type=int, default=0, help="Seed for everything")
    # total environments steps, we don't have episodes anymore!
    parser.add_argument(
        "--total-steps",
        "-s",
        type=int,
        required=True,
        help="The training duration in steps",
    )
    parser.add_argument(
        "--batching",
        action="store_true",
        help="Whether to use batch training",
    )
    parser.add_argument(
        "--eval-games",
        "-eg",
        type=int,
        default=100,
        help="Amount of evaluation games against all opponents. If you specify 10, 10 * 100 games will be played.",
    )

    args, unknown = parser.parse_known_args()

    config = ArgumentConfig(
        name=args.name,
        agent=args.agent,
        mode=args.mode,
        total_steps=args.total_steps,
        backup_frequency=args.backup_freq,
        parallel_envs=args.num_parallel_envs,
        seed=args.seed,
        eval_games=args.eval_games,
        opponent=args.opponent,
        batching=args.batching,
        kwargs=parse_kwargs(unknown),
    )

    return config
