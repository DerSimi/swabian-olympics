# Reinforcement Learning Tournament WT25/26 University of Tübingen

This repository contains the code for the Laser Hockey Tournament of the University of Tübingen. You can find the report [here](report/report.pdf)
<img width="585" height="825" alt="image" src="https://github.com/user-attachments/assets/97d81433-f0ec-42f2-8974-a9a679e6ddd1" />

# For Reviewers

## Overview

- [src/framework](src/framework) contains the framework shared by all agents
- [src/agents](src/agents) contains the agent implementations by name for all three team members.
- [src/checkpoints](src/checkpoints) contains the trained agents used in the final tournament.
- [src/report](src/report) contains latex files and plotting code.

## Setup

Run

```zsh
./setup.sh
```

and source the python environment. Make sure you have uv installed.

If you are missing `swig` you can install it via `sudo apt install swig` or uncomment the line in `setup.sh` (was
disabled for server e.g container was build with it)

### Checkpoints

For revieweing and for compact hand-in file, we only upload checkpoints used in the final tournament.

### Playing Against Weak and Strong

Quickly look in the `checkpoints` folder and check for the name, then you can play 100 games against weak (or strong,
just insert strong):

```zsh
python src/visualize_gameplay.py -a1 crossplay.niklas@td_universal -a2 weak -r 100 --headless
```

or with out activating the python environment:

```zsh
uv run src/visualize_gameplay.py -a1 crossplay.niklas@td_universal -a2 weak -r 100 --headless
```

Instead of the point (.) in the checkpoints folder, just insert @  `[checkpoint name]@[algorithm name]`.

```zsh
crossplay.niklas.td_universal -> crossplay.niklas@td_universal
```

The `headless` argument indicates no visualizations.

### Playing Against each Other

This is fairly simple, for example:

```zsh
python src/visualize_gameplay.py -a1 crossplay.niklas@td_universal -a2 crossplay.simon@sac -r 100 --headless
```

### Start Training

The agents can be selected by the name given from the registry call above their class (e.g. `@register_agent("sac")`).
Existing checkpoints are loaded automatically for training and can be loaded for opponents via the `@` syntax as
described above.

```zsh
python src/main.py --agent sac --opponent weak strong --total-steps 100000 --num-parallel-envs 8
```

All agents can specify custom arguments, that are not defined in the frameworks command parser. For sac, selfplay
training can be started like this:

```zsh
python src/main.py --agent sac --opponent weak strong --total-steps 100000 --num-parallel-envs 8 --normalize_obs=True --selfplay=True --backup-freq -1
```

#### For the N-PACT agent run:

Batching is used to change how the logging works to total env steps instead of update steps since N-PACT uses internal
gradient updates this way logging can visulize the agents progress more accurately. Running the command below will train
the agent as it was trained for the tournament (excluding opponent/previos run checkpoints/ teammate checkpoints in base
pool)

```zsh
uv run src/main.py --agent td_universal --opponent weak strong --total-steps 100000 --num-parallel-envs 8 --batching
```

> **Note**  
> Before starting training, setup the environment file, see .env.sample as an example. For training, only a wandb config
> is required. Name the resulting file `.env`. It should be placed in the project root.

# Team Swabian Olympics Official Badges:
<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/9d6cd8fd-19b9-498a-8024-1c3b18f6d3d7" />
<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/423a7f70-7ce8-461d-b290-24236940d833" />
<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/4b27c96e-9ea9-45ee-a18a-74ad7674b35a" />


