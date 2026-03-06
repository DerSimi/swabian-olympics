from dataclasses import dataclass, field
from typing import List

"""
Config exact set up used for the N-PACT Turnament agent
"""


@dataclass
class UniversalConfig:
    # ==========================================================================
    # General Training
    # ==========================================================================
    inference_mode: bool = False
    action_dim: int = 4
    obs_dim: int = 18
    max_env_steps: int = 250

    actor_lr: float = 7e-4
    actor_learning_starts: int = 0

    critic_lr: float = 3e-4
    critic_learning_starts: int = 25_000
    batch: int = 256
    gamma: float = 0.99
    tau: float = 0.005

    # Set to parallel env amount for 1:1 data efficiency ratio
    grad_steps_env: int = 1
    critic_grad_steps: int = 2  # 4 * 2 = 8 (1:1 data efficiency ratio)
    actor_grad_steps: int = 1  # 4 * 1 = 4 (1:2 data efficiency ratio)
    n_step_returns: int = 3

    hard_update: bool = False

    # ==========================================================================
    # ARCHITECTURE
    # ==========================================================================
    hidden_dims: List[int] = field(default_factory=lambda: [512, 512])
    emb_dim: int = 512
    actor_type: str = "mlp"  # 'mlp'

    n_critics: int = 2
    n_quantiles: int = 20
    quantile_exploration: bool = True
    quantile_exploration_cycle_length: int = 400_000
    top_quantiles_to_drop_start: float = 0.0
    top_quantiles_to_drop_end: float = 0.1
    top_quantiles_to_drop: float = 0.1

    quantile_huber_kappa: int = 1
    ensemble_mode: str = "min"  # 'min', 'mean', 'lcb'
    uncertainty_coeff: float = 0.0

    # ==========================================================================
    # NOISE
    # ==========================================================================
    noise_type: str = "pink"
    expl_noise: float = 0.1
    policy_noise: float = 0.2
    noise_clip: float = 0.5

    # ==========================================================================
    # AUXILIARY PHYSICS LOSS
    # ==========================================================================
    aux_loss: bool = True
    aux_loss_coeff: float = 5.0
    aux_coeff_start: float = 5.0
    aux_coeff_end: float = 0.5
    aux_decay_steps: int = 250_000

    # ==========================================================================
    # PLANNING / CEM
    # ==========================================================================

    # --------------------------------------------------------------------------
    # Experimental feature not used in final agent
    # can potntially be a automatic forced interfierence throughout the whole
    # training activates when ever critics disagree (chooses the action most
    # valueable to the critics e.g not nessesarly reward exploration bonus tweeks this)
    # In experiment one can see that a value of around >3.0 will anneal its self automaticly
    # e.g interveen less and less as critics become more similar probapy 3.5 or 4.0 are
    # the sweet spot eperiments only did 1.0, 1.5, 2.0, 2.5, 3.0 without annealing it
    selective_planning: bool = False
    planning_std_threshold: float = 0.5
    planning_std_threshold_start: float = 0.05  # Help often early
    planning_std_threshold_end: float = 1.0  # Help only in emergencies later
    planning_std_decay_steps: int = 500_000
    # --------------------------------------------------------------------------

    use_planning_teacher: bool = True
    planning_intervention_threshold_start: float = 0.5
    planning_intervention_threshold_end: float = 0.0
    planning_intervention_decay_steps: int = 500_000

    cem_samples: int = 32
    cem_iters: int = 3
    cem_elites: int = 10
    exploration_bonus: float = 0.5

    # ==========================================================================
    # OBSERVATION & DATA
    # ==========================================================================
    time_aware: bool = False  # Because of the competition server not providing actuall on game end hooks
    normalize_obs: bool = True
    mirror_data: bool = True
    replay_buffer: str = "prioritized"  # 'uniform', 'prioritized'
    buffer_size: int = 1_000_000
    alpha: float = 0.6
    beta: float = 0.4

    # ==========================================================================
    # SELF-PLAY & SYSTEM
    # ==========================================================================
    wandb_group: str = "default"
    # 110k steps per self-play samples over time all triangle cycle agent veriations
    # with differnt quantile drop percentage which translates in to more or less
    # agressive agents
    self_play: int = 500_000  # -1 for no self-play
    opponent_pool: str = "safe_weighted"  # "safe_weighted", "uniform", "sequential"
    max_pool_size: int = 100
    base_opponent_percent: float = 0.2

    def update(self, new_config):
        for key, value in new_config.items():
            if hasattr(self, key):
                target_type = type(getattr(self, key))
                if isinstance(value, str):
                    if target_type is bool:
                        if value.lower() == "true":
                            value = True
                        elif value.lower() == "false":
                            value = False
                    elif target_type is int:
                        try:
                            value = int(value)
                        except ValueError:
                            pass
                    elif target_type is float:
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                setattr(self, key, value)
            else:
                print(f"Warning: Config has no attribute '{key}', ignoring.")
                pass
