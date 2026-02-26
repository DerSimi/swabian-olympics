import copy

import torch
import torch.optim as optim

from agents.niklas.actors import MLPActor, StateEncoder
from agents.niklas.configs import UniversalConfig
from agents.niklas.critics import Critic
from agents.niklas.observations import (
    GPUPrioritizedReplayBuffer,
    GPUReplayBuffer,
    ObsProcessor,
    VectorizedNStepCollector,
)
from agents.niklas.planning import Planner
from agents.niklas.utils import (
    CheckpointManager,
    HyperparameterScheduler,
    create_warmup_scheduler,
)
from agents.niklas.utils.noise import create_noise


def setup_config(agent, kwargs):
    if not hasattr(agent, "cfg"):
        agent.cfg = UniversalConfig()

    if "agent_kwargs" in kwargs:
        agent.cfg.update(kwargs["agent_kwargs"])
    agent.cfg.update(kwargs)

    agent.inference_mode = agent.cfg.inference_mode
    agent.time_aware = agent.cfg.time_aware
    agent.mirror_data = agent.cfg.mirror_data
    agent.max_env_steps = agent.cfg.max_env_steps


def setup_dimensions(agent):
    agent.raw_obs_dim = agent.cfg.obs_dim

    agent.obs_dim = agent.raw_obs_dim + (1 if agent.time_aware else 0)

    agent.state_shape = (agent.obs_dim,)

    agent.action_dim = agent.cfg.action_dim


def setup_components(agent):
    agent.processor = ObsProcessor(
        agent.cfg, agent.dev, agent.raw_obs_dim, agent.action_dim
    )
    agent.n_step_collector = VectorizedNStepCollector(
        n_step=agent.cfg.n_step_returns, gamma=agent.cfg.gamma
    )
    agent.hp_scheduler = HyperparameterScheduler(agent.cfg)

    agent.checkpointer = CheckpointManager(agent, agent.run_name, agent.storage_path)

    agent.planner = Planner(agent.cfg, agent.dev, agent.obs_dim, agent.action_dim)

    agent.noise = create_noise(
        agent.action_dim, agent.cfg.noise_type, agent.cfg.expl_noise
    )


def setup_data(agent):
    agent.buffer = None
    if not agent.inference_mode:
        if agent.cfg.replay_buffer == "prioritized":
            agent.buffer = GPUPrioritizedReplayBuffer(
                agent.state_shape,
                agent.action_dim,
                max_size=agent.cfg.buffer_size,
                alpha=agent.cfg.alpha,
                device=agent.dev,
            )
        else:
            agent.buffer = GPUReplayBuffer(
                agent.state_shape,
                agent.action_dim,
                max_size=agent.cfg.buffer_size,
                device=agent.dev,
            )


def setup_stats(agent):
    agent.stats = {
        "loss/critic": torch.tensor(0.0, device=agent.dev),
        "loss/actor": torch.tensor(0.0, device=agent.dev),
        "loss/aux": torch.tensor(0.0, device=agent.dev),
        "grad/critic": 0.0,
        "grad/actor": 0.0,
        "vals/q_mean": torch.tensor(0.0, device=agent.dev),
        "vals/q_std": torch.tensor(0.0, device=agent.dev),
        "vals/target_mean": torch.tensor(0.0, device=agent.dev),
        "vals/batch_reward": torch.tensor(0.0, device=agent.dev),
        "planning/intervention_rate": 0.0,
        "planning/divergence": 0.0,
        "planning/value_gain": 0.0,
        "vals/action_saturation": 0.0,
        "vals/estimation_bias": 0.0,
    }
    agent.learn_calls = 0
    agent.act_calls_train = 0
    agent.planning_executed_count = 0
    agent.grad_step_counter = 0
    agent.actor_update_counter = 0


def build_networks(agent):
    # Encoder
    hidden_dims = agent.cfg.hidden_dims
    emb_dim = agent.cfg.emb_dim

    agent.enc = StateEncoder(
        agent.state_shape,
        emb_dim=emb_dim,
        hidden_dims=hidden_dims,
        use_layernorm=True,
    ).to(agent.dev)
    actual_emb_dim = agent.enc.output_dim

    # Actor
    agent.actor = MLPActor(
        cfg=agent.cfg,
        input_dim=actual_emb_dim,
        act_dim=agent.action_dim,
    ).to(agent.dev)

    # If Inference Mode, Stop Here
    if agent.inference_mode:
        agent.enc.eval()
        agent.actor.eval()
        agent.critics = None
        agent.enc_t = None
        agent.actor_t = None
        agent.critics_t = None
        agent.actor_opt = None
        agent.critic_opt = None
        agent.schedulers = []
        return

    # Critics
    agent.critics = Critic(
        cfg=agent.cfg,
        input_dim=actual_emb_dim,
        action_dim=agent.action_dim,
    ).to(agent.dev)

    # Target Networks
    agent.enc_t = copy.deepcopy(agent.enc)
    agent.actor_t = copy.deepcopy(agent.actor)
    agent.critics_t = copy.deepcopy(agent.critics)

    # Optimizers
    agent.actor_opt = optim.Adam(agent.actor.parameters(), lr=agent.cfg.actor_lr)
    agent.critic_opt = optim.Adam(
        list(agent.critics.parameters()) + list(agent.enc.parameters()),
        lr=agent.cfg.critic_lr,
    )

    # Schedulers
    agent.schedulers = [
        create_warmup_scheduler(opt, agent.cfg.critic_learning_starts)
        for opt in [agent.actor_opt, agent.critic_opt]
    ]
