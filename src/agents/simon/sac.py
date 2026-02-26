import copy

import numpy as np
import torch
from torch.optim import Adam

from agents.simon.core.buffer import TorchReplayBuffer
from agents.simon.core.critic import DoubleCritic, QuantileCritic, quantile_huber_loss_f
from agents.simon.core.policy import SacPolicy
from agents.simon.core.selfplay import Selfplay
from agents.simon.utils.logger import agent_print
from agents.simon.utils.noise import PinkNoise
from agents.simon.utils.running_mean import RunningMeanStdTorch
from framework.agent import AbstractAgent
from framework.registry import register_agent

torch.set_float32_matmul_precision("high")

CONFIG = {
    # Learning Rates
    "policy_lr": 3e-4,
    "critic_lr": 3e-4,
    "alpha_lr": 3e-4,
    # Policy
    "policy_hidden_dim": 512,
    "policy_general_depth": 3,
    "policy_mean_depth": 0,
    # Critic
    "critic_hidden_dim": 512,
    "critic_depth": 3,
    # General
    "learning_starts": 5000,  # we start training after 5000 update steps in the environment
    "batch_size": 256,  # batch size from buffer
    "discount": 0.99,
    "target_entropy": -4,  # -action_dim see https://arxiv.org/pdf/1812.05905
    "tau": 0.005,  # exponential movering average for target updates
    # Quantile Critic
    "qc_num_critics": 5,
    "qc_num_quantils": 25,  # default values from paper
    "qc_top_drop": 2,  # oriented myself with the humanoid experiment in the paper
    # Self Play
    "sp_check_start": 100000,  # every 100k update steps we check if threshold reached
    "sp_check_interval": 100000,  # every 100k update steps we check if self play should start
    "sp_phase": False,  # False: selfplay waiting, True: selfplay complete
    "sp_threshold": 0.80,  # selfplay switches into complete mode against strong
    "sp_continue": 0.60,  # need 60% win rate before new agent is added to the pool!
    "sp_max_agents": 200,
    "sp_fine_tuning": 0.95,  # after 95% of total steps we stop storing anything.
    # Modules
    "normalize_obs": True,  # normalize observation
    "pink_noise": False,  # Pink Noise to improve exploration
    "custom_reward": False,  # Pink Noise to improve exploration
    "quantile_critic": False,  # See https://arxiv.org/abs/2005.04269
    "selfplay": False,
}

REPORT = {
    "log_alpha": float("nan"),
    "alpha_loss": float("nan"),
    "critic_loss": float("nan"),
    "policy_loss": float("nan"),
}


@register_agent("sac")
class SACAgent(AbstractAgent):
    shared_selfplay = False

    def __init__(self, run_name, storage_path, config, **kwargs):
        super().__init__(run_name, storage_path, config, **kwargs)

        if SACAgent.shared_selfplay:
            self.dev = torch.device("cpu")

            # not needed here anyways just to be on the safe side
            self.safe_buffer = False
            agent_print(
                f"Self-Play: I'm {run_name}, only a selfplay agent, skipping most of the setup..., Normalize_obs: {CONFIG['normalize_obs']}"
            )

        for key in [
            "pink_noise",
            "custom_reward",
            "quantile_critic",
            "selfplay",
            "normalize_obs",
        ]:
            if f"{key}=True" in kwargs:
                agent_print(f"{key} activated!")
                CONFIG[key] = True

        # Only these two are ALWAYS needed
        self.policy = SacPolicy(
            obs_dim=self.obs_dim,
            act_dim=self.action_dim,
            hidden_dim=CONFIG["policy_hidden_dim"],
            depth=[CONFIG["policy_general_depth"], CONFIG["policy_mean_depth"]],
        ).to(self.dev)

        self.compiled_forward_inference = torch.compile(
            self.policy.forward_inference, mode="reduce-overhead"
        )

        self.policy = torch.compile(self.policy)

        # Observation normalization
        if CONFIG["normalize_obs"]:
            self.obs_rms = RunningMeanStdTorch(shape=(self.obs_dim,), dev=self.dev)

        if not SACAgent.shared_selfplay:
            # Value parsing (network_width & hidden_depth)
            for key in kwargs.keys():
                if key.startswith("network_width="):
                    val = int(key.split("=")[1])
                    CONFIG["policy_hidden_dim"] = val
                    CONFIG["critic_hidden_dim"] = val
                    agent_print(f"Set network_width to {val}")

                if key.startswith("hidden_depth="):
                    val = int(key.split("=")[1])
                    CONFIG["policy_general_depth"] = val
                    CONFIG["critic_depth"] = val
                    agent_print(f"Set hidden_depth to {val}")

            # self.buffer = ReplayBuffer(self.obs_dim, self.action_dim)
            self.buffer = TorchReplayBuffer(self.obs_dim, self.action_dim, self.dev)
            self.pool = None  # is initalized later

            # Init critic
            if CONFIG["quantile_critic"]:
                self.critic = QuantileCritic(
                    obs_dim=self.obs_dim,
                    act_dim=self.action_dim,
                    hidden_dim=CONFIG["critic_hidden_dim"],
                    depth=CONFIG["critic_depth"],
                    n_critics=CONFIG["qc_num_critics"],
                    n_quantiles=CONFIG["qc_num_quantils"],
                ).to(self.dev)

                self.num_quantiles = (
                    CONFIG["qc_num_critics"] * CONFIG["qc_num_quantils"]
                )
            else:
                self.critic = DoubleCritic(
                    obs_dim=self.obs_dim,
                    act_dim=self.action_dim,
                    hidden_dim=CONFIG["critic_hidden_dim"],
                    depth=CONFIG["critic_depth"],
                ).to(self.dev)

            self.target_critic = copy.deepcopy(self.critic)
            self.critic = torch.compile(self.critic)

            self.critic_optimizer = Adam(
                self.critic.parameters(), lr=CONFIG["critic_lr"]
            )

            # Init policy (whats left of it and not needed in inference mode)
            self.policy_optimizer = Adam(
                self.policy.parameters(), lr=CONFIG["policy_lr"]
            )

            # Adapt alpha automatically
            self.log_alpha = torch.zeros(1, device=self.dev, requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha], lr=CONFIG["alpha_lr"]
            )

            # Pink noise
            if CONFIG["pink_noise"]:
                self.pink_noise = PinkNoise(self.num_env, self.action_dim)
                self.policy.pink_noise = self.pink_noise

            if CONFIG["selfplay"]:
                SACAgent.shared_selfplay = (
                    True  # to notify LATER loaded agents to make loading more efficient
                )
                if config.backup_frequency != -1:
                    raise ValueError(
                        "You fucked up, selfplay does not allow for other backups."
                    )

                self.safe_buffer = False
                agent_print("Disabled buffer storing for selfplay.")

            # This effectively doubles the training speed.
            self.update_compiled = torch.compile(self.update, mode="reduce-overhead")

            agent_print(f"Using device: {self.dev}")

    def act(self, obs, inference=False, **kwargs):
        obs = torch.from_numpy(obs).float().to(self.dev).contiguous()

        if CONFIG["normalize_obs"]:
            obs = (obs - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + 1e-8)
            obs = obs.contiguous()

        if inference:
            # Call inference forward
            with torch.no_grad():
                return self.compiled_forward_inference(obs).cpu().numpy()

        # this will improve exploration in the warmup phase...
        if (
            not inference  # <- call from environment exploration
            and self.time_step < CONFIG["learning_starts"]
        ):
            return np.random.uniform(-1, 1, size=(obs.shape[0], self.action_dim))

        with torch.no_grad():
            return (
                self.policy.forward_exploration(obs, pink_noise=not inference)
                .cpu()
                .numpy()
            )

    def learn(self, obs, action, reward, next_obs, info, done):
        # Reset right env process for pink noise
        if CONFIG["pink_noise"]:
            env_ids = np.where(done.flatten())[0]
            for env_id in env_ids:
                self.pink_noise.reset(env_id)

        self.buffer.add_batch(
            obs,
            action,
            reward,
            next_obs,
            done,
        )  # Buffer lives now on the gpu

        if CONFIG["normalize_obs"]:
            obs_gpu = torch.from_numpy(obs).to(self.dev).float()
            self.obs_rms.update(obs_gpu)

        if self.time_step < CONFIG["learning_starts"]:
            return

        if self.time_step == 5000:
            agent_print("Finished warmup-phase, starting training...")

        obs, action, reward, next_obs, done = self.buffer.sample(CONFIG["batch_size"])

        # Normalize Observation
        if CONFIG["normalize_obs"]:
            obs = (obs - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + 1e-8)
            next_obs = (next_obs - self.obs_rms.mean) / torch.sqrt(
                self.obs_rms.var + 1e-8
            )

        critic_loss, policy_loss, log_alpha, alpha_loss = self.update_compiled(
            obs, action, reward, next_obs, done
        )

        REPORT["critic_loss"] = critic_loss.item()
        REPORT["policy_loss"] = policy_loss.item()
        REPORT["log_alpha"] = log_alpha.item()
        REPORT["alpha_loss"] = alpha_loss.item()

        # 4. Update target networks
        with torch.no_grad():
            for param, target_param in zip(
                self.critic.parameters(), self.target_critic.parameters()
            ):
                target_param.data.lerp_(param.data, CONFIG["tau"])

        # Self Play
        if CONFIG["selfplay"]:
            self.selfplay(info, done)

    def update(self, obs, action, reward, next_obs, done):
        # Calculate alpha once...
        alpha = torch.exp(self.log_alpha).detach()

        # 1. Compute targets for Q functions
        with torch.no_grad():  # DON'T PASS GRADIENT THORUGH target_q!!!
            # Run WITHOUT inference, we need policy smoothing...
            next_actions, next_log_probs = self.policy(next_obs)

            target = self.target_critic(next_obs, next_actions)

            if CONFIG["quantile_critic"]:
                k = self.num_quantiles - CONFIG["qc_top_drop"]
                sorted_target, _ = torch.topk(
                    target.reshape(CONFIG["batch_size"], -1), k, dim=1, largest=False
                )

                # compute target
                target_q = reward + CONFIG["discount"] * (1 - done) * (
                    sorted_target - alpha * next_log_probs
                )
            else:
                target_q1, target_q2 = target
                min_target_q = torch.min(target_q1, target_q2) - alpha * next_log_probs
                target_q = reward + CONFIG["discount"] * (1 - done) * min_target_q

        # 2. Update critic
        if CONFIG["quantile_critic"]:
            q = self.critic(obs, action)
            critic_loss = quantile_huber_loss_f(q, target_q, self.dev)
        else:
            q1, q2 = self.critic(obs, action)
            critic_loss = torch.nn.functional.mse_loss(
                q1, target_q
            ) + torch.nn.functional.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Calculate alpha once...
        alpha = torch.exp(self.log_alpha).detach()

        # 3. Update policy
        policy_action, policy_log_prob = self.policy(obs)
        if CONFIG["quantile_critic"]:
            q = self.critic(obs, policy_action)
            policy_loss = (
                alpha * policy_log_prob - q.mean(2).mean(1, keepdim=True)
            ).mean()
        else:
            q1, q2 = self.critic(obs, policy_action)
            min_q = torch.min(q1, q2)

            policy_loss = -(min_q - alpha * policy_log_prob).mean()  # we want accent

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # 4. Update alpha
        # SAC should lower its entropy over time see https://arxiv.org/pdf/1812.05905
        with torch.no_grad():
            _, policy_log_prob = self.policy(obs)

        alpha_loss = (
            -torch.exp(self.log_alpha)
            * (policy_log_prob.detach() + CONFIG["target_entropy"])
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return (
            critic_loss,
            policy_loss,
            self.log_alpha,
            alpha_loss,
        )

    def selfplay(self, info, done):
        # insert noisy winning signal
        for idx, win in enumerate(info["winner"]):
            if torch.isclose(done[idx], torch.tensor(1.0, device=done.device)):
                self.pool.update_win_rate(idx, win)

        if not CONFIG["sp_phase"] and self.time_step % CONFIG["sp_check_start"] == 0:
            eval = self.pool.win_rate_base_pool(CONFIG["sp_threshold"])

            if eval:
                CONFIG["sp_phase"] = True
                agent_print("Reached win threshold, activating selfplay.")

                self.internal_save()

                # Enable self play phase.
                self.pool.selfplay_phase = True
                return
            else:
                agent_print(
                    "Threshold not reached, continue training against base pool."
                )

        if CONFIG["sp_phase"] and self.time_step % CONFIG["sp_check_interval"] == 0:
            agent_print("Evaluation:")

            latest = self.pool.selfplay_pool[-1]
            wins, _, _ = self.pool.eval(latest, n_games=25)
            agent_print(f"Latest {latest.run_name}: {round(wins * 100)}%")

            eval = self.pool.win_rate_base_pool(CONFIG["sp_threshold"])

            if (
                CONFIG["sp_fine_tuning"] * self.argument_config.total_steps
                <= self.time_step
            ):
                agent_print(
                    "Selfplay is in fine tuning mode, will not add any more agents to pool."
                )
                return

            if (
                wins >= CONFIG["sp_continue"]  # need to beat the latest agent
                and eval  # never forget base pool
            ):
                agent_print(f"Defeated {latest.run_name}, adding to pool...")
                # new check point
                self.internal_save()
            else:
                agent_print(
                    f"Threshold against {latest.run_name} (and base pool) not reached."
                )

    def store_dict(self, agent_descriptor):
        state = {
            "policy": self.policy.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu().numpy(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
        }

        if CONFIG["normalize_obs"]:
            state["obs_rms"] = {
                "mean": self.obs_rms.mean.cpu(),
                "var": self.obs_rms.var.cpu(),
                "count": self.obs_rms.count,
            }

        return state

    def stored_agent(self, agent_descriptor):
        # we instantly add every stored opponent to the pool!
        # Note: This is only triggered in selfplay()
        agent = self.pool.load(agent_descriptor)
        self.pool.add_opponent(agent[0])

        self.pool.visualize_pool()

    def load_dict(self, state_dict, inference=False):
        self.policy.load_state_dict(state_dict["policy"])

        if not inference:
            self.critic.load_state_dict(state_dict["critic"])
            # be careful on load, ...
            self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
            self.critic_optimizer.param_groups[0]["lr"] = CONFIG["critic_lr"]

            self.policy_optimizer.load_state_dict(state_dict["policy_optimizer"])
            self.policy_optimizer.param_groups[0]["lr"] = CONFIG["policy_lr"]

            self.log_alpha.data.copy_(
                torch.tensor(state_dict["log_alpha"], device=self.dev)
            )
            self.alpha_optimizer.load_state_dict(state_dict["alpha_optimizer"])
            self.alpha_optimizer.param_groups[0]["lr"] = CONFIG["alpha_lr"]

        # we need to load this, ..., always!!!
        if CONFIG["normalize_obs"]:
            self.obs_rms.mean = state_dict["obs_rms"]["mean"].to(self.dev)
            self.obs_rms.var = state_dict["obs_rms"]["var"].to(self.dev)
            self.obs_rms.count = state_dict["obs_rms"]["count"]

    def opponent_pool(self, opponents, eval_env, n_env):
        self.pool = Selfplay(
            self,
            self.argument_config,
            opponents,
            eval_env,
            n_env,
            use_selfplay=CONFIG["selfplay"],
            max_agents=CONFIG["sp_max_agents"],
        )

        if self.pool.attempt_continue_training(self.storage_path):
            CONFIG["sp_phase"] = True

            agent_print("Recovered Selfplay checkpoints. Skipping warmup phase...")

        return self.pool

    def print_report(self):
        return {
            "log_alpha": round(REPORT["log_alpha"], 3),
            "alpha_loss": round(REPORT["alpha_loss"], 3),
            "critic_loss": round(REPORT["critic_loss"], 3),
            "policy_loss": round(REPORT["policy_loss"], 3),
        }
