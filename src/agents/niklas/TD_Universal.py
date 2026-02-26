from typing import Any, Dict

import numpy as np
import torch

from agents.niklas.actors import soft_update
from agents.niklas.utils import (
    build_networks,
)
from agents.niklas.utils.setup import (
    setup_components,
    setup_config,
    setup_data,
    setup_dimensions,
    setup_stats,
)
from framework.agent import AbstractAgent
from framework.registry import register_agent


@register_agent("td_universal")
class TDUniversalAgent(AbstractAgent):
    """
    TD Universal: One Agent to Rule Them All
    Config-driven architecture that can instantiate TD3, TD7, TQC, ... at least that was the plan
    and for the most part it works

    Time awarness was a intital feature but later it turned out that the server didnt have a actuall
    hook for a reset after each game thus a new agent needed to be trained without it
    But since the agent gets a punishment reward for existing it didnt need it i guess e.g
    indirect time awarness via reward it was a nice initial idea but later i found out that
    it would cause out of distribution problems if a game ever was longer than 250
    so over all deactivating it was the better choice for a more universal policy
    """

    def __init__(self, run_name, storage_path, config, **kwargs):
        super().__init__(run_name, storage_path, config, **kwargs)
        setup_config(self, kwargs)
        setup_dimensions(self)
        setup_stats(self)
        setup_components(self)
        build_networks(self)
        setup_data(self)

        self.opponent_pool_ref = None
        self.inference_time_step = 0
        self.current_time_step = None
        self.last_dones = None
        print(f"Total Steps: {self.total_steps}")
        print(f"Time Aware: {self.cfg.time_aware}")

    def reset(self):
        """Called by Competition Wrapper or Eval Loop to force a clean state e.g reset time awarness"""
        self.inference_time_step = 0

    def act(self, obs, inference=False, **kwargs):
        """
        Executes a forward pass

        During training, applies additive exploration noise and clips to valid action bounds:
        a = clip(μ_φ(s) + ε, -1, 1)    where    ε ~ N(0, σ²) or pink noise
        """
        info = kwargs.get("info", None)
        n_envs = obs.shape[0]

        if info is not None and "time_step" in info:
            t_curr = np.array(info["time_step"], dtype=np.float32)
            if self.last_dones is not None:
                t_curr[self.last_dones] = 0.0
        else:
            t_curr = np.full(n_envs, self.inference_time_step, dtype=np.float32)

        self.current_time_step = t_curr.copy()

        s_tensor = torch.as_tensor(obs, device=self.dev, dtype=torch.float32)
        s_normed = self.processor.normalize(s_tensor, update_stats=not inference)

        if self.cfg.time_aware:
            t_norm = torch.as_tensor(
                t_curr / self.cfg.max_env_steps, device=self.dev, dtype=torch.float32
            ).view(-1, 1)
            s_ready = torch.cat([s_normed, t_norm], dim=-1)
        else:
            s_ready = s_normed

        # Buffer fill-up phase dont uses actor since it is not trained yet
        if not inference and self.env_step < self.cfg.critic_learning_starts:
            return np.random.uniform(
                -1, 1, size=(obs.shape[0], self.cfg.action_dim)
            ).astype(np.float32)

        params = self.hp_scheduler.get_annealed_params(self.env_step)

        action_tensor, plan_stats = self.planner.decide_and_act(
            actor=self.actor,
            critics=self.critics,
            encoder=self.enc,
            s_normed=s_ready,
            std_threshold=params["std_thresh"],
            intervention_threshold=params["intervention_thresh"],
            inference=inference,
        )

        action = action_tensor.detach().cpu().numpy()

        if not inference:
            saturated = (np.abs(action) > 0.95).mean()
            self.stats["vals/action_saturation"] = saturated

            action = action + self.noise(action.shape)
            if self.last_dones is not None:
                self.noise.reset(self.last_dones)

            self.act_calls_train += 1

            if plan_stats.get("planning/intervention_rate", 0) > 0:
                self.planning_executed_count += 1

            for k, v in plan_stats.items():
                if k in self.stats:
                    self.stats[k] += v
                else:
                    self.stats[k] = v if not torch.is_tensor(v) else v.item()

        if info is None or "time_step" not in info:
            self.inference_time_step += 1

        return np.clip(action, -1, 1)

    def learn(
        self,
        obs,
        action,
        reward,
        next_obs,
        info,
        done,
    ):
        if self.inference_mode or self.buffer is None:
            return

        current_dones = done.flatten().astype(bool)
        prev_dones_mask = (
            self.last_dones
            if self.last_dones is not None
            else np.ones(current_dones.shape, dtype=bool)
        )
        self.last_dones = current_dones

        # Protect for other pools
        if hasattr(self.opponent_pool_ref, "register_outcome"):
            for i, d in enumerate(self.last_dones):
                if d:
                    self.opponent_pool_ref.register_outcome(i, info["winner"][i])

        total_reward = reward.copy()

        # L1 Action Smoothness Penalty
        if (
            getattr(self, "last_actions", None) is None
            or self.last_actions.shape != action.shape
        ):
            self.last_actions = np.zeros_like(action)

        action_diff = np.sum(np.abs(action - self.last_actions), axis=-1)
        penalty = 0.005 * action_diff
        penalty[prev_dones_mask] = 0.0
        total_reward = total_reward - penalty.reshape(total_reward.shape)

        self.last_actions = action.copy()

        if self.cfg.time_aware:
            t_curr = self.current_time_step
            t_next = t_curr + 1.0

            s_store = np.concatenate([obs, t_curr.reshape(-1, 1)], axis=-1)
            ns_store = np.concatenate([next_obs, t_next.reshape(-1, 1)], axis=-1)
        else:
            s_store = obs
            ns_store = next_obs

        n_step_data = self.n_step_collector.step(
            s_store, action, total_reward, ns_store, done
        )

        if n_step_data is not None:
            s_n, a_n, r_n, ns_n, d_n = n_step_data
            self.buffer.add_batch(
                torch.as_tensor(s_n, device=self.dev, dtype=torch.float32),
                torch.as_tensor(a_n, device=self.dev, dtype=torch.float32),
                torch.as_tensor(r_n, device=self.dev, dtype=torch.float32),
                torch.as_tensor(ns_n, device=self.dev, dtype=torch.float32),
                torch.as_tensor(d_n, device=self.dev, dtype=torch.float32),
            )

        if self.env_step < self.cfg.critic_learning_starts:
            return

        params = self.hp_scheduler.get_annealed_params(self.env_step)

        # Update Networks
        for _ in range(self.cfg.grad_steps_env):
            # Critics
            for _ in range(self.cfg.critic_grad_steps):
                (s, a, r, ns, d), weights, indices, emb = self._sample_batch()
                self._update_critic(
                    s,
                    a,
                    r,
                    ns,
                    d,
                    weights,
                    indices,
                    params["q_drop"],
                    params["aux_coeff"],
                    emb,
                )

            # Actor
            for i in range(self.cfg.actor_grad_steps):
                (s, a, _, _, _), _, _, emb = self._sample_batch()
                self._update_actor(s, a, emb)

            # Schedulers
            for sched in self.schedulers:
                sched.step()

            self.grad_step_counter += 1
            self.learn_calls += 1

        # Self-play hook
        if (
            self.cfg.self_play != -1
            and self.env_step > 0
            and self.env_step % self.cfg.self_play == 0
            and self.opponent_pool_ref is not None
        ):
            snapshot = self.checkpointer.save_snapshot(self.env_step, self.cfg)
            if hasattr(self.opponent_pool_ref, "add_opponent"):
                self.opponent_pool_ref.add_opponent(snapshot)
            print(f"[Self-Play] Added Snapshot v{self.env_step} to pool.")

    def _sample_batch(self):
        batch, weights, indices = self.buffer.sample(self.cfg.batch, beta=self.cfg.beta)

        (s, a, r, ns, d), weights = self.processor.mirror_batch(batch, weights)

        s = self.processor.normalize(s, update_stats=False)
        ns = self.processor.normalize(ns, update_stats=False)

        self.enc.train()
        emb = self.enc(s)

        return (s, a, r, ns, d), weights, indices, emb

    def _update_critic(self, s, a, r, ns, d, weights, indices, q_drop, aux_coeff, emb):
        """
        Calculates the Temporal Difference target and updates the critic networks

        Critic Target (TD3 with Target Policy Smoothing):
        y = r + γ(1 - d) * min_i( Q_θi'(s', μ_φ'(s') + ε) )

        Critic Loss (Mean Squared Error with PER weights):
        L(θ_i) = (1/N) Σ w_j * ( Q_θi(s_j, a_j) - y_j )² + L_aux

        PER Priority Update (L1 Error):
        p_j = | E[Q_θ(s_j, a_j)] - y_j | + ε
        """
        cfg = self.cfg

        train_kwargs = cfg.__dict__.copy()
        train_kwargs["top_quantiles_to_drop"] = q_drop
        train_kwargs["aux_loss_coeff"] = aux_coeff

        train_kwargs["gamma"] = cfg.gamma**cfg.n_step_returns

        with torch.no_grad():
            next_emb = self.enc_t(ns)

        with torch.no_grad():
            next_a = self.actor_t.get_target_action(next_emb)

            next_q = self.critics_t(next_emb, next_a)
            if isinstance(next_q, tuple):
                next_q = next_q[0]

            target_y = self.critics.compute_target(r, d, next_q, **train_kwargs)

        q_pred, aux_pred = self.critics(emb, a)
        loss_q, prios = self.critics.compute_loss(q_pred, target_y, weights)

        loss_aux = self.critics.compute_aux_loss(
            aux_pred,
            s,
            ns,
            obs_dim=self.raw_obs_dim,
            aux_loss_coeff=aux_coeff,
        )
        total_critic_loss = loss_q + loss_aux

        self.critic_opt.zero_grad()
        total_critic_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.critics.parameters(), 1.0)
        self.critic_opt.step()

        if aux_coeff > 1e-6:
            self.stats["loss/aux"] += loss_aux.detach() / aux_coeff

        bias = q_pred.mean().detach() - target_y.mean().detach()
        self.stats["loss/critic"] += total_critic_loss.detach()
        self.stats["vals/estimation_bias"] += bias
        self.stats["grad/critic"] += grad_norm.item()
        self.stats["vals/q_mean"] += q_pred.mean().detach()
        self.stats["vals/q_std"] += q_pred.std().detach()
        self.stats["vals/target_mean"] += target_y.mean().detach()
        self.stats["vals/batch_reward"] += r.mean().detach()

        # Update Priorities for Buffer
        if hasattr(self.buffer, "update_priorities"):
            # Use L1 instead of Huber loss for priority update
            current_q_mean = q_pred.mean(dim=-1).mean(dim=0)
            target_q_mean = target_y.mean(dim=-1)
            new_prios = (current_q_mean - target_q_mean).abs().detach()

            if new_prios.shape[0] > indices.shape[0]:
                batch_size = indices.shape[0]
                p_original = new_prios[:batch_size]
                p_mirrored = new_prios[batch_size:]
                new_prios = torch.max(p_original, p_mirrored)

            save_prios = new_prios.cpu().numpy() + 1e-6
            self.buffer.update_priorities(indices, save_prios)

    def _update_actor(self, s, a, emb):
        """
        Update actor network using the Deterministic Policy Gradient
        Freeze critics to prevent unnecessary gradient computation

        Actor Objective (Maximize expected Min-Q-value):
        J(φ) = E_s [ min_i(Q_θi(s, μ_φ(s))) ]

        Gradient Ascent Step (Chain Rule):
        ∇_φ J(φ) ≈ (1/N) Σ [ ∇_a Q_θ1(s, a)|a=μ_φ(s) * ∇_φ μ_φ(s) ]
        """
        # Freeze critics to save memory/computation graph
        for p in self.critics.parameters():
            p.requires_grad = False

        self.actor.train()

        actor_loss = self.actor.compute_loss(
            critics=self.critics, state_emb=emb.detach(), action=a, **self.cfg.__dict__
        )

        self.actor_opt.zero_grad()
        actor_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # Unfreeze critics
        for p in self.critics.parameters():
            p.requires_grad = True

        self.stats["loss/actor"] += actor_loss.detach()
        self.stats["grad/actor"] += grad_norm.item()

        # Soft Updates
        tau = 1.0 if self.cfg.hard_update else self.cfg.tau
        soft_update(self.critics, self.critics_t, tau)
        soft_update(self.actor, self.actor_t, tau)
        soft_update(self.enc, self.enc_t, tau)

    def print_report(self) -> Dict[str, Any]:
        stats = {}
        for k, v in self.stats.items():
            if k.startswith("planning/"):
                if k == "planning/intervention_rate":
                    if self.act_calls_train > 0:
                        stats[k] = v / self.act_calls_train
                    else:
                        stats[k] = 0.0
                elif k in ["planning/divergence", "planning/value_gain"]:
                    if self.planning_executed_count > 0:
                        stats[k] = v / self.planning_executed_count
                    else:
                        stats[k] = 0.0
                else:
                    stats[k] = 0.0
            else:
                if self.learn_calls > 0:
                    if torch.is_tensor(v):
                        stats[k] = v.item() / self.learn_calls
                    else:
                        stats[k] = v / self.learn_calls
                else:
                    stats[k] = 0.0

        for k in self.stats:
            if torch.is_tensor(self.stats[k]):
                self.stats[k].zero_()
            else:
                self.stats[k] = 0.0

        self.learn_calls = 0
        self.act_calls_train = 0
        self.planning_executed_count = 0

        return stats

    def store_dict(self, agent_descriptor: str) -> Dict[str, Any]:
        return self.checkpointer.store_dict()

    def load_dict(self, state_dict: Dict[str, Any], inference: bool = False) -> None:
        self.checkpointer.load_dict(state_dict, inference)

    def opponent_pool(self, opponents, eval_env, n_env) -> Any:
        from agents.niklas.utils.opponent_pools import OPPONENT_POOL_REGISTRY

        self.opponent_pool_ref = OPPONENT_POOL_REGISTRY[self.cfg.opponent_pool](
            agent=self,
            config=self.argument_config,
            opponents=opponents,
            eval_env=eval_env,
            n_env=n_env,
        )
        self.opponent_pool_ref.selected_opponents = [opponents[0]] * n_env

        return self.opponent_pool_ref

    @staticmethod
    def custom_reward(original_reward, obs, terminated, truncated, info):
        bonus = 0.0
        if "winner" in info:
            if info["winner"] == 1:
                bonus += 10.0
            elif info["winner"] == -1:
                bonus -= 10.0

        # TODO refine this e.g move it a bit further out of the goal maybe?
        # Defensive shaping
        agent_x = obs[0]
        agent_y = obs[1]
        puck_x = obs[12]
        pressure = max(0.0, puck_x) / 5.0

        if pressure > 0.0:
            x_penalty = max(0.0, agent_x - (-3.0))
            y_penalty = abs(agent_y)
            bonus -= pressure * ((0.02 * x_penalty) + (0.005 * y_penalty))

        return original_reward + bonus
