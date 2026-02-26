import numpy as np
import torch

from .cem import cem_plan


class Planner:
    def __init__(self, cfg, device, obs_dim, action_dim):
        self.cfg = cfg
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.physics_dim = cfg.obs_dim
        self.time_aware = cfg.time_aware

    def plan(self, encoder, critics, state):
        return cem_plan(
            encoder=encoder,
            critics=critics,
            state=state,
            action_dim=self.action_dim,
            cfg=self.cfg,
            device=self.device,
            obs_dim=self.obs_dim,
        )

    @torch.no_grad()
    def decide_and_act(
        self,
        actor,
        critics,
        encoder,
        s_normed,
        std_threshold,
        intervention_threshold=0.0,
        inference=False,
    ):
        """
        Decide for the act method whether to plan or not
        (intervention planning and/or selective planning)
        """
        stats = {
            "planning/intervention_rate": 0.0,
            "planning/divergence": 0.0,
            "planning/value_gain": 0.0,
        }

        student_action = actor.get_action(encoder(s_normed), deterministic=True)

        if inference:
            return student_action, stats

        do_plan = False

        if self.cfg.use_planning_teacher:
            if np.random.random() < intervention_threshold:
                do_plan = True

        if not do_plan and self.cfg.selective_planning:
            emb = encoder(s_normed)
            qs, _ = critics(emb, student_action)

            if qs.shape[0] > 1:
                disagreement = qs.std(dim=0).max().item()
                if disagreement > std_threshold:
                    do_plan = True

        final_action = student_action

        if do_plan:
            final_action = self.plan(encoder, critics, s_normed)

            stats["planning/intervention_rate"] = 1.0
            diff = (student_action - final_action).pow(2).mean().item()
            stats["planning/divergence"] = diff

            emb = encoder(s_normed)
            q_s, _ = critics(emb, student_action)
            q_t, _ = critics(emb, final_action)
            stats["planning/value_gain"] = q_t.mean().item() - q_s.mean().item()

        return final_action, stats
