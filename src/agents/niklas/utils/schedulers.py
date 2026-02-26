import torch


def create_warmup_scheduler(optimizer, warmup_steps):
    """LR scheduler to help stabilize early training with small replay buffer and new network"""
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-4, end_factor=1.0, total_iters=warmup_steps
    )
    const = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, const], milestones=[warmup_steps]
    )


class HyperparameterScheduler:
    def __init__(self, cfg):
        self.cfg = cfg

    def get_annealed_params(self, step):
        # Planning STD Threshold Decay
        frac = min(step / self.cfg.planning_std_decay_steps, 1.0)
        std_thresh = self.cfg.planning_std_threshold_start + frac * (
            self.cfg.planning_std_threshold_end - self.cfg.planning_std_threshold_start
        )

        # Intervention Threshold Decay
        frac_inter = min(step / self.cfg.planning_intervention_decay_steps, 1.0)
        intervention_thresh = (
            self.cfg.planning_intervention_threshold_start
            + frac_inter
            * (
                self.cfg.planning_intervention_threshold_end
                - self.cfg.planning_intervention_threshold_start
            )
        )

        # Aux Loss Coeff Decay
        aux_coeff = self.cfg.aux_coeff_start + frac * (
            self.cfg.aux_coeff_end - self.cfg.aux_coeff_start
        )

        # Quantile Drop (Triangle Cycle)
        q_drop = 0.0
        cycle = self.cfg.quantile_exploration_cycle_length

        if self.cfg.quantile_exploration and cycle > 0:
            pos = (step % cycle) / cycle

            # 0.0 -> 1.0 -> 0.0
            tri_factor = pos * 2.0 if pos < 0.5 else (1.0 - pos) * 2.0

            q_drop = self.cfg.top_quantiles_to_drop_start + tri_factor * (
                self.cfg.top_quantiles_to_drop_end
                - self.cfg.top_quantiles_to_drop_start
            )
        else:
            q_drop = self.cfg.top_quantiles_to_drop
        q_drop = max(0.0, min(q_drop, 1.0))

        return {
            "aux_coeff": aux_coeff,
            "q_drop": q_drop,
            "std_thresh": std_thresh,
            "intervention_thresh": intervention_thresh,
        }
