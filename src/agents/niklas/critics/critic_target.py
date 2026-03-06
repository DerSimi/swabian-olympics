import torch


def calc_scalar_target(
    r, d, next_q, gamma, ensemble_mode="min", uncertainty_coeff=0.0, **kwargs
):
    if ensemble_mode == "min":
        next_val, _ = torch.min(next_q, dim=0)
    elif ensemble_mode == "mean":
        next_val = torch.mean(next_q, dim=0)
    elif ensemble_mode == "lcb":
        mean = next_q.mean(dim=0)
        std = next_q.std(dim=0)
        next_val = mean - (uncertainty_coeff * std)
    else:
        raise ValueError(f"Unknown ensemble mode: {ensemble_mode}")

    return r + (1 - d) * gamma * next_val


def calc_quantile_target(r, d, next_q, gamma, **kwargs):
    """
    Compute the target for the critic network
    Drop top quantiles to reduce overestimation bias
    """
    q_drop_fraction = kwargs.get("top_quantiles_to_drop", 0.0)
    E, B, Q = next_q.shape

    next_q = next_q.permute(1, 0, 2).reshape(B, E * Q)
    sorted_qs, _ = torch.sort(next_q, dim=1)
    n_total = sorted_qs.shape[1]

    n_keep = int(n_total * (1.0 - q_drop_fraction))

    if n_keep < 1:
        n_keep = 1
    target_atoms = sorted_qs[:, :n_keep]

    return r + (1 - d) * gamma * target_atoms
