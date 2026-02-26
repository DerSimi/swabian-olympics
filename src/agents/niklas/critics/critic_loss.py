import torch
import torch.nn.functional as F


def calc_mse_loss(current_q, target_q, weights=None, **kwargs):
    loss = F.mse_loss(current_q, target_q.unsqueeze(0), reduction="none")

    if weights is not None:
        loss *= weights.view(1, -1, 1)

    return loss.mean(), (target_q.unsqueeze(0) - current_q).abs().mean(0).detach()


def calc_huber_loss(current_q, target_q, weights=None, **kwargs):
    n_quantiles = current_q.shape[-1]
    delta = kwargs.get("quantile_huber_kappa", 1.0)
    device = current_q.device

    taus = (
        torch.arange(n_quantiles, device=device, dtype=torch.float32) + 0.5
    ) / n_quantiles
    taus = taus.view(1, 1, -1, 1)

    curr = current_q.unsqueeze(-1)
    targ = target_q.unsqueeze(0).unsqueeze(2)

    u = targ - curr
    curr_expanded, targ_expanded = torch.broadcast_tensors(curr, targ)

    huber_error = F.huber_loss(
        curr_expanded, targ_expanded, reduction="none", delta=delta
    )

    with torch.no_grad():
        asymmetry = torch.abs(taus - (u < 0).float())

    element_loss = asymmetry * huber_error
    loss = element_loss.sum(dim=-1).mean(dim=-1)

    if weights is not None:
        loss *= weights.view(1, -1)

    return loss.mean(), loss.detach().mean(0)


def calc_physics_aux(
    delta_pred, s, ns, obs_dim, aux_loss_coeff, physics_dim=18, **kwargs
):
    if delta_pred is None:
        return torch.tensor(0.0, device=s.device)

    curr_phys = s[:, :physics_dim]
    next_phys = ns[:, :physics_dim]

    delta_target = next_phys - curr_phys

    if delta_pred.ndim > delta_target.ndim:
        target_expanded = delta_target.unsqueeze(0).repeat(delta_pred.shape[0], 1, 1)
    else:
        target_expanded = delta_target

    loss = F.mse_loss(delta_pred, target_expanded)
    return loss * aux_loss_coeff
