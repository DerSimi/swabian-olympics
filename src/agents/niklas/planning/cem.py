import torch


def cem_plan(
    encoder,
    critics,
    state,
    action_dim,
    cfg,
    device,
    obs_dim,
):
    """
    Stateless CEM Planner with UCB Exploration
    Sample k actions and evaluate them using the critics
    Helps improving the critics as well as getting out of early local optima
    (thats the plan at least :))

    Sample: a ~ N(μ, σ²)
    Score:  S(s, a) = μ_Q(s, a) + (bonus * σ_Q(s, a))
    Update: μ, σ² = fit(Top_K(S))
    """
    batch_size = state.shape[0]
    exploration_bonus = cfg.exploration_bonus

    K = cfg.cem_samples
    Iters = cfg.cem_iters
    Elites = cfg.cem_elites

    total_samples = batch_size * K
    curr_s_norm = state.unsqueeze(1).expand(-1, K, -1).reshape(total_samples, obs_dim)

    mean_a = torch.zeros(batch_size, action_dim, device=device)
    std_a = torch.ones(batch_size, action_dim, device=device)

    for i in range(Iters):
        actions = mean_a.unsqueeze(1) + std_a.unsqueeze(1) * torch.randn(
            batch_size, K, action_dim, device=device
        )
        actions = torch.clamp(actions, -1.0, 1.0)

        flat_actions = actions.view(total_samples, action_dim)

        emb = encoder(curr_s_norm)
        qs, _ = critics(emb, flat_actions)

        qs_mean_quantiles = qs.mean(dim=-1)
        q_mean = qs_mean_quantiles.mean(dim=0)

        if qs_mean_quantiles.shape[0] > 1:
            q_std = qs_mean_quantiles.std(dim=0)
            scores_flat = q_mean + exploration_bonus * q_std
        else:
            scores_flat = q_mean

        scores = scores_flat.view(batch_size, K)

        _, elite_idxs = torch.topk(scores, Elites, dim=1, largest=True)

        idx_expanded = elite_idxs.unsqueeze(-1).expand(-1, -1, action_dim)
        elite_actions = torch.gather(actions, 1, idx_expanded)

        mean_a = elite_actions.mean(dim=1)
        std_a = elite_actions.std(dim=1) + 1e-4

    return mean_a
