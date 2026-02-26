def calc_dpg_loss(actor, critics, state_emb, **kwargs):
    action = actor(state_emb)
    out_c = critics(state_emb, action)
    q_vals = out_c[0]
    if q_vals.ndim == 3 and q_vals.shape[-1] > 1:
        q_scalar = q_vals.mean(dim=-1)
    else:
        q_scalar = q_vals.squeeze(-1)
    if q_scalar.ndim > 1:
        q_val = q_scalar.min(dim=0)[0]
    else:
        q_val = q_scalar
    return -q_val.mean()
