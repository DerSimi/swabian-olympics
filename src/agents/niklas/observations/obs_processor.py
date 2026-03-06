import numpy as np
import torch

# --- Mirror Indices ---
# P1: y (1), angle (2), vy (4), w (5)
# P2: y (7), angle (8), vy (10), w (11)
# Puck: y (13), vy (15)
MIRROR_OBS_INDICES = np.array(
    [
        1,
        2,
        4,
        5,  # P1
        7,
        8,
        10,
        11,  # P2
        13,
        15,  # Puck
    ],
    dtype=np.int32,
)

# Action Mirroring
# [Fx, Fy, Torque, Shoot]
# Flip Fy (1) and Torque (2)
MIRROR_ACT_INDICES = np.array([1, 2], dtype=np.int32)


class RunningMeanStd:
    """
    Tracks mean and variance for normalization of observations
    μ_k = μ_{k-1} + (x_k - μ_{k-1}) / k
    σ²_k = M2_k / k
    """

    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class ObsProcessor:
    def __init__(self, config, device, obs_dim=18, act_dim=4):
        self.cfg = config
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.rms = RunningMeanStd(shape=(obs_dim,))
        self.clip = 5.0

        if self.cfg.mirror_data:
            self.obs_mirror = torch.ones(obs_dim, device=device, dtype=torch.float32)
            self.act_mirror = torch.ones(act_dim, device=device, dtype=torch.float32)

            obs_idx = torch.as_tensor(
                MIRROR_OBS_INDICES, device=device, dtype=torch.long
            )
            act_idx = torch.as_tensor(
                MIRROR_ACT_INDICES, device=device, dtype=torch.long
            )

            self.obs_mirror[obs_idx] = -1.0
            self.act_mirror[act_idx] = -1.0

    def normalize(self, obs_tensor, update_stats=False):
        if not self.cfg.normalize_obs:
            return obs_tensor

        # Separate Physics from Timedata (time is already normalized)
        obs_part = obs_tensor[..., : self.obs_dim]
        meta_part = (
            obs_tensor[..., self.obs_dim :]
            if obs_tensor.shape[-1] > self.obs_dim
            else None
        )

        if update_stats:
            self.rms.update(obs_part.detach().cpu().numpy())

        device = obs_tensor.device
        mean = torch.as_tensor(self.rms.mean, dtype=torch.float32, device=device)
        std = torch.as_tensor(
            np.sqrt(self.rms.var + 1e-8), dtype=torch.float32, device=device
        )

        # Normalize Physics
        normed_obs = (obs_part - mean) / std
        normed_obs = torch.clamp(normed_obs, -self.clip, self.clip)

        # Recombine with Timedata
        if meta_part is not None and meta_part.numel() > 0:
            normed_meta = meta_part / float(self.cfg.max_env_steps)
            return torch.cat([normed_obs, normed_meta], dim=-1)

        return normed_obs

    def mirror_batch(self, batch, weights=None):
        """
        Mirror batch of observations and actions for 2x samples
        """
        if not self.cfg.mirror_data:
            return batch, weights

        s, a, r, ns, d = batch
        p_dim = self.obs_mirror.shape[0]

        # Split Physics andTimedata
        s_phys, s_meta = s[:, :p_dim], s[:, p_dim:]
        ns_phys, ns_meta = ns[:, :p_dim], ns[:, p_dim:]

        # Mirror Physics
        s_phys_mir = s_phys * self.obs_mirror
        ns_phys_mir = ns_phys * self.obs_mirror
        a_mir = a * self.act_mirror

        # Recombine with Timedata
        s_mir = torch.cat([s_phys_mir, s_meta], dim=1)
        ns_mir = torch.cat([ns_phys_mir, ns_meta], dim=1)

        # Batch: [Original | Mirrored]
        s_aug = torch.cat([s, s_mir], dim=0)
        a_aug = torch.cat([a, a_mir], dim=0)
        r_aug = torch.cat([r, r], dim=0)
        ns_aug = torch.cat([ns, ns_mir], dim=0)
        d_aug = torch.cat([d, d], dim=0)

        w_aug = torch.cat([weights, weights], dim=0) if weights is not None else None

        return (s_aug, a_aug, r_aug, ns_aug, d_aug), w_aug
