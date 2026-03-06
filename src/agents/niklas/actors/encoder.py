import numpy as np
import torch.nn as nn


class StateEncoder(nn.Module):
    def __init__(
        self,
        obs_dim,
        emb_dim,
        hidden_dims=[256, 256],
        use_layernorm=False,
        activation=nn.Mish,
    ):
        super().__init__()
        layers = []

        # Keep this for backward compatibility otherwise checkpoints break
        # TODO convert all checkpoints so this is not needed (when i have time)
        if isinstance(obs_dim, tuple):
            layers.append(nn.Flatten())
            obs_dim = int(np.prod(obs_dim))

        for h in hidden_dims:
            layers.append(nn.Linear(obs_dim, h))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(activation())
            obs_dim = h

        layers.append(nn.Linear(obs_dim, emb_dim))
        if use_layernorm:
            layers.append(nn.LayerNorm(emb_dim))

        self.net = nn.Sequential(*layers)
        self.output_dim = emb_dim

    def forward(self, x):
        return self.net(x)
