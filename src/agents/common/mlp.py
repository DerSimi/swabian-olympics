from typing import List, Type

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    A standard Multi-Layer Perceptron.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = None,
        activation: Type[nn.Module] = nn.ReLU,
        output_activation: Type[nn.Module] = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
        layers = []
        last_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(activation())
            last_dim = h

        layers.append(nn.Linear(last_dim, output_dim))

        if output_activation:
            layers.append(output_activation())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
