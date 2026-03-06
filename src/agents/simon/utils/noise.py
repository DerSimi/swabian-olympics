import torch
from torch import Tensor
from pink import PinkNoiseDist


class PinkNoise:
    def __init__(self, n_env: int, action_dim: int):
        self.n_env = n_env
        self.action_dim = action_dim
        # Init noise processes
        self.noise_processes = []

        for _ in range(self.n_env):
            self.noise_processes.append(
                PinkNoiseDist(seq_len=250, action_dim=action_dim)
            )

    def __call__(self, mean: Tensor, log_std: Tensor) -> Tensor:
        actions = []
        log_probs = []

        for env_id in range(self.n_env):
            action, log_prob = self.sample(
                env_id, mean[env_id].unsqueeze(0), log_std[env_id].unsqueeze(0)
            )
            actions.append(action.squeeze(0))
            log_probs.append(log_prob.squeeze(0))
        return torch.stack(actions), torch.stack(log_probs)

    def sample(
        self, env_id: int, mean: Tensor, log_std: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Internal sampling function, mean and log_std does not have a batch axis anymore!!!
        """

        process: PinkNoiseDist = self.noise_processes[env_id]
        action, log_prob = process.log_prob_from_params(mean, log_std)

        return action, log_prob

    def reset(self, env_id: int):
        process: PinkNoiseDist = self.noise_processes[env_id]
        process.gen.reset()
