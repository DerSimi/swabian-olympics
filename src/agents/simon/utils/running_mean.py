import torch


# Stolen and adapted from here: https://gymnasium.farama.org/v0.29.0/_modules/gymnasium/wrappers/normalize/
class RunningMeanStdTorch:
    """Tracks the mean, variance and count of values (PyTorch version)."""

    def __init__(
        self,
        dev,
        epsilon: float = 1e-4,
        shape: tuple = (),
    ):
        """Tracks the mean, variance and count of values."""
        self.mean = torch.zeros(shape, dtype=torch.float32, device=dev)
        self.var = torch.ones(shape, dtype=torch.float32, device=dev)
        self.count = epsilon

        self.dev = dev

    def update(self, x: torch.Tensor):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = x.shape[0]

        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: float
    ):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(
    mean: torch.Tensor,
    var: torch.Tensor,
    count: float,
    batch_mean: torch.Tensor,
    batch_var: torch.Tensor,
    batch_count: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count
