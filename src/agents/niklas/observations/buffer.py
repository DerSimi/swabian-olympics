import numpy as np
import torch


class GPUReplayBuffer:
    def __init__(
        self,
        state_shape,
        action_dim,
        max_size=1000000,
        device="cuda",
    ):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = torch.device(device)

        # Handle state shape
        if isinstance(state_shape, int):
            self.state_shape = (state_shape,)
        else:
            self.state_shape = state_shape

        # Allocate GPU Tensors
        self.state = torch.zeros(
            (max_size, *self.state_shape), dtype=torch.float32, device=self.device
        )
        self.action = torch.zeros(
            (max_size, action_dim), dtype=torch.float32, device=self.device
        )
        self.reward = torch.zeros(
            (max_size, 1), dtype=torch.float32, device=self.device
        )
        self.next_state = torch.zeros(
            (max_size, *self.state_shape), dtype=torch.float32, device=self.device
        )
        self.done = torch.zeros((max_size, 1), dtype=torch.float32, device=self.device)
        print(f"GPUReplayBuffer initialized with max_size={self.max_size}")

    def add_batch(self, state, action, reward, next_state, done):
        """Add a batch of transitions"""
        n = state.shape[0]
        idxs = np.arange(self.ptr, self.ptr + n) % self.max_size

        self.state[idxs] = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        )
        self.action[idxs] = torch.as_tensor(
            action, dtype=torch.float32, device=self.device
        )
        self.reward[idxs] = torch.as_tensor(
            reward, dtype=torch.float32, device=self.device
        ).reshape(-1, 1)
        self.next_state[idxs] = torch.as_tensor(
            next_state, dtype=torch.float32, device=self.device
        )
        self.done[idxs] = torch.as_tensor(
            done, dtype=torch.float32, device=self.device
        ).reshape(-1, 1)

        self.ptr = (self.ptr + n) % self.max_size
        self.size = min(self.size + n, self.max_size)

    def sample(self, batch_size, beta=0.0):
        """
        Sample a batch of transitions
        Returns: (batch, weights, indices)
        """
        idxs = torch.randint(0, self.size, (batch_size,), device=self.device)

        batch = (
            self.state[idxs],
            self.action[idxs],
            self.reward[idxs],
            self.next_state[idxs],
            self.done[idxs],
        )

        # Uniform weights for standard buffer
        weights = torch.ones((batch_size, 1), device=self.device)

        return batch, weights, idxs

    def save(self, path):
        # Move to CPU for saving with numpy
        np.savez_compressed(
            path,
            state=self.state[: self.size].cpu().numpy(),
            action=self.action[: self.size].cpu().numpy(),
            reward=self.reward[: self.size].cpu().numpy(),
            next_state=self.next_state[: self.size].cpu().numpy(),
            done=self.done[: self.size].cpu().numpy(),
            ptr=self.ptr,
            size=self.size,
        )

    def load(self, path):
        if not path.endswith(".npz"):
            path = f"{path}.npz"
        try:
            data = np.load(path)
        except FileNotFoundError:
            return False

        if "size" in data:
            loaded_size = int(data["size"])
        else:
            loaded_size = data["state"].shape[0]

        # Load directly to GPU
        self.state[:loaded_size] = torch.from_numpy(data["state"]).to(self.device)
        self.action[:loaded_size] = torch.from_numpy(data["action"]).to(self.device)
        self.reward[:loaded_size] = torch.from_numpy(data["reward"]).to(self.device)
        self.next_state[:loaded_size] = torch.from_numpy(data["next_state"]).to(
            self.device
        )
        self.done[:loaded_size] = torch.from_numpy(data["done"]).to(self.device)

        if "ptr" in data:
            self.ptr = int(data["ptr"])
        else:
            self.ptr = loaded_size % self.max_size

        self.size = loaded_size
        return True

    def clear(self):
        self.ptr = 0
        self.size = 0


class GPUPrioritizedReplayBuffer(GPUReplayBuffer):
    """
    Prioritized Experience Replay (PER) using a vectorized GPU Sum Tree

    Probability of sampling transition i:
    P(i) = p_i^α / Σ(p_k^α)

    Importance Sampling (IS) weights to correct for bias:
    w_i = (1 / (N · P(i)))^β / max(w_j)
    """

    def __init__(
        self,
        state_shape,
        action_dim,
        max_size=1000000,
        alpha=0.6,
        device="cuda",
    ):
        super().__init__(state_shape, action_dim, max_size, device)
        self.alpha = alpha

        # Tree Capacity (Power of 2)
        tree_capacity = 1
        while tree_capacity < max_size:
            tree_capacity *= 2
        self.tree_capacity = tree_capacity

        # Allocate Tree on GPU
        self.sum_tree = torch.zeros(
            2 * self.tree_capacity, dtype=torch.float32, device=self.device
        )
        self.min_tree = torch.full(
            (2 * self.tree_capacity,),
            float("inf"),
            dtype=torch.float32,
            device=self.device,
        )
        self.max_priority = torch.tensor(1.0, device=self.device, dtype=torch.float32)

        self.tree_depth = int(np.ceil(np.log2(self.tree_capacity)))

        print(
            f"GPUPrioritizedReplayBuffer initialized with alpha={self.alpha}, max_size={self.max_size}"
        )

    def add_batch(self, state, action, reward, next_state, done):
        n = state.shape[0]
        idxs = np.arange(self.ptr, self.ptr + n) % self.max_size
        super().add_batch(state, action, reward, next_state, done)

        # Update Tree
        tree_idxs = idxs + self.tree_capacity
        tree_idxs_t = torch.as_tensor(tree_idxs, device=self.device, dtype=torch.long)

        # New priorities
        priorities = self.max_priority.expand(n)
        powered_p = (priorities**self.alpha) + 1e-6
        self._update_tree_batch(tree_idxs_t, powered_p)

    def _update_tree_batch(self, tree_idxs, priorities):
        """
        Vectorized tree update.
        tree_idxs: [B] Tensor (Global indices in tree array)
        priorities: [B] Tensor
        """
        self.sum_tree[tree_idxs] = priorities
        self.min_tree[tree_idxs] = priorities

        # Propagate Up
        idx = tree_idxs
        for _ in range(self.tree_depth):
            idx = idx // 2
            idx = torch.unique(idx)

            left = 2 * idx
            right = left + 1

            self.sum_tree[idx] = self.sum_tree[left] + self.sum_tree[right]
            self.min_tree[idx] = torch.min(self.min_tree[left], self.min_tree[right])

    def sample(self, batch_size, beta=0.4):
        total_p = self.sum_tree[1]

        if total_p <= 1e-6:
            total_p = 1.0

        segment = total_p / batch_size
        r = torch.rand(batch_size, device=self.device, dtype=torch.float32)
        offsets = (
            torch.arange(batch_size, device=self.device, dtype=torch.float32) * segment
        )
        targets = offsets + (r * segment)
        idxs = torch.ones(batch_size, dtype=torch.long, device=self.device)

        # Vectorized Tree Search
        for _ in range(self.tree_depth):
            left = 2 * idxs
            right = left + 1

            left_val = self.sum_tree[left]

            mask = targets <= left_val
            targets = torch.where(mask, targets, targets - left_val)
            idxs = torch.where(mask, left, right)

        data_idxs = idxs - self.tree_capacity
        data_idxs = torch.clamp(data_idxs, 0, self.max_size - 1)

        priorities = self.sum_tree[idxs]

        priorities = torch.clamp(priorities, min=1e-6)

        batch = (
            self.state[data_idxs],
            self.action[data_idxs],
            self.reward[data_idxs],
            self.next_state[data_idxs],
            self.done[data_idxs],
        )

        min_p_val = self.min_tree[1]

        if min_p_val == float("inf") or min_p_val <= 1e-6:
            min_p_val = torch.tensor(1e-6, device=self.device, dtype=torch.float32)

        min_prob = min_p_val / total_p

        # Calculate max_weight for normalization
        max_weight = (min_prob * self.size) ** (-beta)
        if max_weight <= 1e-8:
            max_weight = 1.0

        probs = priorities / total_p
        weights = (self.size * probs) ** (-beta)
        weights = weights / max_weight
        weights = torch.clamp(weights, min=0.0, max=10.0)
        weights_t = weights.view(-1, 1).to(dtype=torch.float32)

        return batch, weights_t, data_idxs

    def update_priorities(self, indices, priorities):
        if not torch.is_tensor(indices):
            indices = torch.as_tensor(indices, device=self.device, dtype=torch.long)
        if not torch.is_tensor(priorities):
            priorities = torch.as_tensor(
                priorities, device=self.device, dtype=torch.float32
            )

        priorities = torch.nan_to_num(priorities, nan=1e-6)

        current_batch_max = torch.max(priorities)
        self.max_priority = torch.max(self.max_priority, current_batch_max)

        tree_idxs = indices + self.tree_capacity
        powered_priorities = (priorities**self.alpha) + 1e-6
        self._update_tree_batch(tree_idxs, powered_priorities)

    def save(self, path):
        state_dict = {
            "state": self.state[: self.size].cpu().numpy(),
            "action": self.action[: self.size].cpu().numpy(),
            "reward": self.reward[: self.size].cpu().numpy(),
            "next_state": self.next_state[: self.size].cpu().numpy(),
            "done": self.done[: self.size].cpu().numpy(),
            "ptr": self.ptr,
            "size": self.size,
            "sum_tree": self.sum_tree.cpu().numpy(),
            "min_tree": self.min_tree.cpu().numpy(),
            "max_priority": self.max_priority.cpu().item(),
        }
        if not path.endswith(".npz"):
            path += ".npz"
        np.savez_compressed(path, **state_dict)

    def load(self, path):
        if not super().load(path):
            return False

        if not path.endswith(".npz"):
            path += ".npz"
        data = np.load(path)

        if "sum_tree" in data:
            self.sum_tree = torch.as_tensor(data["sum_tree"], device=self.device)
            self.min_tree = torch.as_tensor(data["min_tree"], device=self.device)
            self.max_priority = torch.tensor(
                float(data["max_priority"]), device=self.device, dtype=torch.float32
            )

        return True
