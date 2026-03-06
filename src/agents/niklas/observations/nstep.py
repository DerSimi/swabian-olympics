from collections import deque

import numpy as np


class VectorizedNStepCollector:
    """
    Keep n steps to propagate rewards faster for learning complexer long term strategies

    N-Step Return Target:
    R_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... + γⁿ·max_a Q(s_{t+n}, a)
    """

    def __init__(self, n_step: int = 1, gamma: float = 0.99):
        self.n_step = n_step
        self.gamma = gamma
        self.buffers = {}

    def step(self, s_batch, a_batch, r_batch, ns_batch, d_batch):
        """Add new step and writ the oldest to buffer with propagated reward"""
        if self.n_step == 1:
            return (
                s_batch,
                a_batch,
                r_batch.reshape(-1, 1),
                ns_batch,
                d_batch.reshape(-1, 1),
            )

        n_envs = s_batch.shape[0]
        ret_s, ret_a, ret_r, ret_ns, ret_d = [], [], [], [], []

        for i in range(n_envs):
            if i not in self.buffers:
                self.buffers[i] = deque(maxlen=self.n_step)

            self.buffers[i].append(
                (
                    s_batch[i],
                    a_batch[i],
                    float(r_batch[i].item()),
                    ns_batch[i],
                    bool(d_batch[i].item()),
                )
            )

            if len(self.buffers[i]) == self.n_step:
                s_0, a_0, _, _, _ = self.buffers[i][0]
                R = 0.0
                ns_N = self.buffers[i][-1][3]
                d_N = self.buffers[i][-1][4]

                for k in range(self.n_step):
                    _, _, r_k, ns_k, d_k = self.buffers[i][k]
                    R += r_k * (self.gamma**k)
                    if d_k:
                        ns_N = ns_k
                        d_N = True
                        break

                ret_s.append(s_0)
                ret_a.append(a_0)
                ret_r.append(R)
                ret_ns.append(ns_N)
                ret_d.append(d_N)

            # If environment done, flush the remaining buffer elements
            if d_batch[i]:
                while len(self.buffers[i]) > 0:
                    s_0, a_0, _, _, _ = self.buffers[i][0]
                    R = 0.0
                    ns_N = self.buffers[i][-1][3]
                    d_N = True
                    for k in range(len(self.buffers[i])):
                        _, _, r_k, ns_k, d_k = self.buffers[i][k]
                        R += r_k * (self.gamma**k)
                        if d_k:
                            ns_N = ns_k
                            break

                    if len(self.buffers[i]) < self.n_step:
                        ret_s.append(s_0)
                        ret_a.append(a_0)
                        ret_r.append(R)
                        ret_ns.append(ns_N)
                        ret_d.append(d_N)

                    self.buffers[i].popleft()
                self.buffers[i].clear()

        if len(ret_s) > 0:
            return (
                np.array(ret_s, dtype=np.float32),
                np.array(ret_a, dtype=np.float32),
                np.array(ret_r, dtype=np.float32).reshape(-1, 1),
                np.array(ret_ns, dtype=np.float32),
                np.array(ret_d, dtype=np.float32).reshape(-1, 1),
            )
        return None
