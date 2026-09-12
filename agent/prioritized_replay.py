import numpy as np
import random

class SumTree:
    """
    Binary SumTree data structure for O(log N) prioritized experience sampling.
    Leaves store transition priorities p_i.
    Internal nodes store the sum of their child subtrees.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = np.zeros(capacity, dtype=object)
        self.write_ptr = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write_ptr + self.capacity - 1
        self.data[self.write_ptr] = data
        self.update(idx, priority)

        self.write_ptr = (self.write_ptr + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Proportional Prioritized Experience Replay (PER) Buffer.
    P(i) = p_i^alpha / sum(p_k^alpha)
    w_i = (N * P(i))^(-beta) / max(w_j)
    """
    def __init__(self, capacity=5000, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=0.01):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, transition)

    def sample(self, batch_size=32):
        if self.tree.n_entries == 0:
            return [], [], np.array([], dtype=np.float32)

        batch = []
        idxs = []
        priorities = []
        total_p = max(1e-5, float(self.tree.total()))
        segment = total_p / batch_size

        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            
            # Fallback if unpopulated leaf
            if data is None or isinstance(data, (int, float)):
                rand_idx = random.randint(0, max(0, self.tree.n_entries - 1))
                data = self.tree.data[rand_idx]
                priority = self.max_priority
                idx = rand_idx + self.capacity - 1

            idxs.append(idx)
            priorities.append(priority)
            batch.append(data)

        # Compute Importance Sampling (IS) weights
        sampling_probabilities = np.array(priorities, dtype=np.float32) / total_p
        is_weights = np.power(max(1, self.tree.n_entries) * sampling_probabilities, -self.beta)
        max_w = is_weights.max()
        if max_w > 1e-5:
            is_weights /= max_w
        else:
            is_weights = np.ones_like(is_weights)

        return batch, idxs, is_weights

    def update_priorities(self, idxs, td_errors):
        for idx, error in zip(idxs, td_errors):
            priority = (abs(error) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.n_entries
