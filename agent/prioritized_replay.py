import numpy as np
import random
import torch

class SumTree:
    """
    Binary SumTree data structure for O(log N) prioritized experience sampling.
    Optimized with iterative propagation and retrieval to eliminate Python recursion overhead.
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
        """Iterative priority propagation to tree root."""
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def _retrieve(self, idx, s):
        """Iterative binary search traversal down to leaf node."""
        tree = self.tree
        tree_len = len(tree)
        while True:
            left = 2 * idx + 1
            if left >= tree_len:
                break
            if s <= tree[left]:
                idx = left
            else:
                s -= tree[left]
                idx = left + 1
        return idx

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
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Proportional Prioritized Experience Replay (PER) Buffer.
    P(i) = p_i^alpha / sum(p_k^alpha)
    w_i = (N * P(i))^(-beta) / max(w_j)
    Optimized with contiguous NumPy arrays for zero-copy PyTorch tensor sampling.
    """
    def __init__(self, capacity=5000, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=0.01, state_size=10):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.state_size = state_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0

        # Contiguous NumPy buffers for ultra-fast vectorized tensor conversion
        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_size), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        s = np.asarray(state, dtype=np.float32)
        ns = np.asarray(next_state, dtype=np.float32)
        ptr = self.tree.write_ptr

        # Ensure buffer matches state dimensionality
        if self.states.shape[1] != len(s):
            self.state_size = len(s)
            self.states = np.zeros((self.capacity, self.state_size), dtype=np.float32)
            self.next_states = np.zeros((self.capacity, self.state_size), dtype=np.float32)

        self.states[ptr] = s
        self.actions[ptr] = int(action)
        self.rewards[ptr] = float(reward)
        self.next_states[ptr] = ns
        self.dones[ptr] = float(done)

        # Maintain tuple in tree for full backward compatibility
        transition = (s, int(action), float(reward), ns, bool(done))
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

        offsets = np.arange(batch_size, dtype=np.float32) * segment
        targets = offsets + np.random.uniform(0.0, segment, size=batch_size).astype(np.float32)

        for s in targets:
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

    def sample_tensors(self, batch_size=32, device='cpu'):
        """Directly returns contiguous PyTorch tensors for state-of-the-art training performance."""
        if self.tree.n_entries == 0:
            return None

        idxs = []
        priorities = []
        data_idxs = []
        total_p = max(1e-5, float(self.tree.total()))
        segment = total_p / batch_size

        self.beta = min(1.0, self.beta + self.beta_increment)

        offsets = np.arange(batch_size, dtype=np.float32) * segment
        targets = offsets + np.random.uniform(0.0, segment, size=batch_size).astype(np.float32)

        for s in targets:
            idx, priority, _ = self.tree.get(s)
            d_idx = idx - self.capacity + 1
            if d_idx < 0 or d_idx >= self.tree.n_entries:
                d_idx = random.randint(0, max(0, self.tree.n_entries - 1))
                idx = d_idx + self.capacity - 1
                priority = self.max_priority
            idxs.append(idx)
            priorities.append(priority)
            data_idxs.append(d_idx)

        d_idxs = np.array(data_idxs, dtype=np.int64)
        states = torch.as_tensor(self.states[d_idxs], device=device)
        actions = torch.as_tensor(self.actions[d_idxs], device=device).unsqueeze(1)
        rewards = torch.as_tensor(self.rewards[d_idxs], device=device)
        next_states = torch.as_tensor(self.next_states[d_idxs], device=device)
        dones = torch.as_tensor(self.dones[d_idxs], device=device)

        sampling_probabilities = np.array(priorities, dtype=np.float32) / total_p
        is_weights = np.power(max(1, self.tree.n_entries) * sampling_probabilities, -self.beta)
        max_w = is_weights.max()
        if max_w > 1e-5:
            is_weights /= max_w
        else:
            is_weights = np.ones_like(is_weights)
        weights = torch.as_tensor(is_weights, device=device)

        return states, actions, rewards, next_states, dones, idxs, weights

    def update_priorities(self, idxs, td_errors):
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        max_p = float(np.max(priorities))
        if max_p > self.max_priority:
            self.max_priority = max_p
        for idx, priority in zip(idxs, priorities):
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.n_entries

