import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
import os
try:
    from prioritized_replay import PrioritizedReplayBuffer
except ImportError:
    from agent.prioritized_replay import PrioritizedReplayBuffer

class DuelingQNetwork(nn.Module):
    """
    Dueling Q-Network architecture for Adaptive SDN Unicast Routing.
    Decouples state-value V(s) from candidate-path advantages A(s, a),
    enabling sharp policy differentiation in dense traffic congestion.
    """
    def __init__(self, state_size=10, action_size=4):
        super(DuelingQNetwork, self).__init__()
        self.feature_network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, state):
        features = self.feature_network(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        return values + (advantages - advantages.mean(dim=-1, keepdim=True))

class DQNRoutingAgent:
    """
    Dueling Double Deep Q-Network (D3QN) Agent for Adaptive SDN Unicast Routing.
    Selects optimal routing paths among K-candidate paths based on
    real-time network state (link latency, bandwidth utilization, topology features).
    Supports Prioritized Experience Replay (PER), Polyak target updates, and GPU/CPU acceleration.
    """
    def __init__(self, state_size=10, action_size=4, lr=0.001, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, memory_size=5000, use_per=True, tau=0.005):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = lr
        self.use_per = use_per
        self.tau = tau

        if self.use_per:
            self.memory = PrioritizedReplayBuffer(capacity=memory_size)
        else:
            self.memory = deque(maxlen=memory_size)

        # Automatic device selection (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build policy network and target network (Dueling Architecture)
        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss(reduction='none' if use_per else 'mean')

        self.update_target_counter = 0
        self.target_update_freq = 10
        self.loss_history = []

    def _build_model(self):
        """Constructs Dueling Q-Network with Layer Normalization."""
        return DuelingQNetwork(self.state_size, self.action_size)

    def act(self, state, explore=True):
        """Epsilon-greedy action selection."""
        if explore and random.uniform(0, 1) <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state_tensor)
        self.model.train()
        return torch.argmax(q_values[0]).item()

    def remember(self, state, action, reward, next_state, done):
        """Stores experience tuple in replay memory."""
        s = np.array(state, dtype=np.float32)
        a = int(action)
        r = float(reward)
        ns = np.array(next_state, dtype=np.float32)
        d = bool(done)

        if self.use_per:
            self.memory.add(s, a, r, ns, d)
        else:
            self.memory.append((s, a, r, ns, d))

    def train(self, batch_size=32):
        """Trains the policy network using Double DQN updates on a mini-batch."""
        if len(self.memory) < batch_size:
            return None

        if self.use_per:
            minibatch, idxs, is_weights = self.memory.sample(batch_size)
            weights_tensor = torch.FloatTensor(is_weights).to(self.device)
        else:
            minibatch = random.sample(self.memory, batch_size)
            idxs, weights_tensor = None, None

        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(self.device)
        actions = torch.LongTensor([t[1] for t in minibatch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in minibatch]).to(self.device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(self.device)
        dones = torch.FloatTensor([t[4] for t in minibatch]).to(self.device)

        # Current Q-values: Q(s, a; theta)
        current_q = self.model(states).gather(1, actions).squeeze(1)

        # Double DQN target computation
        with torch.no_grad():
            best_actions = self.model(next_states).argmax(1).unsqueeze(1)
            max_next_q = self.target_model(next_states).gather(1, best_actions).squeeze(1)
            expected_q = rewards + (1.0 - dones) * self.gamma * max_next_q

        if self.use_per:
            elementwise_loss = self.criterion(current_q, expected_q)
            loss = (elementwise_loss * weights_tensor).mean()
            td_errors = (current_q - expected_q).detach().cpu().numpy()
            self.memory.update_priorities(idxs, td_errors)
        else:
            loss = self.criterion(current_q, expected_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        loss_val = loss.item()
        self.loss_history.append(loss_val)

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Polyak soft target update for continuous smooth value target learning
        for target_p, p in zip(self.target_model.parameters(), self.model.parameters()):
            target_p.data.copy_(self.tau * p.data + (1.0 - self.tau) * target_p.data)

        # Periodic hard target network sync
        self.update_target_counter += 1
        if self.update_target_counter % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss_val

    def save(self, filepath):
        """Saves model checkpoint."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)

    def load(self, filepath):
        """Loads model checkpoint with backward compatibility for legacy weights."""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    try:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    except Exception:
                        pass
            except RuntimeError:
                # Gracefully load feature representation from legacy MLP checkpoint
                saved_sd = checkpoint['model_state_dict']
                if '0.weight' in saved_sd and hasattr(self.model, 'feature_network'):
                    with torch.no_grad():
                        if self.model.feature_network[0].weight.shape == saved_sd['0.weight'].shape:
                            self.model.feature_network[0].weight.copy_(saved_sd['0.weight'])
                            self.model.feature_network[0].bias.copy_(saved_sd['0.bias'])
                        if self.model.feature_network[3].weight.shape == saved_sd['3.weight'].shape:
                            self.model.feature_network[3].weight.copy_(saved_sd['3.weight'])
                            self.model.feature_network[3].bias.copy_(saved_sd['3.bias'])
                    self.target_model.load_state_dict(self.model.state_dict())
                else:
                    return False
            self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
            return True
        return False
