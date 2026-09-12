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

class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network for Multicast Tree Construction.
    Decomposes Q(s, a) into state Value V(s) and action Advantage A(s, a):
    Q(s, a) = V(s) + (A(s, a) - 1/|A| * sum(A(s, a')))
    """
    def __init__(self, state_size, action_size):
        super(DuelingDQN, self).__init__()
        
        # Shared feature representation layers
        self.feature = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        # Advantage stream: A(s, a)
        self.advantage = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        
        # Value stream: V(s)
        self.value = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        features = self.feature(x)
        adv = self.advantage(features)
        val = self.value(features)
        return val + adv - adv.mean(dim=1, keepdim=True)


class DQNMulticastAgent:
    """
    Dueling Double DQN Agent for Multicast Tree Construction in SDN.
    Optimizes branch selection and Steiner Tree heuristic weighting across
    multicast destination sets.
    Supports Prioritized Experience Replay (PER) and automatic GPU/CPU device acceleration.
    """
    def __init__(self, state_size=50, action_size=10, lr=0.001, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, memory_size=5000, use_per=True):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = lr
        self.use_per = use_per
        
        if self.use_per:
            self.memory = PrioritizedReplayBuffer(capacity=memory_size)
        else:
            self.memory = deque(maxlen=memory_size)
        
        # Automatic device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize Dueling DQN policy and target networks
        self.model = DuelingDQN(state_size, action_size).to(self.device)
        self.target_model = DuelingDQN(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss(reduction='none' if use_per else 'mean')
        
        self.update_target_counter = 0
        self.target_update_freq = 10
        self.loss_history = []

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
        """Appends experience tuple to replay buffer."""
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
        """Trains Dueling DQN network using Double DQN updates."""
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

        # Current Q-values
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

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Target network update
        self.update_target_counter += 1
        if self.update_target_counter % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss_val

    def save(self, filepath):
        """Saves agent weights."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)

    def load(self, filepath):
        """Loads agent weights."""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
            return True
        return False
