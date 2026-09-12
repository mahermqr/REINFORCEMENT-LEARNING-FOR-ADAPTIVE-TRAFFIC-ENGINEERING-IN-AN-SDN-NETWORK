import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import os

class Actor(nn.Module):
    """
    Actor Network: Maps State -> Continuous Action in [-1.0, 1.0].
    Action < 0: Allow packet flow (benign).
    Action >= 0: Drop/Rate-limit flow (malicious/DDoS).
    """
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Tanh() # Continuous action bounded in [-1, 1]
        )

    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    """
    Critic Network: Evaluates State-Action value Q(s, a).
    """
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_layer = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )
        self.action_layer = nn.Sequential(
            nn.Linear(action_size, 64),
            nn.ReLU()
        )
        self.joint_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, action):
        s_feat = self.state_layer(state)
        a_feat = self.action_layer(action)
        joint = torch.cat([s_feat, a_feat], dim=1)
        return self.joint_layer(joint)


class DDPGSecurityAgent:
    """
    Deep Deterministic Policy Gradient (DDPG) Agent for Real-Time DDoS Detection and Mitigation.
    Employs Actor-Critic architecture with continuous control for fine-grained flow rate policing.
    Supports automatic GPU/CPU device acceleration.
    """
    def __init__(self, state_size=5, action_size=1, actor_lr=0.001, critic_lr=0.002,
                 gamma=0.95, tau=0.005, memory_size=5000, noise_std=0.2, noise_decay=0.999):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.noise_min = 0.02
        self.noise_decay = noise_decay
        self.memory = deque(maxlen=memory_size)
        
        # Device auto-detection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Primary Actor and Critic
        self.actor = Actor(state_size, action_size).to(self.device)
        self.critic = Critic(state_size, action_size).to(self.device)
        
        # Target Actor and Critic
        self.target_actor = Actor(state_size, action_size).to(self.device)
        self.target_critic = Critic(state_size, action_size).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.eval()
        self.target_critic.eval()
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=1e-4)
        self.criterion = nn.SmoothL1Loss()
        
        self.loss_history = []

    def act(self, state, add_noise=True):
        """Generates continuous action with exploratory Gaussian perturbation."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        self.actor.train()
        
        if add_noise:
            noise = np.random.normal(0, self.noise_std, size=self.action_size)
            action = np.clip(action + noise, -1.0, 1.0)
            if self.noise_std > self.noise_min:
                self.noise_std *= self.noise_decay
                
        return float(action[0]) if self.action_size == 1 else action

    def remember(self, state, action, reward, next_state, done):
        """Stores transition in replay buffer."""
        action_val = [action] if isinstance(action, (int, float)) else list(action)
        self.memory.append((
            np.array(state, dtype=np.float32),
            np.array(action_val, dtype=np.float32),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done)
        ))

    def train(self, batch_size=32):
        """Trains Actor and Critic networks on a sampled mini-batch."""
        if len(self.memory) < batch_size:
            return None

        minibatch = random.sample(self.memory, batch_size)

        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(self.device)
        actions = torch.FloatTensor(np.array([t[1] for t in minibatch])).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in minibatch]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(self.device)
        dones = torch.FloatTensor([t[4] for t in minibatch]).unsqueeze(1).to(self.device)

        # 1. Critic Update
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, next_actions)
            expected_q = rewards + (1.0 - dones) * self.gamma * target_q

        current_q = self.critic(states, actions)
        critic_loss = self.criterion(current_q, expected_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # 2. Actor Update (Policy Gradient: maximize Q(s, mu(s)))
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # 3. Polyak Soft Target Network Updates
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        loss_val = (critic_loss.item(), actor_loss.item())
        self.loss_history.append(loss_val)
        return loss_val

    def save(self, filepath):
        """Saves Actor and Critic checkpoints."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'noise_std': self.noise_std
        }, filepath)

    def load(self, filepath):
        """Loads checkpoints."""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
            self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            self.noise_std = checkpoint.get('noise_std', 0.05)
            return True
        return False
