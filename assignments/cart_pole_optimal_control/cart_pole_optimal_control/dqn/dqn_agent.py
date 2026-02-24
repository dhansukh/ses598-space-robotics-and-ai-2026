import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


class QNetwork(nn.Module):
    """Deep Q-Network with 3 hidden layers for cart-pole control."""

    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    """DQN Agent with experience replay and target network."""

    def __init__(
        self,
        state_dim,
        action_dim,
        gamma=0.99,
        lr=1e-3,
        epsilon=1.0,
        min_epsilon=0.05,
        decay=0.9995,
        batch_size=64,
        memory_size=100000,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize Q-Network & Target Network
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

    def select_action(self, state, evaluate=False):
        """Selects action using epsilon-greedy strategy."""
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return torch.argmax(self.q_network(state_tensor)).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Current Q values
        q_values = self.q_network(states).gather(1, actions)

        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = nn.SmoothL1Loss()(q_values, target_q_values)

        # Optimize with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

        return loss.item()

    def update_target_model(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_model(self, filename="dqn_cartpole_trained.pth"):
        torch.save(self.q_network.state_dict(), filename)
        print(f"Model saved as {filename}")

    def load_model(self, filename="dqn_cartpole_trained.pth"):
        self.q_network.load_state_dict(
            torch.load(filename, map_location=self.device)
        )
        self.q_network.eval()
        print(f"Model loaded from {filename}")
