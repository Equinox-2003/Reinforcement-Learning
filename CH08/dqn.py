import random
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Adam

import matplotlib.pyplot as plt
import gymnasium as gym

from replay_buffer import ReplayBuffer

STATE_SIZE = 4
ACTION_SIZE = 2


class QNet(nn.Module):
    def __init__(self, state_size: int = STATE_SIZE, action_size: int = ACTION_SIZE) -> None:
        super().__init__()
        self.l1 = nn.Linear(state_size, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_size)

    def forward(self, states: Tensor) -> Tensor:
        hidden = F.relu(self.l1(states))
        hidden = F.relu(self.l2(hidden))
        return self.l3(hidden)


class DQNAgent:
    def __init__(
        self,
        state_size: int = STATE_SIZE,
        action_size: int = ACTION_SIZE,
        *,
        gamma: float = 0.98,
        lr: float = 0.0005,
        epsilon: float = 0.1,
        buffer_size: int = 10_000,
        batch_size: int = 32,
        device: torch.device | str = "cpu",
    ) -> None:
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.action_size = action_size
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)
        self.qnet = QNet(state_size, action_size).to(self.device)
        self.qnet_target = QNet(state_size, action_size).to(self.device)
        self.qnet_target.eval()
        for parameter in self.qnet_target.parameters():
            parameter.requires_grad_(False)
        self.optimizer = Adam(self.qnet.parameters(), lr=self.lr)

    def get_action(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(self.action_size))

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)
        with torch.no_grad():
            qs = self.qnet(state_tensor)
        return int(qs.argmax(dim=1).item())

    def compute_target(self, rewards: Tensor, next_states: Tensor, dones: Tensor) -> Tensor:
        """补充：计算目标 Q 值"""
        with torch.no_grad():
            next_qs = self.qnet_target(next_states)
            max_next_qs = next_qs.max(dim=1).values
            target = rewards + (1.0 - dones.float()) * self.gamma * max_next_qs
        return target

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> float | None:
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.get_batch()
        
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        all_qs = self.qnet(states)
        current_q = all_qs.gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)
        target = self.compute_target(rewards, next_states, dones)
        loss = F.mse_loss(current_q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def sync_qnet(self) -> None:
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        self.qnet_target.eval()


iters = 100
episodes = 300
sync_interval = 20

env = gym.make('CartPole-v1') 
# env = gym.make('CartPole-v1', render_mode='human') 
agent = DQNAgent(device= 'cuda' if torch.cuda.is_available() else 'cpu')

avg_history = [0] * episodes
for i in range(iters):
    for episode in range(episodes):
        state, _ = env.reset()  
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated  
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward

        if episode % sync_interval == 0:
            agent.sync_qnet()

        avg_history[episode] += total_reward

for i, x in enumerate(avg_history):
    avg_history[i] = x / 100

env.close()

figure, axis = plt.subplots()
axis.set_xlabel("Episode")
axis.set_ylabel("Total Reward")
axis.plot(range(len(avg_history)), avg_history)
figure.tight_layout()
plt.show()  
