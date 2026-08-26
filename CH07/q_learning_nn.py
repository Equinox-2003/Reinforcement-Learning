from pathlib import Path
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from common.gridworld import GridWorld

HEIGHT = 3
WIDTH = 4
STATE_SIZE = HEIGHT * WIDTH
ACTION_SIZE = 4
EPISODES = 1000
MAX_STEPS_PER_EPISODE = 10_000

def one_hot(state: Tuple[int, int]) -> torch.Tensor:
    vector = torch.zeros((1, STATE_SIZE), dtype=torch.float32)
    y, x = state
    vector[0, WIDTH * y + x] = 1.0
    return vector

class QNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(STATE_SIZE, 100)
        self.l2 = nn.Linear(100, ACTION_SIZE)

    def forward(self, state_vector: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.l1(state_vector))
        return self.l2(hidden)

class QLearningAgent:

    def __init__(self) -> None:
        self.gamma = 0.9
        self.lr = 0.01
        self.epsilon = 0.1
        self.action_size = ACTION_SIZE
        self.qnet = QNet()
        self.optimizer = torch.optim.SGD(self.qnet.parameters(), lr=self.lr)

    def get_action(self, state_vector: torch.Tensor) -> int:
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(self.action_size))

        with torch.no_grad():
            qs = self.qnet(state_vector)
        return int(qs.argmax(dim=1).item())

    def compute_target(
        self,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> torch.Tensor:
        with torch.no_grad():
            if done:
                next_q = torch.zeros(
                    1,
                    dtype=next_state.dtype,
                    device=next_state.device,
                )
            else:
                next_q = self.qnet(next_state).max(dim=1).values
            return reward + self.gamma * next_q

    def update(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> float:
        target = self.compute_target(reward, next_state, done)

        qs = self.qnet(state)
        q = qs[:, action]
        loss = F.mse_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

def main() -> None:
    np.random.seed(0)
    torch.manual_seed(0)

    env = GridWorld()
    agent = QLearningAgent()
    loss_history = []

    for _ in range(EPISODES):
        state = one_hot(env.reset())
        total_loss, cnt = 0.0, 0
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            next_state = one_hot(next_state)

            loss = agent.update(state, action, float(reward), next_state, done)
            total_loss += loss
            cnt += 1
            state = next_state

            if cnt >= MAX_STEPS_PER_EPISODE and not done:
                raise RuntimeError(
                    "An episode exceeded MAX_STEPS_PER_EPISODE without "
                    "reaching the goal."
                )

        loss_history.append(total_loss / cnt)

    plt.xlabel("episode")
    plt.ylabel("loss")
    plt.plot(range(len(loss_history)), loss_history)
    plt.show()

    q_values = {}
    with torch.no_grad():
        for state in env.states():
            for action in env.action_space:
                q = agent.qnet(one_hot(state))[:, action]
                q_values[state, action] = float(q.item())
    env.render_q(q_values)


if __name__ == "__main__":
    main()
