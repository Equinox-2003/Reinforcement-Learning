import random
from collections.abc import Sequence
from typing import Any
import numpy as np
import gymnasium as gym
import torch
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.nn import functional as F
from torch.optim import Adam

STATE_SIZE = 4
ACTION_SIZE = 2

def compute_reward_to_go(rewards: Sequence[float], gamma: float) -> list[float]:
    """反向递推并返回 [G_0, G_1, ..., G_T]。"""
    returns = [0.0] * len(rewards)
    reward_to_go = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        reward_to_go = float(rewards[index]) + gamma * reward_to_go
        returns[index] = reward_to_go
    return returns

class Policy(nn.Module):
    def __init__(
        self,
        state_size: int = STATE_SIZE,
        action_size: int = ACTION_SIZE,
    ) -> None:
        super().__init__()
        self.l1 = nn.Linear(state_size, 128)
        self.l2 = nn.Linear(128, action_size)

    def forward(self, states: Tensor) -> Tensor:
        hidden = F.relu(self.l1(states))
        logits = self.l2(hidden)
        return F.softmax(logits, dim=-1)


class Agent:
    """以每个时间步的 reward-to-go 更新策略的 REINFORCE 智能代理。"""

    def __init__(
        self,
        state_size: int = STATE_SIZE,
        action_size: int = ACTION_SIZE,
        *,
        gamma: float = 0.98,
        lr: float = 0.0002,
        device: str | torch.device = "cpu",
    ) -> None:
        self.gamma = gamma
        self.device = torch.device(device)
        self.memory: list[tuple[float, Tensor]] = []
        self.pi = Policy(state_size, action_size).to(self.device)
        self.optimizer = Adam(self.pi.parameters(), lr=lr)

    def get_action(self, state: np.ndarray) -> tuple[int, Tensor]:
        """按 π(a|s) 采样动作，并返回仍带计算图的 log π(a|s)。"""
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        probabilities = self.pi(state_tensor)[0]
        distribution = Categorical(probs=probabilities)
        action_tensor = distribution.sample()

        # REINFORCE 在回合末更新，不能提前切断这条计算图。
        log_prob = distribution.log_prob(action_tensor)
        return int(action_tensor.item()), log_prob

    def add(self, reward: float, log_prob: Tensor) -> None:
        """保存本时间步的奖励与动作对数概率。"""
        self.memory.append((float(reward), log_prob))

    def compute_loss(self) -> Tensor:
        """构造 -Σ G_t log π(A_t|S_t)。"""
        if not self.memory:
            raise RuntimeError("轨迹为空，无法计算 REINFORCE 损失。")

        rewards = [reward for reward, _ in self.memory]
        log_probs = torch.stack([log_prob for _, log_prob in self.memory])
        returns = torch.as_tensor(
            compute_reward_to_go(rewards, self.gamma),
            dtype=log_probs.dtype,
            device=log_probs.device,
        )

        # REINFORCE：每个 log_prob 都乘自己的 G_t
        return -(returns * log_probs).sum()

    def update(self) -> float:
        """在回合结束后更新策略，并清空本回合数据。"""
        loss = self.compute_loss()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.memory.clear()
        return float(loss.detach().item())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(
    env: Any,
    agent: Agent,
    *,
    episodes: int = 3000,
    seed: int | None = None,
    log_interval: int | None = None,
) -> list[float]:
    """运行完整的回合式训练循环并返回每回合总奖励。"""
    if episodes < 1:
        raise ValueError("episodes 必须大于 0。")
    if seed is not None:
        set_seed(seed)

    reward_history: list[float] = []
    for episode in range(episodes):
        reset_seed = seed if episode == 0 else None
        state, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, log_prob = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.add(reward, log_prob)
            state = next_state
            total_reward += reward

        agent.update()
        reward_history.append(total_reward)
        if log_interval and episode % log_interval == 0:
            print(f"episode: {episode}, total reward: {total_reward:.1f}")

    return reward_history


def plot_reward_history(reward_history: Sequence[float]) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    axis.set_xlabel("Episode")
    axis.set_ylabel("Total Reward")
    axis.plot(range(len(reward_history)), reward_history)
    figure.tight_layout()
    plt.show() 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
env = gym.make('CartPole-v1')
agent = Agent(device=device)
episodes = 3000
seed = 42
log_interval=100

set_seed(seed)

reward_his = train(env, agent, episodes=episodes, seed=seed, log_interval=log_interval)
plot_reward_history(reward_his)
