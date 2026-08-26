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


class PolicyNet(nn.Module):
    """Actor：输出离散动作概率的 4 -> 128 -> 2 网络。"""

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


class ValueNet(nn.Module):
    """Critic：为每个状态输出一个标量价值。"""

    def __init__(self, state_size: int = STATE_SIZE) -> None:
        super().__init__()
        self.l1 = nn.Linear(state_size, 128)
        self.l2 = nn.Linear(128, 1)

    def forward(self, states: Tensor) -> Tensor:
        hidden = F.relu(self.l1(states))
        return self.l2(hidden)


class ActorCriticAgent:
    """使用一步 TD error 连接 Actor 与 Critic 的智能代理。"""

    def __init__(
        self,
        state_size: int = STATE_SIZE,
        action_size: int = ACTION_SIZE,
        *,
        gamma: float = 0.98,
        lr_pi: float = 0.0002,
        lr_v: float = 0.0005,
        device: str | torch.device = "cpu",
    ) -> None:
        self.gamma = gamma
        self.device = torch.device(device)
        self.pi = PolicyNet(state_size, action_size).to(self.device)
        self.v = ValueNet(state_size).to(self.device)
        self.optimizer_pi = Adam(self.pi.parameters(), lr=lr_pi)
        self.optimizer_v = Adam(self.v.parameters(), lr=lr_v)

    def get_action(self, state: np.ndarray) -> tuple[int, Tensor]:
        """按 Actor 的概率分布采样动作并返回动作的 log_prob。"""
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        probabilities = self.pi(state_tensor)[0]
        distribution = Categorical(probs=probabilities)
        action_tensor = distribution.sample()
        log_prob = distribution.log_prob(action_tensor)
        return int(action_tensor.item()), log_prob

    def compute_td_target(
        self,
        reward: float,
        next_state: np.ndarray,
        bootstrap_stop: bool,
    ) -> Tensor:
        """计算不参与反向传播的一步 TD target。"""
        reward_tensor = torch.tensor(
            [[float(reward)]], dtype=torch.float32, device=self.device
        )
        if bootstrap_stop:
            # 终止状态没有未来价值，也不应读取可能无意义的 next_state。
            return reward_tensor

        next_state_tensor = torch.as_tensor(
            next_state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            # target 对应 DeZero 的 target.unchain()，不能训练 V(s')。
            next_value = self.v(next_state_tensor)
            target = reward_tensor + self.gamma * next_value
        return target

    def compute_losses(
        self,
        state: np.ndarray,
        log_prob: Tensor,
        reward: float,
        next_state: np.ndarray,
        bootstrap_stop: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """分别构造 Actor loss、Critic loss 和 TD target。"""
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        target = self.compute_td_target(reward, next_state, bootstrap_stop)
        value = self.v(state_tensor)
        td_error = target - value

        value_loss = F.mse_loss(value, target)
        # delta 只负责给 Actor 的动作打分，Actor loss 不能反向修改 Critic。
        policy_loss = -log_prob * td_error.detach().squeeze()
        return policy_loss, value_loss, target

    def update(
        self,
        state: np.ndarray,
        log_prob: Tensor,
        reward: float,
        next_state: np.ndarray,
        bootstrap_stop: bool,
    ) -> tuple[float, float]:
        """用一次转移同时更新 Actor 和 Critic。"""
        policy_loss, value_loss, _ = self.compute_losses(
            state, log_prob, reward, next_state, bootstrap_stop
        )

        self.optimizer_pi.zero_grad(set_to_none=True)
        self.optimizer_v.zero_grad(set_to_none=True)
        value_loss.backward()
        policy_loss.backward()
        self.optimizer_v.step()
        self.optimizer_pi.step()

        return (
            float(policy_loss.detach().item()),
            float(value_loss.detach().item()),
        )

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(
    env: Any,
    agent: ActorCriticAgent,
    *,
    episodes: int = 3000,
    seed: int | None = None,
    log_interval: int | None = None,
) -> list[float]:
    """运行逐时间步更新的 Actor-Critic 训练循环。"""
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

            agent.update(
                state,
                log_prob,
                reward,
                next_state,
                done,
            )
            state = next_state
            total_reward += reward

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
agent = ActorCriticAgent(device=device)
episodes = 3000
seed = 42
log_interval=100

set_seed(seed)

reward_his = train(env, agent, episodes=episodes, seed=seed, log_interval=log_interval)
plot_reward_history(reward_his)
