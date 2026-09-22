"""
Deep Q-Network (DQN) implementation in PyTorch.

This module intentionally avoids high-level RL libraries so every piece of
the algorithm is visible and editable for learning purposes.

Key components:
  - QNetwork     : a small fully-connected network that maps state -> Q(s,a)
  - ReplayBuffer : stores (s, a, r, s', terminated) transitions for replay
  - DQNAgent     : the training loop, epsilon-greedy policy, target-net sync
"""
import random
from collections import deque
from pathlib import Path
from typing import Self

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim

# ── Neural network ────────────────────────────────────────────────────


class QNetwork(nn.Module):
    """Maps a state to one Q-value per action.

    A fully-connected net, state_dim -> hidden -> hidden -> action_dim, with a
    ReLU after each hidden layer. forward() takes a batch of shape
    (B, state_dim) and returns (B, action_dim).
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128) -> None:
        super().__init__()
        # No activation on the output: these are Q-values, not probabilities.
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Replay buffer ────────────────────────────────────────────────────


class ReplayBuffer:
    """Fixed-size FIFO buffer that stores transitions for experience replay."""

    def __init__(self, capacity: int = 100_000) -> None:
        self.buffer: deque[tuple] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, terminated))

    def sample(self, batch_size: int) -> list[tuple]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


# ── Agent ─────────────────────────────────────────────────────────────


class DQNAgent:
    """
    Deep Q-Network agent implemented from scratch.

    Hyperparameters are intentionally exposed as constructor args so you
    can experiment with them directly.
    """

    def __init__(
        self,
        env_id: str,
        *,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_capacity: int = 100_000,
        target_update_freq: int = 10,
        hidden: int = 128,
        explore_repeat: int = 25,
    ) -> None:
        self.env_id = env_id
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.target_update_freq = target_update_freq
        self.hidden = hidden
        self.explore_repeat = explore_repeat
        self.training_episodes = 0

        # Exploration run state, reset each episode in train().
        self._explore_action = 0
        self._explore_steps_left = 0

        env = gym.make(env_id)
        self.state_dim = int(env.observation_space.shape[0])  # type: ignore[index]
        self.action_dim = int(env.action_space.n)  # type: ignore[attr-defined]
        env.close()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = QNetwork(self.state_dim, self.action_dim, hidden).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.action_dim, hidden).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer(buffer_capacity)

    # ── policy ────────────────────────────────────────────────────────

    def _greedy(self, state: np.ndarray) -> int:
        """Argmax over Q(s, .), with no exploration."""
        with torch.no_grad():
            t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.q_net(t).argmax(dim=1).item())

    def select_action(self, state: np.ndarray, *, deterministic: bool = False) -> int:
        """Epsilon-greedy with temporally correlated exploration.

        Drawing a fresh uniform action every step makes consecutive
        exploratory actions independent, which cannot produce the sustained
        runs this environment needs: 20 identical pushes in a row have
        probability (1/3)^20. An exploratory action is instead held for
        `explore_repeat` steps.

        `deterministic=True` is always pure greedy.
        """
        if deterministic:
            return self._greedy(state)

        # Continue the current run.
        if self._explore_steps_left > 0:
            self._explore_steps_left -= 1
            return self._explore_action

        if random.random() < self.epsilon:
            self._explore_action = random.randrange(self.action_dim)
            self._explore_steps_left = self.explore_repeat - 1
            return self._explore_action

        return self._greedy(state)

    def predict(self, obs: np.ndarray, *, deterministic: bool = True) -> tuple[int, None]:
        return self.select_action(obs, deterministic=deterministic), None

    # ── learning step ─────────────────────────────────────────────────

    def _tensor(self, x, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(np.array(x), dtype=dtype, device=self.device)

    def _learn(self) -> float:
        """Sample a mini-batch from the buffer and take one gradient step.

        Returns the batch loss value.
        """
        if len(self.buffer) < self.batch_size:
            return 0.0

        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, terminateds = zip(*batch)

        states_t = self._tensor(states)
        actions_t = self._tensor(actions, torch.int64).unsqueeze(1)
        rewards_t = self._tensor(rewards).unsqueeze(1)
        next_states_t = self._tensor(next_states)
        terminateds_t = self._tensor(terminateds).unsqueeze(1)

        # Q(s, a) from the online net, for the actions taken -> (B, 1).
        current_q = self.q_net(states_t).gather(1, actions_t)

        # max_a' Q(s', a') from the frozen target net, without gradient.
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1, keepdim=True).values

        # Bellman target. (1 - terminated) drops the bootstrap term on real
        # terminal states; the buffer stores `terminated`, not `truncated`.
        target_q = rewards_t + self.gamma * next_q * (1.0 - terminateds_t)

        loss = self.loss_fn(current_q, target_q)  # both sides are (B, 1)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    # ── training loop ─────────────────────────────────────────────────

    def train(self, total_episodes: int = 500, log_interval: int = 10) -> list[float]:
        env = gym.make(self.env_id)
        rewards_history: list[float] = []

        for episode in range(1, total_episodes + 1):
            obs, _ = env.reset()
            total_reward = 0.0
            done = False

            # Runs must not carry over between episodes.
            self._explore_action = 0
            self._explore_steps_left = 0

            while not done:
                action = self.select_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Store `terminated`, not `done`: hitting the 200-step time
                # limit is not a real terminal state, so we must keep
                # bootstrapping through it.
                self.buffer.push(obs, action, float(reward), next_obs, terminated)
                self._learn()

                obs = next_obs
                total_reward += reward

            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            self.training_episodes += 1
            rewards_history.append(total_reward)

            if episode % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

            if episode % log_interval == 0:
                avg = np.mean(rewards_history[-log_interval:])
                print(
                    f"Episode {episode}/{total_episodes} | "
                    f"Avg Reward: {avg:.2f} | "
                    f"Epsilon: {self.epsilon:.4f} | "
                    f"Buffer: {len(self.buffer)}"
                )

        env.close()
        return rewards_history

    # ── persistence ───────────────────────────────────────────────────

    _HPARAMS = (
        "env_id",
        "lr",
        "gamma",
        "epsilon_end",
        "epsilon_decay",
        "batch_size",
        "buffer_capacity",
        "target_update_freq",
        "hidden",
        "explore_repeat",
    )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: getattr(self, k) for k in self._HPARAMS}
        data["q_net_state"] = self.q_net.state_dict()
        data["optimizer_state"] = self.optimizer.state_dict()
        data["epsilon"] = self.epsilon
        data["training_episodes"] = self.training_episodes
        torch.save(data, path)
        print(f"Saved DQN agent to {path}")

    @classmethod
    def load(cls, path: Path) -> Self:
        data = torch.load(path, weights_only=False)
        agent = cls(
            data["env_id"],
            epsilon_start=data["epsilon"],
            **{k: data[k] for k in cls._HPARAMS if k != "env_id"},
        )
        # The target net starts as a copy of the online net; it re-syncs during
        # training anyway, so there is no need to persist it separately.
        agent.q_net.load_state_dict(data["q_net_state"])
        agent.target_net.load_state_dict(data["q_net_state"])
        agent.optimizer.load_state_dict(data["optimizer_state"])
        agent.training_episodes = data["training_episodes"]
        return agent

    def info(self) -> str:
        params = sum(p.numel() for p in self.q_net.parameters())
        return (
            f"DQN agent for {self.env_id}\n"
            f"  Episodes trained  : {self.training_episodes}\n"
            f"  Network params    : {params:,}\n"
            f"  Epsilon           : {self.epsilon:.4f}\n"
            f"  LR / Gamma        : {self.lr} / {self.gamma}\n"
            f"  Batch size        : {self.batch_size}\n"
            f"  Target update     : every {self.target_update_freq} episodes\n"
            f"  Device            : {self.device}"
        )
