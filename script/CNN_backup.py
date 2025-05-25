import torch.nn as nn

import torch.optim as optim


from typing import NamedTuple, Tuple, Union
from collections import deque
import copy
import random
import numpy as np

import torch


# Action Space :Discrete(4)
# Observation Space :Boinput(0, 255, (210, 160, 3), uint8)
# (4, 84, 84)


class network(nn.Module):

    def __init__(
        self,
        actionspace,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__()
        self.device = device

        self.conv = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=8, stride=4),  # input: (4,84,84)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
        ).to(self.device)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9 * 9 * 32, 256),
            nn.ReLU(),
            nn.Linear(256, actionspace),
        ).to(self.device)

    def preprocess(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        obs: (B, C, H, W) か (C, H, W) の Tensor かnp.ndarray
        出力: (B, C, H, W) の float32 Tensor（0-1 正規化済み）
        """
        if isinstance(obs, np.ndarray):
            x = obs.astype(np.float32)
            if x.ndim == 3:
                x = x[None, ...]  # バッチ次元追加

            elif x.ndim == 4:
                pass
            else:
                raise ValueError(f"Unexpected ndarray ndim: {x.ndim}")
            t = torch.from_numpy(x)
        elif isinstance(obs, torch.Tensor):
            t = obs  # assume already BCHW or CHW
            if t.ndim == 3:
                t = t.unsqueeze(0)
        else:
            raise TypeError("obs must be np.ndarray or torch.Tensor")

        return t.to(self.device) / 255.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv(x)
        # x = x.reshape(x.shape[0], -1)
        return self.fc(x)


class experience(NamedTuple):
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    done: torch.Tensor


class replay_buffer:

    def __init__(self, batch_size=1, buffer_size=1000):
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(
        self,
        state: Union[np.ndarray, Tuple],
        action: int,
        reward: float,
        next_state: Union[np.ndarray, Tuple],
        done,
    ):
        self.buffer.append(((state), (action), reward, (next_state), done))

    def get(self) -> experience:
        # ランダムに batch_size 件の経験をサンプリング
        batch = random.sample(self.buffer, self.batch_size)

        # NumPy 配列としてまとめる（HWC のまま）
        states = np.stack([x[0] for x in batch], axis=0).astype(
            np.uint8
        )  # (B, C,H, W,)
        actions = np.array([x[1] for x in batch], dtype=np.int64)  # (B,)
        rewards = np.array([x[2] for x in batch], dtype=np.float32)  # (B,)
        next_states = np.stack([x[3] for x in batch], axis=0).astype(
            np.uint8
        )  # (B, C,H, W,)
        dones = np.array([x[4] for x in batch], dtype=np.bool_)  # (B,)

        # NumPy 配列を Tensor に変換
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)
        

        return experience(states, actions, rewards, next_states, dones)


class agent:

    def __init__(self, action_space, timesteps):
        self.action_network = network(action_space)
        self.target_network = network(action_space)
        self.action_space = action_space
        self.max_buffer = 1000000
        self.batch_size = 32
        self.replay = replay_buffer(
            batch_size=self.batch_size, buffer_size=self.max_buffer
        )

        self.data = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"device : {self.device}")
        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        self.epsilon_decay = (self.epsilon_start - self.epsilon_end) / 250000
        self.epsilon = self.epsilon_start
        self.gamma = 0.99
        self.lr = 2.5 * 1e-4

        self.net_sync()
        self.optimizer = optim.Adam(self.action_network.parameters(), self.lr)

    def set_paramaters(
        self,
        batch_size=32,
        buffer_size=1000000,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_steps=250000,
        gamma=0.99,
        lr=2.5 * 1e-4,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay = (
            self.epsilon_start - self.epsilon_end
        ) / epsilon_decay_steps
        # print (f"epsilon decay : {self.epsilon_decay}")
        self.epsilon = self.epsilon_start

        self.gamma = gamma
        self.lr = lr
        self.device = device

        return

    def net_sync(self):

        self.target_network = copy.deepcopy(self.action_network)
        return

    def update(self):
        if len(self.replay.buffer) < self.batch_size:
            return

        self.data = self.replay.get()

        states = self.action_network.preprocess(self.data.state)  # (B, C, H, W) Tensor
        next_states = self.action_network.preprocess(self.data.next_state)

        q_c = self.action_network(states)
        q = q_c[torch.arange(self.batch_size), self.data.action]

        with torch.no_grad():
            next_q_c = self.target_network(next_states)
            next_q = next_q_c.max(1)[0]
            # ── ここで reward, done を Tensor に変換 ──
            rewards = torch.from_numpy(self.data.reward).float().to(self.device)  # (B,)
            dones = torch.from_numpy(self.data.done.astype(np.float32)).to(
                self.device
            )  # (B,)

            non_final = 1.0 - dones  # Tensor (B,)
            target = rewards + self.gamma * non_final * next_q  # Tensor (B,)

        loss_function = nn.MSELoss()
        loss = loss_function(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return

    def add(self, observation, action, reward, next_observation, terminated):
        self.replay.add(observation, action, reward, next_observation, terminated)

    def set_epsilon(self) -> None:
        self.epsilon -= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_end)
        self.epsilon = min(self.epsilon, self.epsilon_start)

    def save_model(self, path="model.pth") -> None:
        torch.save(self.action_network.state_dict(), path)

    def select_action(self, observation: np.ndarray) -> int:
        # 前処理して Tensor 化
        state = self.action_network.preprocess(observation)  # -> (1,C,H,W) Tensor
        self.set_epsilon()
        # ε-greedy
        if random.random() < self.epsilon:
            return random.randrange(self.action_space)

        with torch.no_grad():
            q_values = self.action_network(state)  # shape (1, n_actions)
        # 最大 Q 値のインデックスを Python int にして返す
        return q_values.argmax(dim=1).item()

    def load_model(self, model_path):
        self.action_network.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
