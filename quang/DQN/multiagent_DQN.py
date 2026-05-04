import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from combine.common.common import ReplayBuffer

# Mạng DQN nhiều đầu (multi-head)
class MultiHeadDQN(nn.Module):
    def __init__(self, state_dim, K, I, hidden_dim=256):
        super().__init__()
        self.K = K
        self.I = I + 1   # +1 nếu có “bỏ trống PRB”
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, self.I) for _ in range(K)])

    def forward(self, state):
        z = self.shared(state)
        q_values = torch.stack([head(z) for head in self.heads], dim=1)
        # Output shape: [batch, K, I+1]
        return q_values


# Agent chính
class MultiHeadDQNAgent:
    def __init__(
        self,
        state_dim,
        num_PRB,
        num_slices,
        device="cpu",
        lr=1e-3,
        gamma=0.99,
        batch_size=64,
        eps_start=1.0,
        eps_end=0.1,
        eps_decay=0.995,
        target_update=10,
    ):
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.num_PRB = num_PRB
        self.num_slices = num_slices
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.learn_step = 0

        # Mạng Q chính & target
        self.policy_net = MultiHeadDQN(state_dim, self.num_PRB, self.num_slices).to(self.device)
        self.target_net = MultiHeadDQN(state_dim, self.num_PRB, self.num_slices).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(forSAC=False)

    def select_action(self, state, eval_mode=False):
        """
        Chọn hành động cho mỗi PRB.
        - Nếu eval_mode=False: dùng epsilon-greedy (random exploration)
        - Nếu eval_mode=True: chỉ khai thác (greedy)
        """
        if not eval_mode and random.random() < self.eps:
            # exploration: chọn ngẫu nhiên 1 slice cho mỗi PRB
            return np.random.randint(0, self.num_slices + 1, size=self.num_PRB)

        # exploitation (greedy)
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_t)  # có thể trả về (1, K, I+1) hoặc (K, I+1)

            # Nếu mạng trả về thêm batch dimension, bóc ra
            if q_values.ndim == 3:
                q_values = q_values[0]

            # Chọn slice có Q-value cao nhất cho từng PRB
            action = torch.argmax(q_values, dim=1).cpu().numpy()

        return action  # shape (K,)


    # Ghi lại transition
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)


    # Học từ batch
    def optimize_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
        ) = self.replay_buffer.sample(self.batch_size)

        state_batch = state_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        done_batch = done_batch.to(self.device)

        # Q hiện tại
        q_values = self.policy_net(state_batch)  # [B, K, I+1]
        action_batch = action_batch.to(self.device)
        chosen_q = q_values.gather(2, action_batch.unsqueeze(2)).squeeze(2)  # [B, K]

        # Q mục tiêu
        with torch.no_grad():
            next_q = self.target_net(next_state_batch)
            max_next_q, _ = next_q.max(2)
            target_q = reward_batch.unsqueeze(1) + self.gamma * (1 - done_batch.unsqueeze(1)) * max_next_q

        loss = nn.functional.smooth_l1_loss(chosen_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.learn_step += 1

        # Cập nhật target
        if self.learn_step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()
    
    def set_power_budget(self, power_budget):
        """
        Cập nhật quota công suất cho RU, gọi từ tầng FrameEnv (SAC level).
        power_budget: np.array [num_slices] hoặc [num_PRB]
        """
        self.power_budget = np.array(power_budget)



