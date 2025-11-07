import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from combine.common.common import MLP, GaussianPolicy, ReplayBuffer


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SACAgent:
    def __init__(self, num_RU, state_dim, num_slices, action_scale=1.0, action_bias=0.0,
                 device='cpu', lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):

        self.state_dim = state_dim
        self.num_slices = num_slices     # số slice => số chiều hành động
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.action_scale = action_scale
        self.action_bias = action_bias
        self.action_shape = (num_RU, num_slices)

        # --- Actor & Critics
        self.actor = GaussianPolicy(state_dim, num_RU * num_slices,
                                    action_scale=self.action_scale,
                                    action_bias=self.action_bias).to(device)

        # Critics Q1, Q2
        self.critic_1 = MLP(state_dim + num_RU * num_slices, 1).to(device)
        self.critic_1.init_weights()
        self.critic_2 = MLP(state_dim + num_RU * num_slices, 1).to(device)
        self.critic_2.init_weights()
        self.target_critic_1 = MLP(state_dim + num_RU * num_slices, 1).to(device)
        self.target_critic_1.init_weights()
        self.target_critic_2 = MLP(state_dim + num_RU * num_slices, 1).to(device)
        self.target_critic_2.init_weights()
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        # --- Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_1_opt = optim.Adam(self.critic_1.parameters(), lr=lr/3)
        self.critic_2_opt = optim.Adam(self.critic_2.parameters(), lr=lr/3)

        # --- Replay buffer (tương tự DQN)
        self.replay_buffer = ReplayBuffer(forSAC=True)

        # --- Entropy tuning
        self.target_entropy = -float(num_RU * num_slices)  # phù hợp cho hành động liên tục
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

    def select_action(self, state, eval_mode=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # Actor output
        if eval_mode:
            with torch.no_grad():
                mean, _ = self.actor.forward(state)
                action = torch.tanh(mean) * self.action_scale + self.action_bias
        else:
            action, _, _ = self.actor.sample(state)

        # Convert to numpy
        action = action.detach().cpu().numpy().reshape(self.action_shape)

        # Đảm bảo không âm (tránh slice bị bỏ đói hoàn toàn)
        epsilon = 1e-4 * np.max(action)
        action = np.maximum(action, epsilon)

        # Co tổng bằng hàm phi tuyến: scale = 1 / (1 + total)
        for r in range(action.shape[0]):
            total = np.sum(action[r])
            scale = 1 / (1 + total)  # càng lớn thì càng co mạnh
            action[r] *= scale

        return action

    # -------------------------------------------------------------
    def update(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return 0.0, 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # ---------- Guard: clip reward để tránh Q nổ ----------
        rewards = torch.clamp(rewards, -10.0, 10.0)

        # ---------- Sample next actions ----------
        next_actions, next_log_probs, _ = self.actor.sample(next_states)
        next_log_probs = next_log_probs.clamp(-20, 2)   # tránh exp(logπ) nổ

        # ---------- Target Q ----------
        with torch.no_grad():
            target_Q1 = self.target_critic_1(torch.cat([next_states, next_actions], dim=-1))
            target_Q2 = self.target_critic_2(torch.cat([next_states, next_actions], dim=-1))
            target_Q = torch.min(target_Q1, target_Q2) - self.alpha * next_log_probs
            target_Q = rewards + (1 - dones) * self.gamma * target_Q
            target_Q = torch.clamp(target_Q, -1e2, 1e2)  # clip an toàn

        # ---------- Current Q ----------
        current_Q1 = self.critic_1(torch.cat([states, actions], dim=-1))
        current_Q2 = self.critic_2(torch.cat([states, actions], dim=-1))

        # ---------- Critic losses ----------
        critic_1_loss = nn.MSELoss()(current_Q1, target_Q)
        critic_2_loss = nn.MSELoss()(current_Q2, target_Q)

        # Guard NaN/Inf
        if torch.isnan(critic_1_loss) or torch.isinf(critic_1_loss):
            print("Skip update: critic_1_loss NaN/Inf")
            return 0.0, 0.0

        # ---------- Update critics ----------
        self.critic_1_opt.zero_grad(set_to_none=True)
        critic_1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), max_norm=5.0)
        self.critic_1_opt.step()

        self.critic_2_opt.zero_grad(set_to_none=True)
        critic_2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), max_norm=5.0)
        self.critic_2_opt.step()

        # ---------- Actor update ----------
        new_actions, log_probs, _ = self.actor.sample(states)
        log_probs = log_probs.clamp(-20, 2)
        Q1_new = self.critic_1(torch.cat([states, new_actions], dim=-1))
        Q2_new = self.critic_2(torch.cat([states, new_actions], dim=-1))
        Q_new = torch.min(Q1_new, Q2_new)

        Q_new_norm = Q_new - Q_new.mean().detach()
        actor_loss = (self.alpha * log_probs - Q_new_norm).mean()

        if torch.isnan(actor_loss) or torch.isinf(actor_loss):
            print("Skip actor update: NaN/Inf loss")
            return 0.0, 0.0

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_opt.step()

        # ---------- alpha update (entropy temperature) ----------
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().clamp(1e-5, 10.0)  # giới hạn alpha

        # ---------- Soft update ----------
        with torch.no_grad():
            for t_p, p in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
                t_p.copy_(t_p * (1 - self.tau) + p * self.tau)
            for t_p, p in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
                t_p.copy_(t_p * (1 - self.tau) + p * self.tau)

        # ---------- Optional debug ----------
        if abs(actor_loss.item()) > 1000 or critic_1_loss.item() > 1e3:
            print(f"actor_loss={actor_loss.item():.3e}, critic_loss={critic_1_loss.item():.3e}, "
                f"reward_mean={rewards.mean().item():.3f}, Q_mean={Q_new.mean().item():.3f}, alpha={self.alpha.item():.3f}")

        return actor_loss.item(), float((critic_1_loss.item() + critic_2_loss.item()) / 2)

