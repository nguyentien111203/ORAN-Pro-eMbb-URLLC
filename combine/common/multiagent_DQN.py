import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from combine.common.common import ReplayBuffer

class MultiHeadDQN(nn.Module):
    def __init__(self, state_dim, num_urllc, num_bwp, num_slice_ue, hidden_dim=256):
        super().__init__()
        self.num_urllc = num_urllc
        self.num_bwp = num_bwp
        self.num_slice_ue = num_slice_ue

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_bwp * num_slice_ue[i]) for i in range(len(num_slice_ue))
        ])

    def forward(self, state):
        z = self.shared(state)
        q_values_per_slice = []
        for i, head in enumerate(self.heads):
            q_slice = head(z) 
            q_slice = q_slice.view(state.size(0), self.num_bwp, self.num_slice_ue[i])
            q_values_per_slice.append(q_slice)
        return q_values_per_slice


class MultiHeadDQNAgent:
    def __init__(self, state_dim, num_slices, num_slice_ue, num_bwp, train_cons, buffer_capacity=10000):
        self.device = torch.device(train_cons["device"])
        self.state_dim = state_dim
        self.num_slices = num_slices
        self.num_slice_ue = num_slice_ue
        self.num_bwp = num_bwp
        self.gamma = train_cons["gamma"]
        self.batch_size = train_cons["batch_size"]
        self.eps = train_cons["eps_start"]
        self.eps_end = train_cons["eps_end"]
        self.eps_decay = train_cons["eps_decay"]
        self.lr = train_cons["lr"]
        self.learn_step = 0
        self.target_update = 2

        self.policy_net = MultiHeadDQN(state_dim, num_slices, num_bwp, num_slice_ue).to(self.device)
        self.target_net = MultiHeadDQN(state_dim, num_slices, num_bwp, num_slice_ue).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.offsets = [0]
        for n in self.num_slice_ue:
            self.offsets.append(self.offsets[-1] + n)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.replay_buffer = ReplayBuffer(forSAC=False, capacity=buffer_capacity)

    def select_action(self, state, BWP_slice):
        num_bwps = self.num_bwp
        num_slices = self.num_slices
        all_allocations = [[None for _ in range(num_bwps)] for _ in range(num_slices)]
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state_t)

        for s_idx in range(num_slices):
            num_ues = self.num_slice_ue[s_idx]
            q_slice_all_bwp = q_values[s_idx].squeeze(0)

            for bwp_idx in range(num_bwps):
                budget = int(BWP_slice[s_idx][bwp_idx])

                if budget <= 0:
                    all_allocations[s_idx][bwp_idx] = np.zeros(num_ues, dtype=int)
                    continue

                q_slice = q_slice_all_bwp[bwp_idx, :num_ues]

                if np.random.rand() < self.eps:
                    probs = np.ones(num_ues) / num_ues
                else:
                    probs = torch.softmax(q_slice, dim=-1).cpu().numpy()
                probs = probs.astype(np.float64)  # Ép sang 64-bit cho độ chính xác cao
                probs = probs / np.sum(probs)     # Ép tổng về chuẩn xác 1.0
                prb_alloc = np.random.multinomial(budget, probs)
                all_allocations[s_idx][bwp_idx] = prb_alloc

        return all_allocations

    def update_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def optimize_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)
        state_batch = state_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        done_batch = done_batch.to(self.device)
        action_batch = action_batch.to(self.device)

        q_values_per_slice = self.policy_net(state_batch) 
        chosen_q_list = []
        
        for i, q_slice in enumerate(q_values_per_slice):
            start = self.num_bwp * self.offsets[i]
            end = self.num_bwp * self.offsets[i+1]
            act = action_batch[:, start:end]
            
            act = act.view(state_batch.size(0), self.num_bwp, self.num_slice_ue[i])
            chosen_q = (q_slice * act).sum(dim=2) 
            chosen_q_list.append(chosen_q)

        chosen_q = torch.stack(chosen_q_list, dim=1)

        with torch.no_grad():
            next_q_values_per_slice = self.target_net(next_state_batch)
            max_next_q_list = []
            for q_slice in next_q_values_per_slice:
                max_next_q, _ = q_slice.max(dim=2) 
                max_next_q_list.append(max_next_q)
            max_next_q = torch.stack(max_next_q_list, dim=1) 
            target_q = reward_batch.unsqueeze(1) + self.gamma * (1 - done_batch.unsqueeze(1)) * max_next_q

        loss = nn.functional.smooth_l1_loss(chosen_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()