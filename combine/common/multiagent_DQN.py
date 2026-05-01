import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from combine.common.common import ReplayBuffer


class MultiHeadDQN(nn.Module):
    def __init__(self, state_dim, num_urllc, num_bwp, num_urllc_ue, hidden_dim=256):
        """
        Mạng DQN nhiều đầu cho URLLC slicing
        Args:
            state_dim (int): số chiều của state
            num_urllc (int): số slice URLLC
            num_bwp (int): số BWP (bandwidth parts)
            num_urllc_ue (list[int]): số UE trong mỗi slice
            hidden_dim (int): số chiều hidden layer
        """
        super().__init__()
        self.num_urllc = num_urllc
        self.num_bwp = num_bwp
        self.num_urllc_ue = num_urllc_ue

        # Mạng chung (shared feature extractor)
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Các head riêng cho từng slice
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_urllc_ue[i]) for i in range(num_urllc)
        ])

    def forward(self, state):
        """
        Forward pass
        Input:
            state: tensor [batch, state_dim]
        Output:
            list các tensor [batch, num_bwp, num_ue_slice_i] cho từng slice
        """
        z = self.shared(state)  # [batch, hidden_dim]

        q_values_per_slice = []
        for i, head in enumerate(self.heads):
            q_slice = head(z)  # [batch, num_ue_slice_i]
            # Thêm chiều num_bwp
            q_slice = q_slice.unsqueeze(1).expand(-1, self.num_bwp, -1)
            # [batch, num_bwp, num_ue_slice_i]
            q_values_per_slice.append(q_slice)

        # Output: list có độ dài num_urllc, mỗi phần tử là [batch, num_bwp, num_ue_slice_i]
        return q_values_per_slice


class MultiHeadDQNAgent:
    def __init__(
        self,
        state_dim,
        num_urllc,
        num_urllc_ue,
        num_bwp,
        train_cons,
        buffer_capacity=10000,
    ):
        self.device = torch.device(train_cons["device"])
        self.state_dim = state_dim
        self.num_urllc = num_urllc
        self.num_urllc_ue = num_urllc_ue
        self.num_bwp = num_bwp
        self.gamma = train_cons["gamma"]
        self.batch_size = train_cons["batch_size"]
        self.eps = train_cons["eps_start"]
        self.eps_end = train_cons["eps_end"]
        self.eps_decay = train_cons["eps_decay"]
        self.lr = train_cons["lr"]
        self.learn_step = 0

        # Mạng Q chính & target (dùng cùng cấu hình)
        self.policy_net = MultiHeadDQN(
            state_dim, num_urllc, num_bwp, num_urllc_ue
        ).to(self.device)
        self.target_net = MultiHeadDQN(
            state_dim, num_urllc, num_bwp, num_urllc_ue
        ).to(self.device)

        # Đồng bộ trọng số ban đầu
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(forSAC=False, capacity=buffer_capacity)


    def select_action(self, state, BWP_slice):
        """
        Phân bổ UE cho các PRB của RU hiện tại dựa trên ngân sách từng slice.
        BWP_slice: list chứa số lượng PRB cho từng slice [s0, s1, ...].
        self.num_urllc_ue: list chứa số UE của từng slice [num_ue_s0, num_ue_s1, ...].
        """
        all_allocations = {}
        
        # 1. Chuẩn bị state và lấy Q-values từ mạng Neural
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            # q_values shape: [total_max_prbs, max_ues_per_slice]
            # Hoặc một cấu hình phù hợp với số UE lớn nhất bạn quản lý
            q_values = self.policy_net(state_t).squeeze(0).cpu().numpy()

        prb_offset = 0 # Điểm bắt đầu dải PRB của slice hiện tại[cite: 2]
        
        # 2. Duyệt qua từng slice dựa trên số lượng slice URLLC[cite: 2]
        for s_idx, budget in enumerate(BWP_slice):
            budget = int(budget)
            if budget <= 0:
                all_allocations[s_idx] = np.array([])
                continue
                
            # Lấy số lượng UE mà slice s phục vụ[cite: 2]
            num_ues = self.num_urllc_ue[s_idx]

            # 3. Thực hiện Epsilon-Greedy để chọn UE cho từng PRB[cite: 2]
            if np.random.rand() < self.eps:
                # Chọn ngẫu nhiên index UE từ 0 đến num_ues-1[cite: 2]
                selected_ues = np.random.randint(0, num_ues, size=budget)
            else:
                # Lấy đoạn Q-values tương ứng với ngân sách PRB của slice s[cite: 2]
                # và giới hạn không gian hành động theo số UE của slice đó (Masking)[cite: 2]
                slice_q_segment = q_values[prb_offset : prb_offset + budget, :num_ues]
                
                # Chọn UE có index mang lại Q-value lớn nhất[cite: 2]
                selected_ues = np.argmax(slice_q_segment, axis=-1)

            all_allocations[s_idx] = selected_ues
            
            # 4. Cập nhật offset để slice tiếp theo nhận đoạn PRB kế tiếp[cite: 2]
            prb_offset += budget

        return all_allocations


    def update_epsilon(self):
        """Giảm epsilon theo decay"""
        self.eps = max(self.eps_end, self.eps * self.eps_decay)


    # Ghi lại transition
    def store_transition(self, state, action, reward, next_state, done):
        """
        Ghi lại transition state, action -> reward + next_state
        """
        self.replay_buffer.push(state, action, reward, next_state, done)


    def optimize_model(self):
        """
        Tối ưu hóa model theo các transition đã lưu
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

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
        action_batch = action_batch.to(self.device)

        # Q hiện tại cho từng slice
        q_values_per_slice = self.policy_net(state_batch)  # list [B, num_bwp, num_ue_slice_i]

        chosen_q_list = []
        for i, q_slice in enumerate(q_values_per_slice):
            # action_batch[:, i, :] chứa hành động cho slice i (theo bwp)
            # shape: [B, num_bwp]
            act = action_batch[:, i, :].unsqueeze(-1)  # [B, num_bwp, 1]
            chosen_q = q_slice.gather(2, act).squeeze(2)  # [B, num_bwp]
            chosen_q_list.append(chosen_q)

        # Ghép lại thành [B, num_urllc, num_bwp]
        chosen_q = torch.stack(chosen_q_list, dim=1)

        # Q mục tiêu
        with torch.no_grad():
            next_q_values_per_slice = self.target_net(next_state_batch)
            max_next_q_list = []
            for q_slice in next_q_values_per_slice:
                max_next_q, _ = q_slice.max(dim=2)  # [B, num_bwp]
                max_next_q_list.append(max_next_q)
            max_next_q = torch.stack(max_next_q_list, dim=1)  # [B, num_urllc, num_bwp]

            target_q = reward_batch.unsqueeze(1).unsqueeze(2) + \
                    self.gamma * (1 - done_batch.unsqueeze(1).unsqueeze(2)) * max_next_q

        # Tính loss
        loss = nn.functional.smooth_l1_loss(chosen_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()



