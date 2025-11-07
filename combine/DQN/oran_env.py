# Mỗi RU sẽ hành động như là 1 independent agent
import gym
from gym import spaces
import numpy as np 
import torch
import random
from combine.DQN.multiagent_DQN import MultiHeadDQN 
from combine.common.common import MLP
from scipy.special import ndtri, lambertw # cho hàm Q^-1, Lambert


class SlotEnv(gym.Env):
    """
    Slot-level environment cho từng RU.
    - Mức này tương tác trực tiếp với DQN agent.
    - Mỗi step tương ứng với 1 slot (subframe nhỏ).
    - Có thể nhận quota power từ SAC thông qua FrameEnv.
    """

    def __init__(self, RU_index, RU, slices, num_urllc, H, T_slot, T_max,
                 eps=0.1, max_steps=20, w_reward=None):
        super(SlotEnv, self).__init__()

        self.RU_index = RU_index
        self.RU = RU
        self.slices = slices
        self.num_urllc = num_urllc
        self.H = H  # channel gains 
        self.T_slot = T_slot
        self.T_max = T_max
        self.eps = eps
        self.max_steps = max_steps

        # Reward weights (merge all weights)
        self.w_reward = w_reward or {"thr": 0.6, "sla": 0.25, "fair": 0.1, "stab": 0.05}

        # Power allocation info
        self.power_budget = np.ones(len(self.slices)) / len(self.slices)  # normalized init
        self.Pmax = RU.Pmax

        # State and action spaces
        self.num_PRB = RU.K
        self.num_slices = len(slices)
        self.state_dim = self.num_PRB + self.num_slices + 2  # channel avg, traffic, etc. adjustable
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([self.num_slices] * self.num_PRB)

        # Step tracking
        self.current_step = 0
        self.last_info = None
        self.dqn_agent = None
        self.eMBBThr = {}
        self.power_alloc = []
        self.x_alloc = []

    # Connection hooks for hierarchical pipeline
    def set_power_budget(self, power_budget):
        """Cập nhật quota power được SAC phân bổ cho từng slice."""
        self.power_budget = np.array(power_budget, dtype=float)

    def assign_dqn_agent(self, agent):
        """Gán DQN agent để dùng khi FrameEnv gọi env.select_action()."""
        self.dqn_agent = agent

    def get_state(self):
        """
        Sinh trạng thái đầu vào cho DQN agent.
        Bao gồm:
            - Trung bình gain của từng slice (chuẩn hóa)
            - Tỷ lệ công suất được SAC cấp cho từng slice
            - Utilization: mức độ sử dụng PRB
            - Load: tỷ lệ backlog / throughput
        """
        # ===== Trung bình channel gain =====
        # H có shape (num_slices, num_PRB)
        ch_avg = np.mean(self.H, axis=1)
        ch_avg = ch_avg / (np.max(ch_avg) + 1e-9)

        # ===== Tỷ lệ công suất được cấp cho từng slice =====
        if np.sum(self.power_budget) > 0:
            slice_ratios = self.power_budget / np.sum(self.power_budget)
        else:
            slice_ratios = np.ones(self.num_slices) / self.num_slices

        # ===== Utilization: tỷ lệ PRB đang dùng =====
        if hasattr(self, "x_alloc") and self.x_alloc is not None:
            util = np.count_nonzero(self.x_alloc) / self.num_PRB
        else:
            util = 0.0

        # ===== Load: tổng buffer / tổng throughput =====
        total_buffer = 0.0
        total_rate = 0.0
        if hasattr(self, "slices"):
            for sl in self.slices:
                total_buffer += getattr(sl, "buffer", 0.0)
                total_rate += getattr(sl, "current_rate", 0.0)

        if total_rate <= 0:
            load = 0.0
        else:
            load = np.clip(total_buffer / (total_rate + 1e-9), 0, 1.5)

        # ===== Ghép thành vector trạng thái =====
        state = np.concatenate([
            ch_avg[:min(len(ch_avg), self.num_slices)],
            slice_ratios,
            [util, load]
        ])

        # ===== Đảm bảo đúng kích thước state_dim =====
        if len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)))
        elif len(state) > self.state_dim:
            state = state[:self.state_dim]

        return state.astype(np.float32)

    def select_action(self, state, eval_mode=False):
        """Nếu có DQN agent thì dùng policy của nó, nếu không thì random."""
        if self.dqn_agent is not None:
            return self.dqn_agent.select_action(state, eval_mode=eval_mode)
        return self.action_space.sample()

    # Core environment logic
    def reset(self):
        self.current_step = 0
        self.last_info = None
        return self.get_state()

    def step(self, action, state):
        """
        Môi trường SlotEnv thực hiện 1 bước (slot):
            - action: vector độ dài num_PRB, mỗi phần tử là index của slice
            - state: state hiện tại (để tính stability)
        Trả về:
            state_next, reward, done, info
        """
        self.current_step += 1
        self.x_alloc = np.array(action)
        self.power_alloc = self.distribute_power_to_prbs(self.x_alloc)
       
        snr = self.calculateSNR(self.x_alloc)
        self.eMBBThr, URLLCCapa = self.calculateSliceCapacity(self.x_alloc, snr)

        # TÍNH QoS, SLA, FAIRNESS, STABILITY
        fairness = self.calculateFairness(self.eMBBThr)
        stability = self.calculateStability(self.x_alloc, state)
        sumeMBBThr = np.sum(
            self.eMBBThr[i] for i in range(self.num_slices)
            if self.slices[i].name == "eMBB"
        )
        util =  np.count_nonzero(self.x_alloc) / self.num_PRB

        # TÍNH REWARD (THỰC TẾ)
        reward = (
            self.w_reward["thr"] * (sumeMBBThr / (self.T_max + 1e-9)) +
            self.w_reward["fair"] * fairness +
            self.w_reward["stab"] * stability +
            self.w_reward["util"] * util
        )


        # CẬP NHẬT STATE & KIỂM TRA DONE
        state_next = self.get_state()
        done = self.current_step >= self.max_steps

        # LOGGING THÔNG TIN
        info = {
            "Throughput_eMBB": self.eMBBThr,
            "Capa_URLLC" : URLLCCapa,
            "Fairness": fairness,
            "Stability": stability,
            "Utilization": util,
        }
        self.last_info = info

        return state_next, reward, done, info

    # Helper computation functions
    def calculateSNR(self, x_alloc):
        """Simple mock SNR model: proportional to power budget and channel gain."""
        # Expand power_budget across PRBs proportionally
        snr = []
        for prb in range(self.num_PRB):
            if x_alloc[prb] > 0:
                snr.append((self.power_alloc[prb] * self.H[x_alloc[prb] - 1][prb]) / (self.RU.B * self.RU.N0))
            else:
                snr.append(0)  
        return snr

    def calculateSliceCapacity(self, x_alloc, snr):
        """Compute slice throughput and URLLC capacity (mock version)."""
        eMBBThr = {}
        for i in range(self.num_slices):
            if self.slices[i].name == "eMBB":
                eMBBThr[i] = 0

        URLLCCapa = np.zeros(len(self.slices))
        for prb in range(self.num_PRB):
            if x_alloc[prb] > 0:
                if self.slices[x_alloc[prb] - 1].name == "eMBB":
                    eMBBThr[x_alloc[prb] - 1] += self.RU.n * self.RU.B *  np.log2(1 + snr[prb])
                else :
                    capacity = np.log2(1 + snr[prb])
                    verlation = (1 - (1 + snr[prb])**-2) * (np.log2(np.e))**2
                    penalty = np.sqrt(verlation / self.RU.n) * abs(ndtri(self.slices[x_alloc[prb] - 1].eps_phy))
                    URLLCCapa[x_alloc[prb]] += self.RU.n * self.T_slot * (capacity - penalty)
        return eMBBThr, URLLCCapa

    def calculateURLLCStatus(self, URLLCCapa=None):
        """
        Cập nhật và tính toán SLA của các URLLC slice.
        - URLLCCapa: dict hoặc list chứa tổng bit phục vụ cho từng URLLC slice
        Trả về: SLA ratio (served / (served + dropped))
        """
        total_served = 0
        total_dropped = 0

        for i, sl in enumerate(self.slices):
            if sl.name == "URLLC":
                # Xác định công suất phục vụ thực tế
                capa_bits = 0.0
                if URLLCCapa is not None:
                    if isinstance(URLLCCapa, dict):
                        capa_bits = URLLCCapa.get(i, 0.0)
                    elif isinstance(URLLCCapa, (list, np.ndarray)) and i < len(URLLCCapa):
                        capa_bits = URLLCCapa[i]

                # Số gói có thể phục vụ trong slot này
                served_now = int(min(capa_bits // sl.packet_size, sl.buffer // sl.packet_size))
                dropped_now = 0

                # Cập nhật buffer và counters
                if served_now > 0:
                    sl.served_packets += served_now
                    sl.buffer = max(0.0, sl.buffer - served_now * sl.packet_size)

                # Giả lập delay và drop nếu buffer tồn tại lâu
                # (đơn giản: xác suất drop tăng khi buffer lớn)
                if sl.buffer > sl.packet_size * 10:  
                    dropped_now = int(sl.buffer // (sl.packet_size * 10))
                    sl.dropped_packets += dropped_now
                    sl.buffer = max(0.0, sl.buffer - dropped_now * sl.packet_size)

                total_served += served_now
                total_dropped += dropped_now

        if total_served + total_dropped == 0:
            return 1.0  # tránh chia 0
        return total_served / (total_served + total_dropped)

    def calculateFairness(self, eMBBThr):
        """Jain Index fairness."""
        thr_values = np.array(list(eMBBThr.values())) + 1e-9
        return (np.sum(thr_values) ** 2) / (len(thr_values) * np.sum(thr_values ** 2) + 1e-9)

    def calculateStability(self, x_alloc, state):
        """Stability metric: reward high if allocation changes little."""
        if not hasattr(self, "x_prev"):
            self.x_prev = np.zeros_like(x_alloc)
        diff_count = np.not_equal(x_alloc, self.x_prev).astype(float)
        stability = np.exp(-np.mean(diff_count/self.num_PRB))
        stability *= np.exp(-(1/self.num_PRB) * (state[-1] + state[-2]))
        self.x_prev = x_alloc
        return stability

    def get_total_throughput(self):
        """Return total throughput of last step."""
        if self.last_info is not None:
            return self.last_info.get("Throughput_eMBB", 0.0)
        return 0.0


        """
        Tính tổng chênh lệch giữa tỷ lệ công suất phân bổ và tỷ lệ công suất thực tế sử dụng.

        Parameters
        ----------
        actual_power_usage : np.ndarray
            Mảng thực tế công suất sử dụng cho mỗi slice, shape giống self.power_budget.
            Ví dụ: shape (num_RU, num_slice)

        Returns
        -------
        total_gap : float
            Tổng chênh lệch tuyệt đối giữa tỷ lệ phân bổ và tỷ lệ thực tế.
        gap_per_slice : np.ndarray
            Mảng chênh lệch theo từng slice, shape (num_RU, num_slice)
        """
        # Tổng công suất mỗi RU
        Pmax_per_RU = np.sum(self.power_budget, axis=1, keepdims=True) + 1e-8  # tránh chia 0

        # Tính tỷ lệ phân bổ và tỷ lệ thực tế
        allocated_ratio = self.power_budget / Pmax_per_RU
        actual_ratio = actual_power_usage / Pmax_per_RU

        # Tính chênh lệch tuyệt đối
        gap_per_slice = np.abs(allocated_ratio - actual_ratio)
        avg_gap = np.sum(gap_per_slice) / (self.num_slices - self.num_urllc)

        return avg_gap

    def distribute_power_to_prbs(self, x_alloc):
        """
        Phân bổ công suất cho từng PRB dựa trên action và power_budget.

        Parameters
        ----------
        action : list[int]
            Mảng phân bổ PRB, mỗi phần tử là:
            - 0: PRB không được phân bổ
            - i > 0: PRB được phân cho slice (i - 1)
            Ví dụ: [1, 2, 0, 1] → PRB 0 và 3 cho slice 0, PRB 1 cho slice 1

        ru_index : int
            Chỉ số RU tương ứng trong self.power_budget 

        Returns
        -------
        prb_power : list[float]
            Mảng công suất phân bổ cho từng PRB, cùng độ dài với action
        """
        prb_power = [0.0] * self.num_PRB

        # Đếm số PRB được phân cho từng slice
        prb_count_per_slice = [0] * self.num_slices
        for slice_id in x_alloc:
            if slice_id > 0:
                prb_count_per_slice[slice_id - 1] += 1

        # Phân bổ công suất đều cho các PRB thuộc mỗi slice
        for prb_idx, slice_id in enumerate(x_alloc):
            if slice_id > 0:
                slice_idx = slice_id - 1
                count = prb_count_per_slice[slice_idx]
                if count > 0:
                    slice_power = self.power_budget[slice_idx] * self.Pmax
                    prb_power[prb_idx] = slice_power / count

        return prb_power
    
    def returnAlloc(self):
        return self.power_alloc, self.x_alloc

    
