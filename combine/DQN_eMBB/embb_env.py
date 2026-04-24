# Mỗi RU sẽ hành động như là 1 independent agent
import gym
from gym import spaces
import numpy as np 
import math
from combine.common.multiagent_DQN import MultiHeadDQNAgent
from combine.common.common import MLP
from scipy.special import ndtri, lambertw # cho hàm Q^-1, Lambert


class SlotEnv(gym.Env):
    """
    Slot-level environment cho từng RU.
    - Mức này tương tác trực tiếp với DQN agent.
    - Mỗi step tương ứng với 1 slot (subframe nhỏ).
    - Có thể nhận quota power từ SAC thông qua FrameEnv.
    """

    def __init__(self, RU_index, RU, slices, num_urllc, H, gain_ru, dist_ru_ue, T_slot, T_max, NF,
                 eps=0.1, max_steps=20, w_reward=None):
        super(SlotEnv, self).__init__()

        self.RU_index = RU_index
        self.RU = RU
        self.slices = slices
        self.num_urllc = num_urllc
        self.H = H  # channel gains 
        self.gain_ru = gain_ru
        self.dist_ru_ue = dist_ru_ue
        self.T_slot = T_slot
        self.T_max = T_max
        self.NF = 10**(NF/10)
        self.eps = eps
        self.max_steps = max_steps

        # Reward weights (merge all weights)
        self.w_reward = w_reward or {"thr": 0.6, "sla": 0.25, "fair": 0.1, "stab": 0.05}

        # Power allocation info
        self.power_budget = np.ones(len(self.slices)) / len(self.slices)  # normalized init
        self.Pmax = RU.Pmax
        self.P_ru = np.zeros(len(gain_ru))
        self.effFactor = 0.5
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

        # Parameters for process of packets
        self.last_arrivals = 0
        self.served_packets = 0
        self.dropped_packets = 0 
        self.lam = 1

        # Constant : priority for each slice based on distance
        self.priority_slice = np.zeros(self.num_slices)

        for u in range(self.num_slices):
            self.priority_slice[u] = 1.0 / (self.dist_ru_ue[u])

        # normalize
        self.priority_slice /= (np.max(self.priority_slice))

    # Connection hooks for hierarchical pipeline
    def set_power_budget(self, power_budget):
        """Cập nhật quota power được SAC phân bổ cho từng slice."""
        self.power_budget = np.array(power_budget, dtype=float)

    def assign_dqn_agent(self, agent):
        """Gán DQN agent để dùng khi FrameEnv gọi env.select_action()."""
        self.dqn_agent = agent

    def get_state(self):
        """
        Generate state for RU-level DQN.
        State components:
            - Average channel gain per slice (normalized)
            - Power ratio per slice (from SAC)
            - PRB utilization
            - URLLC arrival pressure
        """

        # ===== Power ratio per slice =====
        if np.sum(self.power_budget) > 0:
            slice_ratios = self.power_budget / np.sum(self.power_budget)
        else:
            slice_ratios = np.ones(self.num_slices) / self.num_slices

        # ===== PRB utilization =====
        if hasattr(self, "x_alloc") and self.x_alloc is not None:
            util = np.count_nonzero(self.x_alloc) / self.num_PRB
        else:
            util = 0.0

        # ===== URLLC arrival pressure (M/M/1/1) =====
        arrival_pressure = self.last_arrivals / self.num_PRB

        # ===== Concatenate state =====
        state = np.concatenate([
            self.priority_slice,
            slice_ratios,
            np.array([util, arrival_pressure])
        ])

        # ===== Pad / truncate to state_dim =====
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

    def step(self, action):
        """
        One scheduling slot at RU level.
        """
        self.current_step += 1

        # Generate URLLC arrivals at beginning of slot
        self.last_arrivals = np.random.poisson(self.lam)
        arrival_pressure = self.last_arrivals / self.num_PRB

        # Apply DQN action (PRB allocation)
        self.x_alloc = np.array(action)
        self.power_alloc, numprbEachSlice = self.distribute_power_to_prbs(self.x_alloc)

        # PHY & capacity
        I_ru = self.compute_ru_interference(self.P_ru)
        snr = self.calculateSNR(self.x_alloc, I_ru)
        self.eMBBThr, URLLCCapa = self.calculateSliceCapacity(self.x_alloc, snr)

        # Utilization
        utilPower, utilPRB = self.calculateUtil()

        # M/M/1/1 service for URLLC
        served = min(np.count_nonzero(self.x_alloc), self.last_arrivals)
        self.served_packets += served
        self.dropped_packets += max(0, self.last_arrivals - served)

        # Stability (arrival-aware)
        stability = self.calculateStability(self.x_alloc, arrival_pressure)

        # Aggregate eMBB throughput
        sumeMBBThr = np.sum(
            self.eMBBThr[i] for i in range(self.num_slices)
            if self.slices[i].name == "eMBB"
        )

        #penalty for serving prioritised UE
        waste_penalty = np.sum(self.power_budget * (1 - self.priority_slice))

        # Reward
        reward = (
            self.w_reward["thr"] * ((sumeMBBThr + np.sum(URLLCCapa)) / (self.T_max + 1e-9)) +
            self.w_reward["stab"] * stability -
            self.w_reward["utilPower"] * utilPower +
            self.w_reward["utilPRB"] * utilPRB -
            self.w_reward["waste"] * waste_penalty
        )

        # (9) Next state & done
        state_next = self.get_state()
        done = self.current_step >= self.max_steps

        # (10) Info
        info = {
            "eMBB_thr": self.eMBBThr,
            "URLLCCapa": URLLCCapa,
            "arrival_pressure": arrival_pressure,
            "served_urllc": served,
            "stab": stability,
            "utilPower": utilPower,
            "utilPRB": utilPRB
        }

        self.last_info = info
        return state_next, reward, done, info

    # Helper computation functions
    def calculateSNR(self, x_alloc, I_ru):
        """Simple mock SNR model: proportional to power budget and channel gain."""
        # Expand power_budget across PRBs proportionally
        snr = []
        for prb in range(self.num_PRB):
            if x_alloc[prb] > 0:
                snr.append((self.power_alloc[prb] * self.H[x_alloc[prb] - 1][prb]) / (self.NF * self.RU.B * self.RU.N0 + I_ru))
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
                    eMBBThr[x_alloc[prb] - 1] += self.RU.B *  np.log2(1 + snr[prb])
                else :
                    capacity = np.log2(1 + snr[prb])
                    verlation = (1 - (1 + snr[prb])**-2) * (np.log2(np.e))**2
                    penalty = np.sqrt(verlation / self.RU.n) * abs(ndtri(self.slices[x_alloc[prb] - 1].eps_phy))
                    URLLCCapa[x_alloc[prb] - 1] += self.RU.n * (capacity - penalty) * self.T_slot
        return eMBBThr, URLLCCapa

    def calculateURLLCStatus(self, URLLCCapa):
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

    def calculateUtil(self):
        """Jain Index fairness."""
        util_power = sum(self.power_alloc) / self.Pmax
        util_PRB = np.count_nonzero(self.x_alloc) / self.num_PRB
        return util_power, util_PRB

    def calculateStability(self, x_alloc, arrival_pressure):
        """
        Stability metric for RU-level DQN.
        High when PRB allocation changes little,
        relaxed under high arrival pressure.
        """
        if not hasattr(self, "x_prev"):
            self.x_prev = np.zeros_like(x_alloc)

        # Fraction of PRBs whose allocation changed
        change_ratio = np.mean(x_alloc != self.x_prev)

        # Stability: penalize changes, relax when load is high
        stability = np.exp(-change_ratio) * np.exp(-arrival_pressure)

        self.x_prev = x_alloc.copy()
        return stability

    def compute_ru_interference(self, P_ru):
        num_RUs = len(self.gain_ru)
        #if np.count_nonzero(self.gain_ru) == 0:
        #    print("max gain_ru:", np.max(self.gain_ru))
        #    print("min gain_ru:", np.min(self.gain_ru))
        #    print("P_ru (mW):", P_ru)

        I_ru = 0

        for rp in range(num_RUs):
            if rp != self.RU_index:
                I_ru += P_ru[rp] * self.gain_ru[rp] * 0.001

        return I_ru

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

        # Normalize
        for u in range(self.num_slices):
            prb_count_per_slice[u] /= self.num_PRB

        return prb_power, prb_count_per_slice

    def updatePru(self, other_P_ru):
        # Cập nhật công suất sử dụng ở từng RU
        self.P_ru = other_P_ru 

    def returnAlloc(self):
        return self.power_alloc, self.x_alloc

