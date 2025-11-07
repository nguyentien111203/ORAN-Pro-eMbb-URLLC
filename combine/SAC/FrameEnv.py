import numpy as np
import gym
from gym import spaces
from combine.powerAllocator.powerRelated import PowerMediator


class FrameEnv(gym.Env):
    def __init__(self, slot_envs, slices, num_urllc, H, T_max, w_reward, 
                 sla_slices, frame_slots=10, allocation_mode="equal"):
        super(FrameEnv, self).__init__()
        self.slot_envs = slot_envs
        self.slices = slices
        self.num_slices = len(slices)
        self.num_urllc = num_urllc
        self.num_rus = len(slot_envs)
        self.H = H
        self.T_max = T_max
        self.sla_slices = sla_slices
        self.frame_slots = frame_slots
        self.total_power = np.array([env.RU.Pmax for env in slot_envs], dtype=float)
        self.power_mediator = PowerMediator(slot_envs, allocation_mode=allocation_mode)
        self.slot_count = 0

        # state: avg_thr_norm, success_ratio, fairness
        self.state_dim = 4 + 3 * self.num_slices + 2 * self.num_rus + \
            (self.num_rus * self.num_slices) + 1
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.state_dim,), dtype=np.float32)

        # action: slice-level ratios (num_slices,)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.num_slices,), dtype=np.float32)

        # internal
        self.last_state = np.zeros(self.state_dim, dtype=float)
        self.last_quota = np.ones(self.num_slices) / self.num_slices
        self.reward_weights = w_reward or {"thr": 0.5, "sla": 0.3, "stab": 0.1, "fair": 0.1}
        self.frame_count = 0
        self.powerAlloc = []
        self.xAlloc = []

    def reset(self):
        for env in self.slot_envs:
            env.reset()
        self.last_state = np.zeros(self.state_dim, dtype=float)
        self.last_quota = np.ones(self.num_slices) / self.num_slices
        self.frame_count = 0
        return self.last_state

    def step(self, action):
        self.resetAlloc()
        # sanitize and normalize action -> slice ratios
        action = np.clip(np.array(action, dtype=float), 0.0, 1.0)
        if np.sum(action) <= 0:
            action = np.ones_like(action) / len(action)
        else:
            action = action / np.sum(action)

        # convert ratios to slice-level power quotas (use total power across RUs)
        total_power_sum = np.sum(self.total_power)
        slice_power_quota = action * total_power_sum  # vector length num_slices

        # PowerMediator divides slice quota per RU (returns shape [num_rus, num_slices])
        quota_power_per_RU = self.power_mediator.allocate_power(slice_power_quota)

        # push per-RU quotas to slot envs (each RU expects per-slice quota)
        for r, env in enumerate(self.slot_envs):
            # env.set_power_budget expects per-slice vector for that RU
            env.set_power_budget(quota_power_per_RU[r])

        # Run DQN scheduling for frame_slots
        eMBBThr_frame = {}
        URLLCCapa_frame = {}
        for i in range(self.num_slices):
            if self.slices[i].name == "eMBB":
                eMBBThr_frame[i] = 0
            else :
                URLLCCapa_frame[i] = 0

        # Tính toán trên từng frame
        sumeMBB_frame = []
        sla_eMBB = []
        sla_URLLC = []
        jain = []
        util = []
        stab = []

        for slot_index in range(self.frame_slots):
            self.slot_count+=1

            eMBBThr_slot = {}
            URLLCCapa_slot = {}
            num_used = 0
            num_prb = 0
            util_slot = 0
            for i in range(self.num_slices):
                if self.slices[i].name == "eMBB":
                    eMBBThr_slot[i] = 0
                else :
                    URLLCCapa_slot[i] = 0

            for r, env in enumerate(self.slot_envs):
                env.H = self.H[r][slot_index]
                state_local = env.get_state()
                action_local = env.select_action(state_local)  # uses assigned DQN
                next_state_local, _, _, info = env.step(action_local, state_local)
                # env.step should set env.last_info
                env.last_info = info
                # Tính toán các thông số
                eMBBThr = info.get("Throughput_eMBB")
                URLLCCapa = info.get("Capa_URLLC")
                num_used += np.count_nonzero(env.x_alloc)
                num_prb += env.num_PRB
                
                # Cộng các giá trị lại cho từng RU
                for i in range(self.num_slices):
                    if self.slices[i].name == "eMBB":
                        eMBBThr_slot[i]+=eMBBThr[i]
                    else :
                        URLLCCapa_slot[i]+=URLLCCapa[i]
            self.update_buffers(eMBBThr_slot, URLLCCapa_slot)
            # Tính cho từng slot 
            sla_eMBB_slot, jain_slot = self.calculateSLAJaineMBB(eMBBThr_slot)
            sla_URLLC_slot = self.calculateSLAURLLC(URLLCCapa_slot)
            # Cập nhật buffer toàn hệ thống

            sla_eMBB.append(sla_eMBB_slot)
            sla_URLLC.append(sla_URLLC_slot)
            jain.append(jain_slot)
            util_slot = num_used / num_prb
            util.append(util_slot)

            stability = np.exp(-np.mean(np.abs(action - self.last_quota)))
            stab.append(stability)

            sumeMBB_frame.append(sum(eMBBThr_slot[i]/self.frame_slots for i in range(self.num_slices) 
                                        if self.slices[i].name == "eMBB"))

        # reward
        w = self.reward_weights
        reward = (w["thr"] * (np.average(sumeMBB_frame)/self.T_max) +  
                w["sla"] * (np.average(sla_eMBB) - self.sla_slices["eMBB"]) + 
                w["sla"] * (np.average(sla_URLLC) - self.sla_slices["eMBB"]) + 
                w["stab"] * np.average(stab) + 
                w["fair"] * np.average(jain) +
                w["util"] * np.average(util))

        # update state
        self.last_quota = action.copy()
        self.last_state = self.get_state(sumeMBB_frame, sla_URLLC, jain, stab)
        self.frame_count += 1
        done = False  # episodes managed in training loop

        info = {
            "eMBB_thr": sumeMBB_frame,
            "SLA_urllc": sla_URLLC,
            "SLA_embb": sla_eMBB,
            "Jain Index": jain,
            "stability": stab,
            "util": util
        }
        return self.last_state, reward, done, info
    
    def returnAlloc(self):
        return self.powerAlloc, self.xAlloc
    
    def resetAlloc(self):
        if self.slot_count == 0:
            self.powerAlloc, self.xAlloc = [], []

    def update_buffers(self, eMBBThr, URLLCCapa):
        """
        Cập nhật buffer và thống kê served/dropped của tất cả slice
        dựa trên throughput thực tế (eMBBThr, URLLCCapa).

        eMBBThr, URLLCCapa: np.array[num_slices], tổng bit được phục vụ trong slot này.
        """
        for i, sl in enumerate(self.slices):

            # === URLLC Slice ===
            if sl.name == "URLLC":
                pkt_size = getattr(sl, "packet_size", 1e4)
                capacity_bits = URLLCCapa[i] if i < len(URLLCCapa) else 0.0
                buffer_bits = getattr(sl, "buffer", 0.0)

                # Số gói được phục vụ trong slot này (theo capacity)
                served_now = int(min(capacity_bits // pkt_size, buffer_bits // pkt_size))
                sl.served_packets += served_now
                sl.buffer = max(0.0, buffer_bits - served_now * pkt_size)

                # Drop nếu backlog quá lớn (delay vượt ngưỡng)
                delay_limit_pkts = getattr(sl, "delay_limit_pkts", 50)
                max_backlog = pkt_size * delay_limit_pkts
                if sl.buffer > max_backlog:
                    dropped_now = int(sl.buffer // max_backlog)
                    sl.dropped_packets += dropped_now
                    sl.buffer = max(0.0, sl.buffer - dropped_now * pkt_size)
                else:
                    dropped_now = 0

                # Sinh thêm packet (arrival)
                if hasattr(sl, "arrival_prob"):
                    arrival_bits = np.random.binomial(1, sl.arrival_prob) * pkt_size
                else:
                    arrival_bits = np.random.poisson(lam=getattr(sl, "arrival_rate", 0))
                sl.buffer += arrival_bits

            # === eMBB Slice ===
            elif sl.name == "eMBB":
                arrival_bits = np.random.poisson(lam=getattr(sl, "arrival_rate", 0))
                departure_bits = eMBBThr[i] if i < len(eMBBThr) else 0.0

                sl.buffer = max(0.0, sl.buffer + arrival_bits - departure_bits)

                # Drop nếu buffer vượt ngưỡng (hiếm)
                max_backlog = getattr(sl, "max_buffer", 1e7)
                if sl.buffer > max_backlog:
                    overflow_drop = int((sl.buffer - max_backlog) // getattr(sl, "packet_size", 1e4))
                    sl.dropped_packets += overflow_drop
                    sl.buffer = max_backlog

    def calculateSLAJaineMBB(self, eMBBThr):
        served, dropped = 0, 0
        sla = 0
        for i in range(self.num_slices):
            if self.slices[i].name == "eMBB":
                if eMBBThr[i] > self.slices[i].dataRate:
                    served+=1
                else:
                    dropped+=1
        if served + dropped == 0:
            sla = 1
        sla = served / (served + dropped)
        
         # Tính toán Jain index
        jain_index = (np.sum([eMBBThr[r] for r in range(self.num_slices)
            if self.slices[r].name == "eMBB"]) ** 2) / ((self.num_slices - self.num_urllc) * \
                    np.sum([eMBBThr[r] ** 2 for r in range(self.num_slices)
            if self.slices[r].name == "eMBB"]) + 1e-9)
        
        return sla, jain_index

    def calculateSLAURLLC(self, URLLCCapa):
        """
        Tính SLA của toàn hệ thống cho URLLC dựa trên bit thực tế phục vụ (URLLCCapa).
        SLA = served / (served + dropped)
        """
        total_served = 0
        total_dropped = 0

        for i, sl in enumerate(self.slices):
            if sl.name == "URLLC":
                pkt_size = getattr(sl, "packet_size", 1e4)
                capacity_bits = URLLCCapa[i] if i < len(URLLCCapa) else 0.0
                buffer_bits = getattr(sl, "buffer", 0.0)

                # Gói được phục vụ = số gói có thể xử lý trong slot này
                served_now = int(min(capacity_bits // pkt_size, buffer_bits // pkt_size))
                sl.served_packets += served_now
                sl.buffer = max(0.0, buffer_bits - served_now * pkt_size)

                # Gói bị drop do trễ hoặc backlog lớn
                delay_limit_pkts = getattr(sl, "delay_limit_pkts", 50)
                max_backlog = pkt_size * delay_limit_pkts
                if sl.buffer > max_backlog:
                    dropped_now = int(sl.buffer // max_backlog)
                    sl.dropped_packets += dropped_now
                    sl.buffer = max(0.0, sl.buffer - dropped_now * pkt_size)
                else:
                    dropped_now = 0

                total_served += served_now
                total_dropped += dropped_now

        if total_served + total_dropped == 0:
            return 1.0
        return float(np.clip(total_served / (total_served + total_dropped + 1e-9), 0.0, 1.0))

    def get_state(self, sumeMBB_frame, sla_URLLC, jain, stability):
        """
        Sinh state vector chi tiết cho SAC agent ở mức frame.
        Bao gồm thông tin toàn cục và chi tiết từng slice/RU.
        """

        # ==== 1. Thông tin tổng thể ====
        avg_thr = np.average(sumeMBB_frame) / (self.T_max + 1e-9)
        avg_sla_url = np.average(sla_URLLC)
        avg_jain = np.average(jain)
        avg_stability = np.average(stability)

        # ==== 2. Thông tin từng slice ====
        slice_buffers = []
        slice_sla = []
        slice_thr = []

        for sl in self.slices:
            buf = getattr(sl, "buffer", 0.0)
            thr = np.mean(getattr(sl, "rate_history", [0.0])[-5:]) if hasattr(sl, "rate_history") else 0.0
            served = getattr(sl, "served_packets", 0)
            dropped = getattr(sl, "dropped_packets", 0)
            sla = served / (served + dropped + 1e-9)

            slice_buffers.append(buf)
            slice_thr.append(thr)
            slice_sla.append(sla)

        # Chuẩn hóa theo max để tránh scale quá lớn
        buf_norm = np.array(slice_buffers) / (np.max(slice_buffers) + 1e-9)
        thr_norm = np.array(slice_thr) / (self.T_max + 1e-9)
        sla_norm = np.clip(np.array(slice_sla), 0, 1)

        # ==== 3. Thông tin per-RU ====
        ru_utils = []
        ru_powers = []
        for env in self.slot_envs:
            util = np.count_nonzero(env.x_alloc) / (env.num_PRB + 1e-9)
            ru_utils.append(util)
            ru_powers.append(np.sum(getattr(env, "power_budget", np.zeros(self.num_slices))) / (env.Pmax + 1e-9))

        ru_utils = np.clip(ru_utils, 0, 1)
        ru_powers = np.clip(ru_powers, 0, 1)

        # ==== 4. Quota stability ====
        delta_quota = np.mean(np.abs(self.last_quota - getattr(self, "prev_quota", self.last_quota)))
        quota_norm = self.last_quota / (np.sum(self.last_quota) + 1e-9)

        # ==== 5. Gộp thành vector ====
        global_features = np.array([avg_thr, avg_sla_url, avg_jain, avg_stability], dtype=float)
        state = np.concatenate([
            global_features,
            buf_norm, thr_norm, sla_norm,
            ru_utils, ru_powers,
            quota_norm,
            [delta_quota]
        ], dtype=float)

        # Cập nhật self.last_state để log/debug
        self.prev_quota = self.last_quota.copy()
        self.last_state = state.copy()

        return state.astype(np.float32)



