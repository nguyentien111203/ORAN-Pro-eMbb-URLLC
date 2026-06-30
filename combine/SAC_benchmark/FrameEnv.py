import numpy as np
import gymnasium as gym
from gymnasium import spaces
import copy


class FrameEnv(gym.Env):
    def __init__(self, RUs, RU_envs, urllc_slices, embb_slices, H, w_reward, scale_max, frame_slots=10):
        super().__init__()
        self.RUs = RUs
        self.RU_envs = RU_envs
        self.urllc_slices = urllc_slices
        self.embb_slices = embb_slices
        self.H = H
        self.w_reward = w_reward
        self.scale_max = scale_max
        self.frame_slots = frame_slots

        self.num_rus = len(RUs)
        self.slot_count = 0
        self.num_urllc = len(urllc_slices)
        self.num_embb = len(embb_slices)
        self.num_slices = self.num_urllc + self.num_embb
        self.state_dim = 4 + self.num_slices

        self.total_sac_actions = sum(len(self.RUs[r].bwps) for r in range(self.num_rus)) * self.num_slices

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.total_sac_actions,), dtype=np.float32)

        self.BWP_slice = [[[0 for _ in range(len(self.RUs[r].bwps))] for _ in range(self.num_slices)] for r in range(self.num_rus)]
        self.last_BWP_slice = copy.deepcopy(self.BWP_slice)
        self.debug_ep_count = 0

    def reset(self):
        for env in self.RU_envs:
            env.reset()

        self.slot_count = 0

        return self.get_state(
            [np.zeros(len(s.ue_set)) for s in self.embb_slices],
            [np.zeros(len(s.ue_set)) for s in self.urllc_slices],
            np.zeros(1),
            np.zeros(1),
            np.zeros(1),
            np.zeros(1)
        )

    def step(self, action):
        self.resetAlloc()
        action = np.array(action).flatten()
        idx = 0
        sac_quotas = []
        for r in range(self.num_rus):
            num_bwps = len(self.RUs[r].bwps)
            ru_quota = [[0 for _ in range(num_bwps)] for _ in range(self.num_slices)]
            for s in range(self.num_slices):
                for b in range(num_bwps):
                    total_prbs = self.RUs[r].bwps[b].num_prb
                    ru_quota[s][b] = int(action[idx] * total_prbs)
                    idx += 1
            sac_quotas.append(ru_quota)

        URLLC_frame = [np.zeros(len(s.ue_set)) for s in self.urllc_slices]
        eMBB_frame  = [np.zeros(len(s.ue_set)) for s in self.embb_slices]
        costEne, costFrag, costSwit, costGB = (
            np.zeros(self.frame_slots), np.zeros(self.frame_slots),
            np.zeros(self.frame_slots), np.zeros(self.frame_slots)
        )

        # State DQN riêng cho từng RU (RU_Env mới không tự lưu self.state sau step)
        DQNstate = [np.zeros(self.RU_envs[0].state_dim) for _ in range(self.num_rus)]

        for slot_index in range(self.frame_slots):
            for r, env in enumerate(self.RU_envs):
                env.update_H(self.H[r][0])

                # Transpose sac_quotas[r] từ [s][b] sang [s][b] - MultiHeadDQNAgent cần [s][b]
                dqn_action = env.select_action(DQNstate[r], sac_quotas[r])

                # computeOutput trả flatBit (URLLC), flatThr (eMBB) - đã flatten sẵn
                flatBit, flatThr = env.computeOutput(dqn_action)

                # ----- Tự tính 4 tham số cho RU_Env.step() -----
                lat_rates = []
                for s in range(env.num_urllc):
                    for u in range(env.num_urllc_ue[s]):
                        max_lat = getattr(env.urllc_slices[s].ue_set[u], 'max_lat', 1.0)
                        pkt_size = getattr(env.urllc_slices[s].ue_set[u], 'packet_size', 100)
                        idx_ue = sum(env.num_urllc_ue[:s]) + u
                        numBit = flatBit[idx_ue] if idx_ue < len(flatBit) else 1e-9
                        latency = pkt_size / (numBit + 1e-9)
                        lat_rates.append(latency / (max_lat + 1e-9))
                lat_rates = np.array(lat_rates, dtype=np.float32)
                lat_soft = np.clip(lat_rates, 0.0, 1.0)

                thr_rates = []
                for s in range(env.num_embb):
                    for u in range(env.num_embb_ue[s]):
                        min_thr = getattr(env.embb_slices[s].ue_set[u], 'min_thr', 1.0)
                        idx_ue = sum(env.num_embb_ue[:s]) + u
                        thr = flatThr[idx_ue] if idx_ue < len(flatThr) else 0.0
                        thr_rates.append(thr / (min_thr + 1e-9))
                thr_rates = np.array(thr_rates, dtype=np.float32)
                thr_soft = np.clip(thr_rates, 0.0, 1.0)

                DQNstate[r], _, _, info = env.step(lat_rates, thr_rates, lat_soft, thr_soft)

                # ----- Tích lũy throughput / latency cho reward (giữ nguyên công thức benchmark) -----
                offset = 0
                for s in range(self.num_embb):
                    for e in range(len(self.embb_slices[s].ue_set)):
                        req_thr = getattr(self.embb_slices[s].ue_set[e], 'min_thr', 10.0)
                        thr_val = flatThr[offset] if offset < len(flatThr) else 0.0
                        eMBB_frame[s][e] += min(thr_val / (req_thr + 1e-9), 1.0)
                        offset += 1

                offset = 0
                for s in range(self.num_urllc):
                    for u in range(len(self.urllc_slices[s].ue_set)):
                        max_lat = getattr(self.urllc_slices[s].ue_set[u], 'max_lat', 1.0)
                        pkt_size = getattr(self.urllc_slices[s].ue_set[u], 'packet_size', 100)
                        numBit = flatBit[offset] if offset < len(flatBit) else 1e-9
                        latency_ratio = (pkt_size / (numBit + 1e-9)) / (max_lat + 1e-9)
                        URLLC_frame[s][u] += min(latency_ratio, 1.0)
                        offset += 1

                # Cost: cộng đúng 1 lần mỗi RU/slot (không lặp theo UE)
                costEne[slot_index]  += info["costE"]
                costFrag[slot_index] += info["costF"]
                costSwit[slot_index] += info["costS"]
                costGB[slot_index]   += info["costGB"]

        eMBB_frame = [x / (self.frame_slots + 1e-9) for x in eMBB_frame]
        URLLC_frame = [x / (self.frame_slots + 1e-9) for x in URLLC_frame]

        embb_avg = sum(np.clip(ratio, 0.0, 2.0) for s in range(self.num_embb) for ratio in eMBB_frame[s])
        urllc_avg = sum(np.clip(2.0 - ratio, 0.0, 2.0) for s in range(self.num_urllc) for ratio in URLLC_frame[s])

        cost_reward = (
            (1 - np.sum(costEne)  / (self.scale_max[0] * self.frame_slots + 1e-9)) +
            (1 - np.sum(costFrag) / (self.scale_max[1] * self.frame_slots + 1e-9)) +
            (1 - np.sum(costSwit) / (self.scale_max[2] * self.frame_slots + 1e-9)) +
            (1 - np.sum(costGB)   / (self.scale_max[3] * self.frame_slots + 1e-9))
        )

        raw_reward = (self.w_reward["thr"] * embb_avg + self.w_reward["lat"] * urllc_avg + self.w_reward["cost"] * cost_reward)
        reward = raw_reward / 1000.0

        self.slot_count += 1
        done = (self.slot_count >= 50)

        return (
            self.get_state(eMBB_frame, URLLC_frame, costEne, costFrag, costSwit, costGB),
            reward,
            done,
            {"thr": eMBB_frame, "lat": URLLC_frame}
        )

    def resetAlloc(self):
        if self.slot_count == 0 and self.last_BWP_slice is not None:
            self.BWP_slice = copy.deepcopy(self.last_BWP_slice)

    def get_state(self, eMBB_frame, URLLC_frame, costEne, costFrag, costSwit, costGB):
        return np.concatenate([
            [np.min(eMBB_frame[s]) for s in range(self.num_embb)],
            [np.max(URLLC_frame[s]) for s in range(self.num_urllc)],
            [np.average(costEne) / self.scale_max[0], np.average(costFrag) / self.scale_max[1],
             np.average(costSwit) / self.scale_max[2], np.average(costGB) / self.scale_max[3]]
        ]).astype(np.float32)
