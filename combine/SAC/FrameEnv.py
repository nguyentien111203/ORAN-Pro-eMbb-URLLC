import numpy as np
import gym
from gym import spaces
import torch


class FrameEnv(gym.Env):
    def __init__(self, RUs, RU_envs, urllc_slices, embb_slices, H, w_reward, 
                 scale_max, frame_slots=10):
        """
        slot_envs: m
        """
        
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

        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(self.state_dim,), 
            dtype=np.float32
        )
        # action: slice-level ratios 
        self.total_sac_actions = self.num_rus * self.num_slices * sum(len(self.RUs[r].bwps) for r in range(self.num_rus))
        self.action_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(self.total_sac_actions,), 
            dtype=np.float32
        )

        # Ma trận phân bổ số PRB từng BWP về từng slice (để chỉ số của slice urllc trước, sau đó mới đến slice embb)
        self.BWP_slice = [[[0 for _ in range(len(self.RUs[r].bwps))] for _ in range(self.num_slices)] for r in range(self.num_rus)]
        self.last_BWP_slice = None

        self.pac = np.array([ue.pac for s in range(self.num_urllc) 
                        for ue in self.RU_envs[0].urllc_slices[s].ue_set])

        self.lat_target = np.array([ue.lat for s in range(self.num_urllc) 
                               for ue in self.RU_envs[0].urllc_slices[s].ue_set])

        self.thr_min = np.array([ue.thr for s in range(self.num_embb) 
                            for ue in self.RU_envs[0].embb_slices[s].ue_set])
        
        self.URLLC_frame = []
        self.eMBB_frame = []
        for slot in range(self.frame_slots): 
            self.URLLC_frame.append([])
            self.eMBB_frame.append([])
            for s in range(len(self.urllc_slices)):
                self.URLLC_frame[slot].append(np.zeros(len(self.urllc_slices[s].ue_set), np.float64))
            for s in range(len(self.embb_slices)):
                self.eMBB_frame[slot].append(np.zeros(len(self.embb_slices[s].ue_set), np.float64))

        # Tính toán các chi phí trên từng frame
        self.costEne = np.zeros(self.frame_slots)
        self.costFrag = np.zeros(self.frame_slots)
        self.costSwit = np.zeros(self.frame_slots)
        self.costGB = np.zeros(self.frame_slots)


    def reset(self):
        """
        Reset lại môi trường ở từng frame cũng như chi phí, throungput và latency
        """
        for env in self.RU_envs:
            env.reset()
        
        for slot in range(self.frame_slots):
            for s in range(len(self.urllc_slices)):
                self.URLLC_frame[slot][s] = np.zeros(len(self.urllc_slices[s].ue_set))
            for s in range(len(self.embb_slices)):
                self.eMBB_frame[slot][s] = np.zeros(len(self.embb_slices[s].ue_set))

        # Tính toán các chi phí trên từng frame
        self.costEne = np.zeros(self.frame_slots)
        self.costFrag = np.zeros(self.frame_slots)
        self.costSwit = np.zeros(self.frame_slots)
        self.costGB = np.zeros(self.frame_slots)

        self.slot_count = 0


    def step(self, action):
        self.resetAlloc()
        self.reset()
        DQNstate = [np.zeros(self.RU_envs[0].state_dim) * self.num_rus]
        for slot_index in range(self.frame_slots):
            numBits = np.array([0 for s in range(self.num_urllc) 
                                for ue in self.RU_envs[0].urllc_slices[s].ue_set], np.float64)
            # Cập nhật và lấy thông tin từ các RU
            for r, env in enumerate(self.RU_envs):
                # Cập nhật H, thực hiện hành động và tính toán throughput, latency và 
                # các chi phí dựa trên action của SAC
                env.update_H(self.H[slot_index][r])
                DQNaction = env.select_action(DQNstate[r], action[r])

                # computeOutput trả numpy array (slice x ue)
                ruBits, ruThr = env.computeOutput(DQNaction)

                # Tính toán thr và latency
                numBits += ruBits
                self.eMBB_frame[slot_index] += ruThr

            self.URLLC_frame[slot_index] = self.pac / numBits
            embb_rate = self.eMBB_frame[slot_index] / self.thr_min
            urllc_rate = self.URLLC_frame[slot_index] / self.lat_target
            
            # Áp action vào tính toán và 
            for r, env in enumerate(self.RU_envs):
                DQNstate[r], _, _, info = env.step(urllc_rate, embb_rate)

                self.costEne[slot_index] += info["costE"]
                self.costFrag[slot_index] += info["costF"]
                self.costSwit[slot_index] += info["costS"]
                self.costGB[slot_index] += info["costGB"]

            


        # Tính trung bình trong 1 frame
        if self.num_embb != 0:
            eMBB_frame = [x / self.frame_slots for x in eMBB_frame]
        if self.num_urllc != 0:
            URLLC_frame = [x / self.frame_slots for x in URLLC_frame]

        # Tính tỷ lệ trễ và throughput trung bình
        embb_avg = np.sum(eMBB_frame[s][e] / self.embb_slices[s].ue_set[e].Rmin for s in range(len(self.embb_slices)) 
                          for e in range(len(self.embb_slices[s].ue_set)))
        urllc_avg = np.sum(URLLC_frame[s][u] / self.urllc_slices[s].ue_set[u].lat for s in range(len(self.urllc_slices)) 
                          for u in range(len(self.urllc_slices[s].ue_set)))

        # reward
        reward = (self.w_reward["thr"] *  embb_avg 
                + self.w_reward["lat"] *  urllc_avg
                + self.w_reward["cost"] * (4 - (np.sum(self.costEne) / (self.scale_max[0] * self.frame_slots))  \
                                        - (np.sum(self.costFrag) / (self.scale_max[1] * self.frame_slots)) \
                                        - (np.sum(self.costSwit) / (self.scale_max[2] * self.frame_slots)) \
                                        - (np.sum(self.costGB) / (self.scale_max[3] * self.frame_slots))))

        info = {
            "thr": eMBB_frame,
            "lat": URLLC_frame,
            "costE": self.costEne,
            "costF": self.costFrag,
            "costS": self.costSwit,
            "costGB": self.costGB
        }

        # Reset lại phân bổ
        self.last_BWP_slice = self.BWP_slice
        next_state = self.get_state(eMBB_frame, URLLC_frame, self.costEne, self.costFrag, self.costSwit, self.costGB)
        done = self.slot_count == self.frame_slots
        return next_state, reward, info, done
    

    def returnAlloc(self):
        """
        Trả về phân bổ từng BWP của từng RU cho các slice
        """
        return self.BWP_slice
    

    def resetAlloc(self):
        """
        Reset lại phân bổ khi frame mới bắt đầu
        """
        if self.slot_count == 0:
            self.BWP_slice = self.last_BWP_slice


    def get_state(self, eMBB_frame, URLLC_frame, costEne, costFrag, costSwit, costGB):
        """
        Sinh state vector chi tiết cho SAC agent ở mức frame.
        Bao gồm thông tin toàn cục và chi tiết từng slice/RU.
        """
        # Tìm tỷ lệ throughput, latency thấp nhất ở từng slice
        minThrRate = []
        for s in range(len(self.embb_slices)):
            minThrRate.append(np.min(eMBB_frame[s]))
        
        maxLatRate = []
        for s in range(len(self.urllc_slices)):
            maxLatRate.append(np.min(URLLC_frame[s]))

        # Ghép state
        state = np.concatenate([minThrRate, maxLatRate, [np.average(costEne) / self.scale_max[0],
                                np.average(costFrag) / self.scale_max[1], np.average(costSwit) / self.scale_max[2],
                                np.average(costGB) / self.scale_max[3]]])
        return state.astype(np.float32)



