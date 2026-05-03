import numpy as np
import gym
from gym import spaces
import torch


class FrameEnv(gym.Env):
    def __init__(self, RUs, RU_urllc_envs, RU_embb_envs, urllc_slices, embb_slices, H, w_reward, 
                 scale_max, frame_slots=10):
        """
        slot_envs: m
        """
        
        super().__init__()
        self.RUs = RUs
        self.RU_urllc_envs = RU_urllc_envs
        self.RU_embb_envs = RU_embb_envs
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
        

    def reset(self):
        """
        Reset lại môi trường ở từng frame 
        """
        for env in self.RU_embb_envs:
            env.reset()

        for env in self.RU_urllc_envs:
            env.reset()

        self.slot_count = 0


    def step(self, action):
        self.resetAlloc()
        
        # Tính toán throughput và latency trên frame của các UE
        URLLC_frame = []
        eMBB_frame = []
        if len(self.urllc_slices) != 0: 
            for s in range(len(self.urllc_slices)):
                URLLC_frame.append(np.zeros(len(self.urllc_slices[s].ue_set)))
        if len(self.embb_slices) != 0:
            for s in range(len(self.embb_slices)):
                eMBB_frame.append(np.zeros(len(self.embb_slices[s].ue_set)))

        # Tính toán các chi phí trên từng frame
        costEne = np.zeros(self.frame_slots)
        costFrag = np.zeros(self.frame_slots)
        costSwit = np.zeros(self.frame_slots)
        costGB = np.zeros(self.frame_slots)
        
        for slot_index in range(self.frame_slots):
            
            # Cập nhật và lấy thông tin từ các RU
            for r, (Eenv, Uenv) in enumerate(zip(self.RU_embb_envs, self.RU_urllc_envs)):
                # Cập nhật H, thực hiện hành động và tính toán throughput, latency và 
                # các chi phí dựa trên action của SAC
                if self.num_embb != 0:
                    Eenv.update_H(self.H[self.num_urllc : self.num_slices])
                    Eaction = Eenv.select_action(action[r])
                    _, _, Einfo = Eenv.step(Eaction)

                    for s in range(len(self.embb_slices)):
                        for e in range(len(self.embb_slices[s])):
                            eMBB_frame[s][e] += Einfo["Thr"][s][e]

                    costEne[slot_index] += Einfo["costE"]
                    costFrag[slot_index] += Einfo["costF"]
                    costSwit[slot_index] += Einfo["costS"]
                    costGB[slot_index] += Einfo["costGB"]

                if self.num_urllc != 0:
                    Uenv.update_H(self.H[:self.num_urllc])
                    Uaction = Uenv.select_action(action[r])
                    _, _, Uinfo = Uenv.step(Uaction)

                    for s in range(len(self.urllc_slices)):
                        for u in range(len(self.urllc_slices[s])):
                            URLLC_frame += Uinfo["lat"][s][u] 

                    costEne[slot_index] += Uinfo["costE"]
                    costFrag[slot_index] += Uinfo["costF"]
                    costSwit[slot_index] += Uinfo["costS"]
                    costGB[slot_index] += Uinfo["costGB"]

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
                + self.w_reward["cost"] * (4 - (np.sum(costEne) / (self.scale_max[0] * self.frame_slots))  \
                                        - (np.sum(costFrag) / (self.scale_max[1] * self.frame_slots)) \
                                        - (np.sum(costSwit) / (self.scale_max[2] * self.frame_slots)) \
                                        - (np.sum(costGB) / (self.scale_max[3] * self.frame_slots))))

        info = {
            "thr": eMBB_frame,
            "lat": URLLC_frame,
            "costE": costEne,
            "costF": costFrag,
            "costS": costSwit,
            "costGB": costGB
        }

        # Reset lại phân bổ
        self.last_BWP_slice = self.BWP_slice
        next_state = self.get_state(eMBB_frame, URLLC_frame, costEne, costFrag, costSwit, costGB)
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



