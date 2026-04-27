import numpy as np
import gym
from gym import spaces
import torch


class FrameEnv(gym.Env):
    def __init__(self, RU_urllc_envs, RU_embb_envs, urllc_slices, embb_slices, H, w_reward, 
                 scale_max, frame_slots=10):
        """
        slot_envs: m
        """
        
        super().__init__()
        self.RU_urllc_envs = RU_urllc_envs
        self.RU_embb_envs = RU_embb_envs
        self.urllc_slices = urllc_slices
        self.embb_slices = embb_slices
        self.H = H
        self.w_reward = w_reward
        self.scale_max = scale_max
        self.frame_slots = frame_slots

        self.slot_count = 0
        self.num_urllc = len(urllc_slices)
        self.num_embb = len(embb_slices)
        self.num_slices = self.num_urllc + self.num_embb

        # Để sau
        # state: avg_thr_norm, success_ratio, fairness
        self.state_dim = 6 + self.num_slices + 2 * self.num_rus + 1
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.state_dim,), dtype=np.float32)
        # action: slice-level ratios (num_slices,)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.num_slices,), dtype=np.float32)

        # Ma trận phân bổ số PRB từng BWP về từng slice (để chỉ số của slice urllc trước, sau đó mới đến slice embb)
        self.BWP_slice = [[[0 for _ in range(len(self.RU.bwps))] for _ in range(len(self.num_slices))] for _ in range(self.num_rus)]
        self.init_BWP_slice = [[[0 for _ in range(len(self.RU.bwps))] for _ in range(len(self.num_slices))] for _ in range(self.num_rus)]


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
        
        # Tính toán throughput và latency trên frame
        eMBBThr_frame = []
        URLLCCapa_frame = []

        # Tính toán các chi phí trên từng frame
        costEne = []
        costFrag = []
        costSwit = []
        costGB = []
        
        for _ in range(self.frame_slots):
            
            # Cập nhật và lấy thông tin từ các RU
            for r, Eenv, Uenv in len(self.RU_embb_envs), self.RU_embb_envs, self.RU_urllc_envs:
                # Thực hiện hành động và tính toán throughput, latency và các chi phí dựa trên action của SAC
                Eaction = Eenv.select_action(action[r])
                _, _, Einfo = Eenv.step(Eaction)

                Uaction = Uenv.select_action(action[r])
                _, _, Uinfo = Uenv.step(Uaction)

                eMBBThr_frame.append(sum(Einfo["Thr"]))
                URLLCCapa_frame.append(sum(Uinfo["lat"]))

                costEne.append(Einfo["costE"] + Uinfo["costE"])
                costFrag.append(Einfo["costF"] + Uinfo["costF"])
                costSwit.append(Einfo["costS"] + Uinfo["costS"])
                costGB.append(Einfo["costGB"] + Uinfo["costGB"])


        # reward
        reward = (self.w_reward["thr"] * (np.sum(eMBBThr_frame) / (self.scale_max[0] * self.frame_slots)) \
                + self.w_reward["lat"] * (np.sum(URLLCCapa_frame) / (self.scale_max[1] * self.frame_slots))
                - self.w_reward["cost"] * ((np.sum(costEne) / (self.scale_max[2] * self.frame_slots))  \
                                        + (np.sum(costFrag) / (self.scale_max[3] * self.frame_slots)) \
                                        + (np.sum(costSwit) / (self.scale_max[4] * self.frame_slots)) \
                                        + (np.sum(costGB) / (self.scale_max[5] * self.frame_slots))))

        info = {
            "thr": np.sum(eMBBThr_frame) / self.frame_slots,
            "lat": np.sum(URLLCCapa_frame) / self.frame_slots,
            "costE": np.sum(costEne) / self.frame_slots,
            "costF": np.sum(costFrag) / self.frame_slots,
            "costS": np.sum(costSwit) / self.frame_slots,
            "costGB": np.sum(costGB) / self.frame_slots
        }

        next_state = self.get_state()
        return next_state, reward, info
    

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
            self.BWP_slice = self.init_BWP_slice


    def get_state(self):
        """
        Sinh state vector chi tiết cho SAC agent ở mức frame.
        Bao gồm thông tin toàn cục và chi tiết từng slice/RU.
        """

        return state.astype(np.float32)



