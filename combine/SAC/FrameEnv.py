import numpy as np
import gym
from gym import spaces
import torch


class FrameEnv(gym.Env):
    def __init__(self, RU_urllc_envs, RU_embb_envs, urllc_slices, embb_slices, H, inter_RU, w_reward, 
                 cost_switch, cost_gb, scale_max, frame_slots=10):
        """
        slot_envs: m
        """
        
        super().__init__()
        self.RU_urllc_envs = RU_urllc_envs
        self.RU_embb_envs = RU_embb_envs
        self.urllc_slices = urllc_slices
        self.embb_slices = embb_slices
        self.H = H
        self.inter_RU = inter_RU
        self.w_reward = w_reward
        self.cost_switch = cost_switch
        self.cost_gb = cost_gb
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
        # sanitize and normalize action -> slice ratios
        action = np.clip(np.array(action, dtype=float), 0.0, 1.0)
        if np.sum(action) <= 0:
            action = np.ones_like(action) / len(action)

        # PowerMediator divides slice quota per RU (returns shape [num_rus, num_slices])
        quota_power_per_RU = np.array(action).reshape(self.num_rus, self.num_slices)

        p_used = [sum(action[i]) for i in range(self.num_rus)]
        self.P_ru = [self.RU_maxPower[i] * p_used[i] for i in range(self.num_rus)]

        # push per-RU quotas to slot envs (each RU expects per-slice quota)
        for r, env in enumerate(self.slot_envs):
            # env.set_power_budget expects per-slice vector for that RU
            env.set_power_budget(quota_power_per_RU[r])

        # Run DQN scheduling for frame_slots
        eMBBThr_frame = {}
        URLLCCapa_frame = {}
        for i in range(self.num_slices):
            if self.slices[i].name == "eMBB":
                eMBBThr_frame[i] = []
            else :
                URLLCCapa_frame[i] = []

        # Tính toán trên từng frame
        sumeMBB_frame = []
        sla_eMBB = []
        sla_URLLC = []
        util_power = []
        util_prb = []
        stab = []
        # Tính toán mức sử dụng ở từng RU
        eachRUutilPower = []
        eachRUutilPRB = []
        for i in range(len(self.slot_envs)):
            eachRUutilPower.append([])
            eachRUutilPRB.append([])

        for slot_index in range(self.frame_slots):
            self.slot_count+=1

            eMBBThr_slot = {}
            URLLCCapa_slot = {}
            num_used = 0
            num_prb = 0
            util_slot = 0
            power_rate = 0
            for i in range(self.num_slices):
                if self.slices[i].name == "eMBB":
                    eMBBThr_slot[i] = 0
                else :
                    URLLCCapa_slot[i] = 0

            for r, env in enumerate(self.slot_envs):
                env.updatePru(self.RU_maxPower)
                env.H = self.H[r][slot_index]
                state_local = env.get_state()
                action_local = env.select_action(state_local)  # uses assigned DQN
                next_state_local, _, _, info = env.step(action_local)
                # env.step should set env.last_info
                env.last_info = info
                # Tính toán các thông số
                eMBBThr = info.get("eMBB_thr")
                URLLCCapa = info.get("URLLCCapa")
                eachRUutilPower[r].append(info.get("utilPower"))
                eachRUutilPRB[r].append(info.get("utilPRB"))
                num_used += np.count_nonzero(env.x_alloc)
                power_rate += sum(env.power_alloc)
                num_prb += env.num_PRB
                
                # Cộng các giá trị lại cho từng RU
                for i in range(self.num_slices):
                    if self.slices[i].name == "eMBB":
                        eMBBThr_slot[i]+=eMBBThr[i]
                    else :
                        URLLCCapa_slot[i]+=URLLCCapa[i]
            self.update_buffers(eMBBThr_slot)
            # Tính cho từng slot 
            sla_eMBB_slot = self.calculateEMBBSLA(eMBBThr_slot)
            sla_URLLC_slot = self.calculateSLAURLLC()
            # Cập nhật buffer toàn hệ thống
            for i in range(self.num_slices):
                if self.slices[i].name == "eMBB":
                    eMBBThr_frame[i].append(eMBBThr_slot[i])
                else :
                    URLLCCapa_frame[i].append(URLLCCapa_slot[i])

            sla_eMBB.append(sla_eMBB_slot)
            sla_URLLC.append(sla_URLLC_slot)
            util_slot = num_used / num_prb
            util_prb.append(util_slot)
            util_power.append(power_rate / sum(self.total_power))

            stability = np.exp(-np.mean(np.abs(action - self.last_quota)))
            stab.append(stability)

            sumeMBB_frame.append(sum(eMBBThr_slot[i] for i in range(self.num_slices) 
                                        if self.slices[i].name == "eMBB"))

        # reward
        w = self.reward_weights
        reward = (w["thr"] * (np.average(sumeMBB_frame)/self.T_max)  
                - w["slaeMBB"] * max(self.sla_slices["eMBB"] - np.average(sla_eMBB), 0)  
                - w["slaURLLC"] * max(self.sla_slices["URLLC"] - np.average(sla_URLLC), 0) 
                + w["stab"] * np.average(stab)  
                - w["utilPower"] * np.average(util_power)
                + w["utilPRB"] * np.average(util_prb))

        # update state
        self.last_quota = action.copy()
        self.last_state = self.get_state(eMBBThr_frame, URLLCCapa_frame, sla_eMBB, sla_URLLC, 
                                         util_power, util_prb, eachRUutilPower, 
                                         eachRUutilPRB, stab)
        self.frame_count += 1
        done = False  # episodes managed in training loop

        info = {
            "eMBB_thr": sumeMBB_frame,
            "SLA_urllc": sla_URLLC,
            "SLA_embb": sla_eMBB,
            "stability": stab,
            "utilPower": util_power,
            "utilPRB": util_prb
        }
        return self.last_state, reward, done, info
    

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



