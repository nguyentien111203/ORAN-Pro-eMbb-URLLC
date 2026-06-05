import numpy as np
import gym
from gym import spaces
import torch
from combine.SAC.drawSAC import plot_rate


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
        self.last_BWP_slice = [[[0 for _ in range(len(self.RUs[r].bwps))] for _ in range(self.num_slices)] for r in range(self.num_rus)]

        self.pac = np.array([ue.pac for s in range(self.num_urllc) 
                        for ue in self.RU_envs[0].urllc_slices[s].ue_set])

        self.lat_target = np.array([ue.lat for s in range(self.num_urllc) 
                               for ue in self.RU_envs[0].urllc_slices[s].ue_set])

        self.thr_min = np.array([ue.thr for s in range(self.num_embb) 
                            for ue in self.RU_envs[0].embb_slices[s].ue_set])
        
        self.URLLC_frame = []
        self.eMBB_frame = []
        self.embb_frame_rate = []
        self.urllc_frame_rate = []
        for slot in range(self.frame_slots): 
            self.URLLC_frame.append([])
            self.eMBB_frame.append([])
            self.embb_frame_rate.append([])
            self.urllc_frame_rate.append([])
            for s in range(len(self.urllc_slices)):
                self.URLLC_frame[slot].append(np.zeros(len(self.urllc_slices[s].ue_set), np.float64))
                self.urllc_frame_rate[slot].append(np.zeros(len(self.urllc_slices[s].ue_set), np.float64))
            for s in range(len(self.embb_slices)):
                self.eMBB_frame[slot].append(np.zeros(len(self.embb_slices[s].ue_set), np.float64))
                self.embb_frame_rate[slot].append(np.zeros(len(self.embb_slices[s].ue_set), np.float64))

        # Tính toán các chi phí trên từng frame
        self.costEne = np.zeros(self.frame_slots)
        self.costFrag = np.zeros(self.frame_slots)
        self.costSwit = np.zeros(self.frame_slots)
        self.costGB = np.zeros(self.frame_slots)
        self.rate_dict = {
            "min_urllc": [],
            "avg_urllc": [],
            "min_embb": [],
            "avg_embb": []
        }

    def reset(self):
        """
        Reset lại môi trường ở từng frame cũng như chi phí, throungput và latency
        """
        for env in self.RU_envs:
            env.reset()
        
        for slot in range(self.frame_slots):
            for s in range(len(self.urllc_slices)):
                self.URLLC_frame[slot][s] = np.zeros(len(self.urllc_slices[s].ue_set))
                self.urllc_frame_rate[slot][s] = np.zeros(len(self.urllc_slices[s].ue_set))
            for s in range(len(self.embb_slices)):
                self.eMBB_frame[slot][s] = np.zeros(len(self.embb_slices[s].ue_set))
                self.embb_frame_rate[slot][s] = np.zeros(len(self.embb_slices[s].ue_set))

        # Tính toán các chi phí trên từng frame
        self.costEne = np.zeros(self.frame_slots)
        self.costFrag = np.zeros(self.frame_slots)
        self.costSwit = np.zeros(self.frame_slots)
        self.costGB = np.zeros(self.frame_slots)

        self.slot_count = 0


    def step(self, action):
        self.resetAlloc()
        self.reset()
        # Setup trạng thái đầu của các DQN
        DQNstate = []
        for r in range(self.num_rus):
            DQNstate.append(np.zeros(self.RU_envs[0].state_dim))

        # Action ở đây là tỷ lệ phân bổ
        # Cần dịch từ tỷ lệ ra số PRB trước khi đưa vào select_action
        budget_BWP_slice = self.convertRateToPRB(action)
        
        for slot_index in range(self.frame_slots):
            numBits = np.array([1e-7 for s in range(self.num_urllc) 
                                for ue in self.RU_envs[0].urllc_slices[s].ue_set], np.float64)
            Thr = np.array([0 for s in range(self.num_embb) 
                                for ue in self.RU_envs[0].embb_slices[s].ue_set], np.float64)
            # Cập nhật và lấy thông tin từ các RU
            for r, env in enumerate(self.RU_envs):
                # Cập nhật H, thực hiện hành động và tính toán throughput, latency và 
                # các chi phí dựa trên action của SAC
                env.update_H(self.H[r][slot_index])
                DQNaction = env.select_action(DQNstate[r], budget_BWP_slice[r])

                # computeOutput trả numpy array (slice x ue)
                ruBits, ruThr = env.computeOutput(DQNaction)

                # Tính toán thr và latency
                numBits += ruBits
                Thr += ruThr


            # Kết quả dạng phẳng
            flatlatency = self.pac / numBits
            flat_urllc_rate = flatlatency / self.lat_target
            flat_embb_rate = Thr / self.thr_min

            self.rate_dict["avg_embb"].append(np.average(np.average(flat_embb_rate)))
            self.rate_dict["avg_urllc"].append(np.average(np.average(flat_urllc_rate)))
            self.rate_dict["min_embb"].append(np.min(np.min(flat_embb_rate)))
            self.rate_dict["min_urllc"].append(np.min(np.min(flat_urllc_rate)))

            # Tính toán và dịch lại cho từng frame
            offset = 0
            for s in range(len(self.URLLC_frame[slot_index])):
                for u in range(len(self.URLLC_frame[slot_index][s])):
                    self.URLLC_frame[slot_index][s][u] = flatlatency[offset]
                    self.urllc_frame_rate[slot_index][s][u] = flat_urllc_rate[offset]
                    offset += 1

            offset = 0
            for s in range(len(self.eMBB_frame[slot_index])):
                for u in range(len(self.eMBB_frame[slot_index][s])):
                    self.eMBB_frame[slot_index][s][u] = Thr[offset]
                    self.embb_frame_rate[slot_index][s][u] = flat_embb_rate[offset]
                    offset += 1  
            
            # Áp action vào tính toán và 
            for r, env in enumerate(self.RU_envs):
                DQNstate[r], _, _, info = env.step(flat_urllc_rate, flat_embb_rate)

                self.costEne[slot_index] += info["costE"]
                self.costFrag[slot_index] += info["costF"]
                self.costSwit[slot_index] += info["costS"]
                self.costGB[slot_index] += info["costGB"]

        # Tính trung bình trong 1 frame
        
        eMBB_frame_avg = [[np.average([self.embb_frame_rate[slot][s][u] 
                                      for slot in range(self.frame_slots)]) \
                          for u in range(len(self.eMBB_frame[0][s]))] 
                          for s in range(len(self.eMBB_frame[0]))]
        URLLC_frame_avg = [[np.average([self.urllc_frame_rate[slot][s][u] 
                                for slot in range(self.frame_slots)]) \
                          for u in range(len(self.URLLC_frame[0][s]))]
                          for s in range(len(self.URLLC_frame[0]))] 
                          
        eMBB_frame_soft = [[eMBB_frame_avg[s][u] / (1 + (eMBB_frame_avg[s][u])) 
                           for u in range(len(eMBB_frame_avg[s]))]
                           for s in range(len(eMBB_frame_avg))] 
                           
        
        URLLC_frame_soft = [[1 / (1 + (URLLC_frame_avg[s][u])) 
                           for u in range(len(URLLC_frame_avg[s]))]
                           for s in range(len(URLLC_frame_avg))]
                           

        # Tính tỷ lệ trễ và throughput trung bình
        embb_avg = np.sum(eMBB_frame_soft[s][e] for s in range(len(self.embb_slices)) 
                          for e in range(len(self.embb_slices[s].ue_set)))
        urllc_avg = np.sum(URLLC_frame_soft[s][u] for s in range(len(self.urllc_slices)) 
                          for u in range(len(self.urllc_slices[s].ue_set)))

        # reward
        reward = ((self.w_reward["thr"] *  embb_avg 
                + self.w_reward["lat"] *  urllc_avg
                + self.w_reward["cost"] * (4 - (np.average(self.costEne) / (self.scale_max[0]))  \
                                        -(np.average(self.costFrag) / (self.scale_max[1])) \
                                        - (np.average(self.costSwit) / (self.scale_max[2])) \
                                        - (np.average(self.costGB) / (self.scale_max[3])))))

        info = {
            "thr": eMBB_frame_avg,
            "lat": URLLC_frame_avg,
            "costE": self.costEne,
            "costF": self.costFrag,
            "costS": self.costSwit,
            "costGB": self.costGB
        }

        # Reset lại phân bổ
        self.last_BWP_slice = self.BWP_slice
        next_state = self.get_state(eMBB_frame_avg, URLLC_frame_avg, self.costEne, self.costFrag, self.costSwit, self.costGB)
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


    def convertRateToPRB(self, BWP_slice_rate):
        """
        Chuyển tỷ lệ phân bổ của SAC thành số PRB thực tế.
        Input:
            BWP_slice_rate[r][b][s]
                = tỷ lệ PRB của slice s trên BWP b tại RU r
        Output:
            budget_BWP_slice[r][b][s]
                = số PRB integer thực tế
        Đảm bảo:
            sum_s budget_BWP_slice[r][b][s] <= num_prb
        """

        budget_BWP_slice = []
        for r in range(self.num_rus):
            ru_alloc = []
            for b in range(len(self.RUs[r].bwps)):
                num_prb = self.RUs[r].bwps[b].num_prb
                # ---- lấy tỷ lệ ----
                rates = np.array(BWP_slice_rate[r][b], dtype=np.float32)
                # tránh âm
                rates = np.maximum(rates, 0.0)
                # scale sang PRB thực
                raw_alloc = rates * num_prb
                # floor trước
                alloc = np.floor(raw_alloc).astype(int)
                # số PRB còn dư
                remain = num_prb - np.sum(alloc)
                # ---- phân bổ phần dư theo fractional lớn nhất ----
                if remain > 0:
                    frac = raw_alloc - alloc
                    order = np.argsort(frac)[::-1]
                    for idx in order:
                        if remain <= 0:
                            break
                        alloc[idx] += 1
                        remain -= 1

                # ---- nếu vượt budget (rare case do numeric) ----
                elif remain < 0:
                    overflow = -remain
                    order = np.argsort(alloc)[::-1]
                    for idx in order:
                        if overflow <= 0:
                            break
                        take = min(alloc[idx], overflow)
                        alloc[idx] -= take
                        overflow -= take

                ru_alloc.append(alloc.tolist())

            budget_BWP_slice.append(ru_alloc)

        return budget_BWP_slice

    def drawChart(self):
        plot_rate(self.rate_dict, "URLLC", "min")
        plot_rate(self.rate_dict, "URLLC", "avg")
        plot_rate(self.rate_dict, "URLLC", "gap")

        plot_rate(self.rate_dict, "eMBB", "min")
        plot_rate(self.rate_dict, "eMBB", "avg")
        plot_rate(self.rate_dict, "eMBB", "gap")

    def resetRateDict(self):
        self.rate_dict = {
            "min_urllc": [],
            "avg_urllc": [],
            "min_embb": [],
            "avg_embb": []
        }