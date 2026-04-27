# Mỗi RU sẽ hành động như là 1 independent agent
import gym
from gym import spaces
import numpy as np 
from combine.common.multiagent_DQN import MultiHeadDQNAgent


class RU_URLLC_Env(gym.Env):
    """
    Environment cho từng RU cho slice URLLC ở từng frame.
    - Mức này tương tác trực tiếp với DQN agent.
    - Mỗi step tương ứng với 1 slot (subframe nhỏ).
    - Có thể nhận quota power từ SAC thông qua FrameEnv.

    RadioUnit RU : RU hiện tại đang xét 
    set of Slice slices: tập các slice URLLC đang xét
    int num_urllc : số slice urllc
    list H : kênh truyền giữa từng PRB của từng BWP tới từng UE
    inter_RU : hệ số nhân với nhiễu nhiệt để cho ra nhiễu liên RU (đang để nhiễu inter-RU là hằng số)
    inter_factor : hệ số nhân tính trong nhiễu liên BWP
    w_reward : trọng số trong hàm phần thưởng
    cost_switch : hệ số chi phí chuyển BWP
    cost_gb : hệ số chi phí guard band
    scale_max : các giá trị scale cho từng thành phần trong reward
    """

    def __init__(self, RU, slices, num_urllc, H, inter_RU, inter_factor, w_reward, cost_switch, cost_gb, scale_max):
        super(self).__init__()
        self.RU = RU
        self.slices = slices
        self.num_urllc = num_urllc
        self.H = H  
        self.inter_RU = inter_RU
        self.inter_factor = inter_factor
        self.w_reward = w_reward
        self.cost_switch = cost_switch
        self.cost_gb = cost_gb
        self.scale_max = scale_max

        # State và action (cái này từ từ mới code)
        self.state_dim = self.num_PRB + self.num_slices + 2  # channel avg, traffic, etc. adjustable
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([self.num_slices] * self.num_PRB)

        # Tính toán số bit mà các UE nhận được
        self.numBit = [[0 for _ in range(len(self.slices[slice_index].ue_set))] for slice_index in range(self.num_urllc)]  
        
        # Dict các latency của các slice URLLC 
        self.URLLC_Latency = [[0 for _ in range(len(self.slices[slice_index].ue_set))] for slice_index in range(self.num_urllc)]  
        
        # List các số PRB của từng BWP phân bổ cho các UE
        self.alloc = [[[0 for u in range(len(self.slices[i].ue_set))] for b in range(len(self.RU.bwps))] for i in range(len(self.num_urllc))]

        # Tỷ lệ SINR của từng UE với từng PRB của từng BWP
        self.snr = [[[0 for u in range(len(self.slices[i].ue_set))] for b in range(len(self.RU.bwps))] for i in range(len(self.num_urllc))]
        
        self.init_numBit = [[0 for _ in range(len(self.slices[slice_index].ue_set))] for slice_index in range(self.num_urllc)]
        self.index_subframe = 0 # index của subframe đang xét hiện tại

        # Số PRB phân cho mỗi slice, đây cũng là thông tin trao đổi từ SAC về DQN của URLLC 
        self.PRB_slice = [[0 for b in range(len(self.RU.bwps))] for _ in range(self.num_urllc)]

        # Ma trận thể hiện slice URLLC có sử dụng BWP không, các giá trị là 0,1
        self.BWP_slice = [[0 for b in range(len(self.RU.bwps))] for _ in range(self.num_urllc)]
        self.last_BWP_slice = [[0 for b in range(len(self.RU.bwps))] for _ in range(self.num_urllc)]

        # DQN agent, (cái này chưa xét vội)
        self.dqn_agent = MultiHeadDQNAgent(self.state_dim, )


    def assign_dqn_agent(self, agent):
        """Gán DQN agent để dùng khi FrameEnv gọi env.select_action()."""
        self.dqn_agent = agent


    def update_H(self, Hnew):
        """
        Cập nhật kênh truyền từ các UE tới slice trong RU
        """
        self.H = Hnew


    def get_state(self):
        """
        Tạo ma trận state cho DQN URLLC (để sau)
        """

        return state.astype(np.float32)


    def select_action(self, state):
        """Nếu có DQN agent thì dùng policy của nó, nếu không thì random."""
        if self.dqn_urllc_agent is not None:
            return self.dqn_urllc_agent.select_action(state)
        return self.action_space.sample()


    def reset(self):
        """
        Reset lại ma trận phân bổ khi xét đến slot mới (reset lại số bit mà mỗi UE nhận, 
        và ghi đè lên các giá trị SINR, latency và số PRB được phân bổ (alloc))
        """
        self.numBit = self.init_numBit
        return self.get_state()


    def step(self, action):
        """
        Sau khi thực hiện action, tính toán reward, tìm ra state tiếp theo và 
        trả về các thông tin về trễ của từng urllc

        action : hành động phân bổ PRB từ từng slice về các UE

        return state_next : trạng thái tiếp theo
               float reward : reward khi action được thực hiện
               done : đã thực hiện xong trong frame chưa để reset lại index_subframe
               info : thông tin về trễ của từng UE urllc trong các slice
        """
        self.index_subframe += 1

        # Lưu lại phân bổ số PRB từ action
        self.alloc = action

        # Tính toán nhiễu, SINR và trễ khi phân bổ
        interBWP = self.calculateMultiBWPNoise()
        self.calculateSNR(interBWP)
        self.calculateNumBit()
        self.calculateLatency()

        # Tính toán các chi phí
        cEne = self.calculateEnergy()
        cFrag = self.calculateFrag()
        cSwit = self.calculateSwitch()
        cGB = self.calculateGuardBand()

        # Tính toán penalty
        penalty = sum(self.URLLC_Latency[s][u] - self.slices[s].ue_set[u].lat 
                      for s in range(len(self.num_urllc)) for u in range(len(self.slices[s].ue_set)))

        reward = (self.w_reward["lat"] * sum(self.URLLC_Latency) / self.scale_max[1]) + \
                self.w_reward["cost"] * ((cEne / self.scale_max[2]) + (cFrag / self.scale_max[3]) + \
                                         (cSwit / self.scale_max[4]) + (cGB / self.scale_max[5])) + \
                self.w_reward["penal"] * penalty

        # Kiểm tra xem đã sang frame mới chưa
        done = self.index_subframe > 9

        # Đưa ra state tiếp theo
        next_state = self.get_state()

        # Thông tin các thứ
        info = {
            "lat" : self.URLLC_Latency,
            "costE" : cEne,
            "costF" : cFrag,
            "costS" : cSwit,
            "costGB" : cGB,
        } 
        self.last_info = info
        return next_state, reward, done, info
    

    def calculateMultiBWPNoise(self):
        """
        Tính toán nhiễu liên BWP
        """
        interBWP = np.zeros(self.num_urllc)
        for u in range(self.num_urllc):
            for b in self.RU.bwps:
                for bk in self.RU.bwps:
                    if bk != b:
                        interBWP[u] += self.inter_factor * np.abs(bk - b) * self.RU.bwps[bk].p_each_PRB * self.BWP_slice[bk]
        return interBWP


    def calculateSNR(self, interBWP):
        """
        Tính toán lại tỷ lệ SINR ở từng BWP với từng UE trong RU
        """
        for i in range(len(self.num_urllc)):
            for b in range(len(self.RU.bwps)):
                for u in range(len(self.slices[i].ue_set)): 
                    self.snr[i][b][u] = (self.RU.bwps[b].p_each_PRB * self.H[self.index_subframe][i][b][u]) \
                        / (self.RU.bwps[b].bandwidth * self.RU.N0 * (self.interRU + 1) + interBWP[u]) 


    def calculateNumBit(self):
        """
        Tính toán số bit mà các UE nhận được
        """
        for i in range(len(self.num_urllc)):
            for u in range(len(self.slices[i].ue_set)):
                for b in range(len(self.RU.bwps)):
                    self.numBit[i][u] += self.RU.bwps[b].time * self.alloc[i][u] * np.log2(1 + self.snr[i][b][u])


    def calculateLatency(self):
        """
        Tính toán trễ của từng slice urllc
        """
        for i in range(len(self.num_urllc)):
            for u in range(len(self.slices[i].ue_set)):
                self.URLLC_Latency[i][u] = self.slices[i].ue_set[u].pac / self.numBit[i][u]


    def calculateFrag(self):
        """
        Tính toán chi phí phân mảnh PRB
        """
        urllc_frag = 0
        for i in range(len(self.num_urllc)):
            urllc_frag += (np.sum(self.PRB_slice[i]))**2 # Cộng với bình phương số PRB phân cho slice của từng slice
        return urllc_frag
    

    def calculateEnergy(self):
        """
        Tính toán năng lượng tiêu tốn
        """
        urllc_energy = 0
        for i in range(len(self.num_urllc)):
            for u in range(len(self.slices[i].ue_set)):
                for b in range(len(self.RU.bwps)):
                    urllc_energy += self.alloc[i][b][u] * self.RU.bwps[b].p_each_PRB * self.RU.bwps[b].time

        return urllc_energy


    def calculateSwitch(self):
        """
        Tính toán chi phí chuyển BWP 
        """
        urllc_switch = 0
        for i in range(len(self.num_urllc)):
            for b in range(len(self.RU.bwps)):
                urllc_switch += self.cost_switch * self.BWP_slice[i][b] * (1 - self.last_BWP_slice[i][b])
        return urllc_switch


    def calculateGuardBand(self):
        """
        Tính toán chi phí guardband
        """
        urllc_guard = 0
        for i in range(len(self.num_urllc)):
            for b in range(len(self.RU.bwps)):
                for bk in range(len(self.RU.bwps)):
                    if bk != b:
                        urllc_guard += self.cost_gb * np.abs(b - bk) * self.BWP_slice[i][b] * self.BWP_slice[i][bk]
        return urllc_guard


    def returnAlloc(self):
        """
        Hàm trả về phân bổ PRB cho từng UE trong slice
        """
        return self.alloc

"""
Khi có state, action và reward hoàn thiện của DQN URLLC thì sẽ xem xem còn thiếu gì thì bổ sung nhé
"""
