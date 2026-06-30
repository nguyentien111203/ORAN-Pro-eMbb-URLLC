# Mỗi RU sẽ hành động như là 1 independent agent
import gym
from gym import spaces
import numpy as np 
from combine.common.multiagent_DQN import MultiHeadDQNAgent
import matplotlib.pyplot as plt


class RU_Env(gym.Env):
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

    def __init__(self, RU, embb_slices, urllc_slices, H, inter_RU, inter_factor, N0, w_reward, cost_switch, 
                 cost_gb, scale_max, trainCons, frame_slots):
        super().__init__()
        self.RU = RU
        self.embb_slices = embb_slices
        self.urllc_slices = urllc_slices
        self.H = H  
        self.inter_RU = inter_RU
        self.inter_factor = inter_factor
        self.N0 = N0
        self.w_reward = w_reward
        self.cost_switch = cost_switch
        self.cost_gb = cost_gb
        self.scale_max = scale_max
        self.trainCons = trainCons
        self.frame_slots = frame_slots

        # Tính toán các tham số phụ
        self.num_embb = len(self.embb_slices)
        self.num_urllc = len(self.urllc_slices)
        self.num_slices = self.num_embb + self.num_urllc

        # Số PRB phân cho mỗi slice, đây cũng là thông tin trao đổi từ SAC về DQN của URLLC 
        self.PRB_slice = [[0 for b in range(len(self.RU.bwps))] for _ in range(self.num_slices)]
        self.last_PRB_slice = [[0 for b in range(len(self.RU.bwps))] for _ in range(self.num_slices)]

        # State và action 
        self.num_embb_ue = [len(self.embb_slices[s].ue_set) for s in range(self.num_embb)]
        self.num_urllc_ue = [len(self.urllc_slices[s].ue_set) for s in range(self.num_urllc)]

        # Tập số ue mỗi slice (eMBB được tính trước)
        self.num_ue = np.concatenate([self.num_embb_ue, self.num_urllc_ue])
        
        # [QoS_ratios, averSinr, minSinr, 4_costs, psi]
        self.state_dim = 2 * sum(self.num_embb_ue) + 3 * sum(self.num_urllc_ue) + 5  
        
        self.observation_space = spaces.Box(
            low=0, 
            high=np.inf, # Ratios có thể lớn hơn 1
            shape=(self.state_dim,), 
            dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete(max(max(self.num_urllc_ue), max(self.num_embb_ue)) * self.RU.B_r)

        # Tính toán số bit mà các UE urllc nhận được
        self.numBit = [[0 for _ in range(self.num_urllc_ue[s])] for s in range(self.num_urllc)]  
        
        # Dict các latency của các slice URLLC 
        self.URLLC_Latency = [[0 for _ in range(self.num_urllc_ue[s])] for s in range(self.num_urllc)]  

        # Throughput các UE eMBB
        self.eMBB_Thr = np.array([[0 for _ in range(self.num_embb_ue[s])] for s in range(self.num_embb)], np.float64)

        # List các số PRB của từng BWP phân bổ cho các UE
        self.alloc = [[[0 for u in range(self.num_ue[i])] for b in range(len(self.RU.bwps))] for i in range(self.num_slices)]

        # Tỷ lệ SINR của từng UE với từng PRB của từng BWP
        self.snr = [[[0 for u in range(self.num_ue[i])] for b in range(len(self.RU.bwps))] for i in range(self.num_slices)]
        
        self.index_subframe = 0 # index của subframe đang xét hiện tại

        # Ma trận thể hiện slice URLLC có sử dụng BWP không, các giá trị là 0,1
        self.BWP_slice = [[0 for b in range(len(self.RU.bwps))] for _ in range(self.num_slices)]
        self.last_BWP_slice = [[0 for b in range(len(self.RU.bwps))] for _ in range(self.num_slices)]
        
        # DQN agent của RU
        self.dqn_agent = None

        # State của DQN hiện tại
        self.state = []


    def assign_agent(self, agent):
        """Gán DQN agent để dùng khi FrameEnv gọi env.select_action(), bắt buộc"""
        self.dqn_agent = agent


    def update_H(self, Hnew):
        """
        Cập nhật kênh truyền từ các UE tới slice trong RU
        """
        self.H = Hnew


    def get_state(self, urllc_rate, embb_rate, averUE, minUE, cEne, cFrag, cSwit, cGB, stab):
        """
        Tạo ma trận state cho DQN, bắt buộc
        urllc_rate, embb_rate : tỷ lệ phục vụ (latency, throughput)
        averUE : hiệu suất phổ (trung bình trên các UE)
        minUE : hiệu suất phổ (min trên các UE URLLC)
        cEne, cFrag, cSwit, cGB : chi phí các loại (năng lượng, phân mảnh, chuyển BWP, guardband)
        stab : độ ổn định trong quyết định
        """
        # Bẻ thẳng averUE, minUE
        averUE_flat = np.array([averUE[s][u] for s in range(self.num_slices) for u in range(self.num_ue[s])])
        minUE_flat = np.array([minUE[s][u] for s in range(self.num_urllc) for u in range(self.num_urllc_ue[s])])
        
        state = np.concatenate((
            urllc_rate,
            embb_rate,
            averUE_flat,
            minUE_flat,
            np.array([
                cEne/self.scale_max[0],
                cFrag/self.scale_max[1],
                cSwit/self.scale_max[2],
                cGB/self.scale_max[3],
                stab
            ])
        ))
         
        return state.astype(np.float32)


    def select_action(self, state, BWP_slice):
        """
        Nếu có DQN agent thì dùng policy của nó, nếu không thì random.
        BWP_slice : phân bổ PRB từ từng BWP về slice
        """
        if self.dqn_agent is not None:
            return self.dqn_agent.select_action(state, BWP_slice)
        return self.action_space.sample()


    def reset(self):
        """
        Reset lại ma trận phân bổ khi xét đến slot mới (reset lại số bit mà mỗi UE nhận, 
        và ghi đè lên các giá trị SINR, latency và số PRB được phân bổ (alloc))
        """
        self.numBit = [[0 for _ in range(self.num_urllc_ue[s])] for s in range(self.num_urllc)]  
        
        self.URLLC_Latency = [[0 for _ in range(self.num_urllc_ue[s])] for s in range(self.num_urllc)]  

        self.eMBB_Thr = [[0 for _ in range(self.num_embb_ue[s])] for s in range(self.num_embb)]
        self.index_subframe = 0


    def computeOutput(self, action):
        """
        Tính số bit mà các slice nhận được
        """
        # Lưu lại phân bổ số PRB từ action
        self.alloc = action
        self.transBWPslice()

        # Tính toán nhiễu, SINR và trễ khi phân bổ
        interBWP = self.calculateMultiBWPNoise()
        self.calculateSNR(interBWP)
        self.calculateNumBit()
        self.calculateThroughput()

        flatBit = np.array([self.numBit[s][u] for s in range(self.num_urllc) 
                            for u in range(len(self.urllc_slices[s].ue_set))], np.float64)
        
        flatThr = np.array([self.eMBB_Thr[s][u] for s in range(self.num_embb) 
                            for u in range(len(self.embb_slices[s].ue_set))], np.float64)
        #print(action)
        return flatBit, flatThr


    def step(self, totalLatRate, totalThrRate, latSoft, thrSoft):
        """
        Sau khi thực hiện action, tính toán reward, tìm ra state tiếp theo và 
        trả về các thông tin về trễ của từng urllc

        action : hành động phân bổ PRB từ từng slice về các UE

        return state_next : trạng thái tiếp theo
               float reward : reward khi action được thực hiện
               done : đã thực hiện xong trong frame chưa để reset lại index_subframe
               info : thông tin về trễ của từng UE urllc trong các slice, bắt buộc
        """
        self.index_subframe += 1

        self.reset() # Reset lại số bit về ban đầu
        
        # Tính toán các chi phí
        cEne = self.calculateEnergy()
        cFrag = self.calculateFrag()
        cSwit = self.calculateSwitch()
        cGB = self.calculateGuardBand()
        stab = self.calculateStab()
        averUE = self.calculateAverUE()
        minUE = self.calculateMinUE()

        # Tính toán hiệu quả sử dụng tài nguyên
        total_alloc = sum(sum(sum(u) for u in b) for b in self.alloc)
        eff = (sum(thrSoft) + sum(latSoft)) / (total_alloc + 1e-9)
        lamda_eff = 0.5

        reward = (self.w_reward["lat"] * np.average(latSoft) + \
                self.w_reward["thr"] * np.average(thrSoft) + \
                self.w_reward["cost"] * (4 - (cEne/self.scale_max[0]) - (cFrag/self.scale_max[1]) - \
                                         (cSwit/self.scale_max[2]) - (cGB/self.scale_max[3])) + stab) + \
                lamda_eff * eff
        
        #print( "eMBB min, mean, max : ",
        #    np.min(totalThrRate), ' ',
        #    np.mean(totalThrRate), ' ', 
        #    np.max(totalThrRate), '\n'
        #    )

        #print( "urllc min, mean, max : ",  
        #    np.min(totalLatRate), ' ',
        #    np.mean(totalLatRate), ' ',
        #    np.max(totalLatRate), '\n'
        #    )
        
        done = self.index_subframe >= self.frame_slots

        # Đưa ra state tiếp theo và action tiếp
        next_state = self.get_state(totalLatRate, totalThrRate, averUE, minUE, 
                                    cEne, cFrag, cSwit, cGB, stab)
        self.last_PRB_slice = self.PRB_slice

        # Thông tin các thứ
        info = {
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
        leakage = 1e-13
        interBWP = np.zeros(self.num_slices)
        for u in range(self.num_slices):
            for b in range(len(self.RU.bwps)):
                for bk in range(b + 1, len(self.RU.bwps)):
                    if self.RU.bwps[b].band_index != self.RU.bwps[bk].band_index:
                        gapIndex = np.abs(self.RU.bwps[b].band_index - self.RU.bwps[bk].band_index)
                        interBWP[u] += leakage * self.inter_factor * (1/(1+gapIndex)) \
                            * self.RU.bwps[bk].p_each_PRB * self.BWP_slice[u][bk]
        return interBWP


    def calculateSNR(self, interBWP):
        """
        Tính toán lại tỷ lệ SINR ở từng BWP với từng UE trong RU
        """
        for i in range(self.num_slices):
            for b in range(len(self.RU.bwps)):
                for u in range(self.num_ue[i]): 
                    noise = self.RU.bwps[b].bandwidth * self.N0 * (self.inter_RU + 1) + interBWP[u]
                    self.snr[i][b][u] = (self.RU.bwps[b].p_each_PRB * self.H[i][u]) \
                        / noise
        #print("max SINR : ", np.max(self.snr) ,'\n')
        #print("min SINR : ", np.min(self.snr) ,'\n')
        #print("aver SINR : ", np.average(self.snr) ,'\n')


    def calculateNumBit(self):
        """
        Tính toán số bit mà các UE nhận được
        """
        for i in range(self.num_urllc):
            for u in range(self.num_urllc_ue[i]):
                for b in range(len(self.RU.bwps)):
                    self.numBit[i][u] += self.RU.bwps[b].time * self.alloc[i+self.num_embb][b][u] * np.log2(1 + self.snr[i+self.num_embb][b][u])


    def calculateThroughput(self):
        """
        Tính toán throughput của eMBBs
        """
        for i in range(self.num_embb):
            for u in range(self.num_embb_ue[i]):
                for b in range(len(self.RU.bwps)):
                    self.eMBB_Thr[i][u] += self.alloc[i][b][u] * self.RU.bwps[b].bandwidth * \
                                           np.log2(1 + self.snr[i][b][u])


    def calculateFrag(self):
        """
        Tính toán chi phí phân mảnh PRB
        """
        urllc_frag = 0
        for i in range(self.num_urllc):
            urllc_frag += (np.sum(self.PRB_slice[i]))**2 # Cộng với bình phương số PRB phân cho slice của từng slice
        return urllc_frag
    

    def calculateEnergy(self):
        """
        Tính toán năng lượng tiêu tốn
        """
        urllc_energy = 0
        for i in range(self.num_slices):
            for u in range(self.num_ue[i]):
                for b in range(len(self.RU.bwps)):
                    urllc_energy += self.alloc[i][b][u] * self.RU.bwps[b].p_each_PRB * self.RU.bwps[b].time

        return urllc_energy


    def calculateSwitch(self):
        """
        Tính toán chi phí chuyển BWP 
        """
        urllc_switch = 0
        for i in range(self.num_slices):
            for b in range(len(self.RU.bwps)):
                urllc_switch += self.cost_switch * self.BWP_slice[i][b] * (1 - self.last_BWP_slice[i][b])
        return urllc_switch


    def calculateGuardBand(self):
        """
        Tính toán chi phí guardband
        """
        urllc_guard = 0
        for i in range(self.num_slices):
            for b in range(len(self.RU.bwps)):
                for bk in range(len(self.RU.bwps)):
                    if bk != b:
                        urllc_guard += self.cost_gb * np.abs(b - bk) * self.BWP_slice[i][b] * self.BWP_slice[i][bk]
        return urllc_guard


    def calculateStab(self):
        """
        Tính toán chỉ số biến động Psi(t) theo phân bổ.
        
        Args:
            current_allocations (np.ndarray): Mảng phân bổ PRB hiện tại y(t), 
                                            shape: (num_slices, num_ues_per_slice, num_bwps)
            last_allocations (np.ndarray): Mảng phân bổ PRB ở subframe trước y(t-1)
            
        Returns:
            float: Chỉ số Psi(t) trong khoảng (exp(-1), 1]
        """
        # Nếu là subframe đầu tiên, chưa có lịch sử thì coi như ổn định tuyệt đối
        if self.last_PRB_slice is None:
            return 1.0
        
        # Tính tổng số thay đổi 
        changes = np.not_equal(self.PRB_slice, self.last_PRB_slice)
        total_changes = np.sum(changes)
        
        # Tính tổng không gian quyết định 
        total_elements = self.last_PRB_slice.__sizeof__()
        
        # Tính tỷ lệ biến động
        variation_ratio = total_changes / (total_elements + 1e-9)
        
        # 4. Áp dụng hàm exp(-x)
        psi = np.exp(-variation_ratio)
        
        return float(psi)


    def calculateAverUE(self):
        """
        Tính toán hiệu suất phổ, tính theo tỷ lệ SINR trung bình trên 1 UE
        """
        averUE = [[0 for _ in range(self.num_ue[s])] for s in range(self.num_slices)]
        for s in range(self.num_slices):
            for u in range(self.num_ue[s]):
                averSINR = np.average([self.snr[s][b][u] for b in range(len(self.RU.bwps))])
                averUE[s][u] = np.log2(1 + averSINR)
        return averUE

    
    def calculateMinUE(self):
        """
        Tính toán hiệu suất phổ nhỏ nhất, tính theo tỷ lệ SINR nhỏ nhất trên 1 UE
        """
        minUE = [[0 for _ in range(self.num_urllc_ue[s])] for s in range(self.num_urllc)]
        for s in range(self.num_urllc):
            for u in range(self.num_urllc_ue[s]):
                minSINR = np.min([self.snr[s + self.num_embb][b][u] for b in range(len(self.RU.bwps))])
                minUE[s][u] = np.log2(1 + minSINR)
        return minUE


    def transBWPslice(self):
        """
        Hàm cập nhật việc slice có sử dụng BWP không
        """
        for i in range(self.num_slices):
            for b in range(len(self.RU.bwps)):
                if sum(self.alloc[i][b]) > 0:
                    self.BWP_slice[i][b] = 1
                else:
                    self.BWP_slice[i][b] = 0 


    def returnAlloc(self):
        """
        Hàm trả về phân bổ PRB cho từng UE trong slice
        """
        return self.alloc
