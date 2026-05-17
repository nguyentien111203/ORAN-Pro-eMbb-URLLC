# Mỗi RU sẽ hành động như là 1 independent agent
import gym
from gym import spaces
import numpy as np 
from combine.common.multiagent_DQN import MultiHeadDQNAgent
import copy

class RU_Env(gym.Env):
    """
    Environment cho từng RU cho slice ở từng frame.
    - Mức này tương tác trực tiếp với DQN agent.
    - Mỗi step tương ứng với 1 slot (subframe nhỏ).
    - Có thể nhận quota power từ SAC thông qua FrameEnv.

    RadioUnit RU : RU hiện tại đang xét 
    set of Slice slices: tập các slice embb,urllc đang xét
    int num_embb : số slice embb
    int num_urllc: số slice urllc
    list H : kênh truyền giữa từng PRB của từng BWP tới từng UE
    inter_RU : hệ số nhân với nhiễu nhiệt để cho ra nhiễu liên RU (đang để nhiễu inter-RU là hằng số)
    inter_factor : hệ số nhân tính trong nhiễu liên BWP
    w_reward : trọng số trong hàm phần thưởng
    cost_switch : hệ số chi phí chuyển BWP
    cost_gb : hệ số chi phí guard band
    scale_max : các giá trị scale cho từng thành phần trong reward
    """

    def __init__(self, RU, slices, num_urllc, num_embb, H, inter_RU, inter_factor, N0, w_reward, cost_switch, 
                 cost_gb, scale_max, trainCons, frame_slots):
        super().__init__()
        self.RU = RU
        self.slices = slices
        self.num_urllc = num_urllc
        self.num_embb = num_embb
        self.num_slices = num_urllc + num_embb
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

        # Quản lý tài nguyên tổng hợp
        self.PRB_slice = [[0 for b in range(len(self.RU.bwps))] for _ in range(self.num_slices)]
        self.last_PRB_slice = copy.deepcopy(self.PRB_slice)

        # Cấu hình kích thước State và Action 
        self.num_embb_ue = [len(self.slices[s].ue_set) for s in range(self.num_embb)]
        self.num_urllc_ue = [len(self.slices[s].ue_set) for s in range(self.num_embb, self.num_slices)]
        
        # State dim
        self.state_dim = sum(self.num_embb_ue) + sum(self.num_urllc_ue) + \
                         sum(self.num_embb_ue) + sum(self.num_urllc_ue) + 4 + 1
                         
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([self.RU.B_r] * sum(self.num_embb_ue + self.num_urllc_ue))

        # Lưu trữ Metrics QoS và Hiệu suất phổ
        self.eMBB_Thr = [[0.0 for _ in range(n)] for n in self.num_embb_ue]  
        self.urllc_Lat = [[0.0 for _ in range(n)] for n in self.num_urllc_ue]
        self.numBit_urllc = [[0.0 for _ in range(n)] for n in self.num_urllc_ue]  
        
        self.spectral_eff_avg = [[0.0 for _ in range(n)] for n in self.num_embb_ue]
        self.spectral_eff_min = [[0.0 for _ in range(n)] for n in self.num_urllc_ue]

        # Quản lý ma trận phân bổ và trạng thái BWP
        self.alloc = [[[0 for u in range(len(self.slices[i].ue_set))] for b in range(len(self.RU.bwps))] for i in range(self.num_slices)]
        self.snr = [[[0.0 for u in range(len(self.slices[i].ue_set))] for b in range(len(self.RU.bwps))] for i in range(self.num_slices)]
        
        self.BWP_slice = [[0 for b in range(len(self.RU.bwps))] for _ in range(self.num_slices)]
        self.last_BWP_slice = copy.deepcopy(self.BWP_slice)
        
        self.index_subframe = 0 
        self.dqn_agent = None
        self.state = []

    def assign_dqn_agent(self, agent):
        self.dqn_agent = agent

    def update_H(self, Hnew):
        self.H = Hnew


    def get_state(self, rate_ratio,spectral_eff, cEne, cFrag, cSwit, cGB, stab): #hiệu quả phổ cho tất cả slice, hiệu suất phổ min cho slice urllc-> tính state dim
        flat_rate_ratio = np.array([rate_ratio[s][u] for s in range(self.num_embb) for u in range (self.num_embb_ue[s])])
        flat_spectral_eff = np.array([spectral_eff[s][u] for s in range(self.num_embb) for u in range (self.num_embb_ue[s])])
        flat_eff_avg = np.array([e for s in self.spectral_eff_avg for e in s])
        flat_eff_min = np.array([e for s in self.spectral_eff_min for e in s])
        state = np.concatenate((
            flat_rate_ratio,
            flat_spectral_eff,
            flat_eff_avg,
            flat_eff_min, 
            np.array([
                self.scale_max[0] / (cEne + 1e-7),
                self.scale_max[1] / (cFrag + 1e-7),
                self.scale_max[2] / (cSwit + 1e-7),
                self.scale_max[3] / (cGB + 1e-7),
                stab
            ])
        ))
         
        return state.astype(np.float32)


    def reset(self):
        """Reset môi trường về trạng thái ban đầu."""
        self.index_subframe = 0
        self.eMBB_Thr_Ratio = [[0.0 for _ in range(n)] for n in self.num_embb_ue]
        self.URLLC_Lat_Ratio = [[0.0 for _ in range(n)] for n in self.num_urllc_ue]
        self.spectral_eff_avg = [[0.0 for _ in range(n)] for n in self.num_embb_ue]
        self.spectral_eff_min = [[0.0 for _ in range(n)] for n in self.num_urllc_ue]
        
        self.alloc = [[[0 for u in range(len(self.slices[i].ue_set))] for b in range(len(self.RU.bwps))] for i in range(self.num_slices)]
        self.PRB_slice = [[0 for b in range(len(self.RU.bwps))] for _ in range(self.num_slices)]
        self.BWP_slice = [[0 for b in range(len(self.RU.bwps))] for _ in range(self.num_slices)]
        
        self.last_PRB_slice = copy.deepcopy(self.PRB_slice)
        self.last_BWP_slice = copy.deepcopy(self.BWP_slice)
        
        
        return self.get_state(cEne=0.0, cFrag=0.0, cSwit=0.0, cGB=0.0, stab=1.0)
        
    def computeOutput(self, action):
        """
        Tính toán thorughput và latency + numbit gửi môi trường chung để return
        
        """
        self.reset() # Reset lại số bit về ban đầu
        
        # Lưu lại phân bổ số PRB từ action
        self.alloc = action
        self.transBWPslice()

        # Tính toán nhiễu, SINR và trễ khi phân bổ
        interBWP = self.calculateMultiBWPNoise()
        self.calculateSNR(interBWP)
        self.calculatethroughput()
        self.calculatelatency()
        self.calculatenumbit()
        return self.eMBB_Thr, self.numBit, self.urllc_Lat


    def step(self, totalThrRate, totalLatRate):
        """
        Sau khi thực hiện action, tính toán reward, tìm ra state tiếp theo và 
        trả về các thông tin về trễ của từng embb

        action : hành động phân bổ PRB từ từng slice về các UE

        return state_next : trạng thái tiếp theo
               float reward : reward khi action được thực hiện
               done : đã thực hiện xong trong frame chưa để reset lại index_subframe
               info : thông tin về trễ của từng UE urllc trong các slice
        """
        if self.index_subframe < self.frame_slots:
            self.index_subframe += 1
        else :
            self.index_subframe = 0

        #totalthroughput = self.computeOutput(action)
        self.eMBB_Thr = totalThrRate
        self.urllc_Lat = totalLatRate
        self.calculatethroughput()
        self.calculateSpectralEfficiency() #tính cho cả 2 loại
        # có thể thêm hàm min tính cho urllc rồi tính state

        # Tính toán các chi phí
        cEne = self.calculateEnergy()
        cFrag = self.calculateFrag()
        cSwit = self.calculateSwitch()
        cGB = self.calculateGuardBand()
        stab = self.calculateStab()


       # 1. Thành phần eMBB 
        thr_reward = self.w_reward["thr"] * sum(
            totalThrRate[s][u] - 1 
            for s in range(self.num_embb) 
            for u in range(len(self.slices[s].ue_set))
        )
        
        # 2. Thành phần URLLC
        lat_reward = (1 - self.w_reward["lat"]) * sum(
            totalLatRate[l][u] - 1 
            for l in range(self.num_urllc) 
            for u in range(len(self.slices[l].ue_set))
        )
        
        # 3. Thành phần Chi phí 
        cost_reward = self.w_reward["cost"] * (
            (self.scale_max[0] / (cEne + 1e-7)) + 
            (self.scale_max[1] / (cFrag + 1e-7)) + 
            (self.scale_max[2] / (cSwit + 1e-7)) + 
            (self.scale_max[3] / (cGB + 1e-7))
        )
        
        
        reward = thr_reward + lat_reward + cost_reward + stab

        # Kiểm tra xem đã sang frame mới chưa
        done = self.index_subframe > self.frame_slots

        # Đưa ra state tiếp theo và action tiếp
        next_state = self.get_state(totalThrRate, cEne, cFrag, cSwit, cGB, stab)
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
        leakage = 1e-7
        interBWP = np.zeros(self.num_embb)
        for u in range(self.num_embb):
            for b in range(len(self.RU.bwps)):
                for bk in range(b + 1, len(self.RU.bwps)):
                    if self.RU.bwps[b].bandindex != self.RU.bwps[bk].bandindex:
                        gapIndex = np.abs(self.RU.bwps[b].bandindex - self.RU.bwps[bk].bandindex)
                        interBWP[u] += leakage * self.inter_factor * (1/(1+gapIndex)) \
                            * self.RU.bwps[bk].p_each_PRB * self.BWP_slice[bk][u]
        return interBWP


    def calculateSNR(self, interBWP):
        """
        Tính toán lại tỷ lệ SINR ở từng BWP với từng UE trong RU
        """
        for i in range(self.num_slices):
            for b in range(len(self.RU.bwps)):
                for u in range(len(self.slices[i].ue_set)): 
                    self.snr[i][b][u] = (self.RU.bwps[b].p_each_PRB * self.H[i][u]) \
                        / (self.RU.bwps[b].bandwidth * self.N0 * (self.inter_RU + 1) + interBWP[u]) 


    def calculateSpectralEfficiency(self):
        """Tính toán gamma_avg cho eMBB phục vụ cấu trúc State """
        for i in range(self.num_embb):
            for u in range(self.num_embb_ue[i]):
                total_sinr = sum([self.snr[i][b][u] for b in range(len(self.RU.bwps))])
                avg_sinr = total_sinr / len(self.RU.bwps) if len(self.RU.bwps) > 0 else 0
                self.spectral_eff_avg[i][u] = np.log2(1 + avg_sinr)

    def calculatethroughput(self):
        """Tính thông lượng eMBB thực tế tích lũy."""
        for i in range(self.num_embb):
            for u in range(self.num_embb_ue[i]):
                self.eMBB_Thr[i][u] = 0.0
                for b in range(len(self.RU.bwps)):
                    self.eMBB_Thr[i][u] += self.alloc[i][b][u] * self.RU.bwps[b].bandwidth * np.log2(1 + self.snr[i][b][u])

    def calculatelatency(self):
        """Tính toán trễ URLLC dựa trên Payload kích thước gói Eq (24)."""
        for i in range(self.num_embb, self.num_slices):
            idx = i - self.num_embb
            for u in range(self.num_urllc_ue[idx]):
                R_u = sum([self.alloc[i][b][u] * (self.RU.bwps[b].bw * self.RU.bwps[b].time * np.log2(1 + self.snr[i][b][u]))
                           for b in range(len(self.RU.bwps))])
                l_u = self.slices[i].ue_set[u].pac / max(R_u, 1e-6)
                self.urllc_Lat[idx][u] = self.slices[i].ue_set[u].lat / max(l_u, 1e-6)
                
                min_sinr = np.min([self.snr[i][b][u] for b in range(len(self.RU.bwps))])
                self.spectral_eff_min[idx][u] = np.log2(1 + min_sinr)
                    
    def calculatenumbit(self):
        """Tính tổng số bit URLLC phục vụ hàm báo cáo Eq (23)."""
        for i in range(self.num_urllc):
            slice_idx = i + self.num_embb
            for u in range(len(self.slices[slice_idx].ue_set)):
                self.numBit_urllc[i][u] = 0.0
                for b in range(len(self.RU.bwps)):
                    self.numBit_urllc[i][u] += self.RU.bwps[b].time * self.alloc[slice_idx][b][u] * np.log2(1 + self.snr[slice_idx][b][u])
    
        

    def calculateFrag(self):
        """
        Tính toán chi phí phân mảnh PRB
        """
        general_frag = 0
        for i in range(self.num_slices):
            general_frag += (np.sum(self.PRB_slice[i]))**2 # Cộng với bình phương số PRB phân cho slice của từng slice
        return general_frag
    

    def calculateEnergy(self):
        """
        Tính toán năng lượng tiêu tốn
        """
        general_energy = 0
        for i in range(self.num_slices):
            for u in range(len(self.slices[i].ue_set)):
                for b in range(len(self.RU.bwps)):
                    general_energy += self.alloc[i][b][u] * self.RU.bwps[b].p_each_PRB * self.RU.bwps[b].time

        return general_energy


    def calculateSwitch(self):
        """
        Tính toán chi phí chuyển BWP 
        """
        general_switch = 0
        for i in range(self.num_slices):
            for b in range(len(self.RU.bwps)):
                general_switch += self.cost_switch * self.BWP_slice[i][b] * (1 - self.last_BWP_slice[i][b])
        return general_switch


    def calculateGuardBand(self):
        """
        Tính toán chi phí guardband
        """
        general_guard = 0
        for i in range(self.num_slices):
            for b in range(len(self.RU.bwps)):
                for bk in range(len(self.RU.bwps)):
                    if bk != b:
                        general_guard += self.cost_gb * np.abs(b - bk) * self.BWP_slice[i][b] * self.BWP_slice[i][bk]
        return general_guard


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
