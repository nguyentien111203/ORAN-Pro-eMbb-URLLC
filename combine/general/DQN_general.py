import gym
from gym import spaces
import numpy as np 
import copy

class RU_Env(gym.Env):
    def __init__(self, RU, embb_slices, urllc_slices, num_embb, num_urllc, H, inter_RU, inter_factor, N0, w_reward, cost_switch, cost_gb, scale_max, trainCons, frame_slots):
        super().__init__()
        self.RU = RU
        self.embb_slices = embb_slices
        self.urllc_slices = urllc_slices
        self.num_embb = num_embb
        self.num_urllc = num_urllc
        self.num_slices = num_embb + num_urllc
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

        self.PRB_slice = [[0 for b in range(len(self.RU.bwps))] for _ in range(self.num_slices)]
        self.last_PRB_slice = [[0 for b in range(len(self.RU.bwps))] for _ in range(self.num_slices)]

        self.num_embb_ue = [len(s.ue_set) for s in self.embb_slices]
        self.num_urllc_ue = [len(s.ue_set) for s in self.urllc_slices]
        self.num_ue = np.concatenate([self.num_embb_ue, self.num_urllc_ue])
        
        self.state_dim = 2 * sum(self.num_embb_ue) + 3 * sum(self.num_urllc_ue) + 5  
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        
        max_ue = max(max(self.num_urllc_ue) if self.num_urllc_ue else 0, max(self.num_embb_ue) if self.num_embb_ue else 0)
        self.action_space = spaces.MultiDiscrete(max_ue * self.RU.B_r)

        self.alloc = [[[0 for u in range(self.num_ue[i])] for b in range(len(self.RU.bwps))] for i in range(self.num_slices)]
        self.snr = [[[0 for u in range(self.num_ue[i])] for b in range(len(self.RU.bwps))] for i in range(self.num_slices)]
        
        self.BWP_slice = [[0 for b in range(len(self.RU.bwps))] for _ in range(self.num_slices)]
        self.last_BWP_slice = copy.deepcopy(self.BWP_slice)
        
        self.index_subframe = 0
        self.dqn_agent = None
        self.state = np.zeros(self.state_dim, dtype=np.float32)
        
        self.clear_buffers()

    def clear_buffers(self):
        self.numBit = [[0 for _ in range(self.num_urllc_ue[s])] for s in range(self.num_urllc)]  
        self.eMBB_Thr = [[0.0 for _ in range(self.num_embb_ue[s])] for s in range(self.num_embb)]

    def reset(self):
        self.clear_buffers()
        self.index_subframe = 0 
        return np.zeros(self.state_dim, dtype=np.float32)

    def assign_dqn_agent(self, agent):
        self.dqn_agent = agent

    def update_H(self, Hnew):
        self.H = Hnew

    def get_state(self, urllc_rate, embb_rate, averUE, minUE, cEne, cFrag, cSwit, cGB, stab):
        averUE_flat = np.array([averUE[s][u] for s in range(self.num_slices) for u in range(self.num_ue[s])])
        minUE_flat = np.array([minUE[s][u] for s in range(self.num_urllc) for u in range(self.num_urllc_ue[s])])
        state = np.concatenate((
            np.array(urllc_rate), np.array(embb_rate), averUE_flat, minUE_flat,
            np.array([cEne/self.scale_max[0], cFrag/self.scale_max[1], cSwit/self.scale_max[2], cGB/self.scale_max[3], stab])
        ))
        return state.astype(np.float32)

    def select_action(self, state, BWP_slice):
        if self.dqn_agent is not None:
            return self.dqn_agent.select_action(state, BWP_slice)
        return self.action_space.sample()

    def computeOutput(self, action):
        self.alloc = action
        self.transBWPslice()
        interBWP = self.calculateMultiBWPNoise()
        self.calculateSNR(interBWP)
        self.calculateNumBit()
        self.calculateThroughput()
        return self.eMBB_Thr, self.numBit

    def step(self, eMBB_Thr, numBit_urllc):
        self.index_subframe += 1
        
        URLLC_Latency = [[0.0 for u in range(self.num_urllc_ue[s])] for s in range(self.num_urllc)]
        for s in range(self.num_urllc):
            for u in range(self.num_urllc_ue[s]):
                pkt_size = getattr(self.urllc_slices[s].ue_set[u], 'packet_size', 100)
                URLLC_Latency[s][u] = pkt_size / (numBit_urllc[s][u] + 1e-9)

        cEne = self.calculateEnergy()
        cFrag = self.calculateFrag()
        cSwit = self.calculateSwitch()
        cGB = self.calculateGuardBand()
        stab = self.calculateStab()
        averUE = self.calculateAverUE()
        minUE = self.calculateMinUE()

        lat_rates = []
        thr_rates = []
        
        for s in range(self.num_urllc):
            for u in range(self.num_urllc_ue[s]):
                max_lat = getattr(self.urllc_slices[s].ue_set[u], 'max_lat', 1.0)
                lat_ratio = URLLC_Latency[s][u] / (max_lat + 1e-9)
                lat_rates.append(lat_ratio)
                
        for s in range(self.num_embb):
            for u in range(self.num_embb_ue[s]):
                min_thr = getattr(self.embb_slices[s].ue_set[u], 'min_thr', 1.0)
                slot_min_thr = min_thr / self.frame_slots
                thr_ratio = eMBB_Thr[s][u] / (slot_min_thr + 1e-9)
                thr_rates.append(thr_ratio)

        

        thr_term = np.mean(np.clip(thr_rates, 0.0, 2.0)) if len(thr_rates) > 0 else 0
        lat_term = np.mean(np.clip(2.0 - np.array(lat_rates), 0.0, 2.0)) if len(lat_rates) > 0 else 0

        reward = (self.w_reward["lat"] * lat_term) + (self.w_reward["thr"] * thr_term)  + (self.w_reward["cost"] * (1/4) * (4 - (cEne/self.scale_max[0]) - (cFrag/self.scale_max[1]) - (cSwit/self.scale_max[2]) - (cGB/self.scale_max[3]))) + (stab * 0.1)

        reward = (self.w_reward["lat"] * lat_term) + \
                 (self.w_reward["thr"] * thr_term) + \
                 (self.w_reward["cost"] * (4 - (cEne/self.scale_max[0]) - (cFrag/self.scale_max[1]) - \
                                          (cSwit/self.scale_max[2]) - (cGB/self.scale_max[3]))) + (stab)

        done = self.index_subframe >= self.frame_slots
        self.state = self.get_state(lat_rates, thr_rates, averUE, minUE, cEne, cFrag, cSwit, cGB, stab)
        
        self.last_PRB_slice = copy.deepcopy(self.PRB_slice)

        info = {
            "costE" : cEne, "costF" : cFrag, "costS" : cSwit, "costGB" : cGB,
            "lat": URLLC_Latency, "thr": eMBB_Thr
        } 
        
        self.clear_buffers() 
        return self.state, reward, done, info

    def calculateMultiBWPNoise(self):
        leakage = 1e-7
        interBWP = np.zeros(self.num_slices)
        for i in range(self.num_slices):
            for b in range(len(self.RU.bwps)):
                for bk in range(len(self.RU.bwps)):
                    if b != bk and self.RU.bwps[b].band_index != self.RU.bwps[bk].band_index:
                        gapIndex = np.abs(self.RU.bwps[b].band_index - self.RU.bwps[bk].band_index)
                        interBWP[i] += leakage * self.inter_factor * (1/(1+gapIndex)) \
                            * self.RU.bwps[bk].p_each_PRB * self.BWP_slice[i][bk]
        return interBWP
        
    def calculateSNR(self, interBWP):
        for i in range(self.num_slices):
            for b in range(len(self.RU.bwps)):
                for u in range(self.num_ue[i]): 
                    self.snr[i][b][u] = (self.RU.bwps[b].p_each_PRB * self.H[i][u]) \
                        / (self.RU.bwps[b].bandwidth * self.N0 * (self.inter_RU + 1) + interBWP[i]) 

    def calculateNumBit(self):
        for i in range(self.num_urllc):
            for u in range(self.num_urllc_ue[i]):
                for b in range(len(self.RU.bwps)):
                    self.numBit[i][u] += self.RU.bwps[b].time * self.alloc[i+self.num_embb][b][u] * np.log2(1 + self.snr[i+self.num_embb][b][u])

    def calculateThroughput(self):
        for i in range(self.num_embb):
            for u in range(self.num_embb_ue[i]):
                for b in range(len(self.RU.bwps)):
                    self.eMBB_Thr[i][u] += self.alloc[i][b][u] * self.RU.bwps[b].bandwidth * np.log2(1 + self.snr[i][b][u])

    def calculateFrag(self):
        total_frag = 0
        for i in range(self.num_slices):
            total_frag += (np.sum(self.BWP_slice[i]))**2 
        return total_frag
    
    def calculateEnergy(self):
        total_energy = 0
        for i in range(self.num_slices):
            for u in range(self.num_ue[i]):
                for b in range(len(self.RU.bwps)):
                    total_energy += self.alloc[i][b][u] * self.RU.bwps[b].p_each_PRB * self.RU.bwps[b].time
        return total_energy

    def calculateSwitch(self):
        total_switch = 0
        for i in range(self.num_slices):
            for b in range(len(self.RU.bwps)):
                total_switch += self.cost_switch * self.BWP_slice[i][b] * (1 - self.last_BWP_slice[i][b])
        return total_switch

    def calculateGuardBand(self):
        total_guard = 0
        for i in range(self.num_slices):
            for b in range(len(self.RU.bwps)):
                for bk in range(len(self.RU.bwps)):
                    if bk != b:
                        total_guard += self.cost_gb * np.abs(b - bk) * self.BWP_slice[i][b] * self.BWP_slice[i][bk]
        return total_guard

    def calculateStab(self):
        if self.last_PRB_slice is None:
            return 1.0
        changes = np.not_equal(self.PRB_slice, self.last_PRB_slice)
        total_changes = np.sum(changes)
        total_elements = max(np.array(self.last_PRB_slice).size, 1)
        psi = np.exp(- (total_changes / total_elements))
        return float(psi)

    def calculateAverUE(self):
        averUE = [[0 for _ in range(self.num_ue[s])] for s in range(self.num_slices)]
        for s in range(self.num_slices):
            for u in range(self.num_ue[s]):
                averSINR = np.average([self.snr[s][b][u] for b in range(len(self.RU.bwps))])
                averUE[s][u] = np.log2(1 + averSINR)
        return averUE
    
    def calculateMinUE(self):
        minUE = [[0 for _ in range(self.num_urllc_ue[s])] for s in range(self.num_urllc)]
        for s in range(self.num_urllc):
            for u in range(self.num_urllc_ue[s]):
                minSINR = np.min([self.snr[s + self.num_embb][b][u] for b in range(len(self.RU.bwps))])
                minUE[s][u] = np.log2(1 + minSINR)
        return minUE

    def transBWPslice(self):
        for i in range(self.num_slices):
            for b in range(len(self.RU.bwps)):
                prb_sum = sum(self.alloc[i][b])
                self.BWP_slice[i][b] = 1 if prb_sum > 0 else 0 
                self.PRB_slice[i][b] = prb_sum 

    def returnAlloc(self):
        return self.alloc