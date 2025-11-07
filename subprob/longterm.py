import numpy as np
from scipy.special import ndtri, lambertw # cho hàm Q^-1, Lambert
import math
import cvxpy as cp


"""
    Class lập bài toán longterm 

    RUs : Tập các RU
    slices : Tập các slice
    K : Số PRB 1 RU 
    H : Ma trận Gain 
    T_slot : Thời gian một slot
    num_slot : Số slot 1 frame
    sla : Tỷ lệ phục vụ cần đảm bảo (SLA)
    T_max : Throuhgput term (tránh để throughput quá áp đảo)  
    w_Thr : Hệ số tunning cho throughput
    w_accepted : Hệ số tunning cho số slice được chấp thuận
    w_redun_Thr : Hệ số tunning cho phần throughput bị thừa
    w_SLA : Hệ số tunning cho SLA
"""
class LongTermAllocator:
    def __init__(self, RUs, slices, K, H, T_slot, num_slot, T_max, w_reward, sla_slices):
        self.RUs = RUs                  # Tập các RU
        self.K = K                      # Số PRB mỗi RU
        self.H = H                      # Gain 
        self.slices = slices            # Tập các slice
        self.T_slot = T_slot            # Thời gian một slit
        self.num_slot = num_slot        # Số slot được xét
        self.total_PRB = K * len(self.RUs)
        self.w_reward = w_reward              # weight của throughput trong hàm mục tiêu
        self.T_max = T_max
        self.sla_slices = sla_slices
        self.num_URLLC = sum([1 for i in range(len(self.slices)) if self.slices[i].name == "URLLC"])

        self.snr = {}                   # Tỷ số SNR tính cho tất cả
        self.rCapa = {}                 # Số bit từng PRB tính cho tất cả
        self.umin = {}                  # Tốc độ phục vụ tối thiểu
        self.mmin = {}                  # Số PRB tối thiểu cần để đảm bảo tốc độ phục vụ

        self.sol_map = {}
        self.check = True
        
        self.time = 0
        self.objvalue = 0

    # Hàm tính toán SNR
    def calculateSNR(self):
        for r in range(len(self.RUs)):
            for t in range(self.num_slot):
                for i in range(len(self.slices)):
                    for k in range(self.K):
                        p_each = self.RUs[r].Pmax / self.K
                        self.snr[(r, i, k, t)] = (p_each * self.H[r][t][i][k]) / (self.RUs[r].N0 * self.RUs[r].B)

    # Hàm tính toán số bit truyền trong 1 PRB cho URLLC
    def finiteBlocklengthCapacityURLLC(self):
        for r in range(len(self.RUs)):
            for t in range(self.num_slot):
                for i in range(len(self.slices)):
                    if self.slices[i].name == "URLLC":
                        for k in range(self.K):
                            capacity = np.log2(1 + self.snr[(r, i, k, t)])
                            verlation = (1 - (1 + self.snr[(r, i, k, t)])**-2) * (np.log2(np.e))**2
                            penalty = np.sqrt(verlation / self.RUs[r].n) * abs(ndtri(self.slices[i].eps_phy))
                            self.rCapa[(r, i, k, t)] = self.RUs[r].n * (capacity - penalty) * self.T_slot
    
    # Hàm tính toán số bit truyền trong 1 PRB cho eMBB
    def CapacityeMBB(self):
        for i in range(len(self.slices)):
            if self.slices[i].name == "eMBB":
                for r in range(len(self.RUs)):
                    for k in range(self.K):
                        for t in range(self.num_slot):
                            capacity = np.log2(1 + self.snr[(r, i, k, t)]) 
                            bits = self.RUs[r].n * capacity
                            self.rCapa[(r, i, k, t)] = self.RUs[r].n * self.RUs[r].B * capacity

                                                                            
    def createProblem(self):
        # Các biến
        # Biến x^r_i,k
        x = cp.Variable(shape= (len(self.RUs), len(self.slices), self.K, self.num_slot),
                        name = "x", boolean = True)

        # Biến p^r_i,k (bị fixed)
        #p = {r : cp.Variable(shape= (self.num_slice, self.RUs[r].K) ,name = "p") for r in range(len(self.RUs))}

        # Biến pi^i
        pi = cp.Variable(shape = (len(self.slices), self.num_slot), name = f"pi" ,boolean=True)

        # Tính toán tham số từ trước
        self.calculateSNR()
        self.finiteBlocklengthCapacityURLLC()
        self.CapacityeMBB()

        # Ràng buộc
        constraint = []

        # Mỗi PRB chỉ được phân tối đa cho 1 người ở từng slot
        for r in range(len(self.RUs)):
            for k in range(self.RUs[r].K):
                for t in range(self.num_slot):
                    constraint.append(cp.sum([x[(r, i, k, t)] for i in range(len(self.slices))]) <= 1)
        
        # Mối quan hệ giữa x và pi
        for i in range(len(self.slices)):
            for t in range(self.num_slot):
                constraint.append((cp.sum([x[(r, i, k, t)] for r in range(len(self.RUs)) for k in range(self.RUs[r].K)]) / self.total_PRB) <= pi[(i, t)])
                constraint.append((cp.sum([x[(r, i, k, t)] for r in range(len(self.RUs)) for k in range(self.RUs[r].K)]) / self.total_PRB) + 1 - 1e-6 >= pi[(i, t)])

        # Điều kiện về công suất ở từng RU
        for r in range(len(self.RUs)):
            for t in range(self.num_slot):
                lhs = cp.sum([
                    (self.RUs[r].Pmax / self.K) * x[(r, i, k, t)]
                    for i in range(len(self.slices))
                    for k in range(self.K)
                ])
                constraint.append(lhs <= self.RUs[r].Pmax)

        # Điều kiện về URLLC
        # Điều kiện về payload và deadline
        for i in range(len(self.slices)):
            if self.slices[i].name == "URLLC":
                for tau in range(self.num_slot - self.slices[i].D + 1):
                    expr = cp.sum([
                        self.rCapa[(r,i,k,t)] * x[(r,i,k,t)]
                        for r in range(len(self.RUs))
                        for k in range(self.K)
                        for t in range(tau, tau + self.slices[i].D)
                    ])
                    constraint.append(expr >= self.slices[i].L * pi[(i,tau)])

        # Điều kiện về eMBB
        for i in range(len(self.slices)):
            if self.slices[i].name == "eMBB":
                constraint.append(cp.sum([self.rCapa[(r, i, k, t)] * x[(r, i, k, t)] for r in range(len(self.RUs)) 
                                                                                    for k in range(self.K)]) 
                                                                                    >= self.slices[i].dataRate * pi[(i, t)])
        
        eMBBThr = {}
        for i in range(len(self.slices)):
            if self.slices[i].name == "eMBB":
                for t in range(self.num_slot):
                    eMBBThr[(i, t)] =  cp.sum([self.rCapa[(r, i, k, t)] * x[(r, i, k, t)] 
                                            for r in range(len(self.RUs)) for k in range(self.K)])
        
        # Tổng throughput cho các eMBB
        sumeMBBThr = cp.sum([eMBBThr[(i, t)] for i in range(len(self.slices)) 
                             if self.slices[i].name == "eMBB"]) 
        
        # Fairness among eMBB throughput (Jain proxy)
        rateeMBBThr = [sum(eMBBThr[(i, t)]) / self.num_slot * self.slices[i].dataRate for i in range(len(self.slices))
                       if self.slices[i].name == "eMBB"]
        if len(rateeMBBThr) > 1:
            avg_T = cp.sum(rateeMBBThr) / len(rateeMBBThr)
            fair_penalty = cp.sum([cp.abs(rateeMBBThr[i] - avg_T) for i in range(len(rateeMBBThr))]) / (avg_T + 1e-6)
        else:
            fair_penalty = 0

        # Phần tính điều kiện cho SLA
        slaembbVio = (cp.sum([pi[(i,t)] for t in range(self.num_slot) for i in range(len(self.slices)) 
                            if self.slices[i].name == "eMBB"])/self.num_slot) - self.sla_slices["eMBB"]
        slaurllc = (cp.sum([pi[(i,t)] for t in range(self.num_slot) for i in range(len(self.slices)) 
                            if self.slices[i].name == "URLLC"])/self.num_slot) - self.sla_slices["URLLC"]
        

        # Hàm mục tiêu đảm bảo fairness và SLA
        objective = cp.Maximize(self.w_reward["thr"] * (sumeMBBThr/(self.num_slot * self.T_max)) \
                                + self.w_reward["fair"] * fair_penalty \
                                + self.w_reward["sla"] * (slaembbVio + slaurllc))
        
        problem = cp.Problem(objective, constraint)

        return problem
    
    # Giải bài toán và trả về các tham số cần thiết : thời gian, check
    def solve(self):
       
        problem = self.createProblem()

        problem.solve(solver=cp.MOSEK) #verbose = True

        self.sol_map = {v.name(): v.value for v in problem.variables()}
        self.check = self.check_solution()
        
        self.time = problem._solve_time
        self.objvalue = problem.objective.value

        return self.time, self.check, self.sol_map
    
    # Kiểm tra lại giá trị
    def check_solution(self):
        x = {(r, i, k, t): self.sol_map.get(f"x")[r][i][k][t] for r in range(len(self.RUs))
                                                                           for i in range(len(self.slices)) 
                                                                           for k in range(self.RUs[r].K)
                                                                           for t in range(self.num_slot)}
        pi = {(i, t): self.sol_map.get(f"pi")[i][t] for i in range(len(self.slices))
                                                        for t in range(self.num_slot)}
        
        # Mỗi PRB chỉ được phân tối đa cho 1 người ở từng slot
        for r in range(len(self.RUs)):
            for k in range(self.RUs[r].K):
                for t in range(self.num_slot):
                    if (sum(x[(r, i, k, t)] for i in range(len(self.slices))) - 1 > 1e-6):
                        print(1)
                        return False

        # Mối quan hệ giữa x và pi
        for i in range(len(self.slices)):
            for t in range(self.num_slot):
                if (sum(x[(r, i, k, t)] for r in range(len(self.RUs)) for k in range(self.RUs[r].K)) / self.total_PRB) - pi[(i, t)] > 0:
                    print(2)
                    return False
                if (sum(x[(r, i, k, t)] for r in range(len(self.RUs)) for k in range(self.RUs[r].K)) / self.total_PRB) + 1 + 1e-6 - pi[(i, t)] < 0:
                    print(2)
                    return False

        # Điều kiện về URLLC
        # Điều kiện về payload và deadline
        for i in range(len(self.slices)):
            if self.slices[i].name == "URLLC":
                for tau in range(self.num_slot - self.slices[i].D + 1):
                    expr = cp.sum([
                        self.rCapa[(r,i,k,t)] * x[(r,i,k,t)]
                        for r in range(len(self.RUs))
                        for k in range(self.K)
                        for t in range(tau, tau + self.slices[i].D)
                    ])
                    if expr < self.slices[i].L * pi[(i,tau)]:
                        print(3)
                        return False


        # Điều kiện về eMBB
        for i in range(len(self.slices)):
            if self.slices[i].name == "eMBB":
                if sum(self.rCapa[(r, i, k, t)] * x[(r, i, k, t)] for r in range(len(self.RUs)) 
                                           for k in range(self.K)) < self.slices[i].dataRate * pi[(i, t)]:
                    print(4)
                    return False

        return True  # Không có lỗi nào
    
    def debug_allocation(self):
        print("----BÁO CÁO THROUGHPUT----")
        for t in range(self.num_slot):
            total_prb_alloc = 0.0
            print(f"\n--- Slot {t} ---")
            for i in range(len(self.slices)):
                allocated_prbs = 0.0
                allocated_bits = 0.0
                for r in range(len(self.RUs)):
                    for k in range(self.RUs[r].K):
                        xv = self.sol_map.get(f"x", None)[r][i][k][t]
                        if xv is None:
                            val = 0.0
                        else:
                            val = float(xv)
                        allocated_prbs += val
                        allocated_bits += float(self.rCapa[(r,i,k,t)]) * val
                pi_val = self.sol_map.get(f"pi", 0.0)[i][t]
                print(f"slice {i} ({self.slices[i].name}): PRBs={allocated_prbs:.3f}, bits={allocated_bits:.1f}, pi={pi_val}")
                total_prb_alloc += allocated_prbs
            print(f"TOTAL PRB allocated (all slices) in slot {t}: {total_prb_alloc:.3f}  (system total_PRB={self.total_PRB})")
            # check per-PRB uniqueness if you have constraint per (r,k)
            for r in range(len(self.RUs)):
                for k in range(self.RUs[r].K):
                    sum_x_on_prb = 0.0
                    for i in range(len(self.slices)):
                        sum_x_on_prb += self.sol_map.get(f"x", 0.0)[r][i][k][t]
                    if sum_x_on_prb > 1.0001:
                        print(f"Overallocated PRB r={r} k={k} sum_x={sum_x_on_prb:.3f}")



        
