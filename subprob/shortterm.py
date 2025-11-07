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
    longterm_results : Phân bổ PRB cho từng slice
    T_max : Throuhgput term (tránh để throughput quá áp đảo)  
    w_Thr : Hệ số tunning cho throughput
    w_accepted : Hệ số tunning cho số slice được chấp thuận
    w_redun_Thr : Hệ số tunning cho phần throughput bị thừa
"""
class ShortTermAllocator:
    def __init__(self, RUs, slices, K, H, T_slot, num_slot, longterm_results, T_max, w_reward):
        self.RUs = RUs          # Tập các RU
        self.slices = slices    # Tập các slice
        self.K = K              # Số lượng PRB ở mỗi RU
        self.H = H              # Gain
        self.T_slot = T_slot    # Thời gian 1 slot
        self.num_slot = num_slot      # Số slot được xét
        self.longterm_result = longterm_results    # Giá trị biến x trong bài toán
        self.T_max = T_max      # Scale term cho throughput
        self.w_reward = w_reward # Hệ số tunning 
        self.num_URLLC = sum([1 for i in range(len(self.slices)) if self.slices[i].name == "URLLC"])

        self.time = 0
        self.objvalue = 0
        self.x = {}
        self.pi = {}
        self.umin = {}
        self.sol_map = {}
        self.check = True
    
    def extractXPI(self):
        self.x = {(r, i, k, t): self.longterm_result.get(f"x", 0)[r][i][k][t] for r in range(len(self.RUs))
                                                                      for i in range(len(self.slices)) 
                                                                      for k in range(self.RUs[r].K)
                                                                      for t in range(self.num_slot)}
        self.pi = {(i, t): self.longterm_result.get(f"pi", 0)[i][t] for i in range(len(self.slices))
                                                            for t in range(self.num_slot)}


    # Hàm mô hình toán số bit truyền trong 1 PRB cho URLLC
    def finiteBlocklengthCapacityURLLC(self, p):
        rCapa_URLLC = {}
        for r in range(len(self.RUs)):
            for t in range(self.num_slot):
                for i in range(len(self.slices)):
                    if self.slices[i].name == "URLLC":
                        for k in range(self.K):
                            g = self.H[r][t][i][k]/(self.RUs[r].B * self.RUs[r].N0)
                            p0 = self.RUs[r].Pmax / self.RUs[r].K
                            V0 = (1 - 1/(1 + g * p0)**2) * (np.log2(np.e))**2
                            alpha = g / ((1 + g * p0) * np.log(2))
                            beta = np.log1p(g * p0) / np.log(2) - alpha * p0
                            penalty_const = np.sqrt(V0/self.RUs[r].n) * abs(ndtri(self.slices[i].eps_phy))

                            rCapa_URLLC[(r, i, k, t)] = self.RUs[r].n * self.T_slot * (alpha * p[(r, i, k, t)] + beta - penalty_const)
        return rCapa_URLLC
    

    # Hàm mô hình toán số bit truyền trong 1 PRB cho URLLC (với p là giá trị số)
    def finiteBlocklengthCapacityURLLC_after(self, p_value):
        rCapa_URLLC_after = {}
        for r in range(len(self.RUs)):
            for t in range(self.num_slot):
                for i in range(len(self.slices)):
                    if self.slices[i].name == "URLLC":
                        for k in range(self.K):
                            g = self.H[r][t][i][k]/(self.RUs[r].B * self.RUs[r].N0)
                            p0 = self.RUs[r].Pmax / self.RUs[r].K
                            V0 = (1 - 1/(1 + g * p0)**2) * (np.log2(np.e))**2
                            alpha = g / ((1 + g * p0) * np.log(2))
                            beta = np.log1p(g * p0) / np.log(2) - alpha * p0
                            penalty_const = np.sqrt(V0/self.RUs[r].n) * abs(ndtri(self.slices[i].eps_phy))

                            rCapa_URLLC_after[(r, i, k, t)] = self.RUs[r].n * self.T_slot * (alpha * p_value[(r, i, k, t)] + beta - penalty_const)
        return rCapa_URLLC_after
    

    # Hàm tính toán số bit truyền trong 1 PRB cho eMBB theo biến p
    def CapacityeMBB(self, p):
        rCapa_eMBB = {}
        for i in range(len(self.slices)):
            if self.slices[i].name == "eMBB":
                for r in range(len(self.RUs)):
                    for k in range(self.K):
                        for t in range(self.num_slot):
                            g = self.H[r][t][i][k]/(self.RUs[r].B * self.RUs[r].N0)
                            p0 = self.RUs[r].Pmax / self.RUs[r].K
                            alpha = g / ((1 + g * p0) * np.log(2))
                            beta = np.log1p(g * p0) / np.log(2) - alpha * p0
                            rCapa_eMBB[(r, i, k, t)] = self.RUs[r].n * (alpha * p[(r, i, k, t)] + beta)
        return rCapa_eMBB


    # Hàm tính toán số bit truyền trong 1 PRB cho eMBB theo biến p
    def CapacityeMBB_after(self, p_value):
        rCapa_eMBB_after = {}
        for i in range(len(self.slices)):
            if self.slices[i].name == "eMBB":
                for r in range(len(self.RUs)):
                    for k in range(self.K):
                        for t in range(self.num_slot):
                            g = self.H[r][t][i][k]/(self.RUs[r].B * self.RUs[r].N0)
                            p0 = self.RUs[r].Pmax / self.RUs[r].K
                            alpha = g / ((1 + g * p0) * np.log(2))
                            beta = np.log1p(g * p0) / np.log(2) - alpha * p0
                            rCapa_eMBB_after[(r, i, k, t)] = self.RUs[r].n * (alpha * p_value[(r, i, k, t)] + beta)
        return rCapa_eMBB_after


    def createProblem(self):
        # Biến p^r_i,k
        p = cp.Variable(shape= (len(self.RUs), len(self.slices), self.K, self.num_slot),
                        name = "p", nonneg=True)
        
        # Lấy giá trị x, pi
        self.extractXPI()
        # Tính toán SINR, số bit truyền qua 1 PRB của 1 slice dùng cho mô hình toán
        rCapa_URLLC = self.finiteBlocklengthCapacityURLLC(p=p)
        rCapa_eMBB = self.CapacityeMBB(p=p)

        constraint = []

        # Điều kiện về công suất ở từng RU
        for r in range(len(self.RUs)):
            for t in range(self.num_slot):
                constraint.append(cp.sum([self.x[(r, i, k, t)] * p[(r, i, k, t)] for i in range(len(self.slices))
                                                            for k in range(self.K)]) <= self.RUs[r].Pmax)
                
        # Mối liên hệ giữa x và p
        for r in range(len(self.RUs)):
            for t in range(self.num_slot):
                for i in range(len(self.slices)):
                    for k in range(self.K):
                        constraint.append(p[(r, i, k, t)] <= self.RUs[r].Pmax * self.x[(r, i, k, t)])
        
        # Điều kiện về payload và deadline
        for i in range(len(self.slices)):
            if self.slices[i].name == "URLLC":
                for tau in range(self.num_slot - self.slices[i].D + 1):
                    expr = cp.sum([
                        rCapa_URLLC[(r,i,k,t)] * self.x[(r,i,k,t)]
                        for r in range(len(self.RUs))
                        for k in range(self.K)
                        for t in range(tau, tau + self.slices[i].D)
                    ])
                    constraint.append(expr >= self.slices[i].L * self.pi[(i,tau)])

        # Điều kiện về eMBB
        for i in range(len(self.slices)):
            if self.slices[i].name == "eMBB":
                constraint.append(cp.sum([rCapa_eMBB[(r, i, k, t)] for r in range(len(self.RUs)) 
                                                                    for k in range(self.K)]) 
                                                                    >= self.slices[i].dataRate * self.pi[(i, t)])
        
        eMBBThr = {}
        for i in range(len(self.slices)):
            if self.slices[i].name == "eMBB":
                for t in range(self.num_slot):
                    eMBBThr[(i, t)] =  cp.sum([rCapa_eMBB[(r, i, k, t)] * self.x[(r, i, k, t)] 
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

        
        objective = cp.Maximize(self.w_reward["thr"] * (sumeMBBThr/self.T_max) \
                                + self.w_reward["fair"] * fair_penalty)
        
        problem = cp.Problem(objective, constraint)

        return problem
    
    # Giải bài toán và trả về các tham số cần thiết : thời gian, check
    def solve(self):
       
        problem = self.createProblem()

        problem.solve(solver=cp.MOSEK) #, verbose = True)

        self.sol_map = {v.name(): v.value for v in problem.variables()}
        self.check, metrics = self.check_solution()
        
        self.time = problem._solve_time
        self.objvalue = problem.objective.value

        return self.time, self.check, metrics
    
    # Kiểm tra lại giá trị, trả về các metrics
    def check_solution(self):
        self.check = True
        p = {(r, i, k, t): self.sol_map.get(f"p", 0)[r][i][k][t] for r in range(len(self.RUs))
                                                                 for i in range(len(self.slices)) 
                                                                 for k in range(self.RUs[r].K)
                                                                 for t in range(self.num_slot)}
        # Khởi tạo tình throughput 
        eMBBThr = {}
        for t in range(self.num_slot):
            for i in range(len(self.slices)):
                if self.slices[i].name == "eMBB":
                    eMBBThr[(i, t)] = 0

        # Thay giá trị của p vào để tính toán các tham số (các hàm có _after)
        rCapa_URLLC_after = self.finiteBlocklengthCapacityURLLC_after(p_value=p)
        rCapa_eMBB_after = self.CapacityeMBB_after(p_value=p)
        
        # Điều kiện về công suất ở từng RU
        for r in range(len(self.RUs)):
            for t in range(self.num_slot):
                if sum([p[(r, i, k, t)] for i in range(len(self.slices)) for k in range(self.K)]) >= self.RUs[r].Pmax + 1e-3:
                    self.check = False
                
        # Mối liên hệ giữa x và p
        for r in range(len(self.RUs)):
            for t in range(self.num_slot):
                for i in range(len(self.slices)):
                    for k in range(self.K):
                        if (p[(r, i, k, t)] >= self.RUs[r].Pmax * self.x[(r, i, k, t)] + 1e-3):
                            self.check = False
        
        # Điều kiện về payload và deadline
        for i in range(len(self.slices)):
            if self.slices[i].name == "URLLC":
                for tau in range(self.num_slot - self.slices[i].D + 1):
                    expr = sum([
                        rCapa_URLLC_after[(r,i,k,t)] * self.x[(r,i,k,t)]
                        for r in range(len(self.RUs))
                        for k in range(self.K)
                        for t in range(tau, tau + self.slices[i].D)
                    ])
                    if expr <= self.slices[i].L * self.pi[(i,tau)] - 1e-6:
                        self.check =  False

        # Điều kiện về eMBB
        for t in range(self.num_slot):
            for i in range(len(self.slices)):
                if self.slices[i].name == "eMBB":
                    eMBBThr[(i, t)] = sum([rCapa_eMBB_after[(r, i, k, t)] for r in range(len(self.RUs)) 
                                                    for k in range(self.K)])
                    if (eMBBThr[(i, t)] <= self.slices[i].dataRate * self.pi[(i, t)] - 1e-6):
                        self.check =  False
        # Tính throuhgput trung bình
        avg_thr = sum(eMBBThr[(i, t)] for t in range(self.num_slot) for i in range(len(self.slices))
                        if self.slices[i].name == "eMBB") / (self.num_slot * (len(self.slices) - self.num_URLLC))
        
        # Tính SLA
        slaembb = sum(self.pi[(i, t)] for t in range(self.num_slot) for i in range(len(self.slices))
                        if self.slices[i].name == "eMBB") / (self.num_slot * (len(self.slices) - self.num_URLLC))
        slaurllc = sum(self.pi[(i, t)] for t in range(self.num_slot) for i in range(len(self.slices))
                        if self.slices[i].name == "URLLC") / (self.num_slot * self.num_URLLC)
        
        for i in range(len(self.slices)):
            if self.slices[i].name == "eMBB":
                eMBBThr[i] = sum(eMBBThr[(i, t)] for t in range(self.num_slot)) / (self.num_slot * self.slices[i].dataRate)
        # Tính toán Jain index trung bình
        avg_fairness = sum(eMBBThr[i] for i in range(len(self.slices)) if self.slices[i].name == "eMBB")**2 / \
                        ((len(self.slices) - self.num_URLLC) * sum(eMBBThr[i]**2 for i in range(len(self.slices)) if self.slices[i].name == "eMBB"))

        # Tính tỷ lệ sử dụng PRB
        utilRate = np.sum(self.x[(r, i, k, t)] for r in range(len(self.RUs)) for i in range(len(self.slices)) 
                    for k in range(self.RUs[r].K) 
                    for t in range(self.num_slot)) / (self.num_slot * len(self.slices) * len(self.RUs)) 
        self.report_power_usage(p_value=p)
        info = {"avg_throughput" : avg_thr,
                "avg_eMBB": slaembb,
                "avg_urllc": slaurllc,
                "avg_fairness": avg_fairness,
                "avg_util": utilRate}
        return self.check, info

    
    def report_power_usage(self, p_value):
        """
        Báo cáo công suất sử dụng của từng slice trong từng slot.
        p_value: dict hoặc numpy array chứa giá trị công suất p[(r,i,k,t)] sau khi solve.
        """
        print("\n========== BÁO CÁO SỬ DỤNG CÔNG SUẤT ==========\n")

        total_system_power = 0.0

        for t in range(self.num_slot):
            print(f"--- Slot {t} ---")
            slot_total = 0.0

            for i in range(len(self.slices)):
                slice_total = 0.0

                # Tính tổng công suất của slice i trong slot t
                for r in range(len(self.RUs)):
                    for k in range(self.K):
                        p_val = 0.0
                        if isinstance(p_value, dict):
                            val = p_value.get((r, i, k, t))
                            if val is not None:
                                p_val = float(val)
                        else:
                            p_val = float(p_value[r, i, k, t])

                        slice_total += p_val

                slot_total += slice_total
                print(f"Slice {i} ({self.slices[i].name}): P_used = {slice_total:.4f} W")

            total_system_power += slot_total
            print(f"--> Tổng công suất Slot {t}: {slot_total:.4f} W\n")

        avg_power = total_system_power / self.num_slot
        print(f"==> Tổng công suất trung bình hệ thống: {avg_power:.4f} W/slot\n")
