import numpy as np

"""
Một class quản lý tín hiệu về sla và fairness
"""
class GlobalCoordinator:
    def __init__(self, sla=0.9, window=10):
        self.sla = sla  # target SLA threshold
        self.window = window
        self.sla_history = []
        self.fair_history = []
        self.global_sla_shortfall = 0.0
        self.global_fairness_index = 1.0

    def update_global_metrics(self, urllc_met_all, urllc_total_all, throughput_list):
        # Update SLA (URLLC met ratio)
        cur_sla = sum(urllc_met_all) / max(1, sum(urllc_total_all))
        self.sla_history.append(cur_sla)
        if len(self.sla_history) > self.window:
            self.sla_history.pop(0)
        avg_sla = np.mean(self.sla_history)
        self.global_sla_shortfall = max(0.0, self.sla - avg_sla)

        # Update global fairness (Jain)
        throughputs = np.array(throughput_list)
        if len(throughputs) > 0:
            self.global_fairness_index = (throughputs.sum()**2) / (
                len(throughputs) * (throughputs**2).sum() + 1e-9
            )

    def get_global_signals(self):
        return {
            'sla_shortfall': self.global_sla_shortfall,
            'fairness_index': self.global_fairness_index
        }
