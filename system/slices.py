import numpy as np

class BaseSlice:
    """
    Lớp cơ sở cho các slice (eMBB, URLLC, ...)
    Quản lý buffer, tốc độ đến, throughput, và cập nhật dữ liệu.
    """

    def __init__(self, name, arrival_rate=0, packet_size=1e5):
        self.name = name
        self.buffer = np.random.uniform(0, 1e6)       # backlog dữ liệu (bit)
        self.arrival_rate = arrival_rate              # tốc độ đến trung bình (bit/slot)
        self.packet_size = packet_size                 # kích thước gói URLLC
        self.current_rate = 0.0                        # throughput đạt được (bit/slot)
        self.arrival_prob = min(1.0, arrival_rate / 1e7)  # xác suất gói đến (cho URLLC)
        self.served_packets = 0
        self.dropped_packets = 0

    def update_buffer(self):
        """Cập nhật buffer theo arrival và throughput."""
        if self.name == "URLLC":
            arrival = np.random.binomial(1, self.arrival_prob) * self.packet_size
        else:
            arrival = np.random.poisson(lam=self.arrival_rate)

        departure = self.current_rate
        self.buffer = max(0.0, self.buffer + arrival - departure)

    def record_throughput(self, rate):
        self.current_rate = rate


class URLLCSlice(BaseSlice):
    def __init__(self, lam, D, L, eps, eps_phy):
        super().__init__(name="URLLC", arrival_rate=lam * L, packet_size=L)
        self.lam = lam
        self.D = D
        self.L = L
        self.eps = eps
        self.eps_phy = eps_phy
        self.eps_queue = self.eps - self.eps_phy

        # Biến kết quả
        self.m_avg = None
        self.mu = None
        self.S = None
        self.K_reserve = None


class eMBBslice(BaseSlice):
    def __init__(self, dataRate=10):
        # dataRate đơn vị: Mbps (trung bình throughput mong muốn)
        super().__init__(name="eMBB", arrival_rate=dataRate * 1e6 / 10)
        self.dataRate = dataRate
        self.prb_history = []
        self.rate_history = []
