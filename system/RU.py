class RadioUnits :
    def __init__(self, K, B, n, N0, Pmax) -> None:
        self.K = K  # Số PRB
        self.B = B  # Băng thông mỗi PRB
        self.n = n  # Số kênh
        self.N0 = N0    # Noise power density
        self.Pmax = Pmax    # Công suất tối đa