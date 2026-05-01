class BandwidthPart:
    def __init__(self, id, num_prb, band_index, base_bw, base_time, p_each_PRB):
        """
        num_prb: số PRB trong BWP
        bw_factor: hệ số nhân băng thông so với base_bw
        time_factor: hệ số nhân thời gian so với base_time
        p_each_PRB: công suất mỗi PRB ở BWP này
        """
        self.id = id
        self.num_prb = num_prb
        self.bandwidth = (2**band_index) * base_bw
        self.time = (1/(2**band_index)) * base_time
        self.p_each_PRB = p_each_PRB

class RadioUnit:
    def __init__(self, Ru_index, bwps):
        """
        bwps: list các BandwidthPart
        N0: noise power density
        Pmax: công suất tối đa
        """
        self.Ru_index = Ru_index
        self.bwps = bwps

        # Tổng số PRB
        self.B_r = sum(bwp.num_prb for bwp in bwps)


        