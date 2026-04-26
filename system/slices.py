import numpy as np

class UE:
    """
    Lớp cơ sở cho các UE
    Thể hiện yêu cầu (Throughput, latency, ... của UE)

    string serv = "eMBB"/"URLLC" : kiểu dịch vụ
    int id : id của UE trong slice
    int slice : id của slice mà UE thuộc
    """
    def __init__(self, serv, id, slice):
        self.serv = serv
        self.id = id
        self.slice = slice


class eMBBUE(UE):
    """
    Lớp các UE có dịch vụ eMBB
    float thr : Throughput tối thiểu mà UE yêu cầu
    """
    def __init__(self, serv, id, slice, thr):
        super().__init__(serv, id, slice)
        self.thr = thr


class URLLCUE(UE):
    """
    Lớp các UE có dịch vụ URLLC
    float lat : latency tối đa mà UE yêu cầu
    float pac : packet size của UE
    """
    def __init__(self, serv, id, slice, lat, pac):
        super().__init__(serv, id, slice)
        self.lat = lat
        self.pac = pac


class BaseSlice:
    """
    Lớp cơ sở cho các slice (eMBB, URLLC, ...)
    Quản lý buffer, tốc độ đến, throughput, và cập nhật dữ liệu.
    """

    def __init__(self, id, type, ue_set):
        self.id = id
        self.type = type
        self.ue_set = ue_set



