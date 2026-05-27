from system.RU import RadioUnit, BandwidthPart
from system.slices import BaseSlice, eMBBUE, URLLCUE
import numpy as np
import yaml
from input.genTopo import generate_topology, calDistance, plot_topology


def genRU(RU_path, BASE_BW, BASE_TIME):
    """
    Tạo tập các RU từ file cấu hình YAML.
    - RU_path: Đường dẫn file cấu hình các RU.
    - BASE_BW: Băng thông cơ bản mỗi PRB
    - BASE_TIME: time slot cơ bản mỗi PRB
    - N0: Mật độ phổ nhiễu (hằng số chung).
    """
    with open(RU_path, "r", encoding="utf-8") as f:
        # Load đúng key "rus" từ file YAML của bạn
        ru_cfg = yaml.safe_load(f)["rus"] 
    
    ru_list = []
    for r in ru_cfg:
        bwps_obj = []
        for b in r["bwps"]:
            # Khởi tạo BWP với các tham số từ YAML và hằng số hệ thống
            # Lưu ý: "num_prb" hoặc "range" tùy thuộc vào cách bạn gọi trong YAML
            num_prbs = b.get("num_prb","unknown")
            
            bwp = BandwidthPart(
                id=b["id"],
                num_prb=num_prbs,
                band_index=b.get("bandindex"),
                p_each_PRB=b["p_each_PRB"],
                base_bw=BASE_BW,
                base_time=BASE_TIME
            )
            bwps_obj.append(bwp)
        
        # Tạo RadioUnit
        # Sử dụng r.get("Ru_index") để khớp với "Ru_index: ru_0" trong YAML của bạn
        ru_id = r.get("Ru_index", "unknown")
        
        ru_list.append(RadioUnit(Ru_index=ru_id, bwps=bwps_obj))
        
    return ru_list


def genSlices(ue_path, slice_path):
    """
    Tạo tập các slices eMBB và URLLC riêng biệt.
    - ue_path: đường dẫn file ue.yaml
    - slice_path: đường dẫn file slice.yaml
    """
    # 1. Load dữ liệu từ YAML
    with open(ue_path, "r", encoding="utf-8") as f:
        ue_data = yaml.safe_load(f)["ues"]
    
    with open(slice_path, "r", encoding="utf-8") as f:
        slice_data = yaml.safe_load(f)["slices"]

    # Tạo map để lưu các đối tượng Slice theo ID
    slice_dict = {}
    urllc_slices = []
    embb_slices = []

    # 2. Khởi tạo các đối tượng Slice trước
    for s_cfg in slice_data:
        s_id = s_cfg["id"]
        s_type = s_cfg["type"]
        
        # Tạo object Slice (giả sử lớp Slice của bạn nhận các tham số này)
        # Khởi tạo ue_set rỗng để lấp đầy sau
        new_slice = BaseSlice(
            id=s_id,
            type=s_type,
            ue_set=[] 
        )
        
        slice_dict[s_id] = new_slice
        
        # Tách nhóm ngay từ đầu
        if s_type.upper() == "URLLC":
            urllc_slices.append(new_slice)
        else:
            embb_slices.append(new_slice)

    # 3. Gán UE vào đúng Slice dựa trên slice_id
    for u_cfg in ue_data:
        s_id = u_cfg["slice_id"]
        if s_id in slice_dict:
            target_slice = slice_dict[s_id]
            
            # Kiểm tra type của slice để khởi tạo đúng Class UE
            if target_slice.type.upper() == "URLLC":
                new_ue = URLLCUE(
                    id=u_cfg["id"],
                    slice=s_id,
                    lat=u_cfg.get("lat"),
                    pac=u_cfg.get("pac")
                )
            else:
                new_ue = eMBBUE(
                    id=u_cfg["id"],
                    slice=s_id,
                    thr=u_cfg.get("min_thr", 10.0)  # đọc đúng key, có giá trị mặc định
                )
            # Thêm UE vào danh sách ue_set của Slice tương ứng
            slice_dict[s_id].ue_set.append(new_ue)

    return urllc_slices, embb_slices


def generate_channel_gain(R, frame_slot, I, K, model="rayleigh", K_factor=5, seed=None):
    """
    Sinh ma trận độ lợi kênh H[r, frame_slot, i, k] (thực, dương), phản ánh fading + path loss thực tế.

    Parameters
    ----------
    R : int
        Số RU.
    frame_slot : int
        Số slot trong một frame.
    I : int
        Số slice hoặc user.
    K : int
        Số PRB mỗi RU.
    model : str
        Kiểu mô hình fading ("rayleigh", "rician", "uniform", "gaussian").
    K_factor : float
        Hệ số Rician K-factor (dB) nếu chọn mô hình Rician.
    seed : int or None
        Seed để tái lập kết quả.

    Returns
    -------
    H : np.ndarray
        Ma trận H có shape (R, frame_slot, I, K), giá trị dương nhỏ phản ánh fading + path loss.
    """
    if seed is not None:
        np.random.seed(seed)

    # Fading
    if model.lower() == "rayleigh":
        fading = np.sqrt(np.random.randn(R, frame_slot, I, K)**2 +
                         np.random.randn(R, frame_slot, I, K)**2) / np.sqrt(2)

    elif model.lower() == "rician":
        K_lin = 10 ** (K_factor / 10)
        h_los = np.ones((R, frame_slot, I, K))
        h_nlos = np.sqrt(np.random.randn(R, frame_slot, I, K)**2 +
                         np.random.randn(R, frame_slot, I, K)**2) / np.sqrt(2)
        fading = np.sqrt(K_lin / (K_lin + 1)) * h_los + np.sqrt(1 / (K_lin + 1)) * h_nlos

    elif model.lower() == "uniform":
        fading = np.random.uniform(0.1, 1.0, size=(R, frame_slot, I, K))

    elif model.lower() == "gaussian":
        fading = np.abs(np.random.randn(R, frame_slot, I, K))

    else:
        raise ValueError(f"Unknown channel model: {model}")

    # ---- LARGE-SCALE: distances & pathloss (fixed per frame) ----
    # sample distances per RU-UE (shape (R,I));
    ru_pos, ue_pos = generate_topology(num_RUs=R,num_UEs=I)
    plot_topology(ru_pos, ue_pos, du_cu_pos=(0,0), R_cell=50,R_ru=250)
    dist_ue_ru, dist_ru_ru = calDistance(num_UEs=I, num_RUs=R,ru_pos=ru_pos, ue_pos=ue_pos)

    # FSPL in dB 
    f = 3.5
    # f in GHz, distance in meters
    FSPL_dB_ru_ue = 32.4 + 20*np.log10(f) + 20*np.log10(dist_ue_ru)
    FSPL_dB_ru_ru = 32.4 + 20*np.log10(f) + 20*np.log10(dist_ru_ru)

    #FSPL_dB = 120.8 + 37.5*np.log10(d)  # Phục vụ kịch bản thử

    # shadowing (log-normal) per RU-UE (dB)
    sigma_sh = 6.0  # dB, choose 4..8 dB depending LOS/NLOS
    shadowing_dB_ru_ue = np.random.normal(0.0, sigma_sh, size=(R, I))
    shadowing_dB_ru_ru = np.random.normal(0.0, sigma_sh, size=(R, R))

    # total pathloss (dB) and linear power gain (unitless)
    PL_dB_ru_ue = FSPL_dB_ru_ue + shadowing_dB_ru_ue      # shape (R,I)
    path_loss_lin = 10.0 ** (-PL_dB_ru_ue / 10.0)   # power-domain gain. shape (R,I)
    path_loss_lin_exp = path_loss_lin[:, np.newaxis, :, np.newaxis]

    # gain for ru - ru
    gain_ru_ru =  10.0 ** (-(FSPL_dB_ru_ru + shadowing_dB_ru_ru) / 10.0)

    # Tổng độ lợi kênh: fading × path loss
    H = (fading**2) * path_loss_lin_exp

    return H, gain_ru_ru, dist_ue_ru


def generate_pipeline_inputs(RU_path, slice_path, ue_path, consta):
    """
    Tạo RU và các slice từ các đường dẫn
    str RU_path: đường dẫn tới cấu hình các RU
    str slice_path: đường dẫn tới cấu hình các slice
    str ue_path: đường dẫn tới cấu hình các ue
    consta: các hằng số chung của hệ thống

    return:
        ru_list, eMBBlist, URLLClist: danh sách các RU, eMBB và URLLC slice
        num_urllc_ue, num_embb_ue: danh sách số lượng UE các slice
    """
    # --- Tạo RUs ---
    ru_list = genRU(RU_path, consta["BASE_BW_MHz"], consta["BASE_TIME_ms"])

    # --- Tạo slices ---
    URLLClist, eMBBlist = genSlices(ue_path, slice_path)

    num_urllc_ue = [len(URLLClist[s].ue_set) for s in range(len(URLLClist))]
    num_embb_ue = [len(eMBBlist[s].ue_set) for s in range(len(eMBBlist))]

    return ru_list, eMBBlist, URLLClist, num_urllc_ue, num_embb_ue


def calculateScaleMax(RUs, embb_slices, urllc_slices, cost_switch, cost_gb):
    """
    Hàm tính toán giá trị scale cho các thành phần
    """
    num_slices = len(embb_slices) + len(urllc_slices)

    # Chi phí về năng lượng tiêu hao max, phân mảnh PRB và switching giữa các BWP
    cEneMax = 0
    cFrag = 0
    cSwitch = 0
    cGuardB = 0
    for r in RUs:
        cFrag += len(r.bwps)
        bidex = [r.bwps[b].band_index for b in range(len(r.bwps))]
        maxIndex = max(bidex)
        minIndex = min(bidex)
        gapIndex = maxIndex - minIndex
        for b in r.bwps:
            cEneMax += b.num_prb * b.p_each_PRB * b.time
            cSwitch += cost_switch * num_slices
            cGuardB += cost_gb * gapIndex * num_slices
    cFrag = cFrag**2

    # Kiếm tra xem cGuardB có bị 0 hay không (cái này là cái duy nhất có thể bị 0)
    if cGuardB == 0:
        cGuardB = 1

    scaleMax = [cEneMax, cFrag, cSwitch, cGuardB]

    return scaleMax


def generate_h_matrix(num_rus, num_slots, num_slices,
                      num_urllc_ue, num_embb_ue):
    """
    Tạo ma trận H [RU][Slot][Slice][UE]
    - num_urllc_ue: list chứa số UE của từng slice URLLC.
    - num_embb_ue: list chứa số UE của từng slice eMBB.
    """
    # Hợp nhất danh sách số lượng UE của tất cả các slice để dễ lặp
    all_slice_ue_counts = num_urllc_ue + num_embb_ue #
    
    H = []
    for r in range(num_rus):
        slots_h = []
        for t in range(num_slots):
            slices_h = []
            for s in range(num_slices):
                num_ues = all_slice_ue_counts[s]
                
                # 1. Tạo "khoảng cách" ngẫu nhiên cố định cho UE này (để giữ tính ổn định tương đối)
                # Giả lập gain nền từ 0.001 đến 0.1
                base_gain = np.random.uniform(0.001, 0.1, size=num_ues)
                
                # 2. Thêm Fading biến động theo từng time slot (Rayleigh Fading)
                # exponential(1.0) mô phỏng bình phương biên độ Rayleigh
                fading = np.random.exponential(1.0, size=num_ues)
                
                # Gain tổng hợp
                ue_gains = base_gain * fading
                slices_h.append(ue_gains)
                
            slots_h.append(slices_h)
        H.append(slots_h)
        
    return H

# --- Ví dụ cách bạn tách H để đưa vào DQN từng loại ---
# H_matrix = generate_h_matrix(...)

# Lấy H cho một RU 'r' tại slot 't':
# H_urllc = H[r][t][:num_urllc]            # List chứa gain của các slice URLLC
# H_embb  = H[r][t][num_urllc:num_slices]  # List chứa gain của các slice eMBB
