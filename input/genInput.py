from system.RU import RadioUnit, BandwidthPart
from system.slices import BaseSlice, eMBBUE, URLLCUE
import numpy as np
import yaml
from input.genTopo import generate_topology, calDistance, plot_topology

def genRU(RU_path):
    """
    Tạo tập các RU
    str RU_path: đường dẫn file cấu hình (yaml) 
    """
    with open(RU_path, "r", encoding="utf-8") as f:
        ru_cfg = yaml.safe_load(f)["ru"]
    ru_list = []
    for r in ru_cfg:
        bwps = [BandwidthPart(**b) for b in r["bwps"]]
        ru_list.append(RadioUnit(id=r["id"], location=r.get("location",""), bwps=bwps))
    return ru_list


def load_slices_and_ues(slice_path: str, ue_path: str):
    """
    Load file slice_path và ue_path để lấy ra thông tin slice và ue
    """
    with open(slice_path, "r", encoding="utf-8") as f:
        slices_cfg = yaml.safe_load(f)["slices"]
    with open(ue_path, "r", encoding="utf-8") as f:
        ues_cfg = yaml.safe_load(f)["ues"]

    # Tạo UE object theo type
    ue_dict = {}
    for u in ues_cfg:
        if u["serv"].upper() == "URLLC":
            ue_obj = URLLCUE(serv=u["serv"], id=u["id"], slice=u["slice"],
                             lat=u["lat"], pac=u["pac"])
        elif u["serv"].upper() == "EMBB":
            ue_obj = eMBBUE(serv=u["serv"], id=u["id"], slice=u["slice"],
                            thr=u["thr"])
        else:
            raise ValueError(f"Unknown UE type: {u['serv']}")
        ue_dict[u["id"]] = ue_obj

    # Ghép UE vào slice
    slice_list = []
    for s in slices_cfg:
        ue_set = [ue_dict[uid] for uid in s["ue_list"]]
        slice_obj = BaseSlice(id=s["id"], type=s["type"], ue_set=ue_set)
        slice_list.append(slice_obj)

    # Phân loại slice theo type
    embb_slices = [s for s in slice_list if s.type.upper() == "EMBB"]
    urllc_slices = [s for s in slice_list if s.type.upper() == "URLLC"]

    return embb_slices, urllc_slices


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


def generate_pipeline_inputs(RU_path, slice_path, ue_path):
    """
    Tạo RU và các slice từ các đường dẫn
    """
    # --- Tạo RUs ---
    ru_list = genRU(RU_path)

    # --- Tạo slices ---
    eMBBlist, URLLClist = load_slices_and_ues(slice_path, ue_path)

    return ru_list, eMBBlist, URLLClist


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
        maxIndex = np.max(r.bwps[b].index for b in range(len(r.bwps)))
        minIndex = np.max(r.bwps[b].index for b in range(len(r.bwps)))
        gapIndex = maxIndex - minIndex
        for b in r.bwps:
            cEneMax += b.num_prb * b.p_each_PRB * b.time
            cSwitch += cost_switch * num_slices
            cGuardB += cost_gb * gapIndex * num_slices
    cFrag = cFrag**2

    scaleMax = [cEneMax, cFrag, cSwitch, cGuardB]

    return scaleMax

    
