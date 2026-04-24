from system.RU import RadioUnits
from system.slices import URLLCSlice, eMBBslice
import numpy as np
from input.genTopo import generate_topology, calDistance, plot_topology

def generateRUs(num_RUs, K, B, n, N0, P_max):
    set_RU = []
    for i in range(num_RUs):
        RUi = RadioUnits(K, B, n, N0, P_max)
        set_RU.append(RUi)
    
    return set_RU


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


def generate_pipeline_inputs(num_RUs, num_slices, num_URLLC, numPRB, B, n, N0, frame_slot, 
                             P_max, deadline, load, dataRate):
    """
    Sinh dữ liệu đầu vào cho pipeline (RU, slices, channel gain, ...).
    Có thể thay thế sau này bằng đọc từ file .json.
    """
    # --- Tạo RUs ---
    RUs = []
    for _ in range(num_RUs):
        RUs.append(RadioUnits(K=numPRB, B=B, n=n, N0=N0, Pmax=P_max))

    # --- Tạo slices ---
    slices = []
    for i in range(num_slices):
        if i < num_URLLC:  # phần đầu là URLLC
            slices.append(URLLCSlice(D=np.random.randint(1,4), L=load, eps=1e-4, eps_phy=1e-5))
        else:  # cuối cùng là eMBB
            slices.append(eMBBslice(dataRate=dataRate))

    # --- Tạo channel gain matrix ---
    H, gain_ru_ru, dist_ue_ru = generate_channel_gain(num_RUs, frame_slot, num_slices, numPRB)

    return RUs, slices, H, gain_ru_ru, dist_ue_ru
