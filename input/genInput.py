from system.RU import RadioUnits
from system.slices import URLLCSlice, eMBBslice
import numpy as np

def generateRUs(num_RUs, K, B, n, N0, P_max):
    set_RU = []
    for i in range(num_RUs):
        RUi = RadioUnits(K, B, n, N0, P_max)
        set_RU.append(RUi)
    
    return set_RU


import numpy as np

import numpy as np

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

    # Path loss: ngẫu nhiên từ 100 đến 130 dB
    path_loss_db = np.random.uniform(90, 120, size=(R, frame_slot, I, K))
    path_loss_lin = 10 ** (-path_loss_db / 10)

    # Tổng độ lợi kênh: fading × path loss
    H = fading * path_loss_lin

    return H

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
        if i < num_URLLC:  # phần lớn là URLLC
            slices.append(URLLCSlice(lam=30, D=deadline, L=load, eps=1e-4, eps_phy=1e-5))
        else:  # cuối cùng là eMBB
            slices.append(eMBBslice(dataRate=dataRate))

    # --- Tạo channel gain matrix ---
    H = generate_channel_gain(num_RUs, frame_slot, num_slices, numPRB)

    return RUs, slices, H
