import pandas as pd
import json
import numpy as np
import os


def load_system_config(csv_path, line):
    """
    Đọc thông tin cấu hình hệ thống từ CSV
    Returns dictionary of parameters
    """
    cfg = pd.read_csv(csv_path).iloc[line-1]

    config = {
        "num_RUs": int(cfg["num_RUs"]),
        "num_slices": int(cfg["num_slices"]),
        "num_URLLC": int(cfg["num_URLLC"]),
        "num_PRB_per_RU": int(cfg["num_PRB_per_RU"]),
        "Pmax_mW": float(cfg["Pmax_mW"]),
        "deadline": int(cfg["deadline"]),
        "load_URLLC": float(cfg["load_URLLC"]),
        "dataRate_eMBB_Mbps": float(cfg["dataRate_eMBB_Mbps"])
    }

    return config


def load_cons_from_json(json_path):
    """
    Đọc tham số cấu hình từ file JSON, dùng cho hằng số hệ thống và 
    các config cần cho việc train mô hình.
    """
    with open(json_path, 'r') as f:
        cons = json.load(f)

    return cons


def save_gain_matrix(gain_matrix, paras):
    """
    Lưu ma trận gain vào file .npy.
    
    gain_matrix: numpy array hoặc list of lists
    paras : các tham số đặc trưng
    filepath: đường dẫn file .npy
    """
    numRU, num_slices, num_urllc = paras
    filepath = f"./Gain/Gain_{numRU}_{num_slices}_{num_urllc}.npy"
    gain_matrix = np.array(gain_matrix)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.save(filepath, gain_matrix)


def load_gain_matrix(paras):
    """
    Load ma trận gain từ file .npy.

    Return: numpy array
    """
    numRU, num_slices, num_urllc = paras
    filepath = f"./Gain/Gain_{numRU}_{num_slices}_{num_urllc}.npy"
    gain_matrix = np.load(filepath)

    return gain_matrix

