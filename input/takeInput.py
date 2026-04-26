import pandas as pd
import json
import numpy as np
import os


def load_system_config(csv_path, line):
    """
    Đọc thông tin cấu hình hệ thống từ CSV
    csv_path : đường dẫn tới file câu hình csv
    line : cấu hình thứ bao nhiêu
    """
    cfg = pd.read_csv(csv_path).iloc[line-1]

    config = {
        "num_RUs" : cfg["num_RUs"],
        "num_slices" : cfg["num_slices"],
        "ue_URLLC" : cfg["ue_URLLC"],
        "ue_eMBB" : cfg["ue_eMBB"],
        "bwp_index" : cfg["bwp_index"],
        "p_bwp_mW" : cfg["p_bwp_mW"],
        "Rmin" : cfg["Rmin"],
        "Pac" : cfg["Pac"],
        "Lat" : cfg["Lat"]
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

