import csv
import os
import matplotlib.pyplot as plt


def write_results_to_csv(file_path, data, header=None, mode='w'):
    """
    Ghi kết quả vào file CSV.

    Args:
        file_path (str): Đường dẫn tới file CSV.
        data (list): Danh sách các dòng dữ liệu (list of dicts hoặc list of lists).
        header (list, optional): Danh sách tên cột. Bắt buộc nếu data là list of lists.
        mode (str): 'w' để ghi mới, 'a' để ghi tiếp. Mặc định là 'w'.
    """
    rounded_data = [round(x, 3) for x in data]
    write_header = not os.path.exists(file_path) or mode == 'w'

    with open(file_path, mode, newline='', encoding='utf-8') as csvfile:
        if isinstance(rounded_data[0], dict):
            fieldnames = rounded_data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(rounded_data)
        else:
            if header is None:
                raise ValueError("Cần truyền header nếu data là list of lists.")
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(header)
            writer.writerow(rounded_data)


def plot_and_save_metrics(results, save_dir, num_slices, num_urllc):
    """
    Vẽ và lưu từng biểu đồ riêng biệt cho các tiêu chí theo frame.

    Parameters
    ----------
    results : list[dict]
        Danh sách kết quả theo từng frame.
    save_dir : str
        Thư mục lưu ảnh. Mặc định là 'metrics_plots'.
    """
    # Tạo thư mục nếu chưa có
    os.makedirs(save_dir, exist_ok=True)

    # Tách dữ liệu
    metrics = {
        "Decision Time (ms)": results["decision_time_ms"],
        "eMBB Throughput (Mbps)": results["throughput"],
        "SLA eMBB": results["SLA_eMBB"],
        "SLA URLLC": results["SLA_URLLC"],
        "utilPower": results["utilPower"],
        "stability": results["stability"],
        "utilPRB": results["utilPRB"]
    }

    #metrics = {
    #    "Decision Time (ms)": [entry["decision_time_ms"] for entry in results],
    #    "eMBB Throughput (Mbps)": [entry["throughput"] for entry in results],
    #    "SLA eMBB": [entry["SLA_eMBB"] for entry in results],
    #    "SLA URLLC": [entry["SLA_URLLC"] for entry in results],
    #    "Jain Index": [entry["Jain_Index"] for entry in results]
    #}

    # Vẽ và lưu từng biểu đồ
    for title, values in metrics.items():
        plt.figure(figsize=(8, 5))
        plt.plot(range(len(values)), values, marker='o')
        plt.xlabel("Frame")
        plt.ylabel(title)
        plt.title(f"{title} per Frame")
        plt.grid(True)

        # Tạo tên file hợp lệ
        filename = f"{title}_{num_slices}_{num_urllc}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath)
        plt.close()