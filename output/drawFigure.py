import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ===============================
# 1. Cấu hình Global cho bài báo (IEEE/ACM Standard)
# ===============================
plt.rcParams.update({
    "text.usetex": True,            # Dùng LaTeX cho công thức và font
    "font.family": "serif",
    "font.serif": ["Times New Roman"],        # Font chuẩn bài báo
    "font.size": 22,                # Size 22-24 là vừa đẹp khi chèn vào 2-column paper
    "axes.labelsize": 24,
    "legend.fontsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "axes.linewidth": 1.2,          # Làm đậm khung biểu đồ
    "pdf.fonttype": 42,             # Đảm bảo font không bị lỗi khi xuất PDF
    "ps.fonttype": 42
})

# Load data 
df = pd.read_csv(".\output\output copy.csv")
x = df["num_URLLC"].values
bar_width = 0.5

import numpy as np
import matplotlib.pyplot as plt

def plot_scientific_bar_two_alg(x,
                                alg1_y1, alg1_y2,
                                alg2_y1, alg2_y2,
                                alg1_name, alg2_name,
                                metric1_label, metric2_label,
                                ylabel, filename):

    fig, ax = plt.subplots(figsize=(10, 7))

    x = np.array(x)

    # ---- Algorithm 1 ----
    ax.bar(x - 1.5*bar_width, alg1_y1,
           width=bar_width,
           label=f"{alg1_name} - {metric1_label}",
           color='white', edgecolor='black',
           hatch='///', linewidth=1.2)

    ax.bar(x - 0.5*bar_width, alg1_y2,
           width=bar_width,
           label=f"{alg1_name} - {metric2_label}",
           color='#d9d9d9', edgecolor='black',
           hatch='\\\\\\', linewidth=1.2)

    # ---- Algorithm 2 ----
    ax.bar(x + 0.5*bar_width, alg2_y1,
           width=bar_width,
           label=f"{alg2_name} - {metric1_label}",
           color='white', edgecolor='black',
           hatch='xxx', linewidth=1.2)

    ax.bar(x + 1.5*bar_width, alg2_y2,
           width=bar_width,
           label=f"{alg2_name} - {metric2_label}",
           color='#bfbfbf', edgecolor='black',
           hatch='---', linewidth=1.2)

    # ---- Axis formatting ----
    ax.set_xlabel("Number of URLLC Slices")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)

    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(loc='lower center',
              bbox_to_anchor=(0.5, 1.18),
              ncol=2,
              frameon=True,
              edgecolor='black',
              fancybox=False)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)

def plot_scientific_barThr(x,
                           y_alg1, y_alg2,
                           alg1_name, alg2_name,
                           ylabel, filename):

    fig, ax = plt.subplots(figsize=(10, 7))

    bar_width = 1
    x = np.array(x)

    # ---- Algorithm 1 ----
    ax.bar(x - bar_width/2, y_alg1,
           width=bar_width,
           label=alg1_name,
           color='white',
           edgecolor='black',
           hatch='///',
           linewidth=1.2)

    # ---- Algorithm 2 ----
    ax.bar(x + bar_width/2, y_alg2,
           width=bar_width,
           label=alg2_name,
           color='#d9d9d9',
           edgecolor='black',
           hatch='\\\\\\',
           linewidth=1.2)

    # ---- Axis formatting ----
    ax.set_xlabel("Number of URLLC Slices")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)

    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(loc='lower center',
              bbox_to_anchor=(0.5, 1.12),
              ncol=2,
              frameon=True,
              edgecolor='black',
              fancybox=False)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)


# ===============================
# 3. Thực thi vẽ
# ===============================

# Biểu đồ 1: SLA
plot_scientific_bar_two_alg(
    x, df["avg_sla_urllc_ml"], df["avg_sla_embb_flat"],
    df["avg_sla_embb_ml"], df["avg_sla_urllc_flat"],
    "Proposed", "FlatDRL",
    "URLLC SLA", "eMBB SLA",
    "Average SLA", "sla_vs_urllc.pdf"
)

# Biểu đồ 2: Resource Utilization
plot_scientific_bar_two_alg(
    x, df["avg_power_ml"], df["avg_power_flat"],
    df["util_ml"], df["util_flat"],
    "Proposed", "FlatDRL",
    "Power Util", "PRB Util",
    "Average Utilization", "util_vs_urllc.pdf"
)

plot_scientific_barThr(x, df["avg_thr_ml"], df["avg_thr_flat"], "Proposed", "FlatDRL",
                       "Average Throughput (Mbps)", "avg_thr.pdf")

