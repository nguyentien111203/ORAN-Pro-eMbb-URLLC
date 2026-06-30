import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": False,           # Chuyển thành True nếu bạn đã cài xong LaTeX
    "font.family": "serif",
    "font.serif": ["Times"],        
    "font.size": 16,                
    "axes.labelsize": 18,
    "legend.fontsize": 14,
    "axes.linewidth": 1.2,          
})

SAVE_DIR = "./Figures/SAC/"

def moving_average(data, window_size=50):
    """Hàm tính trung bình động (Moving Average) để làm mượt đồ thị"""
    if len(data) < window_size:
        return data
    # Dùng convolution để tính trung bình trượt
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_SACtraining_curves(SAC_rewards_log, num_RU, num_slices, num_urllc, mode):
    os.makedirs(SAVE_DIR, exist_ok=True)
    plt.figure(figsize=(8,5))
    
    # 1. Vẽ Reward thô (Mờ phía sau)
    plt.plot(SAC_rewards_log, color='lightgray', alpha=0.4, label='Raw Reward')
    
    # 2. Vẽ Moving Average (Đậm phía trước, màu xanh lá giống ảnh của bạn)
    window = min(10, max(1, len(SAC_rewards_log) // 10)) # Tự động chọn window size
    smoothed = moving_average(SAC_rewards_log, window_size=window)
    
    # Khớp trục x cho mảng đã làm mượt
    x_smoothed = range(len(SAC_rewards_log) - len(smoothed), len(SAC_rewards_log))
    plt.plot(x_smoothed, smoothed, color='forestgreen', linewidth=1.5, label=f'Moving Average (w={window})')
    
    plt.title("Moving Average Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"./Figures/SAC/SAC_reward_convergence", mode, ".png")

def plot_SACactorlosstraining_curves(SAC_loss_log, num_RU, num_slices, num_urllc, mode):
    os.makedirs(SAVE_DIR, exist_ok=True)
    plt.figure(figsize=(8,5))
    
    # Vẽ Loss màu đỏ giống ảnh
    plt.plot(SAC_loss_log, color='tab:red', linewidth=1.5)
    
    plt.title("SAC Actor Loss")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"./Figures/SAC/SAC_actor_loss_convergence", mode, ".png")

def plot_SACcriticlosstraining_curves(SAC_loss_log, num_RU, num_slices, num_urllc, mode):
    os.makedirs(SAVE_DIR, exist_ok=True)
    plt.figure(figsize=(8,5))
    
    # Vẽ Loss màu đỏ giống ảnh
    plt.plot(SAC_loss_log, color='tab:red', linewidth=1.5)
    
    plt.title("SAC Critic Loss")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"./Figures/SAC/SAC_critic_loss_convergence", mode,".png")