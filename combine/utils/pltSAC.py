import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,            # Dùng LaTeX cho công thức và font
    "font.family": "serif",
    "font.serif": ["Times"],        # Font chuẩn bài báo
    "font.size": 22,                # Size 22-24 là vừa đẹp khi chèn vào 2-column paper
    "axes.labelsize": 24,
    "legend.fontsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "axes.linewidth": 1.2,          # Làm đậm khung biểu đồ
    "pdf.fonttype": 42,             # Đảm bảo font không bị lỗi khi xuất PDF
    "ps.fonttype": 42
})

def plot_SACtraining_curves(SAC_rewards_log, num_RU, num_slices, num_urllc):
    plt.figure(figsize=(10,4))
    plt.plot(SAC_rewards_log, label='SAC reward per frame', color='orange', alpha=0.7)
    plt.title("SAC Frame-Level Reward")
    plt.xlabel("Frame")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig(f"./Figures/SAC/SAC_{num_RU}_{num_slices}_{num_urllc}_training_convergence.png")

def plot_SACactorlosstraining_curves(SAC_loss_log, num_RU, num_slices, num_urllc):
    plt.figure(figsize=(10,4))
    plt.plot(SAC_loss_log, label='SAC actor loss per frame', color='orange', alpha=0.7)
    plt.title("SAC Frame-Level Actor loss")
    plt.xlabel("Frame")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig(f"./Figures/SAC/SAC_{num_RU}_{num_slices}_{num_urllc}_actorlosstraining_convergence.png")

def plot_SACcriticlosstraining_curves(SAC_loss_log, num_RU, num_slices, num_urllc):
    plt.figure(figsize=(10,4))
    plt.plot(SAC_loss_log, label='SAC critic loss per frame', color='orange', alpha=0.7)
    plt.title("SAC Frame-Level Critic loss")
    plt.xlabel("Frame")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig(f"./Figures/SAC/SAC_{num_RU}_{num_slices}_{num_urllc}_criticlosstraining_convergence.png")