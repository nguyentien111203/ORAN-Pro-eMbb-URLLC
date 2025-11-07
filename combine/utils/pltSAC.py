import matplotlib.pyplot as plt


def plot_SACtraining_curves(SAC_rewards_log, num_slices, num_urllc):
    plt.figure(figsize=(10,4))
    plt.plot(SAC_rewards_log, label='SAC reward per frame', color='orange', alpha=0.7)
    plt.title("SAC Frame-Level Reward")
    plt.xlabel("Frame")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig(f"./Figures/SAC_{num_slices}_{num_urllc}_training_convergence.png")

def plot_SACactorlosstraining_curves(SAC_loss_log, num_slices, num_urllc):
    plt.figure(figsize=(10,4))
    plt.plot(SAC_loss_log, label='SAC actor loss per frame', color='orange', alpha=0.7)
    plt.title("SAC Frame-Level Actor loss")
    plt.xlabel("Frame")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig(f"./Figures/SAC_{num_slices}_{num_urllc}_actorlosstraining_convergence.png")

def plot_SACcriticlosstraining_curves(SAC_loss_log, num_slices, num_urllc):
    plt.figure(figsize=(10,4))
    plt.plot(SAC_loss_log, label='SAC critic loss per frame', color='orange', alpha=0.7)
    plt.title("SAC Frame-Level Critic loss")
    plt.xlabel("Frame")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig(f"./Figures/SAC_{num_slices}_{num_urllc}_criticlosstraining_convergence.png")