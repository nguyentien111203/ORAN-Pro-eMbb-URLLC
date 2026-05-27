import matplotlib.pyplot as plt
import numpy as np
import os

def plot_DQNtraining_curves(DQN_rewards_log, RU_index, num_slices, num_urllc):
    os.makedirs("./Figures/DQN", exist_ok=True)
    
    window = min(50, len(DQN_rewards_log))
    moving_avg = np.convolve(DQN_rewards_log, np.ones(window)/window, mode='valid')
    
    plt.figure(figsize=(10, 4))
    plt.plot(DQN_rewards_log, color='lightgray', alpha=0.6, label='Raw Reward')
    plt.plot(range(window-1, len(DQN_rewards_log)), moving_avg,
             color='green', linewidth=2, label=f'Moving Average (w={window})')
    plt.title(f"DQN Greedy Evaluation Reward - RU {RU_index}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"./Figures/DQN/DQN_reward_convergence_RU{RU_index}.png")
    plt.close()


def plot_DQNlosstraining_curves(DQN_loss_log, RU_index, num_slices, num_urllc):
    os.makedirs("./Figures/DQN", exist_ok=True)
    
    plt.figure(figsize=(10, 4))
    plt.plot(DQN_loss_log, color='red', linewidth=1.5, label='DQN loss per episode')
    plt.title(f"DQN Slot-Level Loss - RU {RU_index}")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"./Figures/DQN/DQN_loss_convergence_RU{RU_index}.png")
    plt.close()