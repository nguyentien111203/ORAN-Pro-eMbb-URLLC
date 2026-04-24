import matplotlib.pyplot as plt

# Plot after training
def plot_DQNtraining_curves(DQN_rewards_log, RU_index, num_slices, num_urllc):
    plt.figure(figsize=(10,4))
    plt.plot(DQN_rewards_log, label='Greedy reward', alpha=0.7)
    plt.title(f"DQN Greedy Evaluation reward {RU_index}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig(rf"./Figures/DQN/DQN_RU_{RU_index}_{num_slices}_{num_urllc}")

# Plot after training
def plot_DQNlosstraining_curves(DQN_loss_log, RU_index, num_slices, num_urllc):
    plt.figure(figsize=(10,4))
    plt.plot(DQN_loss_log, label='DQN loss per episode', alpha=0.7)
    plt.title(f"DQN Slot-Level loss in RU {RU_index}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig(rf"./Figures/DQN/DQN_RUloss_{RU_index}_{num_slices}_{num_urllc}")
