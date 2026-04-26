import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import torch
from collections import deque

def train_dqn_urllc(env, agent, num_episodes=2000, log_interval=10):
    """
    Huấn luyện DQN với đánh giá greedy định kỳ (không plot trong hàm này)
    """
    losses = []
    dqn_model_path = f"./combine/DQN/model/best_dqn_RU{getattr(env, 'RU_index', 0)}_{env.num_slices}_{env.num_urllc}.pth"

    window_size = 10                      # độ dài cửa sổ trung bình
    reward_window = deque(maxlen=window_size)
    avg_rewards = []                      # lưu reward trung bình trượt
    losses = []

    for ep in trange(num_episodes, desc=f"Training DQN (RU {getattr(env, 'RU_index', 0)})"):
        state = env.reset()
        done = False
        total_loss = 0.0
        total_reward = 0.0
        steps = 0

        while not done:
            # --- Chọn hành động ---
            action = agent.select_action(state)

            # --- Tương tác môi trường ---
            next_state, reward, done, info = env.step(action)

            # --- Lưu transition và train ---
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.optimize_model()
            if loss is not None:
                total_loss += loss

            state = next_state
            total_reward += reward
            steps += 1

        # --- Sau mỗi episode ---
        agent.eps = max(agent.eps_end, agent.eps * agent.eps_decay)
        avg_loss = total_loss / max(1, steps)
        losses.append(avg_loss)

        # --- Cập nhật reward ---
        reward_window.append(total_reward)

        # Trung bình trượt (trên 10 ep gần nhất)
        moving_avg = np.mean(reward_window)
        avg_rewards.append(moving_avg)

        # --- In tiến trình ---
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"[Episode {ep+1:4d}] Avg Reward (last {window_size}) = {moving_avg:.3f}, "
                f"Last Reward = {total_reward:.3f}, Avg Loss = {avg_loss:.5f}, Eps = {agent.eps:.3f}")

        # Logging
        if (ep + 1) % log_interval == 0:
            print(f"[Ep {ep+1:4d}] Reward = {total_reward:.3f}, Avg Loss = {avg_loss:.4f}, eps = {agent.eps:.3f}")

    return avg_rewards, losses, dqn_model_path




