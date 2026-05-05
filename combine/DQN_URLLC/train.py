import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from collections import deque

def train_dqn_urllc(env, agent, num_slices, num_episodes, initBWP_slice):
    """
    Huấn luyện DQN với đánh giá greedy định kỳ (không plot trong hàm này)
    initBWP_slice : phân bổ PRB từ các RU về các slice ban đầu
    """
    losses = []
    dqn_model_path = f"./combine/DQN/model/best_dqn_RU{getattr(env, 'RU_index', 0)}_{num_slices}_{env.num_urllc}.pth"

    window_size = 10                      # độ dài cửa sổ trung bình
    reward_window = deque(maxlen=window_size)
    avg_rewards = []                      # lưu reward trung bình trượt
    state = np.zeros(env.state_dim)       # bắt buộc
    losses = []

    for ep in trange(num_episodes, desc=f"Training DQN (RU {getattr(env, 'RU_index', 0)})"):
        done = False
        total_loss = 0.0
        total_reward = 0.0
        steps = 0

        while not done:
            # --- Chọn hành động ---
            action = agent.select_action(state, initBWP_slice)

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

    return avg_rewards, losses, dqn_model_path




