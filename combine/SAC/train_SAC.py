import torch
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from collections import deque

def train_sac(env, agent, num_episodes=300, log_interval=10, update_per_step=5, batch_size=64):
    """
    Huấn luyện Soft Actor-Critic cho bài toán phân bổ công suất (Frame-level).
    - env: FrameEnv (SAC -> Power quota -> PowerMediator -> DQN)
    - update_per_step: số lần cập nhật actor/critic mỗi frame
    - batch_size: kích thước batch khi cập nhật
    """

    # --- Cấu hình ---
    window_size = 10         # độ dài cửa sổ trượt
    reward_window = deque(maxlen=window_size)
    actor_window = deque(maxlen=window_size)
    critic_window = deque(maxlen=window_size)

    # --- Bộ nhớ thống kê ---
    rewards_history = []
    avg_rewards = []
    actor_losses = []
    critic_losses = []

    for ep in trange(num_episodes, desc="Training SAC"):
        state = env.reset()
        done = False
        total_reward = 0.0
        actor_loss_sum = 0.0
        critic_loss_sum = 0.0
        steps = 0

        # Mỗi episode = 1 chuỗi frame
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            # Lưu transition
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # Cập nhật actor/critic nếu đủ batch
            if len(agent.replay_buffer) >= batch_size:
                for _ in range(update_per_step):
                    a_loss, c_loss = agent.update(batch_size=batch_size)
                    actor_loss_sum += a_loss if a_loss is not None else 0.0
                    critic_loss_sum += c_loss if c_loss is not None else 0.0

            state = next_state
            total_reward += reward
            steps += 1

            # Giới hạn số frame / episode
            done = steps >= env.frame_slots  # hoặc env.max_frames

        # --- Sau mỗi episode ---
        avg_actor_loss = actor_loss_sum / max(1, steps * update_per_step)
        avg_critic_loss = critic_loss_sum / max(1, steps * update_per_step)

        rewards_history.append(total_reward)
        actor_losses.append(avg_actor_loss)
        critic_losses.append(avg_critic_loss)

        # === Cập nhật trung bình trượt ===
        reward_window.append(total_reward)
        actor_window.append(avg_actor_loss)
        critic_window.append(avg_critic_loss)

        moving_avg_reward = np.mean(reward_window)
        moving_avg_actor = np.mean(actor_window)
        moving_avg_critic = np.mean(critic_window)
        avg_rewards.append(moving_avg_reward)

    # === In log theo tiến độ ===
    if (ep + 1) % log_interval == 0 or ep == 0:
        print(f"[SAC Ep {ep+1:4d}] "
              f"Reward={total_reward:.3f} | "
              f"Reward(avg {window_size})={moving_avg_reward:.3f} | "
              f"ActorLoss={moving_avg_actor:.5f} | "
              f"CriticLoss={moving_avg_critic:.5f}")

    # Sau khi train xong
    print(" SAC training complete.")
    sac_model_path = f"./combine/SAC/model/sac_model_{env.num_rus}_{env.num_slices}_{env.num_urllc}.pth"
    torch.save({
        'actor': agent.actor.state_dict(),
        'critic_1': agent.critic_1.state_dict(),
        'critic_2': agent.critic_2.state_dict(),
        'alpha': agent.alpha,
    }, sac_model_path)

    return avg_rewards, actor_losses, critic_losses, sac_model_path


def evaluate_sac(env, agent, num_episodes=20, eval_mode=True, render=False):
    """
    Đánh giá chính sách SAC (không update).
    """
    reward_list = []
    sla_list = []
    thr_list = []
    fair_list = []
    stab_list = []

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        total_sla = 0.0
        total_thr = 0.0
        total_fair = 0.0
        total_stab = 0.0
        steps = 0

        while not done:
            action = agent.select_action(state, eval_mode=eval_mode)
            next_state, reward, done, info = env.step(action)

            total_reward += reward
            total_sla += info.get("SLA", 0)
            total_thr += info.get("throughput", 0)
            total_fair += info.get("fairness", 0)
            total_stab += info.get("stability", 0)
            steps += 1
            state = next_state

            done = steps >= env.frame_slots  # 1 episode = 1 frame hoặc vài frame

        reward_list.append(total_reward)
        sla_list.append(total_sla / steps)
        thr_list.append(total_thr / steps)
        fair_list.append(total_fair / steps)
        stab_list.append(total_stab / steps)

    results = {
        "avg_reward": np.mean(reward_list),
        "avg_sla": np.mean(sla_list),
        "avg_thr": np.mean(thr_list),
        "avg_fair": np.mean(fair_list),
        "avg_stab": np.mean(stab_list),
    }

    print("[SAC Evaluation]")
    print(f"Reward={results['avg_reward']:.3f} | "
          f"SLA={results['avg_sla']:.3f} | Thr={results['avg_thr']:.3f} | "
          f"Fair={results['avg_fair']:.3f} | Stab={results['avg_stab']:.3f}")

    return results



