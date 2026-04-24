import torch
import numpy as np
from tqdm import trange
from collections import deque

def train_sac(env, agent, num_episodes=300, log_interval=10, update_per_step=1, batch_size=128):
    """
    Huấn luyện Soft Actor-Critic cho bài toán phân bổ công suất (Frame-level).
    env : Môi trường chung trong 1 frame
    agent : SAC agent 
    num_episodes : số episodes dùng để train agent
    log_interval : chu kỳ quan sát các giá trị
    update_per_step: số lần cập nhật actor/critic mỗi frame
    batch_size: kích thước batch khi cập nhật
    """

    # --- Cấu hình ---
    window_size = 10         # độ dài cửa sổ trượt
    reward_window = deque(maxlen=window_size)
    actor_window = deque(maxlen=window_size)
    critic_window = deque(maxlen=window_size)
    warmup_steps = 100  # hoặc 10000
    total_env_steps = 0
    policy_delay = 1


    # --- Bộ nhớ thống kê ---
    rewards_history = []
    avg_rewards = []
    actor_losses = []
    critic_losses = []

    last_actor_loss = torch.tensor(0.0, device=agent.device)   # <-- KHÔNG reset trong episode
    total_env_steps = 0
    debug = False

    for ep in trange(num_episodes, desc="Training SAC"):
        
        state = env.reset()
        done = False
        total_reward = 0.0
        actor_loss_sum = 0.0
        critic_loss_sum = 0.0
        update_count = 0                # <-- Đếm số lần update thực sự
        steps = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            # Save transition
            agent.replay_buffer.push(state, action, reward, next_state, done)

            total_env_steps += 1
            if total_env_steps >= warmup_steps and len(agent.replay_buffer) >= batch_size:
                #debug = True
                for step in range(update_per_step):
                    a_loss, c_loss, last_actor_loss = agent.update(
                        total_env_steps,
                        policy_delay,
                        last_actor_loss,
                        batch_size,
                        debug=debug
                    )
                    actor_loss_sum += float(a_loss)
                    critic_loss_sum += float(c_loss)
                    update_count += 1

            # Step forward
            state = next_state
            total_reward += reward
            steps += 1

            # Force terminate episode for ORAN env
            if steps >= log_interval:   # <-- OK
                done = True

        if update_count > 0:
            avg_actor_loss = actor_loss_sum / update_count
            avg_critic_loss = critic_loss_sum / update_count
        else:
            avg_actor_loss = 0.0
            avg_critic_loss = 0.0

        rewards_history.append(total_reward)
        actor_losses.append(avg_actor_loss)
        critic_losses.append(avg_critic_loss)

        # === Sliding averages ===
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


