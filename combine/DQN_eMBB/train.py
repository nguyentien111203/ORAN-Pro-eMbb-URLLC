import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import torch
from collections import deque


def evaluate_greedy(agent, env, n_eval=5):
    prev_eps = agent.eps
    agent.eps = 0.0  # Greedy evaluation
    returns = []

    for _ in range(n_eval):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
        returns.append(total_reward)

    agent.eps = prev_eps
    return np.mean(returns), np.std(returns)


def train_dqn(env, agent, num_episodes=2000, log_interval=10, eval_interval=10):
    """
    Huấn luyện DQN với đánh giá greedy định kỳ (không plot trong hàm này)
    """
    losses = []
    best_eval = -float("inf")
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

        # === Đánh giá greedy định kỳ ===
        if (ep + 1) % eval_interval == 0:
            eval_mean, eval_std = evaluate_greedy(agent, env, n_eval=5)
            print(f"[Eval @Ep {ep+1}] Greedy Reward = {eval_mean:.3f} ± {eval_std:.3f}")

            # Lưu best model
            if eval_mean > best_eval:
                torch.save(agent.policy_net.state_dict(), dqn_model_path)
                best_eval = eval_mean

    print(f"✅ Training finished. Best greedy reward = {best_eval:.3f}")
    return avg_rewards, losses, dqn_model_path


def evaluate_dqn_agents(slot_envs, dqn_agents, num_episodes=10):
    """
    Đánh giá tất cả DQN agents (slot-level) sau khi train.
    """
    results = []

    for r, env in enumerate(slot_envs):
        agent = dqn_agents[r]
        total_reward = 0.0
        total_thr = 0.0
        total_sla = 0.0
        total_fair = 0.0
        episodes_done = 0

        for ep in range(num_episodes):
            state = env.reset()
            done = False
            while not done:
                action = agent.select_action(state, eval_mode=True)
                next_state, reward, done, info = env.step(action)

                total_reward += reward
                total_thr += info.get("eMBB_thr", 0)
                total_sla += info.get("SLA", 0)
                total_fair += info.get("JainIndex", 0)
                state = next_state

                done = info.get("done", False)

            episodes_done += 1

        results.append({
            "RU": r,
            "avg_reward": total_reward / max(1, episodes_done),
            "avg_thr": total_thr / max(1, episodes_done),
            "avg_sla": total_sla / max(1, episodes_done),
            "avg_fair": total_fair / max(1, episodes_done),
        })

    print("\n[DQN Evaluation Results]")
    for res in results:
        print(f"RU {res['RU']:2d}: Reward={res['avg_reward']:.3f} | "
              f"Thr={res['avg_thr']:.3f} | SLA={res['avg_sla']:.3f} | Fair={res['avg_fair']:.3f}")

    return results


