import numpy as np
from tqdm import trange
import torch

def train_sacBM(env, agent, num_episodes):
    avg_rewards = []
    actor_losses = []
    critic_losses = []
    
    # ==========================================
    # KHAI BÁO BIẾN ĐỂ TRUYỀN VÀO HÀM UPDATE
    # ==========================================
    global_step = 0
    last_actor_loss = 0.0

    for ep in trange(num_episodes, desc="Training SAC (Episodes)"):
        state = env.reset()
        ep_reward = 0
        done = False
        
        # ==========================================
        # FIX 1: Để None ở bước đầu để tránh lỗi 108
        # ==========================================
        last_action = None 
        
        while not done:
            global_step += 1
            
            # 1. Chọn hành động
            action = agent.select_action(state, last_action)
            
            # 2. Tương tác với môi trường
            next_state, reward, done, info = env.step(action)
            
            # 3. Lưu vào Replay Buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # 4. Tối ưu hóa mạng Nơ-ron
            if len(agent.replay_buffer) > agent.batch_size:
                # ==========================================
                # FIX 2: Truyền đủ tham số và đổi đúng thứ tự a_loss, c_loss
                # ==========================================
                a_loss, c_loss, last_actor_loss = agent.update(
                    step=global_step, 
                    policy_delay=2, 
                    last_actor_loss=last_actor_loss, 
                    batch_size=agent.batch_size
                )
                
                if a_loss is not None:
                    actor_losses.append(a_loss)
                if c_loss is not None:
                    critic_losses.append(c_loss)
                    
            # 5. Chuyển trạng thái
            state = next_state
            last_action = action  
            ep_reward += reward

        avg_rewards.append(ep_reward)

    sac_model_path = "sac_model.pth"
    # Lưu trọng số của mô hình sau khi train xong
    torch.save({
        'actor': agent.actor.state_dict(),
        'critic_1': agent.critic_1.state_dict(),
        'critic_2': agent.critic_2.state_dict(),
        'alpha': agent.alpha
    }, sac_model_path)
    return avg_rewards, actor_losses, critic_losses, sac_model_path