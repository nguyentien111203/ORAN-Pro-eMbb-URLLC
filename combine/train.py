# Alternating training: DQN (slot-level) <-> SAC (frame-level)
import numpy as np
import torch
from tqdm import trange

from combine.common.multiagent_DQN import MultiHeadDQNAgent
from combine.SAC.SACagent import SACAgent
from combine.SAC.FrameEnv import FrameEnv
from combine.SAC.train_SAC import train_sac
from combine.utils.pltSAC import plot_SACtraining_curves, plot_SACcriticlosstraining_curves, plot_SACactorlosstraining_curves

# Import môi trường hợp nhất
from combine.general.DQN_general import RU_Env

def buildEnvAgent(RUs, urllc_slices, embb_slices, H, inter_RU, inter_factor, N0,
                  w_reward, cost_switch, cost_gb, scale_max, train_cons, frame_slots):
    """
    Dựng môi trường hợp nhất xử lý đồng thời cả 2 dịch vụ trên 1 RU duy nhất.
    """
    ru_envs = []
    ru_dqn_agents = []
    
    num_urllc = len(urllc_slices)
    num_embb = len(embb_slices)
    
    num_slices_combined = num_urllc + num_embb
    num_urllc_ue = [len(s.ue_set) for s in urllc_slices]
    num_embb_ue = [len(s.ue_set) for s in embb_slices]
    num_ue_combined = num_embb_ue + num_urllc_ue

    for r in range(len(RUs)):
        # Gộp chung danh sách slices (eMBB trước, URLLC sau)
        slices_combined = list(embb_slices) + list(urllc_slices)
        
        # Hợp nhất ma trận nhiễu H
        H_urllc = H[r][0][:num_urllc]
        H_embb = H[r][0][num_urllc:num_urllc + num_embb]
        if isinstance(H_embb, list):
            H_combined = H_embb + H_urllc
        else:
            H_combined = np.concatenate((H_embb, H_urllc), axis=0)

        # Khởi tạo 1 Env duy nhất quản trị RU
        ru_env = RU_Env(
            RUs[r], slices_combined, num_urllc, num_embb, H_combined, 
            inter_RU, inter_factor, N0, w_reward, cost_switch, 
            cost_gb, scale_max, train_cons["forDQN"], frame_slots
        )
        ru_envs.append(ru_env)
        
        # Tạo Agent tập trung
        dqn_agent = MultiHeadDQNAgent(
            ru_env.state_dim, num_slices_combined, num_ue_combined, 
            len(RUs[r].bwps), train_cons["forDQN"]
        )
        ru_env.assign_dqn_agent(dqn_agent)
        ru_dqn_agents.append(dqn_agent)

    num_bwp_ru = [len(RUs[r].bwps) for r in range(len(RUs))]
    
    # Khởi tạo môi trường mức Frame
    frame_env = FrameEnv(RUs, ru_envs, urllc_slices, embb_slices, H, w_reward, scale_max, frame_slots)
    sac_agent = SACAgent(4 + num_slices_combined, len(RUs), num_bwp_ru, num_slices_combined, train_cons["forSAC"])

    return ru_envs, ru_dqn_agents, frame_env, sac_agent


def alternating_training(num_rus, ru_envs, ru_dqn_agents, frame_env, sac_agent, numepDQN, numepSAC):
    """
    Vòng lặp huấn luyện xen kẽ tổng hợp mức Slot (DQN) và mức Frame (SAC).
    """
    
    # ==========================================
    # FIX 1: ĐẢO CHIỀU MA TRẬN BWP_slice (Thành [RU][Slice][BWP])
    # ==========================================
    BWP_slice = [[[frame_env.RUs[r].bwps[b].num_prb / frame_env.num_slices 
                   for b in range(len(frame_env.RUs[r].bwps))] 
                  for _ in range(frame_env.num_slices)] for r in range(num_rus)]

    print(f"-- Đang tiến hành huấn luyện Unified DQN Agents (Slot-level) --")
    for ep in trange(numepDQN, desc="Unified DQN Training Loop"):
        for r in range(num_rus):
            env = ru_envs[r]     
            agent = ru_dqn_agents[r] 
            
            state = env.reset()
            done = False
            
            while not done:
                action = agent.select_action(state, BWP_slice[r])
                totalThrRate, _, totalLatRate = env.computeOutput(action)
                next_state, reward, done, info = env.step(totalThrRate, totalLatRate)
                
                agent.store_transition(state, action, reward, next_state, done)
                agent.optimize_model()
                state = next_state
                
            agent.eps = max(agent.eps_end, agent.eps * agent.eps_decay)

    # ==========================================
    # FIX 2: LƯU TRỌNG SỐ CHO CÁC MẠNG DQN
    # ==========================================
    print("\n-- Lưu mô hình DQN --")
    for r in range(num_rus):
        dqn_model_path = f"dqn_agent_ru_{r}.pth"
        torch.save(ru_dqn_agents[r].policy_net.state_dict(), dqn_model_path)
        print(f"Đã lưu: {dqn_model_path}")
    print("---------------------\n")

    print("-- Đang tiến hành huấn luyện SAC (Frame-level) --")
    avg_rewards, actor_losses, critic_losses, sac_model_path = train_sac(frame_env, sac_agent, numepSAC)
    
    # Save SAC Model
    torch.save(sac_agent.actor.state_dict(), sac_model_path)
    print(f"Đã lưu mô hình SAC tại: {sac_model_path}")
    
    plot_SACtraining_curves(avg_rewards, num_rus, frame_env.num_slices, frame_env.num_urllc)
    plot_SACactorlosstraining_curves(actor_losses, num_rus, frame_env.num_slices, frame_env.num_urllc)
    plot_SACcriticlosstraining_curves(critic_losses, num_rus, frame_env.num_slices, frame_env.num_urllc)
    print("Hoàn thành chu kỳ huấn luyện đan xen hệ thống.")