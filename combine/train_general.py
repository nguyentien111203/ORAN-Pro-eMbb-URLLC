# Alternating training: DQN (slot-level) <-> SAC (frame-level)
import numpy as np
import torch
from tqdm import trange, tqdm

from combine.common.multiagent_DQN import MultiHeadDQNAgent
from combine.SAC.SACagent import SACAgent
from combine.DQN_eMBB.embb_env import RU_eMBB_Env
from combine.DQN_URLLC.urllc_env import RU_URLLC_Env
from combine.SAC.FrameEnv import FrameEnv
from combine.DQN_eMBB.train import train_dqn_embb
from combine.DQN_URLLC.train import train_dqn_urllc
from combine.SAC.train_SAC import train_sac
from combine.utils.pltSAC import plot_SACtraining_curves, plot_SACcriticlosstraining_curves, plot_SACactorlosstraining_curves
from DQN_general import RU_Env
#gộp lại xây dựng chung cho 1 ru
def buildEnvAgent(RUs, urllc_slices, embb_slices, H, inter_RU, inter_factor, N0,
                  w_reward, cost_switch, cost_gb, scale_max, train_cons, frame_slots):
    """
    Dựng các môi trường và agent cho SAC và DQN
    set of Radio Unit RUs : tập các RU
    set of urllc_slices : tập các urllc slice
    set of embb_slices : tập các embb slice
    H : ma trận các kênh truyền
    inter_RU : hệ số nhiễu liên RU
    w_reward : hệ số cho hàm phần thưởng
    cost_switch : hệ số chi phí chuyển BWP
    cost_gb : hệ số guard band
    scale_max : các giá trị scale theo từng thành phần trong scale_max
    train_cons : hằng số trong training mô hình

    giá trị trả về :
    embb_envs, urllc_envs : tập các môi trường cho embb và urllc ở từng RU
    embb_dqn_agents, urllc_dqn_agents : tập các agent embb và urllc ở từng RU
    frame_env : môi trường của SAC
    sac_agent : agent của SAC
    """
    # env cho embb và urllc trong từng RU và từng DQN agents
    ru_envs = [] #chỉnh thành envs
    ru_dqn_agents = [] #agents
    num_urllc = [len(urllc_slices[u].ue_set) for u in range(len(urllc_slices))]
    num_embb = [len(urllc_slices[u].ue_set) for u in range(len(embb_slices))]
    num_slices_combined = num_urllc + num_embb
    
    num_urllc_ue = [len(s.ue_set) for s in urllc_slices]
    num_embb_ue = [len(s.ue_set) for s in embb_slices]
    num_ue_combined = num_embb_ue + num_urllc_ue
    for r in range(len(RUs)):
        # 1. Đồng bộ ma trận kênh truyền H (Ghép eMBB và URLLC)
        H_urllc = H[r][0][:len(urllc_slices)]
        H_embb = H[r][0][len(urllc_slices):len(urllc_slices) + len(embb_slices)]
        
        if isinstance(H_embb, list):
            H_combined = H_embb + H_urllc
        else:
            H_combined = np.concatenate((H_embb, H_urllc), axis=0)

        # 2. Hợp nhất danh sách Slices
        slices_combined = list(embb_slices) + list(urllc_slices)

        # 3. Khởi tạo 1 môi trường RU_Env duy nhất quản lý chung
        ru_env = RU_Env(
            RUs[r], slices_combined, len(urllc_slices), len(embb_slices), H_combined, 
            inter_RU, inter_factor, N0, w_reward, cost_switch, 
            cost_gb, scale_max, train_cons["forDQN"], frame_slots
        )
        ru_envs.append(ru_env)

        # 4. Tạo 1 MultiHeadDQNAgent đại diện xử lý tập trung
        dqn_agent = MultiHeadDQNAgent(
            ru_env.state_dim, num_slices_combined, num_ue_combined, 
            len(RUs[r].bwps), train_cons["forDQN"]
        )
        ru_env.assign_dqn_agent(dqn_agent)
        ru_dqn_agents.append(dqn_agent)

    # 5. Cấu hình môi trường Frame cho thuật toán cấp cao SAC
    num_bwp_ru = [len(RUs[r].bwps) for r in range(len(RUs))]
    
    # Truyền cùng danh sách môi trường ru_envs để SAC đồng bộ hóa trạng thái tổng thể
    frame_env = FrameEnv(RUs, ru_envs, ru_envs, urllc_slices, embb_slices, H, w_reward, scale_max, frame_slots)
    
    sac_agent = SACAgent(
        4 + len(urllc_slices) + len(embb_slices), len(RUs), 
        num_bwp_ru, len(urllc_slices) + len(embb_slices), train_cons["forSAC"]
    )

    # Trả về double-reference để giữ tính tương thích ngược với luồng code ngoài
    return ru_envs, ru_envs, ru_dqn_agents, ru_dqn_agents, frame_env, sac_agent


def alternating_training(num_rus, ru_envs,  ru_dqn_agents, frame_env, sac_agent, numepDQN, numepSAC):
    """
    Vòng lặp huấn luyện đan xen giữa DQN mức Slot và SAC mức Frame.
    """
    # Khởi tạo hạn mức phân bổ tài nguyên ban đầu cho các BWP
    BWP_slice = [[[frame_env.RUs[r].bwps[b].num_prb / frame_env.num_slices 
                   for _ in range(frame_env.num_slices)] 
                  for b in range(len(frame_env.RUs[r].bwps))] for r in range(num_rus)]

    print(f"-- Đang tiến hành huấn luyện Unified DQN Agents (Slot-level) --")
    
    # Chạy vòng lặp huấn luyện trực tiếp cho môi trường hợp nhất
    for ep in trange(numepDQN, desc="Unified DQN Training Loop"):
        for r in range(num_rus):
            env = ru_envs[r] # embb_envs giờ đóng vai trò là unified env chung
            agent =ru_dqn_agents[r]
            
            state = env.reset() # Reset slot
            done = False
            
            while not done:
                # Chọn hành động phân bổ PRB dựa trên quota từ SAC
                action = agent.select_action(state, BWP_slice[r])
                
                # Tính toán các chỉ số QoS thực tế thu được từ hành động
                totalThrRate, _, totalLatRate = env.computeOutput(action)
                
                # Thực hiện bước chuyển đổi slot và tính toán Reward hợp nhất (Eq 32)
                next_state, reward, done, info = env.step(totalThrRate, totalLatRate)
                
                # Lưu vào Replay Buffer và tối ưu mạng neural
                agent.store_transition(state, action, reward, next_state, done)
                agent.optimize_model()
                
                state = next_state
                
            # Suy hao epsilon sau mỗi tập huấn luyện
            agent.eps = max(agent.eps_end, agent.eps * agent.eps_decay)

    # Tiến hành chạy huấn luyện SAC cho điều phối mức Frame
    print("-- Đang tiến hành huấn luyện SAC (Frame-level) --")
    avg_rewards, actor_losses, critic_losses, sac_model_path = train_sac(frame_env, sac_agent, numepSAC)
    
    # Vẽ và xuất đồ thị phân tích kết quả mô phỏng
    plot_SACtraining_curves(avg_rewards, num_rus, frame_env.num_slices, frame_env.num_urllc)
    plot_SACactorlosstraining_curves(actor_losses, num_rus, frame_env.num_slices, frame_env.num_urllc)
    plot_SACcriticlosstraining_curves(critic_losses, num_rus, frame_env.num_slices, frame_env.num_urllc)
    
    print("Training completed")




