# Alternating training: DQN (slot-level) <-> SAC (frame-level)
import numpy as np
import torch
from tqdm import trange, tqdm

from combine.common.multiagent_DQN import MultiHeadDQNAgent
from combine.SAC.SACagent import SACAgent
from combine.DQN.env import RU_Env
from combine.SAC.FrameEnv import FrameEnv
from combine.DQN.train import train_dqn
from combine.SAC.train_SAC import train_sac


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
    envs = []
    agents = []
    num_urllc_ue = [len(urllc_slices[u].ue_set) for u in range(len(urllc_slices))]
    num_embb_ue = [len(urllc_slices[u].ue_set) for u in range(len(embb_slices))]
    num_ue = np.concatenate([num_embb_ue, num_urllc_ue])

    for r in range(len(RUs)):
        # Khởi tạo môi trường và agent cho từng loại slice ở từng RU
        # Kiếm tra xem số urllc thế nào
        env = RU_Env(RUs[r], embb_slices, urllc_slices, H[r][0], 
                    inter_RU, inter_factor, N0, w_reward, cost_switch, 
                    cost_gb, scale_max, train_cons["forDQN"], frame_slots)
        envs.append(env)
        agent = MultiHeadDQNAgent(env.state_dim, len(num_ue), num_ue, len(RUs[r].bwps), train_cons["forDQN"])
        env.assign_agent(agent)
        agents.append(agent)
        

    # Frame env và SAC agent
    num_bwp_ru = [len(RUs[r].bwps) for r in range(len(RUs))]
    frame_env = FrameEnv(RUs, envs, urllc_slices, embb_slices, H, w_reward, scale_max, frame_slots)
    sac_agent = SACAgent(frame_env.state_dim, len(RUs), 
                         num_bwp_ru, len(urllc_slices) + len(embb_slices), train_cons["forSAC"])

    return envs, agents, frame_env, sac_agent


def alternating_training(num_rus, envs, agents, 
                         frame_env, sac_agent, numepDQN, numepSAC):
    """
    Hàm thực hiện train mô hình và lưu
    Input :
    envs : tập môi trường 
    agents : tập agent 
    frame_env : môi trường của sac 
    sac_agent : agent của sac

    Output :
    models_path : tập các đường dẫn tới file lưu model dqn
    model_path : đường dẫn tới file lưu model sac
    """
    
    # Set các file chứa model
    models_path = []

    # Budget cho các slice ban đầu
    BWP_slice = [[[frame_env.RUs[r].bwps[b].num_prb / frame_env.num_slices 
                   for _ in range(frame_env.num_slices)] 
                  for b in range(len(frame_env.RUs[r].bwps))] for r in range(num_rus)]

    # Train DQN ở từng RU
    print(f"-- Training DQN agents --")
    
    reward, losses = train_dqn(envs, agents, numepDQN, BWP_slice)

    # Train SAC chung
    print("-- Training SAC (frame-level) --")
    sac_model_path = train_sac(frame_env, sac_agent, numepSAC)

    print("Training complete.")
    #return embb_models_path, urllc_models_path, sac_model_path






