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


def buildEnvAgent(RUs, urllc_slices, embb_slices, H, inter_RU, inter_factor, w_reward, cost_switch, cost_gb, scale_max, train_cons):
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
    embb_envs = []
    embb_dqn_agents = []
    urllc_envs = []
    urllc_dqn_agents = []
    num_urllc_ue = [len(urllc_slices[u].ue_set) for u in range(len(urllc_slices))]
    num_embb_ue = [len(urllc_slices[u].ue_set) for u in range(len(embb_slices))]

    for r in range(len(RUs)):
        # Khởi tạo môi trường và agent cho từng loại slice ở từng RU
        urllc_env = RU_URLLC_Env(RUs[r], urllc_slices, len(urllc_slices), H[r][0][:len(urllc_slices)], 
                                 inter_RU, inter_factor, w_reward, cost_switch, cost_gb, scale_max, train_cons["forDQN"])
        #embb_env = RU_eMBB_Env(RUs[r], embb_slices, len(embb_slices), H[r][0][len(urllc_slices):len(urllc_slices)+len(embb_slices)], 
        #                       inter_RU, inter_factor, w_reward, cost_switch, cost_gb, scale_max, train_cons["forDQN"])

        urllc_agent = MultiHeadDQNAgent(1, len(urllc_slices), num_urllc_ue, len(RUs[r].bwps), train_cons["forDQN"])
        #embb_agent = MultiHeadDQNAgent(1, len(embb_slices), num_embb_ue, len(RUs[r].bwps), train_cons["forDQN"])

        #embb_envs.append(embb_env)
        urllc_envs.append(urllc_env)
        #embb_dqn_agents.append(embb_agent)
        urllc_dqn_agents.append(urllc_agent)

    # Frame env và SAC agent
    num_bwp_ru = [len(RUs[r].bwps) for r in range(len(RUs))]
    frame_env = FrameEnv(urllc_envs, embb_envs, urllc_slices, embb_slices, H, w_reward)
    sac_agent = SACAgent(1, len(RUs), num_bwp_ru, len(urllc_slices) + len(embb_slices), train_cons["forSAC"])

    return embb_envs, urllc_envs, embb_dqn_agents, urllc_dqn_agents, frame_env, sac_agent


def alternating_training(embb_envs, urllc_envs, embb_dqn_agents, urllc_dqn_agents, frame_env, sac_agent):
    """
    Hàm thực hiện train mô hình và lưu
    Input :
    embb_envs, urllc_envs : tập môi trường embb và urllc 
    embb_dqn_agents, urllc_dqn_agents : tập agent của embb và urllc 
    frame_env : môi trường của sac 
    sac_agent : agent của sac

    Output :
    embb_models_path, urllc_models_path : tập các đường dẫn tới file lưu model dqn
    sac_model_path : đường dẫn tới file lưu model sac
    """
    
    # Set các file chứa model
    embb_models_path = []
    urllc_models_path = []

    # Train DQN ở từng RU
    for r in range(len(embb_envs)):
        print(f"-- Training DQN agents {r} --")
        embb_model = train_dqn_embb(embb_envs[r], embb_dqn_agents[r])
        urllc_model = train_dqn_urllc(urllc_envs[r], urllc_dqn_agents[r])
        embb_models_path.append(embb_model)
        urllc_models_path.append(urllc_model)

    # Train SAC chung
    print("-- Training SAC (frame-level) --")
    sac_model_path = train_sac(frame_env, sac_agent)

    print("Training complete.")
    return embb_models_path, urllc_models_path, sac_model_path






