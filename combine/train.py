# Alternating training: DQN (slot-level) <-> SAC (frame-level)
import numpy as np
import torch
from tqdm import trange, tqdm

from combine.DQN.multiagent_DQN import MultiHeadDQNAgent
from combine.SAC.SACagent import SACAgent
from combine.DQN.oran_env import SlotEnv
from combine.SAC.FrameEnv import FrameEnv
from combine.DQN.train import train_dqn
from combine.SAC.train_SAC import train_sac
from combine.SAC.FrameEnv import PowerMediator 
from combine.utils.plotDQN import plot_DQNtraining_curves, plot_DQNlosstraining_curves
from combine.utils.pltSAC import plot_SACtraining_curves, plot_SACactorlosstraining_curves, plot_SACcriticlosstraining_curves


def build_pipeline(RUs, slices, num_urllc, H, gain_ru_ru, dist_ue_ru,
                   T_slot, w_reward, T_max, NF, sla_slices, frame_slots,
                   max_steps, gamma, learning_rate, eps_DQN,
                   device='cpu', allocation_mode = "proportional"):
    """
    Dựng pipeline cho SAC và DQN
    """
    # slot envs and DQN agents
    slot_envs = []
    dqn_agents = []
    for r in range(len(RUs)):
        env = SlotEnv(RU_index=r, RU=RUs[r], slices=slices, num_urllc=num_urllc, H=H[r][0], gain_ru=gain_ru_ru[r],
                      dist_ru_ue=dist_ue_ru[r], T_slot=T_slot, T_max=T_max, NF=NF, w_reward=w_reward, 
                      eps=eps_DQN, max_steps=max_steps)
        slot_envs.append(env)

        agent = MultiHeadDQNAgent(
            state_dim=env.observation_space.shape[0],
            num_PRB=RUs[r].K,
            num_slices=len(slices),
            lr=learning_rate, gamma=gamma, batch_size=128
        )
        dqn_agents.append(agent)

    # Frame env and SAC agent
    frame_env = FrameEnv(slot_envs=slot_envs, slices=slices, num_urllc=num_urllc, H=H, gain_ru_ru=gain_ru_ru, 
                         T_max=T_max, w_reward=w_reward, 
                         sla_slices=sla_slices, frame_slots=frame_slots, allocation_mode=allocation_mode)
    sac_agent = SACAgent(num_RU=len(RUs), state_dim=frame_env.state_dim, num_slices=len(slices), 
                         action_scale=0.495, action_bias=0.505, device=device)

    return slot_envs, dqn_agents, frame_env, sac_agent


def alternating_training(RUs, slices, num_urllc, H, gain_ru_ru, dist_ue_ru,
                         T_slot, w_reward, T_max, NF, frame_slots, sla_slices,
                         max_steps, gamma, learning_rate, eps_DQN,
                         dqn_pretrain_episodes=1000, dqn_episode_steps=10,
                         sac_train_episodes=200, alt_rounds=1, device='cpu', allocation_mode = "equal"):
    """
    Hàm thực hiện train mô hình và lưu
    """
    slot_envs, dqn_agents, frame_env, sac_agent = build_pipeline(
        RUs, slices, num_urllc, H, gain_ru_ru, dist_ue_ru,
        T_slot, w_reward, T_max, NF, sla_slices, frame_slots,
        max_steps, gamma, learning_rate, eps_DQN,
        device, allocation_mode
    )

    # Alternating rounds
    logs = {"sac_rewards": [], "dqn_rewards": [], "actor_loss": [], "critic_loss": [], "dqn_loss": []}
    dqn_model_paths = []
    
    # (A) Train DQN per RU (fine-tune)
    for r, (env, agent) in enumerate(zip(slot_envs, dqn_agents)):
        print(f"-- Training DQN agents {r} --")
        reward_hist, loss_hist, dqn_model_path = train_dqn(env, agent, num_episodes=dqn_pretrain_episodes)
        logs["dqn_rewards"].append(reward_hist)
        logs["dqn_loss"].append(loss_hist)
        env.assign_dqn_agent(agent)
        dqn_model_paths.append(dqn_model_path)

    # (B) Train SAC on FrameEnv (SAC chooses slice quotas)
    print("-- Training SAC (frame-level) --")
    sac_rewards, a_losses, c_losses, sac_model_path = train_sac(frame_env, sac_agent, num_episodes=sac_train_episodes)
    logs["sac_rewards"] = sac_rewards
    logs["critic_loss"] = c_losses
    logs["actor_loss"] = a_losses

    print("Training complete.")
    return slot_envs, dqn_agents, frame_env, sac_agent, logs, sac_model_path, dqn_model_paths 


def evaluate_full_system(frame_env, sac_agent, num_frames=50, eval_mode=True):
    total_reward = 0.0
    total_thr = 0.0
    total_sla_embb = 0.0
    total_sla_urllc = 0.0
    total_utilPower = 0.0
    total_utilPRB = 0.0
    total_stab = 0.0

    state = frame_env.reset()
    for _ in range(num_frames):
        action = sac_agent.select_action(state, eval_mode=eval_mode)
        state, reward, done, info = frame_env.step(action)
        total_reward += reward
        total_thr += sum(info.get("eMBB_thr", 0.0))
        total_sla_embb += sum(info.get("SLA_embb", 0.0))
        total_sla_urllc += sum(info.get("SLA_urllc", 0.0))
        total_utilPower += sum(info.get("utilPower", 0.0))
        total_utilPRB += sum(info.get("utilPRB", 0.0))
        total_stab += sum(info.get("stability", 0.0))

    results = {
        "avg_reward": total_reward / (num_frames*frame_env.frame_slots),
        "avg_throughput": total_thr / (num_frames*frame_env.frame_slots),
        "avg_SLA_embb": total_sla_embb / (num_frames*frame_env.frame_slots),
        "avg_SLA_urllc": total_sla_urllc / (num_frames*frame_env.frame_slots),
        "avg_utilPower": total_utilPower / (num_frames*frame_env.frame_slots),
        "avg_utilPRB": total_utilPRB / (num_frames*frame_env.frame_slots),
        "avg_stability": total_stab / (num_frames*frame_env.frame_slots)
    }
    return results


def createModel(config, consta, trainCons, RUs, slices, H, gain_ru_ru, dist_ue_ru):
    """
    Tạo ra model SAC và DQN
    Trả về đường dẫn tới mô hình
    """
     # --- Huấn luyện alternating DQN <-> SAC ---
    print(" Bắt đầu huấn luyện pipeline SAC + DQN...")
    slot_envs, dqn_agents, frame_env, sac_agent, logs, sac_model_path, dqn_model_paths = alternating_training(
        RUs=RUs, slices=slices, num_urllc=config["num_URLLC"], H=H,
        gain_ru_ru=gain_ru_ru, dist_ue_ru=dist_ue_ru,
        T_slot=consta["T_slot"], w_reward=consta["w_reward"], T_max=consta["T_max_Mbps"], NF = consta["NF_dB"],
        frame_slots=consta["num_slot_per_frame"], sla_slices=consta["sla_slices"], max_steps=100,
        gamma=trainCons["forSAC"]["gamma"], learning_rate=trainCons["forDQN"]["learning_rate"], 
        eps_DQN=trainCons["forDQN"]["eps_DQN"],
        dqn_pretrain_episodes=trainCons["forDQN"]["dqn_pretrain_episodes"], 
        dqn_episode_steps=trainCons["forDQN"]["dqn_episode_steps"],
        sac_train_episodes=trainCons["forSAC"]["sac_train_episodes"], 
        alt_rounds=trainCons["alt_round"],
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # --- Đánh giá toàn hệ thống ---
    print("\n Đang đánh giá toàn hệ thống (Full-System Evaluation)...")
    eval_results = evaluate_full_system(frame_env, sac_agent, num_frames=30)
    print("\n Kết quả đánh giá:")
    for k, v in eval_results.items():
        print(f"{k}: {v}")

    # --- Vẽ biểu đồ ---
    print("\n Đang vẽ biểu đồ huấn luyện...")
    plot_SACtraining_curves(logs["sac_rewards"], len(RUs), len(slices), config["num_URLLC"])
    plot_SACactorlosstraining_curves(logs["actor_loss"], len(RUs), len(slices), config["num_URLLC"])
    plot_SACcriticlosstraining_curves(logs["critic_loss"], len(RUs), len(slices), config["num_URLLC"])
    for r in range(config["num_RUs"]):
        plot_DQNtraining_curves(logs["dqn_rewards"][r], r, len(slices), config["num_URLLC"])
        plot_DQNlosstraining_curves(logs["dqn_loss"][r], r, len(slices), config["num_URLLC"])

    return sac_model_path, dqn_model_paths




