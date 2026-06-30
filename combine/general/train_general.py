import numpy as np
import torch
from tqdm import trange, tqdm

from combine.common.multiagent_DQN import MultiHeadDQNAgent
from combine.SAC.SACagent import SACAgent
from combine.SAC.FrameEnv import FrameEnv
from combine.SAC.train_SAC import train_sac
from combine.utils.pltSAC import plot_SACtraining_curves, plot_SACcriticlosstraining_curves, plot_SACactorlosstraining_curves
from combine.utils.plotDQN import plot_DQNtraining_curves, plot_DQNlosstraining_curves
from combine.general.DQN_general import RU_Env

def buildEnvAgent(RUs, arg1_slices, arg2_slices, H, inter_RU, inter_factor, N0, w_reward, cost_switch, cost_gb, scale_max, train_cons, frame_slots):
    
    is_embb_first = "embb" in arg1_slices[0].id.lower()
    if is_embb_first:
        embb_slices, urllc_slices = arg1_slices, arg2_slices
    else:
        urllc_slices, embb_slices = arg1_slices, arg2_slices
        
    num_urllc = len(urllc_slices)
    num_embb = len(embb_slices)
    num_slices_combined = num_urllc + num_embb
    
    num_urllc_ue = [len(s.ue_set) for s in urllc_slices]
    num_embb_ue = [len(s.ue_set) for s in embb_slices]
    num_ue_combined = num_embb_ue + num_urllc_ue 

    fixed_H = []
    for r in range(len(RUs)):
        H_r = H[r][0] 
        if is_embb_first:
            H_embb_r, H_urllc_r = H_r[:num_embb], H_r[num_embb:num_embb + num_urllc]
        else:
            H_urllc_r, H_embb_r = H_r[:num_urllc], H_r[num_urllc:num_urllc + num_embb]
            
        H_combined = list(H_embb_r) + list(H_urllc_r)
        fixed_H.append([H_combined])

    ru_envs = []
    ru_dqn_agents = []

    for r in range(len(RUs)):
        ru_env = RU_Env(
            RUs[r], embb_slices, urllc_slices, fixed_H[r][0], 
            inter_RU, inter_factor, N0, w_reward, cost_switch, 
            cost_gb, scale_max, train_cons["forDQN"], frame_slots
        )
        ru_envs.append(ru_env)

        dqn_agent = MultiHeadDQNAgent(
            ru_env.state_dim, num_slices_combined, num_ue_combined, 
            len(RUs[r].bwps), train_cons["forDQN"]
        )
        ru_env.assign_agent(dqn_agent)
        ru_dqn_agents.append(dqn_agent)

    num_bwp_ru = [len(RUs[r].bwps) for r in range(len(RUs))]
    
    frame_env = FrameEnv(RUs, ru_envs, urllc_slices, embb_slices, fixed_H, w_reward, scale_max, frame_slots)
    
    sac_agent = SACAgent(
        5 + 4 * (len(urllc_slices) + len(embb_slices)), len(RUs), 
        num_bwp_ru, len(urllc_slices) + len(embb_slices), train_cons["forSAC"]
    )

    return ru_envs, ru_dqn_agents, frame_env, sac_agent


def alternating_training(num_rus, ru_envs, ru_dqn_agents, frame_env, sac_agent, numepDQN, numepSAC):
    """
    Train DQN trước (slot-level, độc lập per-RU), sau đó train SAC (frame-level).

    Với RU_Env mới, step() cần 4 tham số (totalLatRate, totalThrRate, latSoft, thrSoft)
    thay vì (eMBB_Thr, numBit_urllc) như cũ. Các giá trị này được tự tính ngay sau
    computeOutput() dựa trên yêu cầu QoS của từng UE (min_thr cho eMBB, max_lat cho URLLC),
    không cần phụ thuộc vào FrameEnv — phù hợp để train DQN độc lập trước SAC.
    """
    # Budget mặc định: chia đều PRB cho các slice (vì SAC chưa train ở giai đoạn này)
    BWP_slice = [[[frame_env.RUs[r].bwps[b].num_prb / frame_env.num_slices 
                   for b in range(len(frame_env.RUs[r].bwps))] 
                  for _ in range(frame_env.num_slices)] for r in range(num_rus)]

    print(f"-- Đang tiến hành huấn luyện Unified DQN Agents (Slot-level) --")
    all_dqn_rewards = [[] for _ in range(num_rus)]
    all_dqn_losses  = [[] for _ in range(num_rus)]

    for ep in trange(numepDQN, desc="Unified DQN Training Loop"):
        for r in range(num_rus):
            env   = ru_envs[r]
            agent = ru_dqn_agents[r]

            # RU_Env.reset() mới không trả về gì, state ban đầu = vector 0
            env.reset()
            state = np.zeros(env.state_dim, dtype=np.float32)

            ep_reward = 0.0
            ep_loss   = 0.0
            steps     = 0

            # Dùng for cố định số slot trong 1 frame thay vì while not done,
            # vì index_subframe bị reset về 0 ngay trong step() (chủ ý của anh Tiến,
            # dùng để check sang frame mới, không phải điều kiện dừng episode)
            for slot in range(env.frame_slots):
                action = agent.select_action(state, BWP_slice[r])

                flatBit, flatThr = env.computeOutput(action)

                # ----- Tự tính 4 tham số cho RU_Env.step() (độc lập, không cần FrameEnv) -----
                # totalLatRate / latSoft: tỷ lệ latency thực tế so với ngưỡng max_lat (URLLC)
                lat_rates = []
                for s in range(env.num_urllc):
                    for u in range(env.num_urllc_ue[s]):
                        max_lat = getattr(env.urllc_slices[s].ue_set[u], 'max_lat', 1.0)
                        pkt_size = getattr(env.urllc_slices[s].ue_set[u], 'packet_size', 100)
                        idx = sum(env.num_urllc_ue[:s]) + u
                        numBit = flatBit[idx] if idx < len(flatBit) else 1e-9
                        latency = pkt_size / (numBit + 1e-9)
                        lat_rates.append(latency / (max_lat + 1e-9))
                lat_rates = np.array(lat_rates, dtype=np.float32)
                lat_soft = np.clip(lat_rates, 0.0, 1.0)

                # totalThrRate / thrSoft: tỷ lệ throughput thực tế so với min_thr (eMBB)
                thr_rates = []
                for s in range(env.num_embb):
                    for u in range(env.num_embb_ue[s]):
                        min_thr = getattr(env.embb_slices[s].ue_set[u], 'min_thr', 1.0)
                        idx = sum(env.num_embb_ue[:s]) + u
                        thr = flatThr[idx] if idx < len(flatThr) else 0.0
                        thr_rates.append(thr / (min_thr + 1e-9))
                thr_rates = np.array(thr_rates, dtype=np.float32)
                thr_soft = np.clip(thr_rates, 0.0, 1.0)

                next_state, reward, _, info = env.step(lat_rates, thr_rates, lat_soft, thr_soft)
                done = (slot == env.frame_slots - 1)

                agent.store_transition(state, action, reward, next_state, done)
                loss = agent.optimize_model()

                if loss is not None:
                    ep_loss += loss
                    steps   += 1

                ep_reward += reward
                state = next_state

            agent.eps = max(agent.eps_end, agent.eps * agent.eps_decay)
            all_dqn_rewards[r].append(ep_reward)
            all_dqn_losses[r].append(ep_loss / max(steps, 1))

    for r in range(num_rus):
        plot_DQNtraining_curves(all_dqn_rewards[r], r, frame_env.num_slices, frame_env.num_urllc)
        plot_DQNlosstraining_curves(all_dqn_losses[r], r, frame_env.num_slices, frame_env.num_urllc)

    print("-- Đang tiến hành huấn luyện SAC (Frame-level) --")
    avg_rewards, actor_losses, critic_losses, sac_model_path = train_sac(frame_env, sac_agent, numepSAC)
    
    plot_SACtraining_curves(avg_rewards, num_rus, frame_env.num_slices, frame_env.num_urllc)
    plot_SACactorlosstraining_curves(actor_losses, num_rus, frame_env.num_slices, frame_env.num_urllc)
    plot_SACcriticlosstraining_curves(critic_losses, num_rus, frame_env.num_slices, frame_env.num_urllc)
    
    print("Training completed")