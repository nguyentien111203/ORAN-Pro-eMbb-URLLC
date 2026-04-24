import time
import numpy as np
import torch
import numpy as np
import time
from combine.SAC.FrameEnv import FrameEnv
from combine.DQN.oran_env import SlotEnv
from combine.DQN.multiagent_DQN import MultiHeadDQNAgent
from combine.SAC.SACagent import SACAgent
from combine.SAC.loadSAC import load_sac_components
from combine.DQN.loadDQN import load_dqn_agent


def run_allocation_episode(sac_agent, dqn_agents, frame_env, num_frames=10, eval_mode=True):
    results = []
    total_thr, total_sla_embb, total_sla_urllc, total_jain = 0, 0, 0, 0
    total_decision_time = 0
    total_slots = 0

    state_frame = frame_env.reset()

    for frame_idx in range(num_frames):
        # ===== SAC quyết định quota công suất cho từng slice =====
        start_decision = time.time()
        sac_action = sac_agent.select_action(state_frame, eval_mode=eval_mode)
        decision_time = time.time() - start_decision
        total_decision_time += decision_time

        # ===== Cập nhật quota công suất vào frame_env và slot_envs =====
        state_next, reward, done, info = frame_env.step(sac_action)

        # ===== Lưu log và cộng dồn các chỉ số =====
        frame_metrics = {
            "frame": frame_idx,
            "decision_time_ms": decision_time * 1000,
            "eMBB_thr": info["eMBB_thr"],
            "SLA_embb": info["SLA_embb"],
            "SLA_urllc": info["SLA_urllc"],
            "Jain_Index": info["Jain Index"],
            "URLLC_served": info["URLLC_served"],
            "URLLC_drop": info["URLLC_drop"],
            "stability": info["stability"]
        }
        results.append(frame_metrics)

        total_thr += info["eMBB_thr"]
        total_sla_embb += info["SLA_embb"]
        total_sla_urllc += info["SLA_urllc"]
        total_jain += info["Jain Index"]
        total_slots += frame_env.frame_slots

        state_frame = state_next

    # ===== Tổng hợp kết quả =====
    metrics = {
        "avg_decision_time_ms": total_decision_time / num_frames * 1000,
        "avg_throughput": total_thr / num_frames,
        "avg_SLA_eMBB": total_sla_embb / num_frames,
        "avg_SLA_URLLC": total_sla_urllc / num_frames,
        "avg_Jain_Index": total_jain / num_frames
    }

    print("=== KẾT QUẢ TỔNG HỢP ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return results, metrics



def run_oran_allocation_pipeline(
    sac_model_path: str,
    dqn_model_paths: list,
    slices: list,
    RUs: list,
    H: np.ndarray,
    gain_ru_ru : np.ndarray,
    dist_ru_ue : np.ndarray,
    T_max: float,
    NF : float,
    w_reward: dict,
    sla_slices: dict,
    num_urllc: int,
    frame_slots: int = 10,
    num_frames: int = 10,
    device: str = "cpu",
    eval_mode: bool = True
):
    """
    Tự động khởi tạo SAC + DQN + môi trường ORAN và chạy phân bổ.

    Tham số:
    ----------
    sac_model_path : str
        Đường dẫn tới mô hình SAC đã huấn luyện (chỉ cần actor).
    dqn_model_paths : list[str]
        Danh sách đường dẫn tới các mô hình DQN cho từng RU.
    slices : list
        Danh sách các slice (đối tượng slice).
    RUs : list
        Danh sách các đối tượng RU.
    H : np.ndarray
        Ma trận kênh [num_RU, num_slices, frame_slots, num_PRB].
    T_max : float
        Throughput tối đa để chuẩn hóa reward.
    w_reward : dict
        Trọng số reward.
    sla_slices : dict
        Mục tiêu SLA cho eMBB và URLLC, ví dụ {"eMBB": 0.9, "URLLC": 0.99}.
    num_urllc : int
        Số lượng slice URLLC.
    frame_slots : int
        Số slot trong mỗi frame.
    num_frames : int
        Số frame chạy mô phỏng.
    device : str
        "cpu" hoặc "cuda".
    eval_mode : bool
        Nếu True thì chỉ inference, không training.

    Trả về:
    ---------
    metrics : dict
        Kết quả trung bình gồm throughput, SLA, Jain Index, thời gian quyết định.
    results : list[dict]
        Log chi tiết theo từng frame.
    """
    # ===== Tạo slot_env cho từng RU =====
    slot_envs = []
    for i, RU in enumerate(RUs):
        env = SlotEnv(
            RU_index=i,
            RU=RU,
            slices=slices,
            num_urllc=num_urllc,
            H=H[i][0],           # slot đầu tiên
            gain_ru=gain_ru_ru[i],
            dist_ru_ue=dist_ru_ue[i],
            T_slot=1,
            T_max=T_max,
            NF=NF,
            eps=0.1,
            max_steps=frame_slots,
            w_reward=w_reward
        )
        slot_envs.append(env)

    num_RU = len(RUs)
    num_slices = len(slices)

    # ===== Load mô hình SAC =====
    sac_agent = SACAgent(num_RU, state_dim=6 + num_slices + 2 * num_RU + 1, 
                         num_slices=num_slices, device=device)
    actor, critic_1, critic_2, alpha = load_sac_components(
                                            sac_model_path=sac_model_path,
                                            state_dim= 6 + num_slices + 2 * num_RU + 1,
                                            num_RU=num_RU,
                                            num_slices=num_slices,
                                            action_scale=1.0,
                                            action_bias=0.0,
                                            device="cpu"  # hoặc "cpu"
                                        )

    # ===== Load mô hình DQN cho từng RU =====
    dqn_agents = []
    for ru_id, env in enumerate(slot_envs):
        agent = MultiHeadDQNAgent(
            state_dim=env.state_dim,
            num_PRB=env.RU.K,
            num_slices=num_slices,
            device=device
        )
        policy_net, target_net = load_dqn_agent(
                                                model_path=dqn_model_paths[ru_id],
                                                state_dim= RUs[0].K + num_slices + 2,
                                                num_PRB=RUs[0].K,
                                                num_slices=num_slices,
                                                device="cpu"  # hoặc "cpu"
                                                )
        env.assign_dqn_agent(agent)
        dqn_agents.append(agent)

    # ===== Tạo FrameEnv =====
    frame_env = FrameEnv(
        slot_envs=slot_envs,
        slices=slices,
        num_urllc=num_urllc,
        H=H,
        gain_ru_ru=gain_ru_ru,
        T_max=T_max,
        w_reward=w_reward,
        sla_slices=sla_slices,
        frame_slots=frame_slots,
    )

    # ===== Chạy mô phỏng chính =====
    results = {
            "frame": [],
            "decision_time_ms": [],
            "throughput": [],
            "SLA_eMBB": [],
            "SLA_URLLC": [],
            "utilPower": [],
            "stability": [],
            "utilPRB": []
        }
    decisions = []
    state_frame = frame_env.reset()
    for frame_idx in range(num_frames):
        start_time = time.time()
        sac_action = sac_agent.select_action(state_frame, eval_mode=eval_mode)
        decision_time = time.time() - start_time

        next_state, reward, done, info = frame_env.step(sac_action)
        state_frame = next_state

        # Xử lý kết quả
        results["frame"].append(frame_idx)
        results["decision_time_ms"].append(decision_time)
        results["throughput"] += info["eMBB_thr"]
        results["SLA_eMBB"] += info["SLA_embb"]
        results["SLA_URLLC"] += info["SLA_urllc"]
        results["utilPower"] += info["utilPower"]
        results["stability"] += info["stability"]
        results["utilPRB"] += info["utilPRB"]

    pAlloc, xAlloc = frame_env.returnAlloc()

    # ===== Tổng hợp chỉ số trung bình =====
    metrics = {
        "avg_decision_time_ms": np.average(results["decision_time_ms"]),
        "avg_throughput": np.average(results["throughput"]),
        "avg_SLA_eMBB": np.average(results["SLA_eMBB"]),
        "avg_SLA_URLLC": np.average(results["SLA_URLLC"]),
        "avg_util_power": np.average(results["utilPower"]),
        "avg_stab": np.average(results["stability"]),
        "avg_util_PRB": np.average(results["utilPRB"])
    }

    print("\n=== KẾT QUẢ ĐÁNH GIÁ TRUNG BÌNH ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return metrics, results, pAlloc, xAlloc

