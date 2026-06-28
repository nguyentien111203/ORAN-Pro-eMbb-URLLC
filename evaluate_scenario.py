"""
evaluate_scenario.py
--------------------
Đánh giá một kịch bản mạng bằng 2 bộ model SAC-DQN (framework vs benchmark)
và trả về các KPI theo từng slot.

Cách dùng:
    from evaluate_scenario import evaluate_scenario
    results_main, results_bm = evaluate_scenario(
        RUs, embb_slices, urllc_slices,
        frame_env_main, frame_env_bm,
        sac_agent, sac_agent2,
        num_frames=50,
        consta=consta,
        plot=True
    )
"""

import numpy as np
import matplotlib.pyplot as plt
from input.genInput import generate_h_matrix


# ==============================================================================
# HÀM CHÍNH
# ==============================================================================

def evaluate_scenario(
    RUs,
    embb_slices,
    urllc_slices,
    frame_env_main,     # FrameEnv gán sac_agent + dqn_agents của framework
    frame_env_bm,       # FrameEnv gán sac_agent2 + dqn_agents2 của benchmark
    sac_agent,          # SAC agent framework (đã train)
    sac_agent2,         # SAC agent benchmark (đã train)
    num_frames,         # Số frame cần mô phỏng
    consta,             # Dict hằng số hệ thống (chứa frame_slots, ...)
    plot=True,
    figure_dir="./Figures/evaluate"
):
    """
    Đánh giá kịch bản mạng, trả về KPI theo từng slot cho cả 2 bộ model.

    Đầu ra
    ------
    results_main : dict  — KPI của model framework
    results_bm   : dict  — KPI của model benchmark
    Mỗi dict có cấu trúc:
        {
            "throughput"        : [],  # Tổng throughput hệ thống tại mỗi slot
            "latency"           : [],  # Latency URLLC tại mỗi slot (trung bình các UE)
            "resource_efficiency": [], # Tổng quota SAC / tổng PRB có sẵn tại mỗi frame
            "energy_cost"       : [],  # Tổng energy cost tại mỗi slot
            "fragment_cost"     : [],  # Tổng fragment cost tại mỗi slot
            "switch_cost"       : [],  # Tổng switch cost tại mỗi slot
            "guardband_cost"    : [],  # Tổng guardband cost tại mỗi slot
            "slice_budget"      : [],  # Tổng PRB SAC dùng / tổng PRB có sẵn tại mỗi frame
        }
    """
    import os
    os.makedirs(figure_dir, exist_ok=True)

    frame_slots   = consta.get("num_slot_per_frame", 10)
    num_embb      = len(embb_slices)
    num_urllc     = len(urllc_slices)
    num_embb_ue   = [len(s.ue_set) for s in embb_slices]
    num_urllc_ue  = [len(s.ue_set) for s in urllc_slices]

    # Tổng PRB có sẵn toàn hệ thống (dùng để tính slice_budget)
    total_prb_system = sum(
        bwp.num_prb
        for r in range(len(RUs))
        for bwp in RUs[r].bwps
    )

    results_main = _empty_results()
    results_bm   = _empty_results()

    # last_action để action smoothing hoạt động đúng như lúc training
    last_action_main = None
    last_action_bm   = None

    for frame_idx in range(num_frames):

        # ------------------------------------------------------------------
        # Bước 1: Sinh ma trận H mới cho frame này
        # ------------------------------------------------------------------
        H = generate_h_matrix(
            len(RUs), frame_slots,
            num_embb + num_urllc,
            num_urllc_ue, num_embb_ue
        )

        # Cập nhật H vào cả 2 FrameEnv
        _update_frame_env_H(frame_env_main, H, RUs, num_embb, num_urllc)
        _update_frame_env_H(frame_env_bm,   H, RUs, num_embb, num_urllc)

        # ------------------------------------------------------------------
        # Bước 2: Reset môi trường, lấy state ban đầu cho SAC
        # ------------------------------------------------------------------
        state_main = frame_env_main.reset()
        state_bm   = frame_env_bm.reset()

        # ------------------------------------------------------------------
        # Bước 3: SAC sinh action (budget) cho cả frame
        # last_action=None ở frame đầu, sau đó truyền action frame trước
        # để action smoothing hoạt động đúng như lúc training
        # ------------------------------------------------------------------
        action_main = sac_agent.select_action(state_main, last_action=last_action_main)
        action_bm   = sac_agent2.select_action(state_bm,  last_action=last_action_bm)

        # Tính slice_budget từ quota SAC
        budget_main = _calc_slice_budget(action_main, RUs, num_embb + num_urllc, total_prb_system)
        budget_bm   = _calc_slice_budget(action_bm,   RUs, num_embb + num_urllc, total_prb_system)

        results_main["slice_budget"].append(budget_main)
        results_bm["slice_budget"].append(budget_bm)

        # resource_efficiency = slice_budget (quota SAC / total PRB)
        results_main["resource_efficiency"].append(budget_main)
        results_bm["resource_efficiency"].append(budget_bm)

        # Cập nhật last_action cho frame tiếp theo (action smoothing)
        last_action_main = action_main
        last_action_bm   = action_bm

        # ------------------------------------------------------------------
        # Bước 4: Chạy từng slot trong frame, thu thập KPI
        # ------------------------------------------------------------------
        _run_frame(
            frame_env_main, action_main,
            frame_slots, num_embb, num_urllc,
            results_main
        )
        _run_frame(
            frame_env_bm, action_bm,
            frame_slots, num_embb, num_urllc,
            results_bm
        )

    # ------------------------------------------------------------------
    # Bước 5 (tuỳ chọn): Vẽ biểu đồ
    # ------------------------------------------------------------------
    if plot:
        _plot_results(results_main, results_bm, figure_dir)

    return results_main, results_bm


# ==============================================================================
# HÀM PHỤ TRỢ NỘI BỘ
# ==============================================================================

def _empty_results():
    return {
        "throughput"         : [],
        "latency"            : [],
        "resource_efficiency": [],
        "energy_cost"        : [],
        "fragment_cost"      : [],
        "switch_cost"        : [],
        "guardband_cost"     : [],
        "slice_budget"       : [],
    }


def _update_frame_env_H(frame_env, H, RUs, num_embb, num_urllc):
    """
    Cập nhật H mới vào FrameEnv và tất cả RU_Env bên trong.
    H đầu vào từ generate_h_matrix: H[r][slot][slice] với thứ tự urllc trước, embb sau.
    RU_Env cần: H[slice][ue] với thứ tự embb trước, urllc sau (giống buildEnvAgent).
    """
    num_rus = len(RUs)
    fixed_H = []
    for r in range(num_rus):
        H_r = H[r][0]  # slot 0, shape: [num_slices][num_ue_in_slice]
        # generate_h_matrix trả về urllc trước, embb sau
        H_urllc_r = H_r[:num_urllc]
        H_embb_r  = H_r[num_urllc:num_urllc + num_embb]
        # RU_Env cần embb trước, urllc sau (giống buildEnvAgent)
        H_combined = list(H_embb_r) + list(H_urllc_r)
        fixed_H.append([H_combined])

    frame_env.H = fixed_H
    for r, ru_env in enumerate(frame_env.RU_envs):
        ru_env.update_H(fixed_H[r][0])


def _calc_slice_budget(action, RUs, num_slices, total_prb_system):
    """
    Tính tỷ lệ PRB mà SAC phân bổ so với tổng PRB hệ thống.
    action là vector [0,1] flat của SAC, mỗi phần tử nhân với num_prb của BWP tương ứng.
    """
    action = np.array(action).flatten()
    idx = 0
    total_quota = 0
    for r in range(len(RUs)):
        for s in range(num_slices):
            for b, bwp in enumerate(RUs[r].bwps):
                total_quota += action[idx] * bwp.num_prb
                idx += 1
    return total_quota / (total_prb_system + 1e-9)


def _run_frame(frame_env, sac_action, frame_slots, num_embb, num_urllc, results):
    """
    Chạy 1 frame theo từng slot, thu thập KPI vào results.
    Tái dụng logic step() của FrameEnv nhưng log chi tiết theo slot.
    """
    # Parse SAC action thành quota cho từng RU
    action = np.array(sac_action).flatten()
    idx = 0
    sac_quotas = []
    for r in range(len(frame_env.RUs)):
        num_bwps = len(frame_env.RUs[r].bwps)
        num_slices = frame_env.num_slices
        ru_quota = [[0 for _ in range(num_bwps)] for _ in range(num_slices)]
        for s in range(num_slices):
            for b in range(num_bwps):
                total_prbs = frame_env.RUs[r].bwps[b].num_prb
                ru_quota[s][b] = int(action[idx] * total_prbs)
                idx += 1
        sac_quotas.append(ru_quota)

    # Lặp từng slot
    for slot_index in range(frame_slots):
        slot_throughput  = 0.0
        slot_latency     = []
        slot_energy      = 0.0
        slot_fragment    = 0.0
        slot_switch      = 0.0
        slot_guardband   = 0.0

        for r, env in enumerate(frame_env.RU_envs):
            env.update_H(frame_env.H[r][0])

            dqn_action             = env.select_action(env.state, sac_quotas[r])
            eMBB_Thr, numBit_urllc = env.computeOutput(dqn_action)
            _, _, _, info          = env.step(eMBB_Thr, numBit_urllc)

            # --- Throughput: tổng throughput eMBB tại slot này (tất cả RU) ---
            for s in range(num_embb):
                for e in range(len(frame_env.embb_slices[s].ue_set)):
                    slot_throughput += info["thr"][s][e]

            # --- Latency: latency URLLC tại slot này (tất cả RU, tất cả UE) ---
            for s in range(num_urllc):
                for u in range(len(frame_env.urllc_slices[s].ue_set)):
                    slot_latency.append(info["lat"][s][u])

            # --- Cost: cộng đúng 1 lần mỗi RU/slot (không lặp theo UE) ---
            slot_energy    += info["costE"]
            slot_fragment  += info["costF"]
            slot_switch    += info["costS"]
            slot_guardband += info["costGB"]

        # Ghi nhận KPI của slot này
        results["throughput"].append(slot_throughput)
        results["latency"].append(np.mean(slot_latency) if slot_latency else 0.0)
        results["energy_cost"].append(slot_energy)
        results["fragment_cost"].append(slot_fragment)
        results["switch_cost"].append(slot_switch)
        results["guardband_cost"].append(slot_guardband)


# ==============================================================================
# VẼ BIỂU ĐỒ
# ==============================================================================

def _plot_results(results_main, results_bm, figure_dir):
    """
    Vẽ biểu đồ đường cho các KPI (trừ latency dùng CDF).
    Mỗi KPI tạo 2 file riêng: <key>.png (framework) và <key>-bm.png (benchmark).
    """
    line_metrics = [
        ("throughput",          "Throughput (bps)",    "Throughput theo slot"),
        ("resource_efficiency", "Resource Efficiency", "Resource Efficiency theo frame"),
        ("energy_cost",         "Energy Cost",         "Energy Cost theo slot"),
        ("fragment_cost",       "Fragment Cost",        "Fragment Cost theo slot"),
        ("switch_cost",         "Switch Cost",          "Switch Cost theo slot"),
        ("guardband_cost",      "Guardband Cost",       "Guardband Cost theo slot"),
        ("slice_budget",        "Slice Budget Ratio",   "Slice Budget theo frame"),
    ]

    xlabel = {
        "resource_efficiency": "Frame",
        "slice_budget":        "Frame",
    }

    for key, ylabel, title in line_metrics:
        x_label = xlabel.get(key, "Slot")

        # --- Framework ---
        plt.figure(figsize=(10, 4))
        plt.plot(results_main[key], color="steelblue", linewidth=1.5)
        plt.xlabel(x_label)
        plt.ylabel(ylabel)
        plt.title(f"{title} — Framework")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{figure_dir}/{key}.png", dpi=150)
        plt.close()
        print(f"[PLOT] {figure_dir}/{key}.png")

        # --- Benchmark ---
        plt.figure(figsize=(10, 4))
        plt.plot(results_bm[key], color="tomato", linewidth=1.5)
        plt.xlabel(x_label)
        plt.ylabel(ylabel)
        plt.title(f"{title} — Benchmark")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{figure_dir}/{key}-bm.png", dpi=150)
        plt.close()
        print(f"[PLOT] {figure_dir}/{key}-bm.png")

    # --- CDF Latency: Framework ---
    plt.figure(figsize=(8, 5))
    data = np.sort(results_main["latency"])
    cdf  = np.arange(1, len(data) + 1) / len(data)
    plt.plot(data, cdf, color="steelblue", linewidth=1.5)
    plt.xlabel("Latency (s)")
    plt.ylabel("CDF")
    plt.title("CDF Latency URLLC — Framework")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{figure_dir}/latency.png", dpi=150)
    plt.close()
    print(f"[PLOT] {figure_dir}/latency.png")

    # --- CDF Latency: Benchmark ---
    plt.figure(figsize=(8, 5))
    data = np.sort(results_bm["latency"])
    cdf  = np.arange(1, len(data) + 1) / len(data)
    plt.plot(data, cdf, color="tomato", linewidth=1.5)
    plt.xlabel("Latency (s)")
    plt.ylabel("CDF")
    plt.title("CDF Latency URLLC — Benchmark")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{figure_dir}/latency-bm.png", dpi=150)
    plt.close()
    print(f"[PLOT] {figure_dir}/latency-bm.png")


# ==============================================================================
# MAIN — chạy trực tiếp: python3 evaluate_scenario.py
# ==============================================================================

if __name__ == "__main__":
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import logging
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.serif": []})

    from input.takeInput import load_cons_from_json
    from input.genInput import generate_pipeline_inputs, calculateScaleMax
    from combine.general.train_general import buildEnvAgent
    from combine.SAC.SACagent      import SACAgent      as SACAgent_main
    from combine.SAC.FrameEnv      import FrameEnv      as FrameEnv_main
    from combine.SAC_benchmark.SACagent import SACAgent as SACAgent_bm
    from combine.SAC_benchmark.FrameEnv import FrameEnv as FrameEnv_bm

    print("=== Evaluate Scenario ===\n")

    # 1. Load config
    consta    = load_cons_from_json("./config/cons.json")
    trainCons = load_cons_from_json("./config/trainCons.json")

    # 2. Tạo RUs, slices
    RUs, embb_slices, urllc_slices, num_urllc_ue, num_embb_ue = generate_pipeline_inputs(
        "./config/ru.yaml", "./config/slice.yaml", "./config/ue.yaml", consta
    )
    scale_max = calculateScaleMax(
        RUs, embb_slices, urllc_slices,
        consta["cost_switch"], consta["cost_gb"]
    )

    # 3. Sinh H ban đầu (sẽ được tái sinh mỗi frame bên trong evaluate_scenario)
    H = generate_h_matrix(
        len(RUs), consta["frame_slots"],
        len(embb_slices) + len(urllc_slices),
        num_urllc_ue, num_embb_ue
    )

    # 4. Tạo FrameEnv + agent cho framework (SAC chính)
    ru_envs_main, _, frame_env_main_base, _ = buildEnvAgent(
        RUs, urllc_slices, embb_slices, H,
        consta["inter_RU"], consta["inter_factor"],
        consta["N0_mW_per_MHz"], consta["w_reward"],
        consta["cost_switch"], consta["cost_gb"],
        scale_max, trainCons, consta["frame_slots"]
    )
    num_bwp_ru = [len(RUs[r].bwps) for r in range(len(RUs))]
    frame_env_main = FrameEnv_main(
        RUs, ru_envs_main, urllc_slices, embb_slices,
        frame_env_main_base.H, consta["w_reward"], scale_max, consta["frame_slots"]
    )
    sac_agent = SACAgent_main(
        4 + len(urllc_slices) + len(embb_slices),
        len(RUs), num_bwp_ru,
        len(urllc_slices) + len(embb_slices),
        trainCons["forSAC"]
    )
    # Load model đã train cho framework
    ck_main = torch.load("./sac_model.pth", map_location="cpu")
    sac_agent.actor.load_state_dict(ck_main["actor"])
    sac_agent.critic_1.load_state_dict(ck_main["critic_1"])
    sac_agent.critic_2.load_state_dict(ck_main["critic_2"])
    sac_agent.actor.eval()
    print("[INFO] Đã load sac_model.pth cho framework")

    # 5. Tạo FrameEnv + agent cho benchmark (SAC_benchmark)
    ru_envs_bm, _, frame_env_bm_base, _ = buildEnvAgent(
        RUs, urllc_slices, embb_slices, H,
        consta["inter_RU"], consta["inter_factor"],
        consta["N0_mW_per_MHz"], consta["w_reward"],
        consta["cost_switch"], consta["cost_gb"],
        scale_max, trainCons, consta["frame_slots"]
    )
    frame_env_bm = FrameEnv_bm(
        RUs, ru_envs_bm, urllc_slices, embb_slices,
        frame_env_bm_base.H, consta["w_reward"], scale_max, consta["frame_slots"]
    )
    sac_agent2 = SACAgent_bm(
        4 + len(urllc_slices) + len(embb_slices),
        len(RUs), num_bwp_ru,
        len(urllc_slices) + len(embb_slices),
        trainCons["forSAC"]
    )
    # Load model đã train cho benchmark (dùng chung file nếu chưa có file riêng)
    ck_bm = torch.load("./sac_model.pth", map_location="cpu")
    sac_agent2.actor.load_state_dict(ck_bm["actor"])
    sac_agent2.critic_1.load_state_dict(ck_bm["critic_1"])
    sac_agent2.critic_2.load_state_dict(ck_bm["critic_2"])
    sac_agent2.actor.eval()
    print("[INFO] Đã load sac_model.pth cho benchmark")

    # 6. Chạy đánh giá
    results_main, results_bm = evaluate_scenario(
        RUs, embb_slices, urllc_slices,
        frame_env_main, frame_env_bm,
        sac_agent, sac_agent2,
        num_frames=50,
        consta=consta,
        plot=True,
        figure_dir="./Figures/evaluate"
    )

    print(f"\n=== Hoàn thành! ===")
    print(f"  Throughput trung bình : {np.mean(results_main['throughput']):.4f}")
    print(f"  Latency trung bình    : {np.mean(results_main['latency']):.4f}")
    print(f"  Plots: ./Figures/evaluate/")