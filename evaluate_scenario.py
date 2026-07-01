"""
evaluate_scenario.py
--------------------
Đánh giá một kịch bản mạng bằng 2 bộ model SAC-DQN (framework vs benchmark)
và trả về các KPI theo từng slot.

H của mỗi frame được sinh trực tiếp từ kịch bản di động (mobility) định
nghĩa trong scenario.py: vị trí UE được khởi tạo 1 lần (generate_scenario)
rồi cập nhật mỗi frame (update_position), channel gain được tính lại từ
vị trí hiện tại (calculate_total_channel_gain). Không còn dùng
generate_h_matrix (random độc lập, không có mobility thật).

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

import copy
import numpy as np
import matplotlib.pyplot as plt
from scenario.scenario import (generate_scenario, update_position, calculate_3gpp_pathloss, 
                               calculate_total_channel_gain, load_and_merge_config)


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
    figure_dir="./Figures/evaluate",
    scenario_type="stable",   # "stable" | "low" | "high" (xem mobility trong scenario.py)
    scenario_seed=None,       # seed cho lần khởi tạo vị trí UE đầu tiên
    scenario_config=None,     # config tuỳ chỉnh cho scenario.py; None -> load_and_merge_config
    dt=None,                  # khoảng thời gian giữa 2 frame (s), dùng để update_position
):
    """
    Đánh giá kịch bản mạng, trả về KPI theo từng slot cho cả 2 bộ model.

    H được sinh từ kịch bản di động dùng chung cho cả framework và benchmark
    (cùng vị trí UE tại mỗi frame), nên kết quả so sánh không bị nhiễu bởi
    việc 2 bên thấy 2 kênh truyền khác nhau.

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

    # ------------------------------------------------------------------
    # Khởi tạo kịch bản di động dùng chung cho toàn bộ evaluate.
    # topology phải khớp với số RU thật và tổng số UE thật (embb + urllc),
    # vì ue_set của từng slice là index toàn cục trỏ vào vị trí UE này
    # (UE được sinh tuần tự theo sector trong generate_topology()).
    # ------------------------------------------------------------------
    num_ues_total = sum(num_embb_ue) + sum(num_urllc_ue)

    scn_config = copy.deepcopy(
        scenario_config or load_and_merge_config(consta.get("scenario_config_path", "config.yaml"))
    )
    scn_config["topology"]["num_rus"] = len(RUs)
    scn_config["topology"]["num_ues"] = num_ues_total

    if num_ues_total % len(RUs) != 0:
        raise ValueError(
            f"Tổng số UE ({num_ues_total}) phải chia hết cho số RU ({len(RUs)}) "
            "để generate_topology() trong scenario.py phân bố đều UE theo sector."
        )

    mobility_scenario = generate_scenario(scenario_type, seed=scenario_seed, config=scn_config)

    # dt mặc định: thời lượng 1 frame (frame_slots * slot_duration), nếu không
    # truyền vào trực tiếp hoặc khai báo trong consta.
    frame_dt = dt if dt is not None else consta.get(
        "frame_duration_s", frame_slots * consta.get("slot_duration_s", 1e-3)
    )

    for frame_idx in range(num_frames):

        # ------------------------------------------------------------------
        # Bước 1: Cập nhật vị trí UE theo mobility (frame đầu dùng vị trí khởi
        # tạo, từ frame thứ 2 trở đi UE di chuyển theo velocity/bounds).
        # ------------------------------------------------------------------
        if frame_idx > 0:
            mobility_scenario = update_position(mobility_scenario, frame_dt)

        H = _build_h_from_scenario(
            mobility_scenario, RUs, embb_slices, urllc_slices,
            fc_ghz=scn_config["channel"]["fc_ghz"],
            antenna_gain_dbi=scn_config["channel"]["antenna_gain_dbi"],
            shadow_std_db=scn_config["channel"]["shadow_std_db"],
        )

        # Cập nhật H vào cả 2 FrameEnv (cùng kịch bản di động -> so sánh công bằng)
        _update_frame_env_H(frame_env_main, H)
        _update_frame_env_H(frame_env_bm,   H)

        # ------------------------------------------------------------------
        # Bước 2: Reset môi trường, lấy state ban đầu cho SAC
        # FrameEnv.reset() mới không return state, nên tự tạo vector 0
        # ------------------------------------------------------------------
        frame_env_main.reset()
        frame_env_bm.reset()
        state_main = np.zeros(frame_env_main.state_dim, dtype=np.float32)
        state_bm   = np.zeros(frame_env_bm.state_dim, dtype=np.float32)

        # ------------------------------------------------------------------
        # Bước 3: SAC sinh action (budget) cho cả frame
        # framework (SACAgent mới) không còn dùng last_action,
        # benchmark (SACAgentBM) vẫn cần last_action cho action smoothing
        # ------------------------------------------------------------------
        action_main = sac_agent.select_action(state_main)
        action_bm   = sac_agent2.select_action(state_bm)

        # Tính slice_budget từ quota SAC
        budget_main = _calc_slice_budget(action_main, RUs, num_embb + num_urllc, total_prb_system)
        budget_bm   = _calc_slice_budget(action_bm,   RUs, num_embb + num_urllc, total_prb_system)

        results_main["slice_budget"].append(budget_main)
        results_bm["slice_budget"].append(budget_bm)

        # resource_efficiency = slice_budget (quota SAC / total PRB)
        results_main["resource_efficiency"].append(budget_main)
        results_bm["resource_efficiency"].append(budget_bm)


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


def _build_h_from_scenario(scenario, RUs, embb_slices, urllc_slices,
                            fc_ghz, antenna_gain_dbi, shadow_std_db):
    """
    Tính H[r][slot=0][slice][ue] trực tiếp từ vị trí UE hiện tại của
    `scenario` (sinh/cập nhật bởi scenario.py), thay cho generate_h_matrix
    random trước đây.

    Lưu ý:
    - Trong scenario.py, mọi RU được coi là đồng vị trí tại gốc tọa độ
      (ru_pos = [0,0]), nên distance / path loss giống nhau cho mọi RU.
      Tuy nhiên shadowing + fast fading được random ĐỘC LẬP cho từng RU
      (mỗi RU là 1 chain anten riêng dù đồng vị trí vật lý).
    - slice.ue_set chứa index UE TOÀN CỤC, khớp đúng thứ tự UE mà
      generate_topology() sinh ra (UE được gán tuần tự theo từng RU sector).
    - H trả về theo đúng format RU_Env cần: slice embb trước, urllc sau,
      mỗi giá trị là channel gain dạng linear (không phải dB).
    """
    num_ues = scenario["ue_pos"].shape[0]

    # Vị trí có thể đã đổi do update_position() -> phải tính lại distance/path loss
    distance = np.linalg.norm(scenario["ue_pos"], axis=1)
    scenario["distance"] = distance
    path_loss_db = calculate_3gpp_pathloss(distance, fc_ghz)

    H = []
    for _ in range(len(RUs)):
        gain_info = calculate_total_channel_gain(
            num_ues, path_loss_db,
            antenna_gain_dbi=antenna_gain_dbi,
            shadow_std_db=shadow_std_db,
        )
        gain_linear = gain_info["gain_linear"]

        # RU_Env cần thứ tự: embb trước, urllc sau (giống buildEnvAgent)
        H_r_slot0 = []
        for s in embb_slices:
            H_r_slot0.append([gain_linear[ue_idx] for ue_idx in range(len(s.ue_set))])
        for s in urllc_slices:
            H_r_slot0.append([gain_linear[ue_idx] for ue_idx in range(len(s.ue_set))])

        # Chỉ có 1 "slot" H dùng chung cho cả frame (giữ nguyên hành vi gốc)
        H.append([H_r_slot0])

    return H


def _update_frame_env_H(frame_env, H):
    """
    Cập nhật H (đã đúng format RU_Env cần: embb trước, urllc sau) vào
    FrameEnv và tất cả RU_Env bên trong.
    """
    frame_env.H = H
    for r, ru_env in enumerate(frame_env.RU_envs):
        ru_env.update_H(H[r][0])


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
    # Parse SAC action thành quota cho từng RU (cấu trúc [r][s][b])
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

    # Tham số softening (giống train_dqn)
    beta  = 1.5
    alpha = 3

    # Các hằng số QoS lấy từ RU_Env đầu tiên (dùng chung cho mọi RU)
    env0 = frame_env.RU_envs[0]
    pac = np.array([
        ue.pac
        for s in range(num_urllc)
        for ue in env0.urllc_slices[s].ue_set
    ], dtype=np.float64)

    lat_target = np.array([
        ue.lat
        for s in range(num_urllc)
        for ue in env0.urllc_slices[s].ue_set
    ], dtype=np.float64)

    thr_min = np.array([
        ue.thr
        for s in range(num_embb)
        for ue in env0.embb_slices[s].ue_set
    ], dtype=np.float64)

    # State khởi tạo cho từng RU
    states = [np.zeros(env0.state_dim, dtype=np.float32) for _ in range(len(frame_env.RU_envs))]

    # Lặp từng slot
    for slot_index in range(frame_slots):
        slot_throughput = 0.0
        slot_latency    = []
        slot_energy     = 0.0
        slot_fragment   = 0.0
        slot_switch     = 0.0
        slot_guardband  = 0.0

        # Tổng hợp numBits và totalThr từ tất cả RU (giống train_dqn)
        numBits  = np.array([1e-7 for s in range(num_urllc)
                             for _ in env0.urllc_slices[s].ue_set], dtype=np.float64)
        totalThr = np.zeros_like(thr_min, dtype=np.float64)

        for r, env in enumerate(frame_env.RU_envs):
            env.update_H(frame_env.H[r][0])
            dqn_action      = env.select_action(states[r], sac_quotas[r])
            ruBits, ruThr   = env.computeOutput(dqn_action)
            numBits  += ruBits
            totalThr += ruThr

        # Tính rate và softening (giống train_dqn)
        urllc_lat  = pac / (numBits + 1e-8)
        urllc_rate = urllc_lat / (lat_target + 1e-8)
        embb_rate  = totalThr / (thr_min + 1e-8)

        lat_soft = [1 / (1 + alpha * max(urllc_rate[u] - 1, 0)) for u in range(len(urllc_rate))]
        thr_soft = [float(np.tanh(beta * embb_rate[u])) for u in range(len(embb_rate))]

        # Gọi step đúng 4 tham số, thu thập info từ từng RU
        for r, env in enumerate(frame_env.RU_envs):
            next_state, _, _, info = env.step(urllc_rate, embb_rate, lat_soft, thr_soft)
            states[r] = next_state

            # --- Cost: cộng đúng 1 lần mỗi RU/slot ---
            slot_energy    += info["costE"]
            slot_fragment  += info["costF"]
            slot_switch    += info["costS"]
            slot_guardband += info["costGB"]

        # --- Throughput: tổng eMBB throughput toàn slot ---
        slot_throughput = float(np.sum(totalThr))

        # --- Latency: latency thô URLLC (giây) cho từng UE ---
        slot_latency = urllc_lat.tolist()

        # Ghi nhận KPI của slot này
        results["throughput"].append(slot_throughput)
        results["latency"].extend(slot_latency)
        results["energy_cost"].append(slot_energy)
        results["fragment_cost"].append(slot_fragment)
        results["switch_cost"].append(slot_switch)
        results["guardband_cost"].append(slot_guardband)


# ==============================================================================
# VẼ BIỂU ĐỒ
# ==============================================================================

def _plot_results(results_main, results_bm, figure_dir):
    """
    Vẽ các KPI trên cùng một biểu đồ:
        - Framework (SAC+DQN): màu xanh
        - Benchmark: màu đỏ
    Latency được biểu diễn bằng CDF.
    """

    line_metrics = [
        ("throughput",          "Throughput (bps)",    "Throughput"),
        ("resource_efficiency", "Resource Efficiency", "Resource Efficiency"),
        ("energy_cost",         "Energy Cost",         "Energy Cost"),
        ("fragment_cost",       "Fragment Cost",       "Fragment Cost"),
        ("switch_cost",         "Switch Cost",         "Switch Cost"),
        ("guardband_cost",      "Guardband Cost",      "Guardband Cost"),
        ("slice_budget",        "Slice Budget Ratio",  "Slice Budget"),
    ]

    xlabel = {
        "resource_efficiency": "Frame",
        "slice_budget": "Frame",
    }

    # ======================================================
    # Line plots
    # ======================================================
    for key, ylabel, title in line_metrics:

        plt.figure(figsize=(10, 4))

        plt.plot(
            results_main[key],
            color="steelblue",
            linewidth=1.8,
            label="Framework"
        )

        plt.plot(
            results_bm[key],
            color="tomato",
            linewidth=1.8,
            label="Benchmark"
        )

        plt.xlabel(xlabel.get(key, "Slot"))
        plt.ylabel(ylabel)
        plt.title(title)

        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()

        plt.savefig(f"{figure_dir}/{key}.png", dpi=150)
        plt.close()

        print(f"[PLOT] {figure_dir}/{key}.png")

    # ======================================================
    # Latency CDF
    # ======================================================

    plt.figure(figsize=(8,5))

    data_main = np.sort(results_main["latency"])
    cdf_main = np.arange(1, len(data_main)+1)/len(data_main)

    data_bm = np.sort(results_bm["latency"])
    cdf_bm = np.arange(1, len(data_bm)+1)/len(data_bm)

    plt.plot(
        data_main,
        cdf_main,
        color="steelblue",
        linewidth=2,
        label="Framework"
    )

    plt.plot(
        data_bm,
        cdf_bm,
        color="tomato",
        linewidth=2,
        label="Benchmark"
    )

    plt.xlabel("Latency (s)")
    plt.ylabel("CDF")
    plt.title("CDF of URLLC Latency")

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{figure_dir}/latency.png", dpi=150)
    plt.close()

    print(f"[PLOT] {figure_dir}/latency.png")

