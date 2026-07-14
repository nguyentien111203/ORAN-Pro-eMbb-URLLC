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
    SCENARIO = ["low", "stable", "high"]   # "stable" | "low" | "high" (xem mobility trong scenario.py)
    for scenario_type in SCENARIO: 

        mobility_scenario = generate_scenario(scenario_type, seed=scenario_seed, config=scn_config)

        # dt mặc định: thời lượng 1 frame (frame_slots * slot_duration), nếu không
        # truyền vào trực tiếp hoặc khai báo trong consta.
        frame_dt = dt if dt is not None else consta.get(
            "frame_duration_s", frame_slots * consta.get("slot_duration_s", 1e-3)
        )
        # State khởi đầu
        state_main = np.zeros(frame_env_main.state_dim, dtype=np.float32)
        state_bm   = np.zeros(frame_env_bm.state_dim, dtype=np.float32)

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

            # Tính entropy của action SAC theo công thức H = -sum(p * log2(p))
            results_main["action_entropy"].append(_action_entropy(action_main))
            results_bm["action_entropy"].append(_action_entropy(action_bm))

            # ------------------------------------------------------------------
            # Bước 4: Chạy từng slot trong frame, thu thập KPI
            # ------------------------------------------------------------------

            state_main, _, info_main, _ = frame_env_main.step(action_main)
            state_bm, _, info_bm, _ = frame_env_bm.step(action_bm)

            results_main['throughput'].append(info_main["thr"])
            results_bm['throughput'].append(info_bm["thr"])

            #print(info_main["thr"], '\n')
            #print(info_bm["thr"], '\n')

            results_main['latency'][scenario_type].extend(info_main["lat"])
            results_bm['latency'][scenario_type].extend(info_bm["lat"])

            results_main["energy_cost"].append(info_main["costE"])
            results_bm["energy_cost"].append(info_bm["costE"])

            results_main["fragment_cost"].append(info_main["costF"])
            results_bm["fragment_cost"].append(info_bm["costF"])

            results_main["switch_cost"].append(info_main["costS"])
            results_bm["switch_cost"].append(info_bm["costS"])

            results_main["guardband_cost"].append(info_main["costGB"])
            results_bm["guardband_cost"].append(info_bm["costGB"])

            results_main["resource_efficiency"].append(info_main["resource_eff"])
            results_bm["resource_efficiency"].append(info_bm["resource_eff"])

            frame_dt += 0.01

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
        "latency"            : {
            "low" : [], 
            "stable": [], 
            "high": []
        },
        "resource_efficiency": [],
        "energy_cost"        : [],
        "fragment_cost"      : [],
        "switch_cost"        : [],
        "guardband_cost"     : [],
        "slice_budget"       : [],
        "action_entropy"     : [],  # Entropy của action SAC mỗi frame
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
        ("action_entropy",      "Action Entropy (bits)", "Action Entropy"),
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
    # Transmission Delay CDF
    # ======================================================
    SCENARIO = ["low", "stable", "high"]   # "stable" | "low" | "high" (xem mobility trong scenario.py)
    for scenario_type in SCENARIO:
        plt.figure(figsize=(8, 5))

        # Sort data
        data_main = np.sort(np.asarray(results_main["latency"][scenario_type]))
        data_bm   = np.sort(np.asarray(results_bm["latency"][scenario_type]))

        # ECDF
        cdf_main = np.arange(1, len(data_main) + 1) / len(data_main)
        cdf_bm   = np.arange(1, len(data_bm) + 1) / len(data_bm)

        # Nếu muốn bỏ 0 vì log scale không vẽ được
        eps = 1e-12
        data_main = np.maximum(data_main, eps)
        data_bm   = np.maximum(data_bm, eps)

        # CDF
        plt.step(
            data_main,
            cdf_main,
            where="post",
            color="steelblue",
            linewidth=2,
            label="Framework"
        )

        plt.step(
            data_bm,
            cdf_bm,
            where="post",
            color="tomato",
            linewidth=2,
            linestyle="--",
            label="Benchmark"
        )

        # Nếu delay trải dài nhiều bậc thì bật log scale
        plt.xscale("log")

        # Hoặc nếu không muốn log thì comment dòng trên
        # và dùng đoạn dưới để cắt outlier:
        #
        # xmax = np.percentile(
        #     np.concatenate([data_main, data_bm]),
        #     99.5
        # )
        # plt.xlim(0, xmax)

        plt.xlabel(f"Transmission Delay (s) on {scenario_type} mobility")
        plt.ylabel("Empirical CDF")
        plt.title("CDF of URLLC Transmission Delay")

        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{figure_dir}/latency_{scenario_type}.png", dpi=300)
        plt.close()

        print(f"[PLOT] {figure_dir}/latency_{scenario_type}.png")


# ==============================================================================
# THỐNG KÊ THROUGHPUT (mean, std)
# ==============================================================================

def print_throughput_stats(results_main, results_bm):
    """In mean và std của throughput cho cả 2 model."""
    thr_main = np.array(results_main["throughput"])
    thr_bm   = np.array(results_bm["throughput"])
    print(f"\n=== Throughput Statistics ===")
    print(f"  Framework  — mean: {np.mean(thr_main):.4f}, std: {np.std(thr_main):.4f}")
    print(f"  Benchmark  — mean: {np.mean(thr_bm):.4f},  std: {np.std(thr_bm):.4f}")
    print(f"\n=== Action Entropy Statistics ===")
    ent_main = np.array(results_main["action_entropy"])
    ent_bm   = np.array(results_bm["action_entropy"])
    print(f"  Framework  — mean entropy: {np.mean(ent_main):.4f}, std: {np.std(ent_main):.4f}")
    print(f"  Benchmark  — mean entropy: {np.mean(ent_bm):.4f},  std: {np.std(ent_bm):.4f}")


# ==============================================================================
# HÀM TÍNH ENTROPY
# ==============================================================================

def _entropy(p):
    """
    Tính entropy Shannon: H = -sum(p_i * log2(p_i))
    p là mảng xác suất (phân bổ tài nguyên cho các slice trong 1 BWP).
    """
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 1e-12]  # bỏ số 0 để tránh log(0)
    return -np.sum(p * np.log2(p))


def _action_entropy(action):
    """
    Tính entropy trung bình của toàn action SAC.
    action có shape (num_rus, num_bwps, num_slices) sau softmax —
    mỗi [r, b, :] là phân phối xác suất phân bổ cho các slice trên BWP b của RU r.
    Entropy được tính cho từng cặp (r, b) rồi lấy trung bình.
    """
    action = np.array(action)
    if action.ndim == 1:
        # Nếu chưa reshape, bỏ qua — không đủ thông tin shape
        return 0.0
    ent = []
    for r in range(action.shape[0]):
        for b in range(action.shape[1]):
            ent.append(_entropy(action[r, b]))
    return float(np.mean(ent))
