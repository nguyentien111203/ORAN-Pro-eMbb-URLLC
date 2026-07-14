import torch
import matplotlib
import numpy as np
matplotlib.use("Agg")
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "sans-serif", "font.serif": []})

from input.takeInput import load_cons_from_json
from input.genInput import generate_pipeline_inputs, calculateScaleMax
from combine.train import buildEnvAgent, alternating_training
from combine.SAC.SACagent      import SACAgent      as SACAgent_main
from combine.SAC.FrameEnv      import FrameEnv      as FrameEnv_main
from combine.SAC_benchmark.SACagent import SACAgentBM as SACAgent_bm
from combine.train import alternating_training
from evaluate_scenario import load_and_merge_config, generate_scenario, _build_h_from_scenario, evaluate_scenario, print_throughput_stats

def main():
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

    # 3. Sinh H ban đầu chỉ để khởi tạo kích thước FrameEnv/RU_env.
    #    H "thật" của từng frame được evaluate_scenario() tự sinh bên trong
    #    (dựa trên kịch bản di động của scenario.py), nên ở đây chỉ cần một
    #    kịch bản khởi tạo tạm thời với đúng số RU / số UE.
    num_ues_total = sum(num_embb_ue) + sum(num_urllc_ue)
    init_scn_config = load_and_merge_config(consta.get("scenario_config_path", "./scenario/config.yaml"))
    init_scn_config["topology"]["num_rus"] = len(RUs)
    init_scn_config["topology"]["num_ues"] = num_ues_total
    init_scenario = generate_scenario("stable", seed=None, config=init_scn_config)
    H = _build_h_from_scenario(
        init_scenario, RUs, embb_slices, urllc_slices,
        fc_ghz=init_scn_config["channel"]["fc_ghz"],
        antenna_gain_dbi=init_scn_config["channel"]["antenna_gain_dbi"],
        shadow_std_db=init_scn_config["channel"]["shadow_std_db"],
    )

    # 4. Tạo FrameEnv + agent cho framework (SAC chính)
    ru_envs_main, ru_env_agents, frame_env_main_base, sac_agent_main = buildEnvAgent(
        RUs, urllc_slices, embb_slices, H,
        consta["inter_RU"], consta["inter_factor"],
        consta["N0_mW_per_MHz"], consta["w_reward"],
        consta["cost_switch"], consta["cost_gb"],
        scale_max, trainCons, consta["frame_slots"], "fm"
    )

    # Dùng trực tiếp, TUYỆT ĐỐI KHÔNG LƯU RA FILE RỒI LOAD LẠI!
    alternating_training(len(RUs), ru_envs_main, ru_env_agents, frame_env_main_base, sac_agent_main,
                         trainCons["forDQN"]["dqn_train_episodes"], 
                         trainCons["forSAC"]["sac_train_episodes"], "fm")

    # 5. Tạo FrameEnv + agent cho benchmark (SAC_benchmark)
    ru_envs_bm, ru_agents_bm, frame_env_bm_base, sac_agentbm = buildEnvAgent(
        RUs, urllc_slices, embb_slices, H,
        consta["inter_RU"], consta["inter_factor"],
        consta["N0_mW_per_MHz"], consta["w_reward"],
        consta["cost_switch"], consta["cost_gb"],
        scale_max, trainCons, consta["frame_slots"], "bm"
    )

    # Dùng trực tiếp, TUYỆT ĐỐI KHÔNG LƯU RA FILE RỒI LOAD LẠI!
    alternating_training(len(RUs), ru_envs_bm, ru_agents_bm, frame_env_bm_base, sac_agentbm,
                         trainCons["forDQN"]["dqn_train_episodes"], 
                         trainCons["forSAC"]["sac_train_episodes"], "bm")

    # 6. Chạy đánh giá (kịch bản di động "high" để xem hành vi dưới di chuyển nhanh;
    #    đổi scenario_type="stable"/"low" tuỳ nhu cầu, giống main của scenario.py)
    results_main, results_bm = evaluate_scenario(
        RUs, embb_slices, urllc_slices,
        frame_env_main_base, frame_env_bm_base,
        sac_agent_main, sac_agentbm,
        num_frames=500,
        consta=consta,
        plot=True,
        figure_dir="./Figures/evaluate",
        scenario_seed=None,
        scenario_config=init_scn_config,
    )

    print_throughput_stats(results_main, results_bm)

main()