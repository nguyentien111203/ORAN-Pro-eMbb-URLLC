from input.takeInput import load_cons_from_json, save_gain_matrix, load_gain_matrix
from input.genInput import generate_pipeline_inputs, generate_channel_gain, calculateScaleMax, generate_h_matrix
from combine.train import alternating_training, buildEnvAgent
from tqdm import trange

def main():
    # --- Lấy các hằng số đầu vào ---
    consta = load_cons_from_json(json_path=r"./config/cons.json")
    
    trainCons = load_cons_from_json(json_path=r"./config/trainCons.json")
    # --- Tạo input ---
    RUs, embb_slices, urllc_slices, num_urllc_ue, num_embb_ue = generate_pipeline_inputs("./config/ru.yaml", "./config/slice.yaml",
                                                              "./config/ue.yaml", consta)
    
    scale_max = calculateScaleMax(RUs, embb_slices, urllc_slices, consta["cost_switch"], consta["cost_gb"])
    
    # Tạm để debug
    H = generate_h_matrix(len(RUs), consta["frame_slots"], len(embb_slices) + len(urllc_slices), 
                          num_urllc_ue, num_embb_ue)

    embb_envs, urllc_envs, embb_dqn_agents, urllc_dqn_agents, frame_env, sac_agent = buildEnvAgent(
        RUs, embb_slices, urllc_slices, H, consta["inter_RU"], consta["inter_factor"], consta["w_reward"], consta["cost_switch"],
        consta["cost_gb"], scale_max, trainCons)

    sac_model_path, dqn_model_paths = alternating_training(embb_envs, urllc_envs, embb_dqn_agents, urllc_dqn_agents, frame_env, sac_agent)

    
    #header = ["num_RUs", "num_slices", "num_URLLC", "num_PRB_per_RU", "Pmax_mW",
    #        "avg_thr_ml", "avg_sla_embb_ml", "avg_sla_urllc_ml", "avg_power_ml", "util_ml",
    #        "avg_thr_flat", "avg_sla_embb_flat", "avg_sla_urllc_flat", "avg_power_flat", "util_flat"]
    
    #data = [config["num_RUs"], config["num_slices"], config["num_URLLC"], config["num_PRB_per_RU"],
    #        config["Pmax_mW"], Probmetrics["avg_throughput"], Probmetrics["avg_eMBB"], Probmetrics["avg_urllc"],
    #        Probmetrics["avg_utilPower"], Probmetrics["avg_utilPRB"], solvetime, 
    #        0, 0, 
    #       0, 0, 0,
    #        0, 0]
    
    #data = [config["num_RUs"], config["num_slices"], config["num_URLLC"], config["num_PRB_per_RU"],
    #        config["Pmax_mW"], 
    #        metrics["avg_throughput"], metrics["avg_SLA_eMBB"], 
    #        metrics["avg_SLA_URLLC"], metrics["avg_util_power"], metrics["avg_util_PRB"],
    #        resultsF["avg_thr"], resultsF["avg_sla_embb"], 
    #        resultsF["avg_sla_urllc"], resultsF["avg_util_power"], resultsF["avg_util_prb"]
    #        ]
    
    #write_results_to_csv(file_path=r"./output/output.csv",
    #                    data=data, header=header)
    
    #plot_and_save_metrics(results=results, save_dir="./Figures/metrics",
    #                    num_slices=config["num_slices"], num_urllc=config["num_URLLC"])

        

main()