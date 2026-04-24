from subprob.solve import bigProblems
from input.takeInput import load_system_config, load_cons_from_json, save_gain_matrix, load_gain_matrix
from input.genInput import generate_pipeline_inputs
from baseline.flatDRL.FrameEnv import FrameEnv
from combine.train import createModel 
from combine.allocate import run_oran_allocation_pipeline
from output.printOutput import write_results_to_csv, plot_and_save_metrics
from baseline.runbaseline import runAndEval
import matplotlib.pyplot as plt
import torch
from tqdm import trange

def main():
    # --- Lấy đầu vào ---
    config = load_system_config(csv_path=r"./config/config.csv", line=1)

    consta = load_cons_from_json(json_path=r"./config/cons.json")
    
    trainCons = load_cons_from_json(json_path=r"./config/trainCons.json")
    # --- Tạo input ---
    RUs, slices, H, gain_ru_ru, dist_ue_ru = generate_pipeline_inputs(num_RUs=config["num_RUs"], num_slices=config["num_slices"], 
                                            num_URLLC=config["num_URLLC"], numPRB=config["num_PRB_per_RU"],
                                            B=consta["B_MHz"], n=consta["n"], N0=consta["N0_mW_per_MHz"], 
                                            frame_slot=consta["num_slot_per_frame"],
                                            P_max=config["Pmax_mW"], deadline=config["deadline"],
                                            load=config["load_URLLC"], dataRate=config["dataRate_eMBB_Mbps"])
    
    print("H shape:", H.shape)
    print("H min:", H.min())
    print("H max:", H.max())
    print("H mean:", H.mean())

    save_gain_matrix(H, (config["num_RUs"], config["num_slices"], config["num_URLLC"]))
    #sac_model_path, dqn_model_paths = createModel(config, consta, trainCons, RUs, slices, H, gain_ru_ru, dist_ue_ru)

    # --- Giải bài toán với cvxpy --- 
    #print(" Giải bài toán với cvxpy")
    #H = load_gain_matrix((config["num_RUs"], config["num_slices"], config["num_URLLC"]))

    #problem = bigProblems(RUs=RUs, slices=slices, K=config["num_PRB_per_RU"], H=H, T_slot=consta["T_slot"], num_slot=consta["num_slot_per_frame"],
    #                      w_reward=consta["w_reward"], T_max=consta["T_max_Mbps"], sla_slices=consta["sla_slices"])
    
    #solvetime, longcheck, shortcheck, Probmetrics = problem.solveTwoProblem()


    #metrics, results, pAlloc, xAlloc = run_oran_allocation_pipeline(sac_model_path, dqn_model_paths, slices,
    #                                        RUs, H, gain_ru_ru, dist_ue_ru, T_max=consta["T_max_Mbps"], NF=consta["NF_dB"], 
    #                                        w_reward=consta["w_reward"], 
    #                                        sla_slices=consta["sla_slices"], num_urllc=config["num_URLLC"],
    #                                        frame_slots=consta["num_slot_per_frame"], num_frames=1)
    
    #resultsF = runAndEval(
    #    RUs, slices, H, gain_ru_ru, dist_ue_ru, config, consta)
    
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