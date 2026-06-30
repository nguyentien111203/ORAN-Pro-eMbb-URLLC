import matplotlib
matplotlib.use('Agg')
from input.takeInput import load_cons_from_json
from input.genInput import generate_pipeline_inputs, calculateScaleMax, generate_h_matrix
from combine.general.train_general import buildEnvAgent
from combine.SAC_benchmark.train_SAC import train_sacBM as train_sac
from combine.SAC_benchmark.SACagent import SACAgentBM as SACAgent
from combine.SAC_benchmark.FrameEnv import FrameEnv

def main():
    consta    = load_cons_from_json(json_path=r"./config/cons.json")
    trainCons = load_cons_from_json(json_path=r"./config/trainCons.json")

    RUs, embb_slices, urllc_slices, num_urllc_ue, num_embb_ue = generate_pipeline_inputs(
        "./config/ru.yaml", "./config/slice.yaml", "./config/ue.yaml", consta
    )

    scale_max = calculateScaleMax(RUs, embb_slices, urllc_slices, consta["cost_switch"], consta["cost_gb"])

    H = generate_h_matrix(
        len(RUs), consta["frame_slots"], len(embb_slices) + len(urllc_slices),
        num_urllc_ue, num_embb_ue
    )

    ru_envs, ru_dqn_agents, frame_env_base, _ = buildEnvAgent(
        RUs, urllc_slices, embb_slices, H, consta["inter_RU"], consta["inter_factor"],
        consta["N0_mW_per_MHz"], consta["w_reward"], consta["cost_switch"],
        consta["cost_gb"], scale_max, trainCons, consta["frame_slots"]
    )

    # Dùng FrameEnv và SACAgent từ SAC_benchmark
    num_bwp_ru = [len(RUs[r].bwps) for r in range(len(RUs))]
    frame_env = FrameEnv(
        RUs, ru_envs, urllc_slices, embb_slices,
        frame_env_base.H, consta["w_reward"], scale_max, consta["frame_slots"]
    )
    sac_agent = SACAgent(
        4 + len(urllc_slices) + len(embb_slices),
        len(RUs), num_bwp_ru,
        len(urllc_slices) + len(embb_slices),
        trainCons["forSAC"]
    )

    # Train SAC_benchmark — lưu ra sac_model_benchmark.pth
    print("=== Train SAC Benchmark ===")
    train_sac(frame_env, sac_agent, trainCons["forSAC"]["sac_train_episodes"])
    print("=== Train hoàn thành! File: sac_model_benchmark.pth ===")

if __name__ == "__main__":
    main()