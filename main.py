import matplotlib
matplotlib.use('Agg') 

from input.takeInput import load_cons_from_json
from input.genInput import generate_pipeline_inputs, calculateScaleMax, generate_h_matrix
from combine.general.train_general import alternating_training, buildEnvAgent

def main():
    consta = load_cons_from_json(json_path=r"./config/cons.json")
    trainCons = load_cons_from_json(json_path=r"./config/trainCons.json")
    
    RUs, embb_slices, urllc_slices, num_urllc_ue, num_embb_ue = generate_pipeline_inputs(
        "./config/ru.yaml", "./config/slice.yaml", "./config/ue.yaml", consta
    )
    
    scale_max = calculateScaleMax(RUs, embb_slices, urllc_slices, consta["cost_switch"], consta["cost_gb"])
    
    H = generate_h_matrix(
        len(RUs), consta["frame_slots"], len(embb_slices) + len(urllc_slices), 
        num_urllc_ue, num_embb_ue
    )

    ru_envs, ru_dqn_agents, frame_env, sac_agent = buildEnvAgent(
        RUs, urllc_slices, embb_slices, H, consta["inter_RU"], consta["inter_factor"],
        consta["N0_mW_per_MHz"], consta["w_reward"], consta["cost_switch"],
        consta["cost_gb"], scale_max, trainCons, consta["frame_slots"]
    )

    alternating_training(
        len(RUs), ru_envs, ru_dqn_agents, frame_env, sac_agent, 
        trainCons["forDQN"]["dqn_train_episodes"],
        trainCons["forSAC"]["sac_train_episodes"]
    )

if __name__ == "__main__":
    main()