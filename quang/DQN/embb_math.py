import numpy as np

def get_embb_state(
    R_current, R_min, 
    C_e, C_e_max, 
    C_s, C_s_max, 
    C_f, C_f_max, 
    C_g, C_g_max, 
    y_current, y_prev
):
  
    # 1. Count how many PRB allocations changed since the last subframe (t-1)
    num_changes = np.sum(y_current != y_prev)
    
    # 2. Get the total number of possible allocations 
    total_elements = y_current.size
    # 3. Calculate the exponential decay (with 1e-9 safeguard)
    psi = np.exp(-(num_changes / (total_elements + 1e-9)))


    # Calculate the 5 ratios (with 1e-9 safeguards to prevent ZeroDivisionError)
    ratio_R   = R_current / (R_min + 1e-9)
    ratio_C_e = C_e / (C_e_max + 1e-9)
    ratio_C_s = C_s / (C_s_max + 1e-9)
    ratio_C_f = C_f / (C_f_max + 1e-9)
    ratio_C_g = C_g / (C_g_max + 1e-9)

    # Pack everything into the final 1D array for the DQN
    state_vector = np.array([
        ratio_R, 
        ratio_C_e, 
        ratio_C_s, 
        ratio_C_f, 
        ratio_C_g, 
        psi
    ], dtype=np.float32)

    return state_vector

def get_embb_reward(
    ue_throughputs, R_min, 
    C_e, C_e_max, 
    C_s, C_s_max, 
    C_f, C_f_max, 
    C_g, C_g_max, 
    psi, 
    gamma_weight=0.5, 
    alpha_weight=0.5
):

    
    # 1. Throughput Reward Component 
    # If throughput is lower than R_min, reward is negative (penalty).
    throughput_sum = 0.0
    for R_e in ue_throughputs:
        throughput_sum += (R_e / (R_min + 1e-9)) - 1.0
        
    throughput_reward = gamma_weight * throughput_sum

    # 2. Cost Penalty Component
   
    ratio_C_e = C_e / (C_e_max + 1e-9)
    ratio_C_s = C_s / (C_s_max + 1e-9)
    ratio_C_f = C_f / (C_f_max + 1e-9)
    ratio_C_g = C_g / (C_g_max + 1e-9)
    
   
    cost_component = alpha_weight * (4.0 - ratio_C_e - ratio_C_s - ratio_C_f - ratio_C_g)

    # 3. Final Total Reward

    total_reward = throughput_reward + cost_component + psi
    
    return total_reward