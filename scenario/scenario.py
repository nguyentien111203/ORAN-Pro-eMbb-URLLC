import numpy as np
import matplotlib.pyplot as plt
import yaml
import copy
import json

# ==========================================
# CONFIGURATION
# ==========================================
DEFAULT_CONFIG = {
    'topology': {
        'num_rus': 3,
        'num_ues': 72
    },
    'mobility': {
        'stable': {'speed_min': 0, 'speed_max': 0, 'r_min': 500, 'r_max': 1000},
        'low': {'speed_min': 1, 'speed_max': 5, 'r_min': 500, 'r_max': 1000},
        'high': {'speed_min': 10, 'speed_max': 20, 'r_min': 500, 'r_max': 1000}
    },
    'channel': {
        'fc_ghz': 3.5,
        'antenna_gain_dbi': 15.0,
        'shadow_std_db': 4.0 #3gpp standard
    }
}

def load_and_merge_config(yaml_path="config.yaml"):
    merged_config = copy.deepcopy(DEFAULT_CONFIG)
    try:
        with open(yaml_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
            
            if 'network' in yaml_data:
                num_rus = yaml_data['network'].get('num_rus', merged_config['topology']['num_rus'])
                ues_per_ru = yaml_data['network'].get('ues_per_ru', 24)
                merged_config['topology']['num_rus'] = num_rus
                merged_config['topology']['num_ues'] = num_rus * ues_per_ru
                
            if 'mobility' in yaml_data:
                merged_config['mobility'] = yaml_data['mobility']
                
        print(f"--> Successfully loaded external config from {yaml_path}")
    except FileNotFoundError:
        print(f"--> Warning: {yaml_path} not found. Falling back to internal DEFAULT_CONFIG.")
        
    return merged_config

# ==========================================
# 1. CHANNEL MODEL FUNCTIONS
# ==========================================
def calculate_3gpp_pathloss(distance_m, fc_ghz=3.5):
    d = np.maximum(distance_m, 1.0)
    return 28.0 + 22 * np.log10(d) + 20 * np.log10(fc_ghz)

def calculate_total_channel_gain(num_ues, path_loss_db, antenna_gain_dbi=15.0, shadow_std_db=4.0):
    shadowing_db = np.random.normal(loc=0.0, scale=shadow_std_db, size=num_ues)
    
    h_real = np.random.normal(loc=0, scale=np.sqrt(0.5), size=num_ues)
    h_imag = np.random.normal(loc=0, scale=np.sqrt(0.5), size=num_ues)
    fading_linear = np.abs(h_real + 1j * h_imag)**2
    fading_db = 10 * np.log10(fading_linear + 1e-12) 
    
    total_gain_db = antenna_gain_dbi - path_loss_db + shadowing_db + fading_db
    total_gain_linear = 10 ** (total_gain_db / 10)
    
    return {
        'gain_db': total_gain_db,
        'gain_linear': total_gain_linear,
        'shadowing_db': shadowing_db,
        'fading_db': fading_db
    }

# ==========================================
# 2. TOPOLOGY GENERATION (UPGRADED)
# ==========================================
def generate_topology(num_rus, num_ues, r_min, r_max, balanced=True):
    """
    Scatters UEs uniformly but strictly binds them within 120-degree 
    sectors dedicated to their respective RUs.
    """
    ue_pos = np.zeros((num_ues, 2))
    ue_ru_assoc = np.zeros(num_ues, dtype=int)
    
    if balanced:
        ues_per_ru = num_ues // num_rus
        sector_angle = 2 * np.pi / num_rus  # 120 degrees in radians
        
        for i in range(num_rus):
            start_idx = i * ues_per_ru
            end_idx = start_idx + ues_per_ru
            
            # Associate this chunk of UEs to RU i
            ue_ru_assoc[start_idx:end_idx] = i
            
            # Generate radii (using sqrt for uniform area distribution)
            r = np.sqrt(np.random.uniform(r_min**2, r_max**2, ues_per_ru))
            
            # Generate angles strictly within this RU's sector
            theta_start = i * sector_angle
            theta_end = (i + 1) * sector_angle
            theta = np.random.uniform(theta_start, theta_end, ues_per_ru)
            
            # Convert polar to Cartesian coordinates
            ue_pos[start_idx:end_idx, 0] = r * np.cos(theta)
            ue_pos[start_idx:end_idx, 1] = r * np.sin(theta)
            
    else:
        # Fallback if unbalanced (not standard for this specific 3-sector setup)
        ue_ru_assoc = np.random.randint(0, num_rus, num_ues)
        r = np.sqrt(np.random.uniform(r_min**2, r_max**2, num_ues))
        theta = np.random.uniform(0, 2 * np.pi, num_ues)
        ue_pos[:, 0] = r * np.cos(theta)
        ue_pos[:, 1] = r * np.sin(theta)
            
    return ue_pos, ue_ru_assoc

# ==========================================
# 3. SCENARIO & MOBILITY SYSTEM
# ==========================================
def generate_scenario(scenario_type="stable", seed=None, config=None):
    if seed is not None:
        np.random.seed(seed)
    config = config or DEFAULT_CONFIG
        
    num_rus = config['topology']['num_rus']
    num_ues = config['topology']['num_ues']
    fc_ghz = config['channel']['fc_ghz']
    antenna_gain = config['channel']['antenna_gain_dbi']
    
    mob_config = config['mobility'].get(scenario_type, config['mobility']['stable'])
    r_min = mob_config.get('r_min', 500)
    r_max = mob_config.get('r_max', 1000)
    
    ue_pos, ue_ru_assoc = generate_topology(num_rus, num_ues, r_min, r_max, balanced=True)
    ru_pos = np.array([0, 0]) 
    distance = np.linalg.norm(ue_pos - ru_pos, axis=1)
    
    speed = np.zeros(num_ues)
    direction = np.zeros(num_ues)
    traffic_profiles = np.empty(num_ues, dtype=object)
    ues_per_ru = num_ues // num_rus
    
    for i in range(num_rus):
        start_idx = i * ues_per_ru
        end_idx = start_idx + ues_per_ru
        mid_idx = start_idx + (ues_per_ru // 2)
        
        if scenario_type == "stable":
            traffic_profiles[start_idx:end_idx] = "DL_normal"
            speed[start_idx:end_idx] = 0
            
        elif scenario_type == "low":
            traffic_profiles[start_idx:mid_idx] = "DL_normal"
            traffic_profiles[mid_idx:end_idx] = "DL_Max_Throughput"
            speed[start_idx:end_idx] = np.random.uniform(mob_config['speed_min'], mob_config['speed_max'], ues_per_ru)
            direction[start_idx:end_idx] = np.random.uniform(0, 2 * np.pi, ues_per_ru)
            
        elif scenario_type == "high":
            traffic_profiles[start_idx:end_idx] = "DL_Continuous"
            speed[start_idx:end_idx] = np.random.uniform(mob_config['speed_min'], mob_config['speed_max'], ues_per_ru)
            angles_to_center = np.arctan2(ue_pos[start_idx:end_idx, 1], ue_pos[start_idx:end_idx, 0])
            direction[start_idx:end_idx] = angles_to_center + np.random.choice([0, np.pi], ues_per_ru)

    velocity = np.column_stack((speed * np.cos(direction), speed * np.sin(direction)))
    
    path_loss_db = calculate_3gpp_pathloss(distance, fc_ghz)
    channel_info = calculate_total_channel_gain(num_ues, path_loss_db, antenna_gain)
    channel_info['path_loss_db'] = path_loss_db
    
    return {
        "ue_pos": ue_pos,
        "ue_ru_assoc": ue_ru_assoc,
        "velocity": velocity,
        "distance": distance,
        "scenario_type": scenario_type,
        "channel": channel_info, 
        "traffic": traffic_profiles.tolist(),
        "bounds": {"r_min": r_min, "r_max": r_max}
    }

def update_position(scenario, dt):
    ue_pos = scenario["ue_pos"]
    velocity = scenario["velocity"]
    r_min = scenario["bounds"]["r_min"]
    r_max = scenario["bounds"]["r_max"]
    
    new_pos = ue_pos + velocity * dt
    distances = np.linalg.norm(new_pos, axis=1)
    
    out_of_bounds_outer = distances > r_max
    out_of_bounds_inner = distances < r_min
    out_of_bounds = out_of_bounds_outer | out_of_bounds_inner
    
    if np.any(out_of_bounds):
        velocity[out_of_bounds] = -velocity[out_of_bounds] 
        new_pos[out_of_bounds] = ue_pos[out_of_bounds] + velocity[out_of_bounds] * dt
        
    scenario["ue_pos"] = new_pos
    scenario["velocity"] = velocity
    return scenario

# ==========================================
# 4. VISUALIZATION (UPGRADED)
# ==========================================
def plot_scenario(scenario):
    colors = ['#4A4AEB', '#4A9C4A', '#9C4A9C'] 
    plt.figure(figsize=(8, 8))
    
    r_min = scenario['bounds']['r_min']
    r_max = scenario['bounds']['r_max']
    num_rus = 3 # Hardcoded for the 3-sector visualization
    
    # Plot RU Center
    plt.scatter(0, 0, color='red', marker='^', s=250, label='3 Co-located RUs (0,0)', zorder=4)
    
    # Plot Sector Divider Lines
    for i in range(num_rus):
        angle = i * (2 * np.pi / num_rus)
        plt.plot([0, r_max * np.cos(angle)], [0, r_max * np.sin(angle)], 
                 color='red', linestyle=':', alpha=0.6, zorder=2)
    
    # Plot UEs
    for i in range(num_rus): 
        idx = np.array(scenario['ue_ru_assoc']) == i
        plt.scatter(scenario['ue_pos'][idx, 0], scenario['ue_pos'][idx, 1], 
                    color=colors[i], alpha=0.9, edgecolors='black', 
                    label=f'RU {i} UEs (n={np.sum(idx)})', zorder=3)

    # Plot Boundaries
    outer_boundary = plt.Circle((0, 0), r_max, color='gray', fill=False, linestyle='--', linewidth=1.5, zorder=1)
    inner_boundary = plt.Circle((0, 0), r_min, color='orange', fill=False, linestyle='--', linewidth=1.5, zorder=1)
    plt.gca().add_patch(outer_boundary)
    plt.gca().add_patch(inner_boundary)
    
    plt.title(f'Topology: {scenario["scenario_type"].capitalize()} Mobility Scenario', fontsize=14)
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.xlim(-r_max*1.1, r_max*1.1)
    plt.ylim(-r_max*1.1, r_max*1.1)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')
    plt.gca().set_aspect('equal')
    plt.show()

# ==========================================
# MAIN EXECUTION 
# ==========================================
if __name__ == "__main__":
    active_config = load_and_merge_config("config.yaml")

    target_scenario = "high" #can change to 'low' or 'stable' to see the output
    
    print(f"\n1. Generating {target_scenario.upper()} Mobility Scenario...")
    
    scenario = generate_scenario(target_scenario, seed=None, config=active_config)
    
    def numpy_to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    with open(f"{target_scenario}_scenario_output.json", "w") as f:
        json.dump(scenario, f, indent=4, default=numpy_to_list)

    print("\n" + "="*60)
    print(f"✅ SUCCESS: Data saved to '{target_scenario}_scenario_output.json'")
    print("="*60 + "\n")

    print(f"\n2. Plotting {target_scenario.capitalize()} Topology...")
    plot_scenario(scenario)