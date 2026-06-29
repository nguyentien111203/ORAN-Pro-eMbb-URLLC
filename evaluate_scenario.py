from input.takeInput import load_cons_from_json
from input.genInput import generate_pipeline_inputs, calculateScaleMax, generate_h_matrix
from combine.train import alternating_training, buildEnvAgent
from tqdm import trange
from utils_scenario import _run_frame, _plot_results
from scenario.scenario import scenario_execution, update_position, generate_scenario
import numpy as np

def evaluate_scenario(frame_env,
                      sac_agent,
                      dqn_agents,
                      scenario,
                      train=True):

    results = {
        "throughput": [],
        "latency": [],
        "resource_efficiency": [],
        "slice_budget": [],
        "energy_cost": [],
        "fragment_cost": [],
        "switch_cost": [],
        "guardband_cost": []
    }

    # ======================================
    # Reset scenario & environment
    # ======================================
    scenario.reset()
    frame_env.reset()

    # ======================================
    # Warm-up (Frame đầu)
    # ======================================
    frame_env.load_frame(scenario.current_frame())

    _run_frame(
        frame_env,
        sac_agent,
        dqn_agents,
        results,
        train=train
    )

    # ======================================
    # Evaluation
    # ======================================
    while scenario.has_next():

        frame = scenario.next_frame()

        frame_env.load_frame(frame)

        _run_frame(
            frame_env,
            sac_agent,
            dqn_agents,
            results,
            train=train
        )

    return results



    
   