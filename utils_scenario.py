import numpy as np
import matplotlib.pyplot as plt
from combine.SAC import FrameEnv
from combine.SAC_benchmark import FrameEnv

def _run_frame(frame_env, sac_agent, results):

    action = sac_agent.select_action()
    next_state, reward, info, done = frame_env.step(action)
    
    # ==========================================================
    # Save KPI
    # ==========================================================
    results["throughput"].append(info["thr"])
    
    results["latency"].extend(info["lat"])

    results["energy_cost"].append(info["costE"])
    results["fragment_cost"].append(info["costF"])
    results["switch_cost"].append(info["costS"])
    results["guardband_cost"].append(info["costGB"])
    results["resource_efficiency"].append(info["resource_eff"])
    return results


def runEvaluateEachFrame(frame_env,
                      sac_agent,
                      dqn_agents,
                      scenario):
    
# ==============================================================================
# VẼ BIỂU ĐỒ
# ==============================================================================

def _plot_results(results_main, results_bm, figure_dir):
    """
    Vẽ biểu đồ đường cho các KPI (trừ latency dùng CDF).
    Mỗi KPI tạo 2 file riêng: <key>.png (framework) và <key>-bm.png (benchmark).
    """
    line_metrics = [
        ("throughput",          "Throughput (bps)",    "Throughput theo slot"),
        ("resource_efficiency", "Resource Efficiency", "Resource Efficiency theo frame"),
        ("energy_cost",         "Energy Cost",         "Energy Cost theo slot"),
        ("fragment_cost",       "Fragment Cost",        "Fragment Cost theo slot"),
        ("switch_cost",         "Switch Cost",          "Switch Cost theo slot"),
        ("guardband_cost",      "Guardband Cost",       "Guardband Cost theo slot"),
        ("slice_budget",        "Slice Budget Ratio",   "Slice Budget theo frame"),
    ]

    xlabel = {
        "resource_efficiency": "Frame",
        "slice_budget":        "Frame",
    }

    for key, ylabel, title in line_metrics:
        x_label = xlabel.get(key, "Slot")

        # --- Framework ---
        plt.figure(figsize=(10, 4))
        plt.plot(results_main[key], color="steelblue", linewidth=1.5)
        plt.xlabel(x_label)
        plt.ylabel(ylabel)
        plt.title(f"{title} — Framework")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{figure_dir}/{key}.png", dpi=150)
        plt.close()
        print(f"[PLOT] {figure_dir}/{key}.png")

        # --- Benchmark ---
        plt.figure(figsize=(10, 4))
        plt.plot(results_bm[key], color="tomato", linewidth=1.5)
        plt.xlabel(x_label)
        plt.ylabel(ylabel)
        plt.title(f"{title} — Benchmark")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{figure_dir}/{key}-bm.png", dpi=150)
        plt.close()
        print(f"[PLOT] {figure_dir}/{key}-bm.png")

    # --- CDF Latency: Framework ---
    plt.figure(figsize=(8, 5))
    data = np.sort(results_main["latency"])
    cdf  = np.arange(1, len(data) + 1) / len(data)
    plt.plot(data, cdf, color="steelblue", linewidth=1.5)
    plt.xlabel("Latency (s)")
    plt.ylabel("CDF")
    plt.title("CDF Latency URLLC — Framework")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{figure_dir}/latency.png", dpi=150)
    plt.close()
    print(f"[PLOT] {figure_dir}/latency.png")

    # --- CDF Latency: Benchmark ---
    plt.figure(figsize=(8, 5))
    data = np.sort(results_bm["latency"])
    cdf  = np.arange(1, len(data) + 1) / len(data)
    plt.plot(data, cdf, color="tomato", linewidth=1.5)
    plt.xlabel("Latency (s)")
    plt.ylabel("CDF")
    plt.title("CDF Latency URLLC — Benchmark")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{figure_dir}/latency-bm.png", dpi=150)
    plt.close()
    print(f"[PLOT] {figure_dir}/latency-bm.png")
