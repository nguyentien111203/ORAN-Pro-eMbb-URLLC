"""
benchmark.py
Pipeline benchmark so sánh SAC allocation (với select_action đã sửa)
Chạy: python benchmark.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging                                                         
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)  
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.serif": [],
})
import json
import csv
from tqdm import trange

# ── Import hệ thống ──────────────────────────────────────────────────────────
from input.takeInput import load_cons_from_json
from input.genInput import generate_pipeline_inputs, calculateScaleMax, generate_h_matrix
from combine.general.train_general import buildEnvAgent

# ── Import SAC_benchmark (thay vì SAC gốc) ───────────────────────────────────
from combine.SAC_benchmark.SACagent import SACAgent
from combine.SAC_benchmark.FrameEnv import FrameEnv

OUTPUT_DIR = "./output/benchmark"
FIGURE_DIR = "./Figures/benchmark"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


# ── 1. Chạy một episode, trả về metrics ──────────────────────────────────────
def run_one_episode(frame_env, sac_agent):
    state = frame_env.reset()
    done = False
    last_action = None
    ep_reward = 0.0

    # Accumulators
    urllc_ratios, embb_ratios = [], []
    total_prb_used, total_prb_available = 0.0, 0.0

    while not done:
        action = sac_agent.select_action(state, last_action)
        next_state, reward, done, info = frame_env.step(action)

        # thu thập metric từ info
        thr_list = info["thr"]   # eMBB: list[array per slice]
        lat_list = info["lat"]   # URLLC: list[array per slice]

        for s_arr in thr_list:
            embb_ratios.extend(s_arr.tolist())
        for s_arr in lat_list:
            urllc_ratios.extend(s_arr.tolist())

        # resource utilization: tính tổng PRB được dùng / tổng PRB available
        action_flat = np.array(action).flatten()
        idx = 0
        for r in range(frame_env.num_rus):
            num_bwps = len(frame_env.RUs[r].bwps)
            for s in range(frame_env.num_slices):
                for b in range(num_bwps):
                    total_prb = frame_env.RUs[r].bwps[b].num_prb
                    total_prb_used      += action_flat[idx] * total_prb
                    total_prb_available += total_prb
                    idx += 1

        ep_reward += reward
        state = next_state
        last_action = action

    # Tính metrics
    urllc_arr = np.array(urllc_ratios)
    embb_arr  = np.array(embb_ratios)

    metrics = {
        "reward":               ep_reward,
        "urllc_avg_ratio":      float(np.mean(urllc_arr))      if len(urllc_arr) else 0.0,
        "urllc_min_ratio":      float(np.min(urllc_arr))       if len(urllc_arr) else 0.0,
        "embb_avg_thr_ratio":   float(np.mean(embb_arr))       if len(embb_arr)  else 0.0,
        "embb_min_thr_ratio":   float(np.min(embb_arr))        if len(embb_arr)  else 0.0,
        "spectrum_efficiency":  float(np.mean(embb_arr))       if len(embb_arr)  else 0.0,
        "resource_utilization": float(total_prb_used / (total_prb_available + 1e-9)),
    }
    return metrics


# ── 2. Chạy N episode benchmark ───────────────────────────────────────────────
def run_benchmark(frame_env, sac_agent, num_episodes=50):
    all_metrics = []
    for ep in trange(num_episodes, desc="Benchmark SAC"):
        m = run_one_episode(frame_env, sac_agent)
        all_metrics.append(m)
        print(f"  Ep {ep+1:3d} | reward={m['reward']:.4f} | "
              f"urllc_avg={m['urllc_avg_ratio']:.3f} | "
              f"embb_avg={m['embb_avg_thr_ratio']:.3f} | "
              f"util={m['resource_utilization']:.3f}")
    return all_metrics


# ── 3. Lưu log CSV ────────────────────────────────────────────────────────────
def save_log(all_metrics, path):
    keys = list(all_metrics[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"[LOG] Đã lưu log → {path}")


# ── 4. In bảng metric tổng hợp ───────────────────────────────────────────────
def print_summary(all_metrics):
    keys = list(all_metrics[0].keys())
    print("\n" + "="*60)
    print(f"{'Metric':<30} {'Mean':>10} {'Std':>10} {'Min':>8} {'Max':>8}")
    print("="*60)
    for k in keys:
        vals = [m[k] for m in all_metrics]
        print(f"{k:<30} {np.mean(vals):>10.4f} {np.std(vals):>10.4f} "
              f"{np.min(vals):>8.4f} {np.max(vals):>8.4f}")
    print("="*60)


# ── 5. Vẽ plot cơ bản ────────────────────────────────────────────────────────
def plot_metrics(all_metrics):
    keys   = list(all_metrics[0].keys())
    colors = ['steelblue','forestgreen','tomato','orange','purple','brown','teal']
    episodes = range(1, len(all_metrics) + 1)

    for i, k in enumerate(keys):
        vals = [m[k] for m in all_metrics]
        plt.figure(figsize=(8, 4))
        plt.plot(episodes, vals, color=colors[i % len(colors)], linewidth=1.5)
        plt.title(k.replace("_", " ").title())
        plt.xlabel("Episode")
        plt.ylabel(k)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        path = os.path.join(FIGURE_DIR, f"{k}.png")
        plt.savefig(path)
        plt.close()
        print(f"[PLOT] Đã lưu → {path}")


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("=== Benchmark SAC Allocation ===\n")

    # Load config
    consta     = load_cons_from_json("./config/cons.json")
    trainCons  = load_cons_from_json("./config/trainCons.json")

    # Tạo môi trường
    RUs, embb_slices, urllc_slices, num_urllc_ue, num_embb_ue = generate_pipeline_inputs(
        "./config/ru.yaml", "./config/slice.yaml", "./config/ue.yaml", consta
    )
    scale_max = calculateScaleMax(RUs, embb_slices, urllc_slices,
                                  consta["cost_switch"], consta["cost_gb"])
    H = generate_h_matrix(
        len(RUs), consta["frame_slots"],
        len(embb_slices) + len(urllc_slices),
        num_urllc_ue, num_embb_ue
    )

    ru_envs, ru_dqn_agents, frame_env, _ = buildEnvAgent(
        RUs, urllc_slices, embb_slices, H,
        consta["inter_RU"], consta["inter_factor"],
        consta["N0_mW_per_MHz"], consta["w_reward"],
        consta["cost_switch"], consta["cost_gb"],
        scale_max, trainCons, consta["frame_slots"]
    )

    # Tạo SAC agent từ SAC_benchmark
    num_bwp_ru = [len(RUs[r].bwps) for r in range(len(RUs))]
    sac_agent = SACAgent(
        4 + len(urllc_slices) + len(embb_slices),
        len(RUs),
        num_bwp_ru,
        len(urllc_slices) + len(embb_slices),
        trainCons["forSAC"]
    )

    # Chạy benchmark
    all_metrics = run_benchmark(frame_env, sac_agent, num_episodes=50)

    # Lưu kết quả
    log_path = os.path.join(OUTPUT_DIR, "benchmark_log.csv")
    save_log(all_metrics, log_path)
    print_summary(all_metrics)
    plot_metrics(all_metrics)

    print("\n=== Benchmark hoàn thành! ===")
    print(f"  Log  : {log_path}")
    print(f"  Plots: {FIGURE_DIR}/")


if __name__ == "__main__":
    main()