import numpy as np
from tqdm import trange
from collections import deque
from combine.DQN.drawDQN import plot_rate

def train_dqn(envs, agents, num_episodes, initBWP_slice):

    num_ru = len(envs)
    num_slices = envs[0].num_slices
    num_embb = envs[0].num_embb
    num_urllc = envs[0].num_urllc

    states = [np.zeros(envs[0].state_dim) for _ in range(num_ru)]

    losses = [[] for _ in range(num_ru)]
    avg_rewards = [[] for _ in range(num_ru)]
    reward_windows = [deque(maxlen=50) for _ in range(num_ru)]  # tăng window
    rate_dict = {
        "min_urllc": [],
        "avg_urllc": [],
        "min_embb": [],
        "avg_embb": []
    }
    # static info
    pac = np.array([
        ue.pac
        for s in range(num_urllc)
        for ue in envs[0].urllc_slices[s].ue_set
    ])

    lat_target = np.array([
        ue.lat
        for s in range(num_urllc)
        for ue in envs[0].urllc_slices[s].ue_set
    ])

    thr_min = np.array([
        ue.thr
        for s in range(num_embb)
        for ue in envs[0].embb_slices[s].ue_set
    ])

    # ================= TRAIN =================
    for ep in trange(num_episodes, desc="Training DQNs"):

        done = False
        step = 0
        max_steps = 10   # IMPORTANT: tạo episode thật

        while not done:

            # ================= ACTION =================
            actions = [
                agents[r].select_action(states[r], initBWP_slice[r])
                for r in range(num_ru)
            ]
            #print("action : ", actions[0])
            # ================= OUTPUT =================
            numBits = np.array([
                1e-7
                for s in range(num_urllc)
                for ue in envs[0].urllc_slices[s].ue_set
            ], np.float64)

            totalThr = np.zeros_like(thr_min, dtype=np.float64)

            for r in range(num_ru):
                ruBits, ruThr = envs[r].computeOutput(actions[r])
                numBits += ruBits
                totalThr += ruThr

            # ================= METRICS =================
            urllc_lat = pac / (numBits + 1e-8)

            urllc_rate = urllc_lat / (lat_target + 1e-8)
            embb_rate = totalThr / (thr_min + 1e-8)

            # ================= STEP =================
            next_states = [None] * num_ru
            rate_dict["avg_embb"].append(np.average(embb_rate))
            rate_dict["avg_urllc"].append(np.average(urllc_rate))
            rate_dict["min_embb"].append(np.min(embb_rate))
            rate_dict["min_urllc"].append(np.min(urllc_rate))

            for r in range(num_ru):

                next_state, reward, done_env, _ = envs[r].step(
                    urllc_rate, embb_rate
                )


                agents[r].store_transition(
                    states[r],
                    actions[r],
                    reward,
                    next_state,
                    done_env
                )

                loss = agents[r].optimize_model()
                if loss is not None:
                    losses[r].append(loss)

                reward_windows[r].append(reward)

                next_states[r] = next_state

            states = next_states

            step += 1
            done = (step >= max_steps)   # FIXED EPISODE LOGIC

        for r in range(num_ru):
            ep_reward = np.mean(reward_windows[r])
            avg_rewards[r].append(ep_reward)

        # ================= EPSILON DECAY =================
        for agent in agents:
            agent.eps = max(agent.eps_end, agent.eps * agent.eps_decay)

    plot_rate(rate_dict, "URLLC", "min")
    plot_rate(rate_dict, "URLLC", "avg")
    plot_rate(rate_dict, "URLLC", "gap")

    plot_rate(rate_dict, "eMBB", "min")
    plot_rate(rate_dict, "eMBB", "avg")
    plot_rate(rate_dict, "eMBB", "gap")

    return avg_rewards, losses




