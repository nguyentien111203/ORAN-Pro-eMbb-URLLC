import numpy as np
from tqdm import trange
from collections import deque
from combine.general.drawDQN import plot_rate

def train_dqn(envs, agents, num_episodes, initBWP_slice):

    num_ru = len(envs)
    num_slices = envs[0].num_slices
    num_embb = envs[0].num_embb
    num_urllc = envs[0].num_urllc
    beta = 1.5
    alpha = 3
    states = [np.zeros(envs[0].state_dim) for _ in range(num_ru)]

    losses = [[] for _ in range(num_ru)]
    avg_losses = [[] for _ in range(num_ru)]
    avg_rewards = [[] for _ in range(num_ru)]
    reward_windows = [[] for _ in range(num_ru)]  # tăng window
    loss_windows = [[] for _ in range(num_ru)]  # tăng window
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

            lat_soft = []
            thr_soft = []
            for u in range(len(urllc_rate)):
                lat_soft.append(1 / (1 + alpha * max(urllc_rate[u]-1,0)))
            for u in range(len(embb_rate)):
                thr_soft.append(np.tanh(beta * embb_rate[u]))
            
            for r in range(num_ru):

                next_state, reward, done_env, _ = envs[r].step(
                    urllc_rate, embb_rate, lat_soft, thr_soft
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
                    loss_windows[r].append(loss)

                reward_windows[r].append(reward)
                
                next_states[r] = next_state

            states = next_states

            step += 1
            done = (step >= max_steps)   # FIXED EPISODE LOGIC

        for r in range(num_ru):
            ep_reward = np.mean(
                [x for x in reward_windows[r]]
            )

            avg_rewards[r].append(ep_reward)

            if loss_windows[r]:
                avg_losses[r].append(
                    np.mean(loss_windows[r])
                )
            else:
                avg_losses[r].append(0)

            reward_windows[r].clear()
            loss_windows[r].clear()

        # ================= EPSILON DECAY =================
        for agent in agents:
            agent.eps = max(agent.eps_end, agent.eps - agent.eps_decay)

    plot_rate(rate_dict, "URLLC", "min")
    plot_rate(rate_dict, "URLLC", "avg")
    plot_rate(rate_dict, "URLLC", "gap")

    plot_rate(rate_dict, "eMBB", "min")
    plot_rate(rate_dict, "eMBB", "avg")
    plot_rate(rate_dict, "eMBB", "gap")

    return avg_rewards, avg_losses




