import numpy as np
from tqdm import trange
from collections import deque

def train_dqn(envs, agents, num_episodes, initBWP_slice):

    num_ru = len(envs)
    num_slices = envs[0].num_slices
    num_embb = envs[0].num_embb
    num_urllc = envs[0].num_urllc

    # --- init ---
    states = [np.zeros(envs[0].state_dim) for r in range(num_ru)]

    losses = [[] for _ in range(num_ru)]
    avg_rewards = [[] for _ in range(num_ru)]
    reward_windows = [deque(maxlen=10) for _ in range(num_ru)]

    # --- static info (vector hóa trước) ---
    offsets = [0]
    for n in envs[0].num_ue:
        offsets.append(offsets[-1] + n)

    pac = np.array([ue.pac for s in range(num_urllc) for ue in envs[0].urllc_slices[s].ue_set])

    lat_target = np.array([ue.lat for s in range(num_urllc)for ue in envs[0].urllc_slices[s].ue_set])

    thr_min = np.array([ue.thr for s in range(num_embb) for ue in envs[0].embb_slices[s].ue_set])

    # ================= TRAIN =================
    for ep in trange(num_episodes, desc="Training DQNs"):

        done = False

        while not done:

            # ================= PHASE 1: ACTION =================
            actions = [
                agents[r].select_action(states[r], initBWP_slice[r])
                for r in range(num_ru)
            ]

            # ================= PHASE 2: COMPUTE OUTPUT =================
            # numBits: (slice, ue)
            numBits = np.array([0 for s in range(num_urllc) for ue in envs[0].urllc_slices[s].ue_set], np.float64)
            totalThr = np.array([0 for s in range(num_embb) for ue in envs[0].embb_slices[s].ue_set], np.float64)

            for r in range(num_ru):
                # computeOutput trả numpy array (slice x ue)
                ruBits, ruThr = envs[r].computeOutput(actions[r])
                numBits += ruBits
                totalThr += ruThr

            # ================= PHASE 3: GLOBAL METRIC =================
            urllc_lat = pac / numBits
            urllc_rate = lat_target / urllc_lat 
            embb_rate = totalThr / thr_min

            # ================= PHASE 4: UPDATE + TRAIN =================
            next_states = [None] * num_ru

            for r in range(num_ru):
                next_state, reward, done, _ = envs[r].step(urllc_rate, embb_rate)

                agents[r].store_transition(
                    states[r], actions[r], reward, next_state, done
                )

                loss = agents[r].optimize_model()

                if loss is not None:
                    losses[r].append(loss)

                reward_windows[r].append(reward)
                avg_rewards[r].append(np.mean(reward_windows[r]))

                next_states[r] = next_state

            # update state đồng bộ
            states = next_states
            done = True # Xong 1 ep rồi

        # ================= EPSILON DECAY =================
        for agent in agents:
            agent.eps = max(agent.eps_end, agent.eps * agent.eps_decay)
        

    return avg_rewards, losses




