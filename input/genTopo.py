import numpy as np
import matplotlib.pyplot as plt

import numpy as np

import numpy as np

def generate_topology(
    num_RUs,
    num_UEs,
    R_ru=0,         # bán kính phân bố RU quanh DU/CU (m)
    R_ue=1000,         # bán kính phục vụ của mỗi RU (m)
    d_min_ru=0,     # khoảng cách tối thiểu giữa các RU (m)
    seed=None,
    max_iter=10000
):
    """
    Sinh topology O-RAN:
    - DU/CU tại (0,0)
    - RU phân bố quanh DU/CU với khoảng cách tối thiểu
    - UE gắn với RU phục vụ, nằm trong vùng phủ RU

    Returns:
    - ru_pos: (num_RUs, 2)
    - ue_pos: (num_UEs, 2)
    - ue_ru_assoc: (num_UEs,) chỉ số RU phục vụ
    """

    if seed is not None:
        np.random.seed(seed)

    # ---------- Generate RU positions ----------
    ru_pos = []
    it = 0

    while len(ru_pos) < num_RUs and it < max_iter:
        it += 1

        r = np.sqrt(np.random.uniform(0, 1)) * R_ru
        a = np.random.uniform(0, 2*np.pi)
        candidate = np.array([r * np.cos(a), r * np.sin(a)])

        if all(np.linalg.norm(candidate - np.array(p)) >= d_min_ru for p in ru_pos):
            ru_pos.append(candidate)

    if len(ru_pos) < num_RUs:
        raise RuntimeError(
            f"Cannot place {num_RUs} RUs with d_min={d_min_ru} m inside R_ru={R_ru} m"
        )

    ru_pos = np.array(ru_pos)

    # ---------- Generate UE positions around serving RU ----------
    ue_pos = []

    for _ in range(num_UEs):
        ru_idx = np.random.randint(num_RUs)
        ru = ru_pos[ru_idx]

        r = np.sqrt(np.random.uniform(0, 1)) * R_ue
        a = np.random.uniform(0, 2*np.pi)
        ue = ru + np.array([r * np.cos(a), r * np.sin(a)])

        ue_pos.append(ue)

    ue_pos = np.array(ue_pos)

    return ru_pos, ue_pos



def calDistance(num_UEs, num_RUs, ru_pos, ue_pos):
    # Distance UE–RU
    dist_ue_ru = np.zeros((num_RUs, num_UEs))
    for u in range(num_UEs):
        for r in range(num_RUs):
            dist_ue_ru[r, u] = np.linalg.norm(ue_pos[u] - ru_pos[r])

    # Distance RU–RU
    dist_ru_ru = np.zeros((num_RUs, num_RUs))
    for r in range(num_RUs):
        for rp in range(num_RUs):
            dist_ru_ru[r, rp] = np.linalg.norm(ru_pos[r] - ru_pos[rp])

    print("min d : ", np.min(dist_ue_ru),'\n')
    print("max d : ", np.max(dist_ue_ru),'\n')
    print("aver d : ", np.average(dist_ue_ru),'\n')
    return dist_ue_ru, dist_ru_ru


def plot_topology(ru_pos, ue_pos, du_cu_pos=(0, 0), R_cell=None, R_ru=None):
    """
    Vẽ bản đồ phân bố UE, RU và DU/CU
    """
    plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 20,
    "text.usetex": False
    })

    plt.figure(figsize=(10, 10))

    # UE
    plt.scatter(
        ue_pos[:, 0], ue_pos[:, 1],
        c='tab:blue', s=25, alpha=0.6,
        label='UE'
    )

    # RU
    plt.scatter(
        ru_pos[:, 0], ru_pos[:, 1],
        c='tab:red', s=120, marker='^',
        label='RU'
    )

    # DU / CU
    plt.scatter(
        du_cu_pos[0], du_cu_pos[1],
        c='black', s=180, marker='*',
        label='DU/CU'
    )

    # Vẽ cell boundary (nếu có)
    if R_cell is not None:
        circle = plt.Circle(
            du_cu_pos, R_cell,
            color='gray', fill=False, linestyle='--', alpha=0.5
        )
        plt.gca().add_patch(circle)

    # Vẽ RU deployment boundary (nếu có)
    if R_ru is not None:
        circle = plt.Circle(
            du_cu_pos, R_ru,
            color='red', fill=False, linestyle=':', alpha=0.5
        )
        plt.gca().add_patch(circle)

    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend(loc='lower center')
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.title('Spatial Deployment of DU/CU, RUs, and UEs')

    plt.savefig(r"./UERUmap.pdf")

