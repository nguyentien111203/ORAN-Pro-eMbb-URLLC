import gym
import numpy as np
from gym import spaces

import numpy as np

class PowerMediator:
    """
    Trung gian giữa SAC và DQN.
    Nhận quota power theo slice (do SAC quyết định),
    chia power cụ thể cho từng RU sao cho tổng công suất mỗi RU ≤ Pmax.
    """

    def __init__(self, slot_envs, allocation_mode="equal"):
        self.slot_envs = slot_envs
        self.allocation_mode = allocation_mode  # 'equal' | 'proportional' | 'snr_based'

    def allocate_power(self, slice_power_quota):
        """
        slice_power_quota: np.array [num_RU, num_slices] — công suất mỗi slice tại mỗi RU do SAC quyết định.
        Trả về quota_power_per_RU cùng shape [num_RU, num_slices].
        """
        slice_power_quota = np.array(slice_power_quota, dtype=float)
        num_rus = len(self.slot_envs)
        num_slices = self.slot_envs[0].num_slices

        # ✅ Kiểm tra shape hợp lệ
        if slice_power_quota.shape != (num_rus, num_slices):
            raise ValueError(f"slice_power_quota phải có shape ({num_rus}, {num_slices}), hiện là {slice_power_quota.shape}")

        quota_power_per_RU = np.zeros_like(slice_power_quota, dtype=float)

        # --- Tính tổng công suất từng RU ---
        total_ru_power = np.array([env.RU.Pmax for env in self.slot_envs], dtype=float)
        total_power_sum = np.sum(total_ru_power) if np.sum(total_ru_power) > 0 else 1.0

        # --- Phân bổ công suất ---
        if self.allocation_mode == "equal":
            # Mỗi RU dùng đúng tỉ lệ slice_power_quota[r, s]
            for r in range(num_rus):
                quota_power_per_RU[r, :] = slice_power_quota[r, :] * self.slot_envs[r].RU.Pmax

        elif self.allocation_mode == "proportional":
            ru_ratio = total_ru_power / total_power_sum
            for r in range(num_rus):
                quota_power_per_RU[r, :] = slice_power_quota[r, :] * ru_ratio[r] * np.sum(total_ru_power)

        elif self.allocation_mode == "snr_based":
            avg_snr = np.array([getattr(env, "get_avg_snr", lambda: 1.0)() for env in self.slot_envs])
            snr_ratio = avg_snr / (np.sum(avg_snr) + 1e-9)
            for r in range(num_rus):
                quota_power_per_RU[r, :] = slice_power_quota[r, :] * snr_ratio[r] * np.sum(total_ru_power)

        else:
            raise ValueError(f"Unknown allocation_mode: {self.allocation_mode}")

        # --- Giới hạn công suất từng RU ---
        for r, env in enumerate(self.slot_envs):
            total_ru_alloc = np.sum(quota_power_per_RU[r])
            if total_ru_alloc > env.RU.Pmax:
                quota_power_per_RU[r] *= env.RU.Pmax / (total_ru_alloc + 1e-9)

        return quota_power_per_RU
