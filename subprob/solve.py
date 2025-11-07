import numpy as np
from subprob.longterm import LongTermAllocator
from subprob.shortterm import ShortTermAllocator


class bigProblems :
    def __init__(self, RUs, slices, K, H, T_slot, num_slot, w_reward, T_max, sla_slices) -> None:
        self.RUs = RUs
        self.slices = slices
        self.K = K
        self.H = H
        self.T_slot = T_slot
        self.num_slot = num_slot
        self.w_reward = w_reward
        self.T_max = T_max
        self.sla_slices = sla_slices

    def solveTwoProblem(self):
        # Long-term allocation
        allocator = LongTermAllocator(RUs=self.RUs, slices=self.slices,
                                    K=self.K, H=self.H, T_slot=self.T_slot, num_slot=self.num_slot, 
                                    w_reward=self.w_reward, T_max=self.T_max, sla_slices=self.sla_slices)
        longtime, longcheck, longterm_results = allocator.solve()

        allocator.debug_allocation()

        shortAllocator = ShortTermAllocator(RUs=self.RUs, slices=self.slices,
                                    K=self.K, H=self.H, T_slot=self.T_slot, num_slot=self.num_slot, 
                                    longterm_results=longterm_results, 
                                    T_max=self.T_max, w_reward=self.w_reward)
        shorttime, shortcheck, Probmetrics = shortAllocator.solve()

        return shorttime + longtime, longcheck, shortcheck, Probmetrics
