import torch.nn as nn
from torch.distributions import Normal
import torch
import random
import numpy as np
from collections import deque


#  Networks
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(256, 256), activation=nn.ReLU):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), activation()]
            prev_dim = h
        layers += [nn.Linear(prev_dim, output_dim)] 
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
    def init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0.0)
    

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(256, 256),
                 log_std_min=-5.0, log_std_max=0.3,
                 action_scale=1.0, action_bias=0.0, device="cpu"):
        super().__init__()
        self.net = MLP(state_dim, 2 * action_dim, hidden_dims)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.register_buffer("action_scale", torch.tensor(action_scale).to(device))
        self.register_buffer("action_bias", torch.tensor(action_bias).to(device))

    def forward(self, state):
        mean_logstd = self.net(state)
        mean, log_std = torch.chunk(mean_logstd, 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        # ---- 1) Sample pre-tanh ----
        x_t = normal.rsample()

        # clamp pre-tanh to avoid extreme saturation
        x_t = torch.clamp(x_t, -8.0, 8.0)

        # ---- 2) Apply tanh squash ----
        y_t = torch.tanh(x_t)
        # avoid exact ±1.0 to keep log(1 - y^2) finite
        y_t = torch.clamp(y_t, -0.999, 0.999)

        # ---- 3) Compute correct log-prob with tanh correction ----
        # raw normal log_prob
        raw_lp = normal.log_prob(x_t)                     # [B, act_dim]
        raw_lp_sum = raw_lp.sum(dim=-1, keepdim=True)     # [B, 1]

        # tanh jacobian correction
        jac = torch.log(1 - y_t.pow(2) + 1e-6)            # <= 0
        jac = torch.clamp(jac, min=-3.0, max=0.0)         # avoid huge spikes
        jac_sum = jac.sum(dim=-1, keepdim=True)           # [B, 1]

        # final correct log_prob
        log_prob = raw_lp_sum - jac_sum                   # [B, 1]

        # ---- 4) Produce final action ----
        action = y_t * self.action_scale + self.action_bias

        # ---- 5) Debug (optional) ----
        #with torch.no_grad():
        #    frac_saturated = (y_t.abs() > 0.995).float().mean().item()
        #    max_abs_y = float(y_t.abs().max().item())
            #print(f"[DBG] tanh saturate fraction = {frac_saturated:.4f}, max|y| = {max_abs_y:.4f}")

        return action, log_prob, mean




# Replay Buffer
class ReplayBuffer:
    def __init__(self, forSAC, capacity=50000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        self.forSAC = forSAC

    def push(self, state, action, reward, next_state, done):
        # 0 đại diện cho việc không phân bổ
        state = np.array(state, dtype=np.float32).flatten()
        next_state = np.array(next_state, dtype=np.float32).flatten()
        action = np.array(action, dtype=np.float32).flatten()
        reward = float(reward)
        done = float(done)
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(action) if self.forSAC else torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done),
        )

    def __len__(self):
        return len(self.buffer)

