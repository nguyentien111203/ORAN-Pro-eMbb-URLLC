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

        # Sample
        x_t = normal.rsample()
        x_t = torch.clamp(x_t, -8.0, 8.0)

        # Tanh squash (SAC chuẩn)
        y_t = torch.tanh(x_t)

        # Log probability với correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        action = y_t

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
        action_flat = np.concatenate([
            ue_alloc
            for slice_alloc in action
            for ue_alloc in slice_alloc
        ])
        done = 1 if done else 0
        reward = float(reward)
        done = float(done)   

        self.buffer.append((state, action_flat, reward, next_state, done))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # stack để đảm bảo cùng shape
        states = torch.FloatTensor(np.stack(states))
        next_states = torch.FloatTensor(np.stack(next_states))

        # action có thể nhiều chiều, nên cũng stack
        if self.forSAC:
            actions = torch.FloatTensor(np.stack(actions))
        else:
            actions = torch.LongTensor(np.stack(actions))

        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)


        return states, actions, rewards, next_states, dones


    def __len__(self):
        return len(self.buffer)

