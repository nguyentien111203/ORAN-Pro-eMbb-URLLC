import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from combine.common.common import MLP, GaussianPolicy, ReplayBuffer


class SACAgentBM:
    def __init__(self, state_dim, num_rus, num_bwp_ru, num_slices, train_cons):
        """
        Agent cho SAC
        int state_dim : số chiều của state
        int num_rus : số RU
        list(int) num_bwp_ru : số BWP mỗi RU 
        int num_slices : số slice
        """

        self.state_dim = state_dim
        self.num_rus = num_rus
        self.num_bwp_ru = num_bwp_ru
        self.num_slices = num_slices    
        self.device = train_cons["device"]
        self.gamma = train_cons["gamma"]
        self.tau = train_cons["tau"]
        self.alpha = train_cons["alpha"]
        self.action_scale = train_cons["action_scale"]
        self.action_bias = train_cons["action_bias"]
        self.lr = train_cons["lr"]
        self.actor_lr = train_cons["actor_lr"]
        self.critic_lr = train_cons["critic_lr"]
        self.alpha_lr = train_cons["alpha_lr"]

        self.action_dim = np.sum(num_bwp_ru) * num_slices

        # --- Actor & Critics
        self.actor = GaussianPolicy(state_dim, self.action_dim,
                                    action_scale=self.action_scale,
                                    action_bias=self.action_bias).to(self.device)

        # Critics Q1, Q2
        q_input_dim = state_dim + self.action_dim  
        self.critic_1 = MLP(q_input_dim, 1).to(self.device)
        self.critic_1.init_weights()
        self.critic_2 = MLP(q_input_dim, 1).to(self.device)
        self.critic_2.init_weights()
        self.target_critic_1 = MLP(q_input_dim, 1).to(self.device)
        self.target_critic_1.init_weights()
        self.target_critic_2 = MLP(q_input_dim, 1).to(self.device)
        self.target_critic_2.init_weights()

        # copy weights to targets
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_1_opt = optim.Adam(self.critic_1.parameters(), lr=self.critic_lr)
        self.critic_2_opt = optim.Adam(self.critic_2.parameters(), lr=self.critic_lr)

        # --- Replay buffer (same as before)
        self.replay_buffer = ReplayBuffer(forSAC=True)

        # --- Entropy tuning (correct, register log_alpha as Parameter)
        self.target_entropy = -float(self.action_dim)

        init_alpha = float(self.alpha) if hasattr(self, "alpha") else float(0.3)
        # register as nn.Parameter so it's in model.parameters()/state_dict()
        self.log_alpha = torch.nn.Parameter(
            torch.tensor(np.log(init_alpha), dtype=torch.float32, device=self.device)
        )
        # scalar float for immediate use in ops
        self.alpha = float(self.log_alpha.exp().item())

        # optimizer for log_alpha
        self.alpha_opt = optim.Adam([self.log_alpha], lr=self.alpha_lr)

    def select_action(self, state):

        state = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device
        )

        with torch.no_grad():
            action_sample, _, _ = self.actor.sample(state)

        # [-1,1] -> [0,1]
        raw_action = (action_sample + 1.0) / 2.0

        action = raw_action.reshape(
            self.num_rus,
            max(self.num_bwp_ru),
            self.num_slices
        )

        # Mỗi slice chỉ chọn một BWP
        for r in range(self.num_rus):
            for s in range(self.num_slices):

                best_bwp = torch.argmax(action[r, :, s]).item()

                keep = action[r, best_bwp, s].clone()

                action[r, :, s] = 0.0
                action[r, best_bwp, s] = keep

        # Chuẩn hóa để dùng hết PRB của từng BWP
        for r in range(self.num_rus):
            for b in range(self.num_bwp_ru[r]):

                total = action[r, b].sum()

                if total > 1e-8:
                    action[r, b] /= total

        return action.detach().cpu().numpy()

    def update(self, step, policy_delay, last_actor_loss, batch_size, debug=False):
        """
        Stable SAC update with per-dim entropy tuning.
        Returns: actor_loss_scalar (float), critic_loss_scalar (float), updated_last_actor_loss (torch.Tensor scalar)
        """
        if len(self.replay_buffer) < batch_size:
            return 0.0, 0.0, last_actor_loss

        # ----------------------------
        # Sample batch and convert to tensors
        # ----------------------------
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        # local alpha value (do NOT mutate self.alpha here)
        alpha_val = float(self.log_alpha.exp().clamp(1e-8, 10.0))

        # ----------------------------
        # 1) Compute target Q value (use next actions)
        # ----------------------------
        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor.sample(next_states)  # next_log_pi expected [B,1] (sum)
            # ensure shape and clamp for safety
            next_log_pi = next_log_pi.view(-1, 1).clamp(min=-100.0, max=100.0)

            target_q1 = self.target_critic_1(torch.cat([next_states, next_actions], dim=-1))
            target_q2 = self.target_critic_2(torch.cat([next_states, next_actions], dim=-1))
            target_q_min = torch.min(target_q1, target_q2)

            target_q = target_q_min - alpha_val * next_log_pi
            target_q = rewards + (1.0 - dones) * self.gamma * target_q

            # optional clamp if you keep seeing huge numbers (uncomment while debugging)
            # target_q = torch.clamp(target_q, -1e6, 1e6)

        # ----------------------------
        # 2) Compute current Q and critic loss
        # ----------------------------
        current_q1 = self.critic_1(torch.cat([states, actions], dim=-1))
        current_q2 = self.critic_2(torch.cat([states, actions], dim=-1))


        critic_loss_fn = nn.SmoothL1Loss()
        critic_1_loss = critic_loss_fn(current_q1, target_q)
        critic_2_loss = critic_loss_fn(current_q2, target_q)
        critic_loss = 0.5 * (critic_1_loss + critic_2_loss)

        # Guard NaN/Inf
        if not torch.isfinite(critic_1_loss).all() or not torch.isfinite(critic_2_loss).all():
            if debug:
                print("Skipping update: critic loss NaN/Inf", critic_1_loss, critic_2_loss)
            return 0.0, 0.0, last_actor_loss

        # ----------------------------
        # 3) Update critics
        # ----------------------------
        self.critic_1_opt.zero_grad(set_to_none=True)
        critic_1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), max_norm=0.5)
        self.critic_1_opt.step()

        self.critic_2_opt.zero_grad(set_to_none=True)
        critic_2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), max_norm=0.5)
        self.critic_2_opt.step()

        # ----------------------------
        # 4) Actor update (delayed)
        # ----------------------------
        actor_loss_tensor = torch.tensor(0.0, device=self.device)

        # sample new actions & log_prob_sum & mean for current states
        # IMPORTANT: sample() must return (action, log_prob_sum, mean)
        new_actions, log_pi_sum, mean = self.actor.sample(states)

        # enforce correct shape and clamp
        log_pi_sum = log_pi_sum.view(-1, 1).clamp(min=-100.0, max=100.0)

        if step % max(1, policy_delay) == 0:
            q1_new = self.critic_1(torch.cat([states, new_actions], dim=-1))
            q2_new = self.critic_2(torch.cat([states, new_actions], dim=-1))
            q_new_min = torch.min(q1_new, q2_new)

            # actor objective: minimize (alpha * log_pi_sum - Q)
            actor_loss_tensor = (alpha_val * log_pi_sum - q_new_min).mean()

            # small regularizers to avoid tanh saturation & very large means
            mean_penalty_coeff = 1e-4
            mean_penalty = mean.pow(2).mean() * mean_penalty_coeff
            action_penalty_coeff = 5e-4
            action_penalty = (new_actions / max(1.0, float(self.action_scale))).pow(2).mean() * action_penalty_coeff

            actor_loss_tensor = actor_loss_tensor + mean_penalty + action_penalty

            # guards before backward
            if not torch.isfinite(actor_loss_tensor).all():
                if debug:
                    print("Skipping actor update: NaN/Inf actor_loss")
                actor_loss_tensor = torch.tensor(0.0, device=self.device)
            else:
                # clip huge actor_loss (prevent numeric explosion)
                if torch.abs(actor_loss_tensor) > 1e6:
                    if debug:
                        print("Clamping huge actor_loss before backward:", float(actor_loss_tensor.item()))
                    actor_loss_tensor = actor_loss_tensor.clamp(-1e6, 1e6)

                self.actor_opt.zero_grad(set_to_none=True)
                actor_loss_tensor.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                self.actor_opt.step()

            # save last actor loss as detached scalar tensor
            last_actor_loss = actor_loss_tensor.detach().clone()
            if last_actor_loss.dim() != 0:
                last_actor_loss = last_actor_loss.mean().detach()
        else:
            # reuse last_actor_loss for logging only
            if isinstance(last_actor_loss, torch.Tensor):
                actor_loss_tensor = last_actor_loss.detach().clone().to(self.device)
                if actor_loss_tensor.dim() != 0:
                    actor_loss_tensor = actor_loss_tensor.mean()
            else:
                actor_loss_tensor = torch.tensor(float(last_actor_loss), device=self.device)

        # ----------------------------
        # 5) Dynamic target entropy & Alpha update (per-dim)
        # ----------------------------
        # compute per-dim log-prob from the current summed log prob
        log_pi_per_dim = (log_pi_sum / float(self.action_dim)).clamp(min=-10.0, max=5.0)

        with torch.no_grad():
            avg_per_dim_ent = -log_pi_per_dim.mean().item()
            self.target_entropy = -0.1 * float(np.tanh(avg_per_dim_ent))

        # clamp log_alpha (in-place) then compute alpha loss using per-dim log-prob
        with torch.no_grad():
            self.log_alpha.data.clamp_(-16.0, 2.0)

        target_entropy_tensor = torch.tensor(self.target_entropy, dtype=torch.float32, device=self.device)
        alpha_loss = -(self.log_alpha * (log_pi_per_dim + target_entropy_tensor).detach()).mean()

        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        # update scalar alpha for use elsewhere (consistent with alpha_val usage)
        self.alpha = float(self.log_alpha.exp().clamp(1e-8, 10.0).item())

        # ----------------------------
        # 6) Soft target update
        # ----------------------------
        with torch.no_grad():
            for t_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
                t_param.copy_(t_param * (1.0 - self.tau) + param * self.tau)
            for t_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
                t_param.copy_(t_param * (1.0 - self.tau) + param * self.tau)

        # ----------------------------
        # 7) Debugging & return
        # ----------------------------
        if debug:
            try:
                stats = {
                    "actor_loss": float(actor_loss_tensor.item()),
                    "critic_loss": float(critic_loss.item()),
                    "Q_cur_mean": float(current_q1.mean().item()),
                    "target_Q_mean": float(target_q.mean().item()),
                    "alpha": float(self.alpha),
                    "log_pi_sum_mean": float(log_pi_sum.mean().item()),
                    "log_pi_per_dim_mean": float(log_pi_per_dim.mean().item())
                }
                print("SAC update stats:", stats)
                if log_pi_sum.mean().item() > 0:
                    print("WARNING: log_pi_sum > 0 (should not happen):", log_pi_sum.mean().item())
            except Exception:
                pass

        return float(actor_loss_tensor.item()), float(critic_loss.item()), last_actor_loss

