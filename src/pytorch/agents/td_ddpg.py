import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

import numpy as np
import pandas as pd

import gymnasium as gym
from tqdm import tqdm

import os

class OrnsteinUhlenbeckNoise:
    def __init__(self,theta: float,sigma: float,base_scale: float,mean: float = 0,std: float = 1) -> None:
        super().__init__()
        self.state = 0
        self.theta = theta
        self.sigma = sigma
        self.base_scale = base_scale

        self.distribution = Normal(loc=torch.tensor(mean, dtype=torch.float32),
                                   scale=torch.tensor(std, dtype=torch.float32))

    def sample(self, size:torch.Size = torch.Size([1,1])) -> torch.Tensor:
        if hasattr(self.state, "shape") and self.state.shape != torch.Size(size):
            self.state = 0
        self.state += -self.state * self.theta + self.sigma * self.distribution.sample(size)

        return self.base_scale * self.state
  
class DDPGMemory:
    def __init__(self, state_dim:int, action_dim:int, buffer_length:int):
        self.memory_buffer_length = buffer_length
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((buffer_length, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_length, action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_length, 1), dtype=np.float32)
        self.next_states = np.zeros((buffer_length, state_dim), dtype=np.float32)
        self.dones = np.zeros((buffer_length, 1), dtype=np.float32)

    def add_sample(self, state, action, reward, next_state, done):
        idx = self.ptr
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        
        self.ptr = (self.ptr + 1) % self.memory_buffer_length
        self.size = min(self.size + 1, self.memory_buffer_length)
    
    def sample_memory(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.states[indices], dtype=torch.float32),
            torch.tensor(self.actions[indices], dtype=torch.float32),
            torch.tensor(self.rewards[indices], dtype=torch.float32),
            torch.tensor(self.next_states[indices], dtype=torch.float32),
            torch.tensor(self.dones[indices], dtype=torch.float32)
        )    

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, policy_lr=1e-3, action_lim=2.0,device='cpu'):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
    
        self.linear_layer_1 = nn.Linear(self.state_dim, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.action_layer = nn.Linear(300, self.action_dim)

        self.optimizer = optim.Adam(self.parameters(), policy_lr)
        if type(action_lim) is not torch.Tensor:
            action_lim = torch.tensor(action_lim, dtype=torch.float32)
        self.action_lim = action_lim
        self.to(device)
        
    def forward(self, inputs):
        x = F.relu(self.linear_layer_1(inputs))
        x = F.relu(self.linear_layer_2(x))
        
        return self.action_lim * torch.tanh(self.action_layer(x))

class Value(nn.Module):
    def __init__(self, state_dim, action_dim, value_lr=1e-3, device='cpu'):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.linear_layer_1 = nn.Linear(self.state_dim + self.action_dim, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.linear_layer_3 = nn.Linear(300, 1)

        self.optimizer = optim.Adam(self.parameters(), value_lr)
        self.to(device)
    
    def forward(self, inputs):
        x = F.relu(self.linear_layer_1(inputs))
        x = F.relu(self.linear_layer_2(x))
        return self.linear_layer_3(x).squeeze()

def init_model_weights(model:nn.Module, mean=0.0, std=0.1, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "weight" in name:
                nn.init.normal_(param, mean=mean, std=std)
            elif "bias" in name:
                nn.init.normal_(param, mean=mean, std=std)
class TD3:
    """Twin Delayed DDPG with minimal API changes. Re‑uses the same Policy and Value classes."""

    def __init__(
        self,
        policy_network: nn.Module,
        target_policy: nn.Module,
        value_network_1: nn.Module,
        target_value_1: nn.Module,
        value_network_2: nn.Module,
        target_value_2: nn.Module,
        discount_factor: float = 0.99,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        seed: int | None = None,
        device: str = "cpu",
    ):
        self.pi = policy_network.to(device)
        self.pi_t = target_policy.to(device)

        self.q1 = value_network_1.to(device)
        self.q1_t = target_value_1.to(device)
        self.q2 = value_network_2.to(device)
        self.q2_t = target_value_2.to(device)

        self.gamma = discount_factor
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.device = device
        self.total_it = 0  # counts gradient steps

        if seed is not None:
            torch.manual_seed(seed)

        # History for plotting
        self.pi_loss: list[float] = []
        self.q1_loss: list[float] = []
        self.q2_loss: list[float] = []

    # ---------------------------------------
    #   target parameter soft‑update helper
    # ---------------------------------------
    @staticmethod
    def _soft_update(target: nn.Module, source: nn.Module, tau: float):
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.data.copy_(tau * s_param.data + (1.0 - tau) * t_param.data)

    # ---------------------------------------------------------------------
    #   single training call (can be invoked every env step, like DDPG)
    # ---------------------------------------------------------------------
    def train(self, memory_buffer, batch_size: int, epochs: int = 1, tau: float = 0.005):
        models = [self.pi, self.pi_t, self.q1, self.q1_t, self.q2, self.q2_t]
        for m in models:
            m.train()

        for _ in range(epochs):
            (
                states,
                actions,
                rewards,
                next_states,
                dones,
            ) = memory_buffer.sample_memory(batch_size)

            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.view(-1).to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.view(-1).to(self.device)

            # ---------------------------------------------------------
            #  Critic targets   y = r + γ (1‑d) · min(Q₁ᵀ, Q₂ᵀ)(s', a')
            #  with target‑policy smoothing noise on a'.
            # ---------------------------------------------------------
            with torch.no_grad():
                noise = (
                    Normal(0, self.policy_noise)
                    .sample(actions.shape)
                    .clamp(-self.noise_clip, self.noise_clip)
                    .to(self.device)
                )
                next_actions = (self.pi_t(next_states) + noise).clamp(-1.0, 1.0)
                next_state_actions = torch.cat([next_states, next_actions], dim=1)

                q1_target = self.q1_t(next_state_actions)
                q2_target = self.q2_t(next_state_actions)
                min_q_target = torch.min(q1_target, q2_target)
                y = rewards + self.gamma * (1 - dones) * min_q_target

            # ----------------------------
            #  Update critics (every step)
            # ----------------------------
            sa = torch.cat([states, actions], dim=1)
            q1_current = self.q1(sa)
            q2_current = self.q2(sa)
            q1_loss = F.mse_loss(q1_current, y)
            q2_loss = F.mse_loss(q2_current, y)

            self.q1.optimizer.zero_grad()
            q1_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 5.0)
            self.q1.optimizer.step()

            self.q2.optimizer.zero_grad()
            q2_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 5.0)
            self.q2.optimizer.step()

            # Record critic losses once per call (mean of the two)
            self.q1_loss.append(q1_loss.detach().cpu().item())
            self.q2_loss.append(q2_loss.detach().cpu().item())

            # ---------------------------------------------------------
            #  Delayed actor & target updates
            # ---------------------------------------------------------
            self.total_it += 1
            if self.total_it % self.policy_freq == 0:
                # Actor update (maximize Q₁)
                actions_pi = self.pi(states)
                q1_pi = self.q1(torch.cat([states, actions_pi], dim=1))
                pi_loss = -q1_pi.mean()

                self.pi.optimizer.zero_grad()
                pi_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.pi.parameters(), 5.0)
                self.pi.optimizer.step()

                # Soft‑update all targets
                self._soft_update(self.pi_t, self.pi, tau)
                self._soft_update(self.q1_t, self.q1, tau)
                self._soft_update(self.q2_t, self.q2, tau)

                self.pi_loss.append(pi_loss.detach().cpu().item())

            # If actor not updated this step, append placeholders for alignment
            else:
                self.pi_loss.append(float("nan"))