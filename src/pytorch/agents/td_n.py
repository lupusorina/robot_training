import gymnasium as gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
from torch.distributions import Normal


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
  

    
class TDNMemory:
    def __init__(self, order: int, state_dim: int, action_dim: int, buffer_length: int):
        self.memory_buffer_length = buffer_length
        self.ptr = 0
        self.size = 0
        self.n = order
        self.states = np.zeros((buffer_length, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_length, action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_length, 1), dtype=np.float32)
        self.next_states = np.zeros((buffer_length, state_dim), dtype=np.float32)
        self.dones = np.zeros((buffer_length, 1), dtype=np.float32)

        self.state_dim = state_dim
        self.action_dim = action_dim

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
        assert self.size >= self.n, "Not enough samples in memory to sample a full trajectory."
        start_idx_list = []
        # # print(self.ptr)
        for _ in range(batch_size):
            if self.size < self.memory_buffer_length:  # Buffer not fully filled
                # sample from any position that allows a full n-step trajectory
                idx = np.random.randint(0, self.size - self.n + 1)
            else:  # Buffer is full
                if np.random.choice([True, False]):
                    if self.ptr >= self.n:
                        # sample from any position that allows a full n-step trajectory
                        idx = np.random.randint(0, self.ptr - self.n + 1)
                    else:
                        idx = np.random.randint(self.ptr, self.memory_buffer_length + self.ptr - self.n)
                else:
                    idx = np.random.randint(self.ptr, self.memory_buffer_length)

            start_idx_list.append(idx)

        start_indices = np.array(start_idx_list)
        # # print(start_indices)
        # For each start index, get a trajectory of length n (or less if we hit the buffer boundary)
        batch_states = np.zeros((batch_size, self.n, self.state_dim), dtype=np.float32)
        batch_actions = np.zeros((batch_size, self.n, self.action_dim), dtype=np.float32)
        batch_rewards = np.zeros((batch_size, self.n, 1), dtype=np.float32)
        batch_next_states = np.zeros((batch_size, self.n, self.state_dim), dtype=np.float32)
        batch_dones = np.zeros((batch_size, self.n, 1), dtype=np.float32)

        batch_sample = 0
        for start_idx in start_indices:
            # # print('start_idx: ', start_idx)
            next_idx = start_idx
            for i in range(self.n):
                # Calculate next index in the trajectory
                batch_states[batch_sample, i,:] = self.states[next_idx]
                batch_actions[batch_sample, i,:] = self.actions[next_idx]
                batch_rewards[batch_sample, i,:] = self.rewards[next_idx]
                batch_next_states[batch_sample, i,:] = self.next_states[next_idx]
                batch_dones[batch_sample, i,:] = self.dones[next_idx]
                # # print('size of memory: ',self.size)
                # # print(next_idx)
                next_idx = (next_idx + 1) % self.size
            batch_sample += 1

        
        return (
            batch_states,
            batch_actions,
            batch_rewards,
            batch_next_states,
            batch_dones
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
        x = functional.relu(self.linear_layer_1(inputs))
        x = functional.relu(self.linear_layer_2(x))
        
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
        x = functional.relu(self.linear_layer_1(inputs))
        x = functional.relu(self.linear_layer_2(x))
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

class TDN:
    def __init__(self,order:int, policy_network:Policy, target_policy:Policy,
                 value_network:Value, target_value_function:Value, 
                 discount_factor:float, seed=None, device='cpu'):
        
        self.pi = policy_network.to(device=device)
        self.pi_t = target_policy.to(device=device)
        self.q = value_network.to(device=device)
        self.q_t = target_value_function.to(device=device)
        self.gamma = discount_factor
        self.pi_loss = []
        self.q_loss = []
        self.device = device
        self.seed = seed
        self.n = order

    
    def soft_update(self,target_network, network, tau):
        for target_param, source_param in zip(target_network.parameters(), network.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


    def train(self,memory_buffer:TDNMemory, batch_size:int, epochs:int):

        models = [self.pi, self.pi_t, self.q, self.q_t]
        for model in models:
            model.train()
        
        for epoch in range(epochs):
            # sample a batch from memory
            sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = memory_buffer.sample_memory(batch_size)

            sampled_states = torch.tensor(sampled_states, dtype=torch.float32, device=self.device)
            sampled_actions = torch.tensor(sampled_actions, dtype=torch.float32, device=self.device)
            sampled_rewards = torch.tensor(sampled_rewards, dtype=torch.float32, device=self.device).squeeze(-1)
            sampled_next_states = torch.tensor(sampled_next_states, dtype=torch.float32, device=self.device)
            sampled_dones = torch.tensor(sampled_dones, dtype=torch.float32, device=self.device).squeeze(-1)
            
            # sampled_actions = sampled_actions.to(self.device)
            # sampled_rewards = sampled_rewards.view(-1).to(self.device)
            # sampled_next_states = sampled_next_states.to(self.device)
            # sampled_dones = sampled_dones.view(-1).to(self.device)
            
            # compute target values
            with torch.no_grad():
                next_actions = self.pi_t.forward(sampled_next_states[:, -1, :])
                # print("next_actions shape: ", next_actions.shape)
                next_state_action_pairs = torch.cat([sampled_next_states[:,-1,:], next_actions], dim=1) #attention dim
                y = self.q_t.forward(next_state_action_pairs)
                # print("y shape: ", y.shape)
                # print("sampled_rewards shape: ", sampled_rewards.shape)
                # print("sampled_dones shape: ", sampled_dones.shape)
                # y = sampled_rewards + self.gamma * (1 - sampled_dones) * target_q_values
                for i in range(self.n-2, -1, -1):
                    y = sampled_rewards[:, i] + self.gamma * (1 - sampled_dones[:, i]) * y

            # compute critic loss
            # print("sampled_states shape: ", sampled_states[:,0,:].shape)
            # print("sampled_actions shape: ", sampled_actions[:,0,:].shape)
            state_action_pairs = torch.cat([sampled_states[:,0,:], sampled_actions[:,0,:]], dim=1)
            critic_values = self.q.forward(state_action_pairs)
            # print("critic_values shape: ", critic_values.shape)
            # print("y shape: ", y.shape)
            critic_loss = functional.mse_loss(critic_values, y)

            # optimization step (critic)
            self.q.optimizer.zero_grad()
            critic_loss.backward()
            # nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
            self.q.optimizer.step()

            # compute policy (actor) loss
            actions = self.pi.forward(sampled_states[:,0,:])
            state_action_pairs = torch.cat([sampled_states[:,0,:], actions], dim=1)
            critic_values = self.q.forward(state_action_pairs)

            policy_loss = -critic_values.mean()

            # optimization step (policy)
            self.pi.optimizer.zero_grad()
            policy_loss.backward()
            # nn.utils.clip_grad_norm_(self.pi.parameters(), 5.0)
            self.pi.optimizer.step()

            # update target networks
            self.soft_update(self.pi_t, self.pi, tau=0.005)
            self.soft_update(self.q_t, self.q, tau=0.005)

        
        self.pi_loss.append(policy_loss.detach().cpu().numpy().item())
        self.q_loss.append(critic_loss.detach().cpu().numpy().item())

