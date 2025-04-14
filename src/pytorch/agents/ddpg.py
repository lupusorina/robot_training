import math
from typing import Dict, Sequence
import torch
from torch import nn
import torch.nn.functional as F

class ReplayBuffer(nn.Module):
  def __init__(self, obs_dim, act_dim, max_sz=1_000_000, device='cpu'):
    super().__init__()
    self.device = device
    self.max_sz = max_sz

    self.register_buffer('ptr_buf', torch.zeros(1, dtype=torch.long))
    self.register_buffer('sz_buf', torch.zeros(1, dtype=torch.long))

    self.register_buffer('obs_buf',torch.zeros((max_sz, obs_dim), dtype=torch.float32, device=device))
    self.register_buffer('next_obs_buf',torch.zeros((max_sz, obs_dim), dtype=torch.float32, device=device))
    self.register_buffer('act_buf',torch.zeros((max_sz, act_dim), dtype=torch.float32, device=device))
    self.register_buffer('reward_buf', torch.zeros((max_sz, 1), dtype=torch.float32, device=device))
    self.register_buffer('done_buf', torch.zeros((max_sz, 1), dtype=torch.int, device=device))
  
  @torch.jit.unused
  def current_ptr(self)->int:
    return int(self.ptr_buf[0].item())
  
  @torch.jit.unused
  def current_sz(self)->int:
    return int(self.sz_buf[0].item())
  
  @torch.jit.export
  def store(self, obs:torch.Tensor, act:torch.Tensor, reward:torch.Tensor,
            next_obs:torch.Tensor, done:torch.Tensor):
    idx = int(self.ptr_buf[0].item())
    sz = int(self.sz_buf[0].item())

    self.obs_buf[idx] = torch.clone(obs)
    self.next_obs_buf[idx] = torch.clone(next_obs)
    self.act_buf[idx] = torch.clone(act)
    self.reward_buf[idx] = torch.clone(reward)
    self.done_buf[idx] = torch.clone(done)

    idx = (idx + 1)%self.max_sz #points to the next index and goes to 0 once we get to end of buffer
    sz = min(sz+1,self.max_sz) #current size of buffer --> when to start training.

    self.ptr_buf[0] = idx
    self.sz_buf[0] = sz

  @torch.jit.export
  def sample_batch(self, batch_sz:int=64):#, idxs:torch.Tensor=None): #rn all on GPU, if buffer to big put buffer on cpu send batch to gpu
    size_int = int(self.sz_buf[0].item())
    if size_int == 0:
      batch = dict(observations = self.obs_buf[:0],
                   next_observations = self.next_obs_buf[:0],
                   actions = self.act_buf[:0],
                   rewards = self.reward_buf[:0],
                   dones = self.done_buf[:0])
    else:
      # if idxs.all() == None:
      idxs = torch.randint(low=0, high = size_int, size=(batch_sz,))
      batch = dict(observations = self.obs_buf[idxs],
                   next_observations = self.next_obs_buf[idxs],
                   actions = self.act_buf[idxs],
                   rewards = self.reward_buf[idxs],
                   dones = self.done_buf[idxs])
    return batch#, idxs


class Agent(nn.Module):
  """ Standard DDPG implementation"""

  def __init__(self, policy_layers:Sequence[int],
               value_layers:Sequence[int], memory:ReplayBuffer,
               discounting:float,tau:float, reward_scaling:float,
               action_limit:float, device:str):
    super(Agent, self).__init__()
    
    policy = []
    for w1,w2 in zip(policy_layers, policy_layers[1:]):
      policy.append(nn.Linear(w1,w2))
      policy.append(nn.SiLU())
    policy.pop()

    self.policy_b = nn.Sequential(*policy)

    self.policy_t = nn.Sequential(*policy)
    self.policy_t.load_state_dict(self.policy_b.state_dict())

    value = []
    for w1, w2 in zip(value_layers, value_layers[1:]):
      value.append(nn.Linear(w1,w2))
      value.append(nn.SiLU())
    value.pop()
    
    self.value_b = nn.Sequential(*value)

    self.value_t = nn.Sequential(*value)
    self.value_t.load_state_dict(self.value_b.state_dict())


    self.gamma = discounting
    self.tau = tau
    self.act_lim = action_limit
    self.reward_scaling = reward_scaling
    self.device = device

    self.memory = memory

  @classmethod
  def action_postprocess(cls,x):
    return cls.act_lim*torch.tanh(x)
  
  @torch.jit.export
  def get_action(self, observation):
    logits = self.policy_b(observation)
    action = self.action_postprocess(logits)
    return logits, action
   
  @torch.jit.export
  def critic_loss(self, buffer_batch:Dict[str,torch.Tensor]):
    
    observation = buffer_batch['observations']
    action = buffer_batch['actions']
    reward = buffer_batch['rewards']
    next_observations = buffer_batch['next_observations']
    done_flag = buffer_batch['dones']
    
    with torch.no_grad():
      next_action = self.action_postprocess(self.policy_t(next_observations))
      next_value_input = torch.concatenate((next_observations,next_action), dim=1)
      next_discounted_return = self.value_t(next_value_input)
      y = reward + (1-done_flag) * self.gamma * next_discounted_return
   
    state_action_pair = torch.concatenate([observation,action], dim=1)
    discounted_return = self.value_b(state_action_pair)
    v_loss = F.mse_loss(input=discounted_return, target=y)
    return v_loss
  
  @torch.jit.export
  def policy_loss(self, buffer_batch:Dict[str, torch.Tensor]):
    observation = buffer_batch['observations']

    _,actions_b = self.get_action(observation)
    state_action_pair = torch.concatenate([observation,actions_b], dim=1)
    critic_values = self.value_b(state_action_pair)
    p_loss = -critic_values.mean()
    return p_loss
  
  # @torch.jit.export
  # def soft_update(self):
  #   with torch.no_grad():
  #     for params, target_params in zip(self.value_b.parameters(), self.value_t.parameters()):
  #       target_params.data.copy_(self.tau*params.data + (1.0-self.tau) * target_params.data)

  #     for params, target_params in zip(self.policy_b.parameters(), self.policy_t.parameters()):
  #       target_params.data.copy_(self.tau*params.data + (1.0-self.tau) * target_params.data)







  
    