import math
from typing import Dict, Sequence
import torch
from torch import nn
import torch.nn.functional as F
import collections
from typing import Callable

StepData = collections.namedtuple(
    'StepData',
    ('obs', 'next_obs', 'privileged_obs', 'action', 'reward', 'done', 'truncation'))

def sd_map(f: Callable[..., torch.Tensor], *sds) -> StepData:
  """Map a function over each field in StepData."""
  items = {}
  keys = sds[0]._asdict().keys()
  for k in keys:
    items[k] = f(*[sd._asdict()[k] for sd in sds])
  return StepData(**items)

class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.td = None
        
    def add_trajectory_data(self, new_td):
        """Add new trajectory data to the buffer."""
        if self.td is None:
            self.td = new_td
        else:
            # Concatenate along the batch dimension
            def concat_buffers(old, new):
                return torch.cat([old, new], dim=0)
            self.td = sd_map(concat_buffers, self.td, new_td)
            
            # Trim buffer if it exceeds max size
            if self.td.obs.shape[0] > self.max_size:
                def trim_buffer(data):
                    return data[-self.max_size:]
                self.td = sd_map(trim_buffer, self.td)
    
    def sample(self, batch_size):
        """Sample a batch of transitions from the buffer."""
        if self.td is None or self.td.obs.shape[0] < batch_size:
            return None
            
        indices = torch.randint(0, self.td.obs.shape[0], (batch_size,))
        
        def sample_data(data):
            return data[indices]
            
        return sd_map(sample_data, self.td)
    
    def current_sz(self):
        """Return current size of the buffer."""
        if self.td is None:
            return 0
        # Check all components have same size
        sizes = [data.shape[0] for data in self.td._asdict().values()]
        assert all(s == sizes[0] for s in sizes), "Buffer components have different sizes"
        return sizes[0]

class Agent(nn.Module):

  def __init__(self,
               policy_layers: Sequence[int],
               value_layers: Sequence[int],
               discounting: float = 0.99,
               tau:float = 0.005,
               device:str = 'cpu'):
    super(Agent, self).__init__()
    
    policy = []
    for w1, w2 in zip(policy_layers, policy_layers[1:]):
      policy.append(nn.Linear(w1,w2))
      policy.append(nn.ReLU())
    policy.pop()

    # Behaviour policy.
    self.policy_b = nn.Sequential(*policy)

    # Target policy.
    self.policy_t = nn.Sequential(*policy)
    
    # Copy the weights from the behaviour policy to the target policy.
    self.policy_t.load_state_dict(self.policy_b.state_dict())

    # Value function.
    value = []
    for w1, w2 in zip(value_layers, value_layers[1:]):
      value.append(nn.Linear(w1,w2))
      value.append(nn.ReLU())
    value.pop()
    
    # Behaviour value function.
    self.value_b = nn.Sequential(*value)

    # Target value function.
    self.value_t = nn.Sequential(*value)
    
    # Copy the weights from the behaviour value function to the target value function.
    self.value_t.load_state_dict(self.value_b.state_dict())

    # Hyperparameters.
    self.gamma = discounting
    self.tau = tau

    self.device = device

  @torch.jit.export
  def action_postprocess(self, x):
    return torch.tanh(x)
  
  @torch.jit.export
  def get_logits_action(self, observation):
    logits = self.policy_b(observation)
    action = self.action_postprocess(logits)
    return logits, action
   
  @torch.jit.export
  def critic_loss(self, buffer_batch:Dict[str, torch.Tensor]):
    
    observation = buffer_batch['obs']
    action = buffer_batch['action']
    reward = buffer_batch['reward'].unsqueeze(-1)  # Add dimension to match [256, 1]
    next_observations = buffer_batch['next_obs']
    done_flag = buffer_batch['done'].unsqueeze(-1)  # Add dimension to match [256, 1]
    
    with torch.no_grad():
      next_action = self.action_postprocess(self.policy_t(next_observations))
      next_value_input = torch.cat((next_observations,next_action), dim=1)
      next_discounted_return = self.value_t(next_value_input)
      y = reward + (1 - done_flag) * self.gamma * next_discounted_return
    state_action_pair = torch.cat([observation, action], dim=1)
    discounted_return = self.value_b(state_action_pair)
    v_loss = F.mse_loss(input=discounted_return, target=y)
    return v_loss
  
  @torch.jit.export
  def policy_loss(self, buffer_batch:Dict[str, torch.Tensor]):
    observation = buffer_batch['obs']

    _,actions_b = self.get_logits_action(observation)
    state_action_pair = torch.cat([observation,actions_b], dim=1)
    critic_values = self.value_b(state_action_pair)
    p_loss = -critic_values.mean()
    return p_loss
