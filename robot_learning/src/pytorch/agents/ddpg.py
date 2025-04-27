import math
from typing import Dict, Sequence
import torch
from torch import nn
import torch.nn.functional as F
import collections
from typing import Callable

StepData = collections.namedtuple(
  'StepData',
  ('obs', 'next_obs', 'privileged_obs', 'next_privileged_obs', 'action', 'reward', 'done', 'truncation'))

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
                    # Generate random indices to keep
                    indices = torch.randperm(data.shape[0])[:self.max_size]
                    return data[indices]
                self.td = sd_map(trim_buffer, self.td)

    def sample(self, batch_size) -> StepData:
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

class Policy(nn.Module):
  def __init__(self, input_dim, output_dim, hidden_layers=[400, 300], device='cpu'):
    super().__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim

    # Create layers dynamically based on hidden_layers list
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_layers:
        layers.extend([
            nn.Linear(prev_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            nn.ReLU()
        ])
        prev_dim = hidden_dim

    # Add final action layer
    layers.append(nn.Linear(prev_dim, output_dim))

    # Create sequential model
    self.network = nn.Sequential(*layers)
    self.to(device)

  def forward(self, inputs):
    return torch.tanh(self.network(inputs))

class Value(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=[400, 300], device='cpu'):
      super().__init__()
      self.input_dim = input_dim
      self.output_dim = output_dim

      # Create layers dynamically based on hidden_layers list
      layers = []
      prev_dim = input_dim
      for hidden_dim in hidden_layers:
          layers.extend([
              nn.Linear(prev_dim, hidden_dim),
              # nn.LayerNorm(hidden_dim),
              nn.ReLU()
          ])
          prev_dim = hidden_dim

      # Add final value layer
      layers.append(nn.Linear(prev_dim, output_dim))

      # Create sequential model
      self.network = nn.Sequential(*layers)
      self.to(device)

    def forward(self, inputs):
      return self.network(inputs)

def init_model_weights(model:nn.Module, low=0.0, high=0.01, seed=42):
  if seed is not None:
      torch.manual_seed(seed)
  for name, param in model.named_parameters():
      if param.requires_grad:
          if "weight" in name:
              nn.init.uniform_(param, a=low, b=high)
          elif "bias" in name:
              nn.init.constant_(param, val=low)
  return model

class Agent(nn.Module):

  def __init__(self,
               policy_layers: Sequence[int],
               value_layers: Sequence[int],
               discounting: float = 0.99, # From original paper.
               action_size: int = 1,
               tau:float = 0.001,         # From original paper.
               device:str = 'cpu',
               seed:int = 42):
    super(Agent, self).__init__()

    # Extract input/output dimensions and hidden layers
    policy_input_dim = policy_layers[0]
    policy_output_dim = policy_layers[-1]
    policy_hidden_layers = policy_layers[1:-1]
    
    value_input_dim = value_layers[0]
    value_output_dim = value_layers[-1]
    value_hidden_layers = value_layers[1:-1]
    
    # Create policy networks
    self.policy_b = Policy(policy_input_dim, policy_output_dim, policy_hidden_layers, device)
    self.policy_t = Policy(policy_input_dim, policy_output_dim, policy_hidden_layers, device)
    
    # Initialize target networks with behavior networks
    self.policy_t = soft_update(self.policy_t, self.policy_b, 1.0)  # tau=1.0 for complete copy

    # Create value networks
    self.value_b = Value(value_input_dim, value_output_dim, value_hidden_layers, device)
    self.value_t = Value(value_input_dim, value_output_dim, value_hidden_layers, device)

    # Initialize target networks with behavior networks
    self.value_t = soft_update(self.value_t, self.value_b, 1.0)  # tau=1.0 for complete copy

    # Hyperparameters.
    self.gamma = discounting
    self.tau = tau
    self.device = device

    # Normalization.
    self.num_steps = torch.zeros((), device=device)
    self.running_mean = torch.zeros(policy_input_dim, device=device)
    self.running_variance = torch.zeros(policy_input_dim, device=device)

    self.num_steps_privileged = torch.zeros((), device=device)
    self.running_mean_privileged = torch.zeros(value_input_dim - action_size, device=device)
    self.running_variance_privileged = torch.zeros(value_input_dim - action_size, device=device)

  @torch.jit.export
  def update_normalization(self, observation):
    # TODO: make this not duplicated
    self.num_steps += observation.shape[0] * observation.shape[1]
    input_to_old_mean = observation - self.running_mean
    mean_diff = torch.sum(input_to_old_mean / self.num_steps, dim=(0, 1))
    self.running_mean = self.running_mean + mean_diff
    input_to_new_mean = observation - self.running_mean
    var_diff = torch.sum(input_to_new_mean * input_to_old_mean, dim=(0, 1))
    self.running_variance = self.running_variance + var_diff

  @torch.jit.export
  def update_normalization_privileged(self, privileged_observation):
    self.num_steps_privileged += privileged_observation.shape[0] * privileged_observation.shape[1]
    input_to_old_mean_privileged = privileged_observation - self.running_mean_privileged
    mean_diff_privileged = torch.sum(input_to_old_mean_privileged / self.num_steps_privileged, dim=(0, 1))
    self.running_mean_privileged = self.running_mean_privileged + mean_diff_privileged
    input_to_new_mean_privileged = privileged_observation - self.running_mean_privileged
    var_diff_privileged = torch.sum(input_to_new_mean_privileged * input_to_old_mean_privileged, dim=(0, 1))
    self.running_variance_privileged = self.running_variance_privileged + var_diff_privileged

  @torch.jit.export
  def normalize(self, observation):
    variance = self.running_variance / (self.num_steps + 1.0)
    variance = torch.clip(variance, 1e-6, 1e6)
    return ((observation - self.running_mean) / variance.sqrt()).clip(-5, 5)

  @torch.jit.export
  def normalize_privileged(self, privileged_observation):
    variance = self.running_variance_privileged / (self.num_steps_privileged + 1.0)
    variance = torch.clip(variance, 1e-6, 1e6)
    return ((privileged_observation - self.running_mean_privileged) / variance.sqrt()).clip(-5, 5)

  @torch.jit.export
  def action_postprocess(self, x):
    return torch.tanh(x)

  @torch.jit.export
  def get_logits_action(self, observation):
    observation = self.normalize(observation)
    logits = self.policy_b(observation)
    action = self.action_postprocess(logits)
    return logits, action

  @torch.jit.export
  def critic_loss(self, buffer_batch:Dict[str, torch.Tensor]):
    sampled_observation = buffer_batch['obs']
    sampled_privileged_observation = buffer_batch['privileged_obs']
    sampled_observation = self.normalize(sampled_observation)
    sampled_privileged_observation = self.normalize_privileged(sampled_privileged_observation)
    sampled_action = buffer_batch['action']
    sampled_rewards = buffer_batch['reward']
    sampled_next_observations = buffer_batch['next_obs']
    sampled_next_privileged_observation = buffer_batch['next_privileged_obs']
    sampled_terminated = buffer_batch['done']
    sampled_truncated = buffer_batch['truncation']

    with torch.no_grad():
      next_action = self.action_postprocess(self.policy_t(sampled_next_observations))
      next_value_input = torch.cat((sampled_next_privileged_observation, next_action), dim=1)
      target_q_values = self.value_t(next_value_input)
      # print('target_q_values', target_q_values.T)
      # y_i = r_i + gamma * Q'(s_{i+1}, μ'(s_{i+1}))
      # r_i + gamma * Q'(s_{i+1}, μ'(s_{i+1})) - Q(s_i, a_i)
      target_values = sampled_rewards + (1 - sampled_terminated) * self.gamma * target_q_values
      # print('sampled_rewards', sampled_rewards.T)

    sampled_state_action_pair = torch.cat([sampled_privileged_observation, sampled_action], dim=1)
    critic_values = self.value_b(sampled_state_action_pair)
    v_loss = F.mse_loss(input=critic_values, target=target_values)

    return v_loss

  @torch.jit.export
  def policy_loss(self, buffer_batch: Dict[str, torch.Tensor]):
    sampled_observations = buffer_batch['obs']
    sampled_observations = self.normalize(sampled_observations)
    sampled_privileged_observations = buffer_batch['privileged_obs']
    sampled_privileged_observations = self.normalize_privileged(sampled_privileged_observations)
    _, actions_b = self.get_logits_action(sampled_observations)
    state_action_pair = torch.cat([sampled_privileged_observations, actions_b], dim=1)
    critic_values = self.value_b(state_action_pair)
    p_loss = -critic_values.mean()
    return p_loss


def soft_update(target_network, behavior_network, tau):
  """Update target network parameters with behavior network parameters using soft update."""
  with torch.no_grad():
    for target_param, behavior_param in zip(target_network.network.parameters(), behavior_network.network.parameters()):
      target_param.data.copy_(tau * behavior_param.data + (1.0 - tau) * target_param.data)
  return target_network

if __name__ == '__main__':

  # Test 1: Inference policy_b
  agent = Agent(policy_layers=[3, 32, 64, 128, 1], value_layers=[4, 32, 64, 128, 1], discounting=0.99, tau=0.001, device='cpu')
  B = 100
  obs = torch.randn((B, 3))
  print('obs', obs.shape)
  action = agent.policy_b(obs)
  print('action', action.shape)

  # Value function
  sampled_privileged_observations = torch.randn(B, 3)
  actions_b = torch.randn(B, 1)
  state_action_pair = torch.cat([sampled_privileged_observations, actions_b], dim=1)
  print('state_action_pair', state_action_pair.shape)
  critic_values = agent.value_b(state_action_pair)
  print('critic_values', critic_values.shape)
