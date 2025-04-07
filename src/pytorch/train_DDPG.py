import gymnasium as gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
from torch.distributions import Normal

from biped_np import *

from tqdm import tqdm




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
    def __init__(self, state_dim, action_dim, policy_lr=1e-3, device='cpu'):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
    
        self.linear_layer_1 = nn.Linear(self.state_dim, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.action_layer = nn.Linear(300, self.action_dim)

        self.optimizer = optim.Adam(self.parameters(), policy_lr)
        self.to(device)
        
    def forward(self, inputs):
        x = functional.relu(self.linear_layer_1(inputs))
        x = functional.relu(self.linear_layer_2(x))
        
        return 2 * torch.tanh(self.action_layer(x))

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

class DDPG:
    def __init__(self, policy_network:Policy, target_policy:Policy,
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

    
    def soft_update(self,target_network, network, tau):
        for target_param, source_param in zip(target_network.parameters(), network.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


    def train(self,memory_buffer:DDPGMemory, batch_size:int, epochs:int):

        models = [self.pi, self.pi_t, self.q, self.q_t]
        for model in models:
            model.train()
        
        for epoch in range(epochs):
            # sample a batch from memory
            sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = memory_buffer.sample_memory(batch_size)

            sampled_states = sampled_states.to(self.device)
            sampled_actions = sampled_actions.to(self.device)
            sampled_rewards = sampled_rewards.view(-1).to(self.device)
            sampled_next_states = sampled_next_states.to(self.device)
            sampled_dones = sampled_dones.view(-1).to(self.device)
            
            # compute target values
            with torch.no_grad():
                next_actions = self.pi_t.forward(sampled_next_states)
                next_state_action_pairs = torch.cat([sampled_next_states, next_actions], dim=1)
                target_q_values = self.q_t.forward(next_state_action_pairs)
                y = sampled_rewards + self.gamma * (1 - sampled_dones) * target_q_values

            # compute critic loss
            state_action_pairs = torch.cat([sampled_states, sampled_actions], dim=1)
            critic_values = self.q.forward(state_action_pairs)
            critic_loss = functional.mse_loss(critic_values, y)

            # optimization step (critic)
            self.q.optimizer.zero_grad()
            critic_loss.backward()
            # nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
            self.q.optimizer.step()

            # compute policy (actor) loss
            actions = self.pi.forward(sampled_states)
            state_action_pairs = torch.cat([sampled_states, actions], dim=1)
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


class SparsePendulumRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, obs, action):
        # Take a step using the underlying environment
        cos_theta, sin_theta, thdot = obs[0], obs[1], obs[2]
        th = np.arctan2(sin_theta, cos_theta) 
        th = angle_normalize(th)
        cost = - (10*np.tanh(10*th**2) + 0.1*thdot**2 + 0.001*action**2)
        obs_, _, terminated, truncated, info = self.env.step(action)
        
        return obs_, cost.squeeze(), terminated, truncated, info

def angle_normalize(x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi



# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# print(f"Using device: {device}")
device = 'cpu'
NB_TRAINING_CYCLES = 1
NOISE = 'OrnsteinUhlenbeck' # 'Gaussian' or 'OrnsteinUhlenbeck'
PLOTTING = False


if __name__ == '__main__':
    
    if NOISE == 'Gaussian':
        noise = Normal(loc=0, scale=0.2)
    elif NOISE == 'OrnsteinUhlenbeck':
        noise = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.2, base_scale=0.1)
    else:
        raise ValueError('Noise must be either Gaussian or OrnsteinUhlenbeck')
    
    env = Biped()
    env.visualize_mujoco = True
    env.reset_model()
    
    

    state_dim = env.observation_size[0]
    action_dim = env.action_size
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    
    
    action_low = env._soft_q_j_min
    action_high = env._soft_q_j_max
    print(f"Action low: {action_low}, Action high: {action_high}")
    
    training_steps = 30000
    warm_up = 0
    discount_gamma = 0.99
    buffer_length = 15000
    batch_size = 100

    list_of_all_the_data = []

    for cycles in range(NB_TRAINING_CYCLES):
        seed_torch = np.random.randint(0, 2**32 - 1)
        torch.manual_seed(seed_torch)
        seed_np = np.random.randint(0, 2**32 - 1)
        np.random.seed(seed_np)
        print(f'\nUsing seed {seed_np} for numpy and {seed_torch} for torch')

        behavior_policy = Policy(state_dim=state_dim, action_dim=action_dim, policy_lr=1e-3)
        target_policy = Policy(state_dim=state_dim, action_dim=action_dim, policy_lr=1e-3)
          
        behavior_q = Value(state_dim=state_dim, action_dim=action_dim, value_lr=1e-3)
        target_q = Value(state_dim=state_dim, action_dim=action_dim, value_lr=1e-3)

        models = [behavior_policy, behavior_q]
        for model in models:
            init_model_weights(model, seed=seed_torch)

        target_policy.load_state_dict(behavior_policy.state_dict())
        target_q.load_state_dict(behavior_q.state_dict())

        
        agent = DDPG(policy_network=behavior_policy, target_policy=target_policy,
                    value_network=behavior_q, target_value_function=target_q,
                    discount_factor=discount_gamma, seed=seed_torch, device=device)
        
        memory = DDPGMemory(state_dim=state_dim, action_dim=action_dim, buffer_length=buffer_length)


        obs = env.reset_model()
        episodic_returns = []
        cumulative_reward = 0

        for t in tqdm(range(training_steps), desc=f"Cycle {cycles+1}", unit="step"):
            with torch.no_grad():
                action = behavior_policy.forward(torch.tensor(obs, dtype=torch.float32, device=device))
                expl_noise = noise.sample(action.shape)
                noisy_action = action.cpu().numpy() + expl_noise.cpu().numpy()
                clipped_action = np.clip(noisy_action,
                                            a_min=action_low,
                                            a_max=action_high)
                
                obs_, reward, termination, truncation, _ = env.step(clipped_action)
                done = termination or truncation
                cumulative_reward += reward
                
                memory.add_sample(state=obs, action=clipped_action, reward=reward, next_state=obs_, done=done)
            
            if t>=warm_up and len(memory.states) >= batch_size:
                agent.train(memory_buffer=memory, batch_size=batch_size,epochs=1)
            
            if done:
                episodic_returns.append(cumulative_reward)
                cumulative_reward = 0
                obs = env.reset_model()
            else:
                obs = obs_.copy()
        
        for i in range(len(agent.pi_loss)):
            list_of_all_the_data.append({
                'cycle': cycles + 1,
                'policy_loss': agent.pi_loss[i],
                'q_loss': agent.q_loss[i],
                'return': episodic_returns[i] if i < len(episodic_returns) else np.nan,
            })
        
    env = None

    biped = Biped()
    obs = biped.reset_model()
    for _ in tqdm(range(100000)):
        # action = np.random.uniform(-1, 1, biped.action_size)
        with torch.no_grad():
            action = behavior_policy.forward(torch.tensor(obs, dtype=torch.float32, device=device))
            action = action.cpu().numpy()
        action = np.clip(action, a_min=action_low, a_max=action_high)
        obs, rewards, done, _, _ = biped.step(action)

        # if done:
        #     print("Done!")
        #     break
    # df = pd.DataFrame(list_of_all_the_data)

    # DATA_FOLDER = 'Data/CSVs/Metrics/tests'
    # if not os.path.exists(DATA_FOLDER):
    #     os.makedirs(DATA_FOLDER)

    # df.to_csv(f'{DATA_FOLDER}/{NOISE}.csv', index=False)

    # print('Saved data to CSV')
    
    # # Plotting
    # if PLOTTING:
    #     print('Plotting...')
    #     fig, ax = plt.subplots(3, 1, sharex=False, figsize=(15, 8))
    #     plotter = DDPGMetrics(data=f'{DATA_FOLDER}/{NOISE}.csv', show=False, title=f'{NOISE} added Noise', smooth=2)
    #     plotter.plot_losses(ax=ax)
    #     plt.tight_layout()
    #     plt.show()
    