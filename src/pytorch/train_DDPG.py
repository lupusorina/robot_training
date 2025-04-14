import gymnasium as gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
from torch.distributions import Normal

from envs.biped_np import *
from MLE_train import ContinuousActionNN

from tqdm import tqdm
import os
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

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
    def __init__(self, state_dim, action_dim, policy_lr=4e-4, device='cpu'):
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
    def __init__(self, state_dim, action_dim, value_lr=4e-3, device='cpu'):
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


RESULTS = 'results'
if not os.path.exists(RESULTS):
    os.makedirs(RESULTS)
time_now = datetime.now().strftime('%Y%m%d-%H%M%S')
if not os.path.exists(os.path.join(RESULTS, time_now)):
    os.makedirs(os.path.join(RESULTS, time_now))
FOLDER_RESULTS = os.path.join(RESULTS, time_now)
ABS_FOLDER_RESUlTS = os.path.abspath(FOLDER_RESULTS)
FOLDER_RESTORE_CHECKPOINT = os.path.abspath(RESULTS + '/20250318-173452/000151388160')
print(f"Saving results to {ABS_FOLDER_RESUlTS}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

NB_TRAINING_CYCLES = 1
NOISE = 'OrnsteinUhlenbeck' # 'Gaussian' or 'OrnsteinUhlenbeck'
PLOTTING = False

if __name__ == '__main__':
    
    env = Biped(visualize=False)
    env.reset()

    state_dim = env.observation_size[0]
    action_dim = env.action_size
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")

    mle_weight_path = 'MLE_model/P(a|s).pth'
    expert = ContinuousActionNN(state_dim=state_dim,action_dim=action_dim)
    expert.load_state_dict(torch.load(mle_weight_path, map_location=device, weights_only=True))
    expert.eval()

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
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)

        behavior_policy = Policy(state_dim=state_dim, action_dim=action_dim, policy_lr=1e-3, device=device)
        target_policy = Policy(state_dim=state_dim, action_dim=action_dim, policy_lr=1e-3, device=device)

        behavior_q = Value(state_dim=state_dim, action_dim=action_dim, value_lr=1e-3, device=device)
        target_q = Value(state_dim=state_dim, action_dim=action_dim, value_lr=1e-3, device=device)

        models = [behavior_policy, behavior_q]
        # for model in models:
        #     init_model_weights(model, seed=seed) # Sorina: I don't think this is needed

        target_policy.load_state_dict(behavior_policy.state_dict())
        target_q.load_state_dict(behavior_q.state_dict())


        agent = DDPG(policy_network=behavior_policy, target_policy=target_policy,
                    value_network=behavior_q, target_value_function=target_q,
                    discount_factor=discount_gamma, seed=seed, device=device)

        memory = DDPGMemory(state_dim=state_dim, action_dim=action_dim, buffer_length=buffer_length)


        obs,_ = env.reset()
        episodic_returns = []
        cumulative_reward = 0
        print( "... Training Starts ...")
        for t in tqdm(range(training_steps), desc=f"Cycle {cycles+1}", unit="step"):
            with torch.no_grad():
                robot_state = torch.tensor(obs, dtype=torch.float32, device=device)
                # print(f"Robot state: {robot_state.shape}")
                action = behavior_policy.forward(robot_state)
                expert_action,_ = expert(robot_state)
                # print(expert_action)
                noisy_action = action.cpu().numpy() + expert_action.cpu().numpy()
                
                # TODO: clip needs to be done differently, considering also qpos_joints
                # clipped_action = np.clip(noisy_action,
                #                             a_min=action_low,
                #                             a_max=action_high)

                obs_, reward, termination, truncation, _ = env.step(noisy_action)
                done = termination or truncation
                cumulative_reward += reward
                
                memory.add_sample(state=obs, action=noisy_action, reward=reward, next_state=obs_, done=done)

            if t>=warm_up and len(memory.states) >= batch_size:
                agent.train(memory_buffer=memory, batch_size=batch_size,epochs=1)
            
            if done:
                episodic_returns.append(cumulative_reward.item())
                print('Cumulative Reward: ', cumulative_reward.item())
                cumulative_reward = 0
                obs,_ = env.reset()
            else:
                obs = obs_.copy()

        for i in range(len(agent.pi_loss)):
            list_of_all_the_data.append({
                'cycle': cycles + 1,
                'policy_loss': agent.pi_loss[i],
                'q_loss': agent.q_loss[i],
                'return': episodic_returns[i] if i < len(episodic_returns) else np.nan,
            })

        # Plot the reward.
        fig, ax = plt.subplots(1, 1)
        ax.plot(episodic_returns)
        ax.set_title(f'{NOISE} added Noise')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Episodic Returns')
        plt.savefig(f'{ABS_FOLDER_RESUlTS}/{NOISE}_{cycles}.png')
        plt.close()

        # Save the results.
        df = pd.DataFrame(list_of_all_the_data)
        df.to_csv(f'{ABS_FOLDER_RESUlTS}/{NOISE}_{cycles}.csv', index=False)

        # Save the behaviour policy.
        torch.save(behavior_policy.state_dict(), f'{ABS_FOLDER_RESUlTS}/policy_{cycles}.pth')

    # Test the policy.
    env = None

    test_env = Biped()
    obs,_ = test_env.reset()
    rollout = []
    for _ in tqdm(range(10000)):
        # action = np.random.uniform(-1, 1, test_env.action_size)
        with torch.no_grad():
            action = behavior_policy.forward(torch.tensor(obs, dtype=torch.float32, device=device))
            action = action.cpu().numpy()
        action = np.clip(action, a_min=action_low, a_max=action_high)
        obs, rewards, done, _, _ = test_env.step(action)

        state = {
            'qpos': test_env.data.qpos.copy(),
            'qvel': test_env.data.qvel.copy(),
            'xfrc_applied': test_env.data.xfrc_applied.copy()
        }
        rollout.append(state)

    render_every = 1 # int.
    fps = 1/ test_env.sim_dt / render_every
    traj = rollout[::render_every]

    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[2] = True
    scene_option.geomgroup[3] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False

    frames = test_env.render(
        traj,
        camera="track",
        scene_option=scene_option,
        width=640,
        height=480,
    )

    # media.show_video(frames, fps=fps, loop=False)
    # ABS_FOLDER_RESUlTS = epath.Path(RESULTS_FOLDER_PATH) / latest_folder
    # NOTE: To make the code run, you need to call: MUJOCO_GL=egl python3 biped_np.py
    media.write_video(f'{ABS_FOLDER_RESUlTS}/joystick_testing.mp4', frames, fps=fps)
    print('Video saved')
