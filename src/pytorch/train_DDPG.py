import gymnasium as gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
from torch.distributions import Normal

from envs.biped_np import *

from tqdm import tqdm
import os
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

import argparse
import json

import os
from moviepy.video.io import ImageSequenceClip

from MLE_train import ContinuousActionNN
from CNF_train import ConditionalNormalizingFlow

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
    def __init__(self, state_dim:int, action_dim:int, BUFFER_LENGTH:int):
        self.memory_BUFFER_LENGTH = BUFFER_LENGTH
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((BUFFER_LENGTH, state_dim), dtype=np.float32)
        self.actions = np.zeros((BUFFER_LENGTH, action_dim), dtype=np.float32)
        self.rewards = np.zeros((BUFFER_LENGTH, 1), dtype=np.float32)
        self.next_states = np.zeros((BUFFER_LENGTH, state_dim), dtype=np.float32)
        self.dones = np.zeros((BUFFER_LENGTH, 1), dtype=np.float32)

    def add_sample(self, state, action, reward, next_state, done):
        idx = self.ptr
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        
        self.ptr = (self.ptr + 1) % self.memory_BUFFER_LENGTH
        self.size = min(self.size + 1, self.memory_BUFFER_LENGTH)
    
    def sample_memory(self, BATCH_SIZE):
        indices = np.random.randint(0, self.size, size=BATCH_SIZE)
        return (
            torch.tensor(self.states[indices], dtype=torch.float32),
            torch.tensor(self.actions[indices], dtype=torch.float32),
            torch.tensor(self.rewards[indices], dtype=torch.float32),
            torch.tensor(self.next_states[indices], dtype=torch.float32),
            torch.tensor(self.dones[indices], dtype=torch.float32)
        )    

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, action_lim=1.0, policy_lr=4e-4, device='cpu'):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = torch.tensor(action_lim, dtype=torch.float32, device=device)

        self.linear_layer_1 = nn.Linear(self.state_dim, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.action_layer = nn.Linear(300, self.action_dim)

        self.optimizer = optim.Adam(self.parameters(), policy_lr)
        self.to(device)

    def forward(self, inputs):
        x = functional.relu(self.linear_layer_1(inputs))
        x = functional.relu(self.linear_layer_2(x))
        
        return self.action_lim * torch.tanh(self.action_layer(x))

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


    def train(self,memory_buffer:DDPGMemory, BATCH_SIZE:int, epochs:int):

        models = [self.pi, self.pi_t, self.q, self.q_t]
        for model in models:
            model.train()
        
        for epoch in range(epochs):
            # sample a batch from memory
            sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = memory_buffer.sample_memory(BATCH_SIZE)

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


if __name__ == '__main__':

    DEFAULT_PARAMS_BIPED = {
        'noise': 'OrnsteinUhlenbeck',
        'training_steps': 1000,
        'warm_up': 0,
        'discount_factor': 0.99,
        'buffer_length': 15000,
        'batch_size': 100,
    }

    DEFAULT_PARAMS_PENDULUM = {
        'noise': 'Gaussian',
        'training_steps': 30_000,
        'warm_up': 0,
        'discount_factor': 0.99,
        'buffer_length': 15000,
        'batch_size': 100,
    }
    DEFAULT_PARAMS_ANT = DEFAULT_PARAMS_PENDULUM

    parser = argparse.ArgumentParser(description='Train DDPG on a given environment')
    parser.add_argument('--env_name', type=str, default='Ant-v4',
                      choices=['Ant-v5', 'Pendulum-v1', 'biped'],
                      help='Name of the environment to train on')
    parser.add_argument('--nb_training_cycles', type=int, default=1,
                        help='Number of training cycles')
    parser.add_argument('--expert_model', type=str, default=None,
                        help='Path to expert model in TrainedExperts folder to load for testing or fine-tuning')


    args = parser.parse_args()
    if args.env_name == 'biped':
        params = DEFAULT_PARAMS_BIPED
    elif args.env_name == 'Pendulum-v1':
        params = DEFAULT_PARAMS_PENDULUM
    elif args.env_name == 'Ant-v4':
        params = DEFAULT_PARAMS_ANT
    else:
        raise ValueError(f"Environment {args.env_name} not supported")
    
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    for key, value in params.items():
        print(f"{key}: {value}")

    # Save the arguments in a file.
    with open(os.path.join(ABS_FOLDER_RESUlTS, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f)
    with open(os.path.join(ABS_FOLDER_RESUlTS, 'params.json'), 'w') as f:
        json.dump(params, f)

    ENV_NAME = args.env_name
    NOISE = params['noise']
    TRAINING_STEPS = params['training_steps']
    WARM_UP = params['warm_up']
    DISCOUNT_FACTOR = params['discount_factor']
    BUFFER_LENGTH = params['buffer_length']
    BATCH_SIZE = params['batch_size']
    NB_TRAINING_CYCLES = args.nb_training_cycles
    EXPERT_MODEL = args.expert_model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    
    if ENV_NAME == 'Pendulum-v1' or ENV_NAME == 'Ant-v5':
        env = gym.make(ENV_NAME)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_low = env.action_space.low
        action_high = env.action_space.high
    elif ENV_NAME == 'biped': #is this is ok for action clipping?
        env = Biped(visualize=False)
        state_dim = env.observation_size[0]
        action_dim = env.action_size
        action_low = env._soft_q_j_min
        action_high = env._soft_q_j_max

    # Load expert model if provided
        if EXPERT_MODEL:
            experts_dir = os.path.join(os.path.dirname(__file__), 'TrainedExperts')
            expert_path = os.path.join(experts_dir, f'{EXPERT_MODEL}.pth')
            print(f"Loading expert model from: {expert_path}")
            if EXPERT_MODEL == 'MLE/{ENV_NAME}':
                expert = ContinuousActionNN(state_dim=state_dim, action_dim=action_dim)
            elif EXPERT_MODEL == 'CNF/{ENV_NAME}':
                expert = ConditionalNormalizingFlow(condition_dim=state_dim, n_flows=10, latent_dim=action_dim)
        else:
            if NOISE == 'Gaussian':
                noise = Normal(loc=0, scale=0.2)
            elif NOISE == 'OrnsteinUhlenbeck':
                noise = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.2, base_scale=0.1)
            else:
                raise ValueError('Noise must be either Gaussian or OrnsteinUhlenbeck')



    env.reset()

    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    print(f"Action low: {action_low}, Action high: {action_high}")

    list_of_all_the_data = []

    for cycles in range(NB_TRAINING_CYCLES):
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)

        behavior_policy = Policy(state_dim=state_dim, action_dim=action_dim, action_lim=action_high, policy_lr=1e-3, device=device)
        target_policy = Policy(state_dim=state_dim, action_dim=action_dim, action_lim=action_high, policy_lr=1e-3, device=device)

        behavior_q = Value(state_dim=state_dim, action_dim=action_dim, value_lr=1e-3, device=device)
        target_q = Value(state_dim=state_dim, action_dim=action_dim, value_lr=1e-3, device=device)

        models = [behavior_policy, behavior_q]
        for model in models:
            init_model_weights(model, seed=seed) # Sorina: I don't think this is needed

        target_policy.load_state_dict(behavior_policy.state_dict())
        target_q.load_state_dict(behavior_q.state_dict())


        agent = DDPG(policy_network=behavior_policy, target_policy=target_policy,
                    value_network=behavior_q, target_value_function=target_q,
                    discount_factor=DISCOUNT_FACTOR, seed=seed, device=device)

        memory = DDPGMemory(state_dim=state_dim, action_dim=action_dim, BUFFER_LENGTH=BUFFER_LENGTH)


        obs, _ = env.reset()
        episodic_returns = []
        cumulative_reward = 0

        for t in tqdm(range(TRAINING_STEPS), desc=f"Cycle {cycles+1}", unit="step"):
            with torch.no_grad():
                action = behavior_policy.forward(torch.tensor(obs, dtype=torch.float32, device=device))
                if EXPERT_MODEL == 'MLE/{ENV_NAME}':
                    if ENV_NAME == 'Pendulum-v1':
                        theta = np.arctan2(obs[1], obs[0])
                        theta_dot = obs[2]
                        new_state = np.array([theta, theta_dot]).reshape(1, -1)
                        expert_action = expert.sample(torch.tensor(new_state, dtype=torch.float32, device=device))
                    else:
                        expert_action = expert.sample(torch.tensor(obs, dtype=torch.float32, device=device))
                elif EXPERT_MODEL == 'CNF/{ENV_NAME}':
                    expert_action = expert.sample(num_samples=1, condition=torch.tensor(obs, dtype=torch.float32, device=device))
                else:
                    expert_action = noise.sample(action.shape).to(device)

                # print(expert_action)
                noisy_action = action.cpu().numpy() + expert_action.cpu().numpy()
                
                # TODO: clip needs to be done differently, considering also qpos_joints
                if ENV_NAME != 'biped': 
                    noisy_action = np.clip(noisy_action,
                                                a_min=action_low,
                                                a_max=action_high)

                obs_, reward, termination, truncation, _ = env.step(noisy_action)
                done = termination or truncation
                cumulative_reward += reward
                
                memory.add_sample(state=obs, action=noisy_action, reward=reward, next_state=obs_, done=done)

            if t>=WARM_UP and len(memory.states) >= BATCH_SIZE:
                agent.train(memory_buffer=memory, BATCH_SIZE=BATCH_SIZE,epochs=1)
            
            if done:
                episodic_returns.append(cumulative_reward.item())
                print('Cumulative Reward: ', cumulative_reward.item())
                cumulative_reward = 0
                obs, _ = env.reset()
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

    ### Testing the policy.

    def run_test(env_name):
        test_env = gym.make(env_name, render_mode='rgb_array')
        test_env.reset()

        frames = []
        obs, _= env.reset()
        for _ in range(200):
            action = behavior_policy.forward(torch.tensor(obs, dtype=torch.float32, device=device))
            action = action.detach().numpy()
            obs, reward, terminated, truncated, info = test_env.step(action)
            frames.append(test_env.render())
        test_env.close()

        video_path = os.path.join(ABS_FOLDER_RESUlTS, f"{env_name}_video.mp4")
        clip = ImageSequenceClip.ImageSequenceClip(frames, fps=30)
        clip.write_videofile(video_path, codec="libx264")
        print(f"Saved video for episode")

    # Load the policy.
    policy_path = f'{ABS_FOLDER_RESUlTS}/policy_0.pth'
    behavior_policy.load_state_dict(torch.load(policy_path, map_location=device, weights_only=True))
    behavior_policy.eval()

    # Run the test.
    if ENV_NAME == 'biped':
        test_env = Biped(visualize=False)
        obs, _ = test_env.reset()
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

            if done:
                print('Robot fell down')
                break

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
        # NOTE: To make the code run, you need to call: MUJOCO_GL=egl python3 train_DDPG.py
        media.write_video(f'{ABS_FOLDER_RESUlTS}/behaviour_robot_after_training.mp4', frames, fps=fps)
        print('Video saved')

    elif ENV_NAME == 'Pendulum-v1' or ENV_NAME == 'Ant-v5':
        run_test(env_name=ENV_NAME)
