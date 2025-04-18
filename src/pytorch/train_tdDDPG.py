import gymnasium as gym
import numpy as np

import torch
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

from agents.td_ddpg import *

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
        'training_steps': 10000,
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
    parser.add_argument('--env_name', type=str, default='Pendulum-v1',
                      choices=['Ant-v4', 'Pendulum-v1', 'biped'],
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
    EXPERT_MODEL = f'MLE/{ENV_NAME}'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    
    if ENV_NAME == 'Pendulum-v1' or ENV_NAME == 'Ant-v4':
        env = gym.make(ENV_NAME)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_low = env.action_space.low
        action_high = env.action_space.high
    elif ENV_NAME == 'biped': #is this is ok for action clipping?
        env = Biped(visualize=False)
        state_dim = 46#env.observation_size[0]//2
        action_dim = env.action_size
        action_low = env._soft_q_j_min
        action_high = env._soft_q_j_max

    # Load expert model if provided
    if EXPERT_MODEL:
        if ENV_NAME == 'Pendulum-v1':
                state_dim = state_dim-1
        experts_dir = os.path.join(os.path.dirname(__file__), 'TrainedExperts')
        expert_path = os.path.join(experts_dir, f'{EXPERT_MODEL}.pth')
        print(f"Loading expert model from: {expert_path}")
        if EXPERT_MODEL == f'MLE/{ENV_NAME}':
            print('mle expert')
            expert = ContinuousActionNN(state_dim=state_dim, action_dim=action_dim)
            expert.to(device=device)
        elif EXPERT_MODEL == f'CNF/{ENV_NAME}':
            print('cnf expert')
            expert = ConditionalNormalizingFlow(condition_dim=state_dim, n_flows=10, latent_dim=action_dim)
            expert.to(device=device)
        expert.eval()
    else:
        if NOISE == 'Gaussian':
            print('def noise')
            noise = Normal(loc=0, scale=0.2)
        elif NOISE == 'OrnsteinUhlenbeck':
            noise = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.2, base_scale=0.1)
        else:
            raise ValueError('Noise must be either Gaussian or OrnsteinUhlenbeck')
      



    env.reset()
    if ENV_NAME == 'biped':
        state_dim = 46#env.observation_space.shape[0]
    else:
        state_dim = env.observation_space.shape[0]

    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    print(f"Action low: {action_low}, Action high: {action_high}")

    list_of_all_the_data = []

    for cycles in range(NB_TRAINING_CYCLES):
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)

        behavior_policy = Policy(state_dim=state_dim, action_dim=action_dim)
        target_policy = Policy(state_dim=state_dim, action_dim=action_dim)
        
        behavior_q_1 = Value(state_dim=state_dim, action_dim=action_dim)
        target_q_1 = Value(state_dim=state_dim, action_dim=action_dim)
        
        behavior_q_2 = Value(state_dim=state_dim, action_dim=action_dim)
        target_q_2 = Value(state_dim=state_dim, action_dim=action_dim)

        # New behaviorPolicy and behaviorQ network weights for each cycle: N(0, 0.1)
        init_model_weights(behavior_policy, seed=seed)
        init_model_weights(behavior_q_1, seed=seed)
        init_model_weights(behavior_q_2, seed=seed)

        # PolicyWeights_t <-- PolicyWeights_b | QWeights_t <-- QWeights_b
        target_policy.load_state_dict(behavior_policy.state_dict())
        
        target_q_1.load_state_dict(behavior_q_1.state_dict())
        target_q_2.load_state_dict(behavior_q_2.state_dict())

        agent = TD3(
            policy_network=behavior_policy,
            target_policy=target_policy,
            value_network_1=behavior_q_1,
            target_value_1=target_q_1,
            value_network_2=behavior_q_2,
            target_value_2=target_q_2,
            seed=seed
        )

        memory = DDPGMemory(state_dim=state_dim, action_dim=action_dim, BUFFER_LENGTH=BUFFER_LENGTH)


        obs, _ = env.reset()
        episodic_returns = []
        cumulative_reward = 0

        for t in tqdm(range(TRAINING_STEPS), desc=f"Cycle {cycles+1}", unit="step"):
            with torch.no_grad():
                
                if ENV_NAME == 'Pendulum-v1':
                    state = np.array([np.arctan2(obs[1],obs[2]), obs[2]])
                    expert_input = torch.tensor(state, dtype=torch.float32, device=device).reshape(1, -1)
                elif ENV_NAME == 'biped':
                    obs = obs[:46].copy()
                    expert_input = torch.tensor(obs,dtype=torch.float32,device=device).reshape(1, -1)
                else:
                    expert_input = torch.tensor(obs, dtype=torch.float32, device=device).reshape(1, -1)
                
                if EXPERT_MODEL == f'MLE/{ENV_NAME}':
                        expert_action,_,_ = expert.sample(expert_input)
                elif EXPERT_MODEL == f'CNF/{ENV_NAME}':
                        base_mean, _ = expert.conditional_base(expert_input)
                        expert_action,_ = expert.inverse(base_mean, expert_input)
                else:
                    expert_action = noise.sample(action.shape).to(device)

                # print(expert_action)
                action = behavior_policy.forward(torch.tensor(obs, dtype=torch.float32, device=device))
                noisy_action = action.cpu().numpy() + expert_action.cpu().numpy().squeeze()
                
                # TODO: clip needs to be done differently, considering also qpos_joints
                if ENV_NAME != 'biped': 
                    noisy_action = np.clip(noisy_action,
                                                a_min=action_low,
                                                a_max=action_high)

                full_obs_, reward, termination, truncation, _ = env.step(noisy_action)
                if ENV_NAME == 'biped':    
                    obs_ = full_obs_[:46]
                else:
                    obs_ = full_obs_.copy()
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
                obs = full_obs_.copy()

        for i in range(len(agent.pi_loss)):
            list_of_all_the_data.append({
                'cycle': cycles + 1,
                'policy_loss': agent.pi_loss[i],
                'q1_loss': agent.q1_loss[i],
                'q2_loss': agent.q2_loss[i],
                'return': episodic_returns[i] if i < len(episodic_returns) else np.nan,
            })

        # Plot the reward.
        if EXPERT_MODEL:
            TYPE = ENV_NAME
        else:
            TYPE = NOISE
        fig, ax = plt.subplots(1, 1)
        ax.plot(episodic_returns)
        ax.set_title(f'{NOISE} added Noise')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Episodic Returns')
        plt.savefig(f'{ABS_FOLDER_RESUlTS}/{TYPE}_{cycles}.png')
        plt.close()

        # Save the results.
        df = pd.DataFrame(list_of_all_the_data)
        df.to_csv(f'{ABS_FOLDER_RESUlTS}/{TYPE}_{cycles}.csv', index=False)

        # Save the behaviour policy.
        torch.save(behavior_policy.state_dict(), f'{ABS_FOLDER_RESUlTS}/policy_{cycles}.pth')

    ### Testing the policy.

    def run_test(env_name):
        test_env = gym.make(env_name, render_mode='rgb_array')
        test_env.reset()

        frames = []
        obs, _= env.reset()
        for _ in range(1000):
            action = behavior_policy.forward(torch.tensor(obs, dtype=torch.float32, device=device))
            action = action.detach().cpu().numpy()
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
        # print(obs.shape)
        # quit()
        rollout = []
        for _ in tqdm(range(10000)):
            # action = np.random.uniform(-1, 1, test_env.action_size)
            non_priv_state = obs[:46]
            with torch.no_grad():
                action = behavior_policy.forward(torch.tensor(non_priv_state, dtype=torch.float32, device=device))
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

    elif ENV_NAME == 'Pendulum-v1' or ENV_NAME == 'Ant-v4':
        run_test(env_name=ENV_NAME)
