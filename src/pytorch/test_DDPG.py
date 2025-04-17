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


from etils import epath

RESULTS_FOLDER_PATH = os.path.abspath('results')

# Sort by date and get the latest folder.
folders = sorted(os.listdir(RESULTS_FOLDER_PATH))
latest_folder = folders[-1]
FULL_PATH = os.path.join(RESULTS_FOLDER_PATH, latest_folder)
print(f'Latest folder: {FULL_PATH}')

POLICY_NAME = 'policy_0.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


if __name__ == '__main__':

    test_env = Biped()
    state_dim = test_env.observation_size[0]
    action_dim = test_env.action_size

    action_low = test_env._soft_q_j_min
    action_high = test_env._soft_q_j_max

    # Load the policy.
    policy_path = f'{FULL_PATH}/policy_0.pth'
    behavior_policy = Policy(state_dim=state_dim, action_dim=action_dim, policy_lr=1e-3, device=device)
    behavior_policy.load_state_dict(torch.load(policy_path, map_location=device, weights_only=True))
    behavior_policy.eval()

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
    # ABS_FOLDER_RESUlTS = epath.Path(RESULTS_FOLDER_PATH) / latest_folder
    # NOTE: To make the code run, you need to call: MUJOCO_GL=egl python3 test_DDPG.py
    media.write_video(f'{FULL_PATH}/behaviour_robot_testing.mp4', frames, fps=fps)
    print('Video saved')
