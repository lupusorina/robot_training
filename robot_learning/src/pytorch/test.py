import gymnasium as gym
import torch
import os
from moviepy.video.io import ImageSequenceClip
from agents.ppo import Agent
import mujoco
import mediapy as media

import numpy as np

from robot_learning.src.pytorch.utils_np import TorchWrapper
from robot_learning.src.pytorch.utils_np import EpisodeWrapper
from robot_learning.src.pytorch.utils_np import VmapWrapper
from robot_learning.src.pytorch.utils_np import AutoResetWrapper
from robot_learning.src.pytorch.utils_np import VectorGymWrapper
import robot_learning.src.pytorch.utils_np as utils_np

import jax
import jax.random

from robot_learning.src.jax.envs.biped import Biped

# Policy path.
ABS_FOLDER_RESUlTS = os.path.abspath('results')
folders = sorted(os.listdir(ABS_FOLDER_RESUlTS))
latest_folder = folders[-1]
FULL_PATH = os.path.join(ABS_FOLDER_RESUlTS, latest_folder)
print(f'Latest folder: {FULL_PATH}')

# Device.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

auto_reset = True

# Load environment.
env_name = 'biped'
env = Biped()
ctrl_dt = env.ctrl_dt
render_fn = env.render

# Batch the environment.
env = VmapWrapper(env, batch_size=1)

# Automatically reset the environment at episode end.
if auto_reset:
    env = AutoResetWrapper(env)

# Convert the jax env to a gym env.
env = VectorGymWrapper(env)
action_size = env.action_space.shape[-1]

# Convert the gym env to a torch env.
env = TorchWrapper(env, device=device)

obs = env.reset()
priviliged_state_size = obs['privileged_state'].shape[-1]
obs_size = obs['state'].shape[-1]

# Create the agent.
policy_layers = [obs_size, 256, 128, action_size * 2]
value_layers = [priviliged_state_size, 256, 128, 1]

agent = Agent(policy_layers, value_layers, device=device)
agent.policy.load_state_dict(torch.load(os.path.join(FULL_PATH, 'ppo_model_pytorch_9.pth')))

# Set to evaluation mode.
agent.policy.eval()

# Put policy to device.
agent.policy.to(device)

out = env.reset()
privileged_obs = out['privileged_state']
obs = out['state']

print('Sizes   priviliged state', priviliged_state_size)
print('        action', action_size)
print('        observation', obs_size)

# Test policy.
time_counter = 0
time_list = []
frames = []
rollout = []

episodes = 0
episode_reward = 0

done_switch = []

print_next_step = False

for i in range(1000):
    print('i = ', i)
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
    _, action = agent.get_logits_action(obs_tensor)
    sampled_action = Agent.dist_postprocess(action)
    obs_dict, rewards, done, _, info = env.step(sampled_action)
    obs = obs_dict['state']

    state = {
        'qpos': info['qpos'].cpu().numpy(), 
        'qvel': info['qvel'].cpu().numpy(),
        'xfrc_applied': info['xfrc_applied'].cpu().numpy()
    }

    rollout.append(state)

    # Keep track of time.
    time_counter += ctrl_dt

    # Save data.
    time_list.append(time_counter)

render_video = True
if render_video:
    render_every = 1 # int.
    fps = 1/ ctrl_dt / render_every

    traj = rollout[::render_every]

    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[2] = True
    scene_option.geomgroup[3] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False

    frames = render_fn(
        traj,
        camera="track",
        scene_option=scene_option,
        width=640,
        height=480,
    )

    # NOTE: To make the code run, you need to call: MUJOCO_GL=egl python3 test.py
    media.write_video(f'{FULL_PATH}/joystick_testing.mp4', frames, fps=fps)
    print('Video saved')

