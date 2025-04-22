import gymnasium as gym
import torch
import os
from moviepy.video.io import ImageSequenceClip
from agents.ppo import Agent
import mujoco
import mediapy as media

import numpy as np

# Define environment.
env_name = 'biped'
if env_name == 'ant':
    env = gym.make("Ant-v5", render_mode='rgb_array')
    obs_size = env.observation_space.shape[0]
    privileged_obs_size = env.observation_space.shape[0]
    print(f'Observation size: {obs_size}')
    print(f'Privileged observation size: {privileged_obs_size}')
    print(f'Action size: {env.action_space.shape[-1]}')
elif env_name == 'biped':
    from envs.biped_np import Biped
    env = Biped()
    obs_size = env.observation_size[0]
    output = env.reset()
    print(output)
    privileged_obs_size = output[0].shape[0]
    print(f'Observation size: {obs_size}')
    print(f'Privileged observation size: {privileged_obs_size}')
    print(f'Action size: {env.action_space.shape[-1]}')
else:
    raise ValueError(f"Environment {env_name} not supported")

# Load policy.
ABS_FOLDER_RESUlTS = os.path.abspath('results')
folders = sorted(os.listdir(ABS_FOLDER_RESUlTS))
latest_folder = folders[-1]
FULL_PATH = os.path.join(ABS_FOLDER_RESUlTS, latest_folder)
print(f'Latest folder: {FULL_PATH}')

# Create the agent.
policy_layers = [obs_size, 256, 128, 64, env.action_space.shape[-1] * 2]
value_layers = [privileged_obs_size, 256, 128, 64, 1]

agent = Agent(policy_layers, value_layers)
agent.policy.load_state_dict(torch.load(os.path.join(FULL_PATH, 'ppo_model_pytorch.pth')))

# Test policy.
out = env.reset()
privileged_obs = out[0]
obs = env.get_observation()

time_counter = 0
time_list = []
frames = []
rollout = []

episodes = 0
episode_reward = 0

for i in range(1000):
    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    _, action = agent.get_logits_action(obs_tensor)
    priviliged_obs, rewards, done, _, _ = env.step(Agent.dist_postprocess(action).detach().numpy())
    obs = env.get_observation()
    state = {
        'qpos': env.data.qpos.copy(),
        'qvel': env.data.qvel.copy(),
        'xfrc_applied': env.data.xfrc_applied.copy()
    }

    rollout.append(state)

    # Keep track of time.
    time_counter += env.ctrl_dt

    # Save data.
    time_list.append(time_counter)
    
    # Metrics.
    episodes += np.sum(done)
    episode_reward += np.sum(rewards)

print(f'Episodes: {episodes}')
print(f'Episode reward: {episode_reward}')
print(f'Average reward: {episode_reward / episodes}')

render_video = True
if render_video:
    render_every = 1 # int.
    fps = 1/ env.ctrl_dt / render_every

    traj = rollout[::render_every]

    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[2] = True
    scene_option.geomgroup[3] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False

    frames = env.render(
        traj,
        camera="track",
        scene_option=scene_option,
        width=640,
        height=480,
    )

    # media.show_video(frames, fps=fps, loop=False)
    # ABS_FOLDER_RESUlTS = epath.Path(RESULTS_FOLDER_PATH) / latest_folder
    # NOTE: To make the code run, you need to call: MUJOCO_GL=egl python3 test.py
    media.write_video(f'{FULL_PATH}/joystick_testing.mp4', frames, fps=fps)
    print('Video saved')

