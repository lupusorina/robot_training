import gymnasium as gym
import torch
import os
from moviepy.video.io import ImageSequenceClip
from agents.ppo import Agent

# Define environment.
env_name = 'ant'
if env_name == 'ant':
    env = gym.make("Ant-v5", render_mode='rgb_array')
    obs_size = env.observation_space.shape[0]
    privileged_obs_size = env.observation_space.shape[0]
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
obs = out[0]
frames = []

for i in range(1000):
    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    _, action = agent.get_logits_action(obs_tensor)
    obs, reward, done, truncation, info = env.step(action.detach().numpy())

    frames.append(env.render())
    
env.close()

video_path = os.path.join(FULL_PATH, 'test_video.mp4')
clip = ImageSequenceClip.ImageSequenceClip(frames, fps=30)
clip.write_videofile(video_path, codec="libx264")
print(f"Saved video!")