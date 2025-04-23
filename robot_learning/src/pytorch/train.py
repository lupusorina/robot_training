from IPython.display import clear_output

import collections
from datetime import datetime
import os
import time
from typing import Any, Callable, Dict, Optional
import matplotlib.pyplot as plt

import numpy as np

from torch import optim
from agents.ppo import Agent
import gymnasium as gym
import pandas as pd

import tqdm as tqdm
import torch

import mujoco
import mediapy as media

from robot_learning.src.pytorch.utils_np import TorchWrapper
from robot_learning.src.pytorch.utils_np import EpisodeWrapper
from robot_learning.src.pytorch.utils_np import VmapWrapper
from robot_learning.src.pytorch.utils_np import AutoResetWrapper
from robot_learning.src.pytorch.utils_np import VectorGymWrapper
import robot_learning.src.pytorch.utils_np as utils_np

from robot_learning.src.jax.envs.biped import Biped

RESULTS = 'results'
if not os.path.exists(RESULTS):
    os.makedirs(RESULTS)
time_now = datetime.now().strftime('%Y%m%d-%H%M%S')
if not os.path.exists(os.path.join(RESULTS, time_now)):
    os.makedirs(os.path.join(RESULTS, time_now))
FOLDER_RESULTS = os.path.join(RESULTS, time_now)
ABS_FOLDER_RESUlTS = os.path.abspath(FOLDER_RESULTS)
print(f"Saving results to {ABS_FOLDER_RESUlTS}")

StepData = collections.namedtuple(
    'StepData',
    ('observation', 'privileged_observation', 'logits', 'action', 'reward', 'done', 'truncation'))


def sd_map(f: Callable[..., torch.Tensor], *sds) -> StepData:
  """Map a function over each field in StepData."""
  items = {}
  keys = sds[0]._asdict().keys()
  for k in keys:
    items[k] = f(*[sd._asdict()[k] for sd in sds])
  return StepData(**items)


def eval_unroll(idx: int, agent: Agent, envs: gym.Env, length: int, obs_size: int, num_envs: int):
  print('In eval unroll')
  """Return number of episodes and average reward for a single unroll."""
  output_reset = envs.reset()

  # Get the observation from the privileged observation (first part)
  obs = output_reset['state']
  episodes = torch.zeros((), device=agent.device)
  episode_reward = torch.zeros((), device=agent.device)

  rollout_envs = {
    f'rollout_env_{i}': [] for i in range(np.min([num_envs, 10]))
  }
  for _ in range(length):
    _, action = agent.get_logits_action(obs)
    obs_dict, reward, done, _, info = envs.step(Agent.dist_postprocess(action))
    obs = obs_dict['state']
    episodes += torch.sum(done)
    episode_reward += torch.sum(reward)

    # Save the state for plotting.
    for i in range(np.min([num_envs, 10])):
      rollout_envs[f'rollout_env_{i}'].append({
        'qpos': info['qpos'][i].cpu().numpy(),
        'qvel': info['qvel'][i].cpu().numpy(),
        'xfrc_applied': info['xfrc_applied'][i].cpu().numpy()
      })

  print('End of eval unroll')
  return episodes, episode_reward / episodes, rollout_envs


def train_unroll(agent: Agent, obs_dict: torch.Tensor, env: gym.Env, num_unrolls: int, unroll_length: int, obs_size: int):
  """Return step data over multple unrolls."""
  sd = StepData([], [], [], [], [], [], [])
  for _ in range(num_unrolls):

    obs = obs_dict['state']
    privileged_obs = obs_dict['privileged_state']

    one_unroll = StepData([obs], [privileged_obs], [], [], [], [], [])
    for _ in range(unroll_length):
      logits, action = agent.get_logits_action(obs)
      obs_dict, reward, done, _, info = env.step(Agent.dist_postprocess(action))
      obs = obs_dict['state']
      privileged_obs = obs_dict['privileged_state']
      one_unroll.observation.append(obs)
      one_unroll.privileged_observation.append(privileged_obs)
      one_unroll.logits.append(logits)
      one_unroll.action.append(action)
      one_unroll.reward.append(reward)
      one_unroll.done.append(done)
      one_unroll.truncation.append(info['truncation'])
    one_unroll = sd_map(torch.stack, one_unroll)
    sd = sd_map(lambda x, y: x + [y], sd, one_unroll)
  td = sd_map(torch.stack, sd)
  return td

def train(
    env_name: str = 'biped',
    num_envs: int = 8192,
    episode_length: int = 1000,
    device: str = 'cuda',
    num_timesteps: int = 150_000_000,
    eval_frequency: int = 10,
    unroll_length: int = 20,
    batch_size: int = 256,
    num_minibatches: int = 32,
    num_update_epochs: int = 4,
    reward_scaling: float = 1.0,
    entropy_cost: float = 0.005,
    discounting: float = 0.97,
    learning_rate: float = 3e-4,
    action_repeat: int = 1,
    clip_epsilon: float = 0.3,
    progress_fn: Optional[Callable[[int, Dict[str, Any]], None]] = None,
):
  """Trains a policy via PPO."""

  # Here we create a wrapper to convert the jax env to a gym env.
  # Create the environment [jax env].
  env = Biped()
  ctrl_dt = env.ctrl_dt
  render_fn = env.render

  # Maintains episode step count and sets done at episode end.
  env = EpisodeWrapper(env, episode_length, action_repeat)
  
  # Batch the environment.
  env = VmapWrapper(env, batch_size=num_envs)

  # Automatically reset the environment at episode end.
  env = AutoResetWrapper(env)
  
  # Convert the jax env to a gym env.
  env = VectorGymWrapper(env)
  action_size = env.action_space.shape[-1]

  # Convert the gym env to a torch env.
  env = TorchWrapper(env, device=device)

  obs = env.reset()
  priviliged_state_size = obs['privileged_state'].shape[-1]
  obs_size = obs['state'].shape[-1]

  print('Sizes   priviliged state', priviliged_state_size)
  print('        action', action_size)
  print('        observation', obs_size)

  # Create the agent.
  policy_layers = [obs_size, 256, 128, action_size * 2]
  value_layers = [priviliged_state_size, 256, 128, 1]
  agent = Agent(policy_layers, value_layers, entropy_cost, discounting,
                reward_scaling, clip_epsilon, device)
  agent = torch.jit.script(agent.to(device))
  optimizer = optim.Adam(agent.parameters(), lr=learning_rate)

  sps = 0
  total_steps = 0
  total_loss = 0
  for eval_i in range(eval_frequency + 1):
    print(f'eval_i: {eval_i}')
    if progress_fn:
      t = time.time()
      with torch.no_grad():
        episode_count, episode_reward, rollout_envs = eval_unroll(eval_i, agent, env, episode_length, obs_size, num_envs)
      duration = time.time() - t
      # TODO: only count stats from completed episodes
      episode_avg_length = num_envs * episode_length / episode_count
      eval_sps = num_envs * episode_length / duration
      progress = {
          'eval/episode_reward': episode_reward,
          'eval/completed_episodes': episode_count,
          'eval/avg_episode_length': episode_avg_length,
          'speed/sps': sps,
          'speed/eval_sps': eval_sps,
          'losses/total_loss': total_loss,
      }
      progress_fn(total_steps, progress)

      # Visualize the rollout.
      render_video = True
      if render_video:
        media_files_list = []
        max_num_envs = np.min([num_envs, 10]) # Avoid too many videos to save.
        all_frames = []
        for idx_env in range(max_num_envs):
          rollout_env = rollout_envs[f'rollout_env_{idx_env}']

          render_every = 1 # int.
          fps = 1/ ctrl_dt / render_every
          traj = rollout_env[::render_every]

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

          # Save individual video
          # media.write_video(f'{ABS_FOLDER_RESUlTS}/joystick_testing_idx_env_epoch_{eval_i}_idx_env_{idx_env}.mp4', frames, fps=fps)
          media_files_list.append(f'{ABS_FOLDER_RESUlTS}/joystick_testing_idx_env_epoch_{eval_i}_idx_env_{idx_env}.mp4')
          all_frames.append(np.array(frames))  # Ensure frames are numpy arrays
          print(f'Video nb {idx_env} saved')

        # Arrange frames in a 2-row grid
        num_videos = len(all_frames)
        num_cols = (num_videos + 1) // 2  # Ceiling division
        frame_shape = all_frames[0].shape
        final_frames = np.zeros((frame_shape[0], frame_shape[1] * 2,
                                 frame_shape[2] * num_cols, frame_shape[3]),
                                 dtype=all_frames[0].dtype)

        for i, frames in enumerate(all_frames):
            row = i // num_cols
            col = i % num_cols
            final_frames[:, row*frame_shape[1]:(row+1)*frame_shape[1],
                        col*frame_shape[2]:(col+1)*frame_shape[2]] = frames

        media.write_video(f'{ABS_FOLDER_RESUlTS}/joystick_testing_idx_envs_epoch_{eval_i}.mp4', final_frames, fps=fps)

    if eval_i == eval_frequency:
      break

    obs = env.reset()

    num_steps = batch_size * num_minibatches * unroll_length
    num_epochs = num_timesteps // (num_steps * eval_frequency)
    num_unrolls = batch_size * num_minibatches // num_envs
    total_loss = 0
    t = time.time()
    for _ in range(num_epochs):
      td = train_unroll(agent,
                        obs,
                        env,
                        num_unrolls,
                        unroll_length,
                        obs_size)

      # Make unroll first.
      def unroll_first(data):
        data = data.swapaxes(0, 1)
        return data.reshape([data.shape[0], -1] + list(data.shape[3:]))
      td = sd_map(unroll_first, td)

      # update normalization statistics
      agent.update_normalization(td.observation)
      agent.update_normalization_privileged(td.privileged_observation)

      for _ in range(num_update_epochs):
        # shuffle and batch the data
        with torch.no_grad():
          permutation = torch.randperm(td.observation.shape[1], device=device)
          def shuffle_batch(data):
            data = data[:, permutation]
            data = data.reshape([data.shape[0], num_minibatches, -1] +
                                list(data.shape[2:]))
            return data.swapaxes(0, 1)
          epoch_td = sd_map(shuffle_batch, td)

        for minibatch_i in range(num_minibatches):
          td_minibatch = sd_map(lambda d: d[minibatch_i], epoch_td)
          loss = agent.loss(td_minibatch._asdict())
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          total_loss += loss

    duration = time.time() - t
    total_steps += num_epochs * num_steps
    total_loss = total_loss / (num_epochs * num_update_epochs * num_minibatches)
    sps = num_epochs * num_steps / duration

    # Save the model
    print(f'saving model at {total_steps}')
    torch.save(agent.policy.state_dict(), ABS_FOLDER_RESUlTS + f'/ppo_model_pytorch_{eval_i}.pth')

def progress(num_steps, metrics):
  print(f'steps: {num_steps}, \
        reward: {metrics["eval/episode_reward"]}, \
        eval_sps: {metrics["speed/eval_sps"]}, \
        train_sps: {metrics["speed/sps"]}')
  times.append(datetime.now())
  xdata.append(num_steps)
  ydata.append(metrics['eval/episode_reward'].cpu().numpy())
  eval_sps.append(metrics['speed/eval_sps'])
  train_sps.append(metrics['speed/sps'])
  clear_output(wait=True)
  # plt.xlim([0, 30_000_000])
  # plt.ylim([0, 2000])
  plt.xlabel('# environment steps')
  plt.ylabel('reward per episode')
  plt.plot(xdata, ydata)
  plt.savefig(ABS_FOLDER_RESUlTS + '/ppo_training.png')

  # save to pandas df
  df = pd.DataFrame({'x': xdata, 'y': ydata, 'eval_sps': eval_sps, 'train_sps': train_sps})
  df.to_csv(ABS_FOLDER_RESUlTS + '/ppo_training.csv', index=False)

# def progress(_, metrics):
#   if 'training/sps' in metrics:
#     train_sps.append(metrics['training/sps'])

if __name__ == '__main__':
  # temporary fix to cuda memory OOM

  utils_np.set_seed()

  os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
  print('Start the PPO training ...')
  xdata = []
  ydata = []
  eval_sps = []
  train_sps = []
  times = [datetime.now()]

  train(progress_fn=progress)

  print(f'time to jit: {times[1] - times[0]}')
  print(f'time to train: {times[-1] - times[1]}')
  print(f'eval steps/sec: {np.mean(eval_sps)}')
  print(f'train steps/sec: {np.mean(train_sps)}')

  print('Training is done!')

