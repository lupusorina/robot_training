from biped_np import Biped

from IPython.display import clear_output

import collections
from datetime import datetime
import os
import time
from typing import Any, Callable, Dict, Optional
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import optim
from ppo import Agent
import gymnasium as gym
import pandas as pd

import tqdm as tqdm

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

StepData = collections.namedtuple(
    'StepData',
    ('observation', 'logits', 'action', 'reward', 'done', 'truncation'))


def sd_map(f: Callable[..., torch.Tensor], *sds) -> StepData:
  """Map a function over each field in StepData."""
  items = {}
  keys = sds[0]._asdict().keys()
  for k in keys:
    items[k] = f(*[sd._asdict()[k] for sd in sds])
  return StepData(**items)


def eval_unroll(idx, agent, env, length):
  print('In eval unroll')
  """Return number of episodes and average reward for a single unroll."""
  observation, _ = env.reset()
  observation = torch.tensor(observation, device=agent.device, dtype=torch.float32)
  episodes = torch.zeros((), device=agent.device)
  episode_reward = torch.zeros((), device=agent.device)
  observation_list = []
  for i in range(length):
    _, action = agent.get_logits_action(observation)
    observation, reward, done, _, info = env.step(Agent.dist_postprocess(action).detach().cpu().numpy())
    observation = torch.tensor(observation, device=agent.device, dtype=torch.float32)
    done = torch.tensor(done, device=agent.device, dtype=torch.float32)
    reward = torch.tensor(reward, device=agent.device, dtype=torch.float32)
    episodes += torch.sum(done)
    episode_reward += torch.sum(reward)
    observation_list.append(observation)
  print('End of eval unroll')
  return episodes, episode_reward / episodes


def train_unroll(agent, env, observation, num_unrolls, unroll_length):
  """Return step data over multple unrolls."""
  sd = StepData([], [], [], [], [], [])
  for _ in range(num_unrolls):
    # Convert initial observation to tensor
    observation = torch.tensor(observation, device=agent.device, dtype=torch.float32)
    one_unroll = StepData([observation], [], [], [], [], [])
    for _ in range(unroll_length):
      logits, action = agent.get_logits_action(observation)
      observation, reward, done, truncation, info = env.step(Agent.dist_postprocess(action).detach().cpu().numpy())
      observation = torch.tensor(observation, device=agent.device, dtype=torch.float32)
      reward = torch.tensor(reward, device=agent.device, dtype=torch.float32)
      done = torch.tensor(done, device=agent.device, dtype=torch.float32)
      truncation = torch.tensor(truncation, device=agent.device, dtype=torch.float32)
      one_unroll.observation.append(observation)
      one_unroll.logits.append(logits)
      one_unroll.action.append(action)
      one_unroll.reward.append(reward)
      one_unroll.done.append(done)
      one_unroll.truncation.append(truncation)
    one_unroll = sd_map(torch.stack, one_unroll)
    sd = sd_map(lambda x, y: x + [y], sd, one_unroll)
  td = sd_map(torch.stack, sd)
  return observation, td

def train(
    env_name: str = 'ant',
    num_envs: int = 256,
    episode_length: int = 1000,
    device: str = 'cuda',
    num_timesteps: int = 30_000_000,
    eval_frequency: int = 10,
    unroll_length: int = 5,
    batch_size: int = 1024,
    num_minibatches: int = 32,
    num_update_epochs: int = 4,
    reward_scaling: float = .1,
    entropy_cost: float = 1e-2,
    discounting: float = .97,
    learning_rate: float = 3e-4,
    progress_fn: Optional[Callable[[int, Dict[str, Any]], None]] = None,
):
  """Trains a policy via PPO."""

  if env_name == 'ant':
    env = gym.make_vec("Ant-v5",
                     num_envs=num_envs,
                     vectorization_mode="async", # Note sync is slow, so we use async
                     wrappers=(gym.wrappers.TimeAwareObservation,))
  elif env_name == 'pendulum':
    env = gym.make_vec('Pendulum-v1',
                       num_envs=num_envs,
                       vectorization_mode="async", # Note sync is slow, so we use async
                       wrappers=(gym.wrappers.TimeAwareObservation,))
  else:
    raise ValueError(f'Unknown environment: {env_name}')

  # Env warmup.
  obs, _ = env.reset()
  action = torch.zeros(env.action_space.shape).to(device)
  obs, reward, done, truncation, info = env.step(action.detach().cpu().numpy())

  # Create the agent.
  policy_layers = [
      env.observation_space.shape[-1], 64, 64, env.action_space.shape[-1] * 2
  ]
  value_layers = [env.observation_space.shape[-1], 64, 64, 1]
  agent = Agent(policy_layers, value_layers, entropy_cost, discounting,
                reward_scaling, device)
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
        episode_count, episode_reward = eval_unroll(eval_i, agent, env, episode_length)
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

    if eval_i == eval_frequency:
      break

    observation, _ = env.reset()
    num_steps = batch_size * num_minibatches * unroll_length
    num_epochs = num_timesteps // (num_steps * eval_frequency)
    num_unrolls = batch_size * num_minibatches // num_envs
    total_loss = 0
    t = time.time()
    for _ in range(num_epochs):
      observation, td = train_unroll(agent, env, observation, num_unrolls,
                                     unroll_length)

      # make unroll first
      def unroll_first(data):
        data = data.swapaxes(0, 1)
        return data.reshape([data.shape[0], -1] + list(data.shape[3:]))
      td = sd_map(unroll_first, td)

      # update normalization statistics
      agent.update_normalization(td.observation)

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
    torch.save(agent.policy.state_dict(), ABS_FOLDER_RESUlTS + '/ppo_model_pytorch.pth')

def progress(num_steps, metrics):
  print(f'steps: {num_steps}, \
        reward: {metrics["eval/episode_reward"]}, \
        eval_sps: {metrics["speed/eval_sps"]}, \
        train_sps: {metrics["speed/sps"]}')
  times.append(datetime.now())
  xdata.append(num_steps)
  ydata.append(metrics['eval/episode_reward'].cpu())
  eval_sps.append(metrics['speed/eval_sps'])
  train_sps.append(metrics['speed/sps'])
  clear_output(wait=True)
  plt.xlim([0, 30_000_000])
  plt.ylim([0, 2000])
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
  os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
  print('Start training')
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

  train_sps = []



