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
from src.pytorch.agents.ddpg import Agent, ReplayBuffer
import gymnasium as gym
import pandas as pd
import envs

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

# StepData = collections.namedtuple(
#     'StepData',
#     ('observation', 'logits', 'action', 'reward', 'done', 'truncation'))

# def sd_map(f: Callable[..., torch.Tensor], *sds) -> StepData:
#   """Map a function over each field in StepData."""
#   items = {}
#   keys = sds[0]._asdict().keys()
#   for k in keys:
#     items[k] = f(*[sd._asdict()[k] for sd in sds])
#   return StepData(**items)

def warm_up(agent, env, n_steps, memory_buffer):
    print('warm up buffer')
    observation, _ = env.reset()
    observation = torch.tensor(observation, device=agent.device, dtype=torch.float32)
    for i in range(n_steps):
        action = env.action_space.sample()
        observation_, reward, termination, truncation, info = env.step(action)
        done = np.logical_or(termination,truncation)
        next_observation = torch.tensor(observation_, device=agent.device, dtype=torch.float32)
        action_tensor = torch.tensor(action, device=agent.device, dtype=torch.float32)
        done_tensor = torch.tensor(done, device=agent.device, dtype=torch.float32)
        reward_tensor = torch.tensor(reward, device=agent.device, dtype=torch.float32)
        for j in range(env.num_envs):
            memory_buffer.store(
            obs=observation[j,:], 
            act=action_tensor[j,:], 
            reward=reward_tensor[j], 
            next_obs=next_observation[j,:], 
            done=done_tensor[j])
        
        if done.any(): # reset only envs that have a done flag active
                # print(done)
                # env_idx = done.nonzero().squeeze()
                env_idx = np.where(done)[0]
                # print(env_idx)
                reset_observation,_ = env.reset()
                observation_[env_idx] = reset_observation[env_idx]
                # print('new obs: ', observation_[env_idx])
                # print('reset obs: ', reset_observation[env_idx])

                # print('obs env 1: ', observation_[0])
                # print('obs reset env 1: ', reset_observation[0])
                
        observation = torch.tensor(observation_, device=agent.device, dtype=torch.float32)
    

def eval_unroll(idx, agent, env, length):
  print('In eval unroll')
  """Return number of episodes and average reward for a single unroll."""
  observation, _ = env.reset()
  observation = torch.tensor(observation, device=agent.device, dtype=torch.float32)
  episodes = torch.zeros((), device=agent.device)
  episode_reward = torch.zeros((), device=agent.device)
  observation_list = []
#   R = torch.zeros(observation.shape[0], device=agent.device, dtype=torch.float32)
#   print(R.shape)
  for i in range(length):
    with torch.no_grad():
        _, action = agent.get_action(observation)
        observation, reward, termination, truncation, info = env.step(action.cpu().numpy())
        # print('info: ',info)
        # print(reward)
        done = np.logical_or(termination,truncation)
        # print(done)
        observation = torch.tensor(observation, device=agent.device, dtype=torch.float32)
        done = torch.tensor(done, device=agent.device, dtype=torch.float32)
        reward = torch.tensor(reward, device=agent.device, dtype=torch.float32)
        # R += reward
        # print(R)
        episodes += torch.sum(done)
        episode_reward += torch.sum(reward)
        # print(episode_reward)
        observation_list.append(observation)
  print('End of eval unroll')
  return episodes, episode_reward / episodes


def train_unroll(agent, env, observation):
    """Collect experience and store in the replay buffer. Only one step per env
    as in baseline ddpg"""
    # print('in train unroll')
    observation_tensor = torch.tensor(observation, device=agent.device, dtype=torch.float32)
    
    # Get action from policy
    with torch.no_grad():
        _, action = agent.get_action(observation_tensor)
        noise = torch.normal(mean=0.0, std=0.1, size=action.shape, device=agent.device)
        action_hat = torch.clip(action + noise, -agent.action_lim, agent.action_lim)

    # Execute action in environment
    next_observation, reward, termination, truncation, _ = env.step(action_hat.cpu().numpy())
    # Convert to tensors
    done = np.logical_or(termination,truncation)
    next_observation_tensor = torch.tensor(next_observation, device=agent.device, dtype=torch.float32)
    reward_tensor = torch.tensor(reward, device=agent.device, dtype=torch.float32)
    done_tensor = torch.tensor(done, device=agent.device, dtype=torch.float32)
    # Store transition in replay buffer
    for i in range(env.num_envs):
        agent.memory.store(
            obs=observation_tensor[i,:], 
            act=action_hat[i,:], 
            reward=reward_tensor[i], 
            next_obs=next_observation_tensor[i,:], 
            done=done_tensor[i]
        )
    
    return next_observation, done #return last new observation after step as well as done_ tensor --> is episode from each env over? need to reset that env


def train(
    env_name: str = 'ant',
    num_envs: int = 100,
    episode_length: int = 1000,
    device: str = 'cuda',
    num_timesteps: int = 1_000_000,
    warm_up_steps: int = 100,
    eval_frequency: int = 10,
    unroll_length: int = 1,
    batch_size: int = 256,
    num_minibatches: int = 10,
    num_update_epochs: int = 3,
    reward_scaling: float = .1,
    discounting: float = .99,
    learning_rate: float = 1e-3,
    max_memory: int = 2_000_000,
    progress_fn: Optional[Callable[[int, Dict[str, Any]], None]] = None,
):
    """Trains a policy via DDPG."""

    # Environment setup (same as before)
    if env_name == 'ant':
        env = gym.make_vec("Ant-v5",
                        num_envs=num_envs,
                        vectorization_mode="async",
                        wrappers=(gym.wrappers.TimeAwareObservation,))
    elif env_name == 'pendulum':
        env = gym.make_vec('Pendulum-v1',
                        num_envs=num_envs,
                        vectorization_mode="async",
                        wrappers=(gym.wrappers.TimeAwareObservation,))
    elif env_name == 'biped':
        env = gym.make_vec('Biped-custom',
                    num_envs=num_envs,
                    vectorization_mode="async")

    # Env warmup.
    observation, _ = env.reset()
    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.shape[-1]
    act_lim = env.single_action_space.high  # hard coded for now for the ant
    
    buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, max_sz=max_memory, device=device)
    
    # Create the agent
    policy_layers = [
        env.observation_space.shape[-1], 400, 300, env.action_space.shape[-1]
    ]
    value_layers = [env.observation_space.shape[-1] + env.action_space.shape[-1], 400, 300, 1]
    agent = Agent(policy_layers, value_layers, memory=buffer, discounting=discounting,
                tau=0.005, action_limit=act_lim, device=device)
    agent = torch.jit.script(agent.to(device))
    print("agent created and jitted")

    actor_optimizer = optim.Adam(agent.policy_b.parameters(), lr=learning_rate)
    critic_optimizer = optim.Adam(agent.value_b.parameters(), lr=learning_rate)
    
    sps = 0
    total_steps = 0
    total_loss = 0
    
    # Main training loop

    if warm_up_steps:
        warm_up(agent, env, warm_up_steps, agent.memory)
    
    for eval_i in range(eval_frequency + 1):
        print(f'eval_i: {eval_i}/{eval_frequency+1}')
        
        # Run evaluation
        if progress_fn:
            t = time.time()
            with torch.no_grad():
                episode_count, episode_reward = eval_unroll(eval_i, agent, env, episode_length)
            duration = time.time() - t
            episode_avg_length = num_envs * episode_length / episode_count
            eval_sps = num_envs * episode_length / duration if duration > 0 else 0
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
        print('Evaluation complete')

        # Training
        num_steps = batch_size * num_minibatches #unused for ddpg --> related to nb of steps in traj for ppo?
        num_epochs = num_timesteps // (num_steps * eval_frequency)
        print(num_epochs)
        total_v_loss = 0
        total_p_loss = 0
        t = time.time()

        # clean gradients before training
        critic_optimizer.zero_grad()
        actor_optimizer.zero_grad()

        for epoch in range(num_epochs):
            for _ in range(num_steps):
                observation_, done = train_unroll(agent, env, observation)
                # reset any finished envs
                if done.any():
                    idxs = np.where(done)[0]
                    reset_obs, _ = env.reset()
                    observation_[idxs] = reset_obs[idxs]
                observation = observation_
            # Collect data
            # observation_, done = train_unroll(agent, env, observation)
            # print('Experience collection complete')

            if agent.memory.current_sz() > batch_size:
                for _ in range(num_update_epochs):
               #once enough transitions stored, can train
                    
                    train_data_batch = agent.memory.sample_batch(batch_sz=batch_size)
                    v_loss = agent.critic_loss(train_data_batch)
                    critic_optimizer.zero_grad()
                    v_loss.backward()
                    critic_optimizer.step()
                    total_v_loss += v_loss.detach()
                    # print('Value loss:', v_loss.item())
                    # critic_optimizer.zero_grad()
                    
                    # Update actor
                    # actor_optimizer.zero_grad()
                    p_loss = agent.policy_loss(train_data_batch)
                    actor_optimizer.zero_grad()
                    p_loss.backward()
                    actor_optimizer.step()
                    total_p_loss += p_loss.detach()
                    # print('Policy loss:', p_loss.item())
                    # actor_optimizer.zero_grad()

                    # Soft update target networks
                    with torch.no_grad():
                        for params, target_params in zip(agent.value_b.parameters(), agent.value_t.parameters()):
                            target_params.data.copy_(agent.tau*params.data + (1.0-agent.tau) * target_params.data)

                        for params, target_params in zip(agent.policy_b.parameters(), agent.policy_t.parameters()):
                            target_params.data.copy_(agent.tau*params.data + (1.0-agent.tau) * target_params.data)
                        # print("soft update of target netwroks")
                        # print('Update cycle complete')
            
            if done.any(): # reset only envs that have a done flag active
                env_idx = np.where(done)[0]
                reset_observation,_ = env.reset()
                observation_[env_idx] = reset_observation[env_idx]
                
            observation = observation_    
        
        duration = time.time() - t
        total_steps += num_epochs * num_steps
        denom = (num_epochs * num_update_epochs * num_minibatches)
        if denom > 0:
            total_loss = (total_v_loss + total_p_loss) / denom
        sps = num_epochs * num_steps / duration if duration > 0 else 0

        # Save the model
        print(f'saving model at {total_steps}')
        torch.save(agent.policy_b.state_dict(), ABS_FOLDER_RESUlTS + f'/ddpg_model_pytorch_{total_steps}.pth')

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
    df.to_csv(ABS_FOLDER_RESUlTS + '/ddpg_ant_training.csv', index=False)


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

