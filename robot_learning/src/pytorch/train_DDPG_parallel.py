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
from robot_learning.src.pytorch.agents.ddpg import Agent, ReplayBuffer
from robot_learning.src.pytorch.agents.ddpg import StepData, sd_map
import gymnasium as gym
import pandas as pd

from robot_learning.src.pytorch.utils_np import TorchWrapper
from robot_learning.src.pytorch.utils_np import EpisodeWrapper
from robot_learning.src.pytorch.utils_np import VmapWrapper
from robot_learning.src.pytorch.utils_np import AutoResetWrapper
from robot_learning.src.pytorch.utils_np import VectorGymWrapper
import robot_learning.src.pytorch.utils_np as utils_np
from robot_learning.src.pytorch.utils_np import OUNoise

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


def warm_up(env, warm_up_steps, memory, num_envs, action_size):
    """Warm up the replay buffer with random actions."""
    print(f"Warming up replay buffer with {warm_up_steps} steps...")

    # Initialize empty trajectory data structure
    warm_up_td = None

    output_reset = env.reset()
    obs = output_reset['state']
    privileged_obs = output_reset['privileged_state']

    warmup_rollout_envs = {
        f'rollout_env_{i}': [] for i in range(np.min([num_envs, 10]))
        } # For video generation.

    # Run environment steps.
    for step in range(warm_up_steps):

        # Take random actions.
        action = torch.randn((num_envs, action_size), device=env.device)
        action = torch.clamp(action, -1.0, 1.0)

        # Step environment.
        next_obs_dict, reward, done, _, info = env.step(action)
        next_obs = next_obs_dict['state']
        next_privileged_obs = next_obs_dict['privileged_state']

        # Create trajectory data for this step (step_td -> step trajectory data).
        step_td = StepData(
            obs=obs,                                    # (num_envs, obs_dim)
            privileged_obs=privileged_obs,              # (num_envs, privileged_obs_dim)
            next_obs=next_obs,                          # (num_envs, obs_dim)
            action=action,                              # (num_envs, action_dim)
            reward=reward.unsqueeze(-1),                # (num_envs, 1)
            done=done.unsqueeze(-1),                    # (num_envs, 1)
            truncation=info['truncation'].unsqueeze(-1) # (num_envs, 1)
        )

        obs = next_obs.clone()
        privileged_obs = next_privileged_obs.clone()

        # Add to warm-up buffer.
        if warm_up_td is None:
            warm_up_td = step_td
        else:
            # Concatenate along the batch dimension.
            def concat_buffers(old, new):
                return torch.cat([old, new], dim=0)
            warm_up_td = sd_map(concat_buffers, warm_up_td, step_td)

        # Save the state for video generation.
        for i in range(np.min([num_envs, 10])):
            warmup_rollout_envs[f'rollout_env_{i}'].append({
                'qpos': info['qpos'][i].cpu().numpy(),
                'qvel': info['qvel'][i].cpu().numpy(),
                'xfrc_applied': info['xfrc_applied'][i].cpu().numpy()
            })

    # Add warm-up data to memory
    memory.add_trajectory_data(warm_up_td)
    print(f"Memory size after warm-up: {memory.current_sz()}")
    print(f"Warm-up complete. Added {warm_up_steps * num_envs} transitions to replay buffer.")
    return warmup_rollout_envs


def eval_unroll(idx: int, agent: Agent, envs: gym.Env, length: int, num_envs: int):
  """Return number of episodes and average reward for a single unroll."""
  print('In eval unroll')
  print('num_envs', num_envs)
  output_reset = envs.reset()
  obs = output_reset['state']

  episodes = torch.zeros((), device=agent.device)
  episode_reward = torch.zeros((), device=agent.device)

  rollout_envs = {
    f'rollout_env_{i}': [] for i in range(np.min([num_envs, 10]))
  }
  for _ in range(length):
    _, action = agent.get_logits_action(obs)
    next_obs_dict, reward, done, _, info = envs.step(action)
    next_obs = next_obs_dict['state']

    episodes += torch.sum(done)
    episode_reward += torch.sum(reward)

    # Save the state for plotting.
    for i in range(np.min([num_envs, 10])):
      rollout_envs[f'rollout_env_{i}'].append({
        'qpos': info['qpos'][i].cpu().numpy(),
        'qvel': info['qvel'][i].cpu().numpy(),
        'xfrc_applied': info['xfrc_applied'][i].cpu().numpy()
      })
    obs = next_obs.clone()

  return episodes, episode_reward / episodes, rollout_envs


def train_unroll(agent: Agent, env: gym.Env, unroll_length: int, noise: OUNoise, obs: torch.Tensor, privileged_obs: torch.Tensor):
    """Return step data over multple unrolls."""
    
    transitions_td = None

    # Run environment steps
    for _ in range(unroll_length):
        with torch.no_grad():
            _, action = agent.get_logits_action(obs)
            # Convert noise sample to tensor and move to correct device
            noise_sample = torch.from_numpy(noise.sample()).to(agent.device)
            action = action + noise_sample
            next_obs_dict, reward, done, _, info = env.step(action)

            # Ensure all tensors are on the same device
            next_obs = next_obs_dict['state'].to(agent.device)
            next_privileged_obs = next_obs_dict['privileged_state'].to(agent.device)
            reward = reward.to(agent.device)
            done = done.to(agent.device)
            truncation = info['truncation'].to(agent.device)

        # Create trajectory data for this step
        step_td = StepData(
            obs=obs,
            privileged_obs=privileged_obs,
            next_obs=next_obs,
            action=action,
            reward=reward.unsqueeze(-1),
            done=done.unsqueeze(-1),
            truncation=truncation.unsqueeze(-1)
        )
        obs = next_obs.clone()
        privileged_obs = next_privileged_obs.clone()
        
        # Add to buffer.
        if transitions_td is None:
            transitions_td = step_td
        else:
            # Concatenate along the batch dimension.
            def concat_buffers(old, new):
                return torch.cat([old, new], dim=0)
            transitions_td = sd_map(concat_buffers, transitions_td, step_td)

    return obs, privileged_obs, transitions_td


def train(
    env_name: str = 'ant',
    num_envs: int = 100,
    episode_length: int = 1000,
    nb_episodes: int = 10,
    device: str = 'cuda',
    num_timesteps: int = 150_000_000,
    warm_up_steps: int = 200,
    minibatch_size: int = 1024,                  # From original paper. 
    discounting: float = .99,                   # From original paper.
    action_repeat: int = 1,
    tau: float = 0.001,                         # From original paper.
    learning_rate_actor: float = 1e-4,          # From original paper.
    learning_rate_critic: float = 1e-3,         # From original paper.
    max_memory: int = 2e6,                     # From original paper.
    gradient_steps: int = 5,
    progress_fn: Optional[Callable[[int, Dict[str, Any]], None]] = None,
):
    """Trains a policy via DDPG."""

    if env_name == 'biped':
        from robot_learning.src.jax.envs.biped import Biped
        env = Biped()
    elif env_name == 'ant':
        from robot_learning.src.jax.envs.ant import Ant
        env = Ant()
    else:
        raise ValueError(f'Environment {env_name} not supported')

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
    print('        observation', obs_size)
    print('        action', action_size)
    print('Device: ', device)

    # Create the agent.
    policy_layers = [obs_size, 256, 128, action_size]
    value_layers = [obs_size + action_size, 256, 128, 1]
        
    agent = Agent(policy_layers, value_layers, discounting=discounting,
                tau=tau, device=device)
    agent = torch.jit.script(agent.to(device))

    memory = ReplayBuffer(max_size=max_memory)

    # Exploration noise.
    ou_noise = OUNoise(action_dim=action_size)

    # Optimizers.
    actor_optimizer = optim.Adam(agent.policy_b.parameters(), lr=learning_rate_actor)
    critic_optimizer = optim.Adam(agent.value_b.parameters(), lr=learning_rate_critic)

    sps = 0
    total_steps = 0
    total_loss = 0

    nb_episodes = num_timesteps // (num_envs * episode_length)
    print('Training hyperparameters:')
    print(f'    nb_episodes: {nb_episodes}')
    print(f'    num_timesteps: {num_timesteps}')
    print(f'    num_envs: {num_envs}')
    print(f'    episode_length: {episode_length}')
    print(f'    minibatch_size: {minibatch_size}')
    print(f'    gradient_steps: {gradient_steps}')
    print(f'    learning_rate_actor: {learning_rate_actor}')
    print(f'    learning_rate_critic: {learning_rate_critic}')
    print(f'    discounting: {discounting}')
    print(f'    tau: {tau}')

    if warm_up_steps:
        warmup_rollout_envs = warm_up(env, warm_up_steps, memory, num_envs, action_size)
        render_video_warmups = False
        if render_video_warmups:
            print('Rendering warmup video ...')
            utils_np.generate_video(render_fn=render_fn,
                                    rollout_envs=warmup_rollout_envs,
                                    num_envs=num_envs,
                                    ctrl_dt=ctrl_dt,
                                    eval_i=0,
                                    append_to_filename='warmup',
                                    folder_name=ABS_FOLDER_RESUlTS)
    
    # MAIN LOOP.
    for episode_i in range(nb_episodes + 1):
        print(f'episode_i: {episode_i}/{nb_episodes+1}')

        # Training one episode.
        print('Training ...')

        total_v_loss = 0
        total_p_loss = 0

        t = time.time()

        # Initialize a random process for action exploration.
        ou_noise.reset()

        # Reset the environment.
        output_reset = env.reset()

        # Receive initial observation.
        obs = output_reset['state']
        privileged_obs = output_reset['privileged_state']

        # clean gradients before training
        print('Starting training')
        for _ in range(episode_length):
            # Collect experience
            obs, privileged_obs, transitions_td = train_unroll(agent,
                                        env,
                                        unroll_length=1,
                                        noise=ou_noise,
                                        obs=obs,
                                        privileged_obs=privileged_obs)

            # Store transition.
            memory.add_trajectory_data(transitions_td)

            for _ in range(gradient_steps):
                # Sample a minibatch of transitions.
                train_data_batch = memory.sample(batch_size=minibatch_size)

                v_loss = agent.critic_loss(train_data_batch._asdict())
                critic_optimizer.zero_grad()
                v_loss.backward()
                critic_optimizer.step()
                total_v_loss += v_loss.detach()

                p_loss = agent.policy_loss(train_data_batch._asdict())
                actor_optimizer.zero_grad()
                p_loss.backward()
                actor_optimizer.step()
                total_p_loss += p_loss.detach()

                # Soft update target networks
                with torch.no_grad():
                    for params, target_params in zip(agent.value_b.parameters(), agent.value_t.parameters()):
                        target_params.data.copy_(agent.tau * params.data + (1.0 - agent.tau) * target_params.data)

                    for params, target_params in zip(agent.policy_b.parameters(), agent.policy_t.parameters()):
                        target_params.data.copy_(agent.tau * params.data + (1.0 - agent.tau) * target_params.data)

        duration = time.time() - t
        total_steps += num_envs * episode_length  # Each episode processes num_envs * episode_length steps.
        denom = episode_length * gradient_steps  # We do gradient_steps updates per episode.
        if denom > 0:
            total_loss = (total_v_loss + total_p_loss) / denom
        sps = num_envs * episode_length / duration if duration > 0 else 0

        # Save the model
        print(f'saving model at {total_steps}')
        torch.save(agent.policy_b.state_dict(), ABS_FOLDER_RESUlTS + f'/ddpg_model_pytorch_{total_steps}.pth')
        
        # Run evaluation.
        if progress_fn:
            t = time.time()
            with torch.no_grad():
                episode_count, episode_reward, rollout_envs = eval_unroll(episode_i, agent, env, episode_length, num_envs)
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

            # Visualize the rollout.
            render_video = False
            if render_video:
                print('Rendering eval video ...')
                utils_np.generate_video(render_fn=render_fn,
                                        rollout_envs=rollout_envs,
                                        num_envs=num_envs,
                                        ctrl_dt=ctrl_dt,
                                        eval_i=episode_i,
                                        folder_name=ABS_FOLDER_RESUlTS)

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
    plt.xlabel('# environment steps')
    plt.ylabel('reward per episode')
    plt.plot(xdata, ydata)
    plt.savefig(ABS_FOLDER_RESUlTS + '/ddpg_training.png')

    # save to pandas df
    df = pd.DataFrame({'x': xdata, 'y': ydata, 'eval_sps': eval_sps, 'train_sps': train_sps})
    df.to_csv(ABS_FOLDER_RESUlTS + '/ddpg_training.csv', index=False)


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