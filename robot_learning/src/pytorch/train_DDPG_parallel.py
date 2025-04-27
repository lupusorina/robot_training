from datetime import datetime
import os
import time
from typing import Any, Callable, Dict, Optional
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import optim
from robot_learning.src.pytorch.agents.ddpg import Agent, ReplayBuffer
from robot_learning.src.pytorch.agents.ddpg import StepData, sd_map, soft_update, init_model_weights
import gymnasium as gym
import pandas as pd

from robot_learning.src.pytorch.utils_np import TorchWrapper
from robot_learning.src.pytorch.utils_np import EpisodeWrapper
from robot_learning.src.pytorch.utils_np import VmapWrapper
from robot_learning.src.pytorch.utils_np import AutoResetWrapper
from robot_learning.src.pytorch.utils_np import VectorGymWrapper
import robot_learning.src.pytorch.utils_np as utils_np
from robot_learning.src.pytorch.utils_np import OrnsteinUhlenbeckActionNoise

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

def eval_unroll(agent: Agent, envs: gym.Env, length: int, num_envs: int):
  """Return number of episodes and average reward for a single unroll."""
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
      })
    obs = next_obs.clone()

  return episodes, episode_reward / episodes, rollout_envs


def collect_experience(agent: Agent,
                 env: gym.Env,
                 num_collected_steps: int,
                 noise: OrnsteinUhlenbeckActionNoise,
                 num_envs: int,
                 obs,
                 privileged_obs,
                 action_size: int,
                 warm_up: bool = False):
    """Return step data over multple unrolls."""

    transitions_td = None

    # Run environment steps
    for _ in range(num_collected_steps):
        with torch.no_grad():
            if warm_up == True:
                # Sample noise and convert to PyTorch tensor
                noise_samples = np.stack([noise() for _ in range(num_envs)], axis=0)
                action = torch.from_numpy(noise_samples).float().to(agent.device)
                assert action.shape == (num_envs, action_size)
            else:
                _, action = agent.get_logits_action(obs)
                # Sample noise and convert to PyTorch tensor
                noise_samples = np.stack([noise() for _ in range(num_envs)], axis=0)
                noise_sample = torch.from_numpy(noise_samples).float().to(agent.device)
                assert noise_sample.shape == (num_envs, action_size)
                action = action + noise_sample # (num_envs, action_size)
            action = torch.clamp(action, -1.0, 1.0)
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
            next_privileged_obs=next_privileged_obs,
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
    env_name: str = 'biped',
    num_envs: int = 100,
    episode_length: int = 1000,
    seed: int = 0,
    device: str = 'cuda',
    total_timesteps: int = 10_150_000,
    warm_up_steps: int = 100,
    batch_size: int = 256,
    discounting: float = .97,                   # From original paper.
    action_repeat: int = 1,
    tau: float = 0.005,                         # From SKRL
    learning_rate_actor: float = 5e-4,          # From original paper.
    learning_rate_critic: float = 5e-4,         # From original paper.
    max_memory: int = 1e8,
    gradient_steps: int = 1,
    num_collected_steps: int = 1,
    progress_fn: Optional[Callable[[int, Dict[str, Any]], None]] = None,
):
    """Trains a policy via DDPG."""
    
    # Set seeds.
    torch.manual_seed(seed)
    np.random.seed(seed)

    if env_name == 'biped':
        from robot_learning.src.jax.envs.biped import Biped
        env = Biped()
    elif env_name == 'ant':
        from robot_learning.src.jax.envs.ant import Ant
        env = Ant()
    elif env_name == 'pendulum':
        from robot_learning.src.jax.envs.pendulum import Pendulum
        env = Pendulum()
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
    policy_layers = [obs_size, 400, 300, action_size]
    value_layers = [priviliged_state_size + action_size, 400, 300, 1]
    agent = Agent(policy_layers, value_layers, discounting=discounting,
                  action_size=action_size,
                tau=tau, device=device)    
    # agent.policy_b = init_model_weights(agent.policy_b, low=0.0, high=0.001, seed=seed)
    # agent.value_b = init_model_weights(agent.value_b,  low=0.0, high=0.001, seed=seed)

    agent = torch.jit.script(agent.to(device))

    # Initialize the memory.
    memory = ReplayBuffer(max_size=max_memory)

    # Optimizers.
    actor_optimizer = optim.Adam(agent.policy_b.parameters(), lr=learning_rate_actor)
    critic_optimizer = optim.Adam(agent.value_b.parameters(), lr=learning_rate_critic)

    # Exploration noise.
    ou_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(action_size),
                                                sigma=0.1 * np.ones(action_size),
                                                theta=0.15)

    sps = 0
    total_steps = 0
    total_loss = 0

    # nb_episodes = num_timesteps // (num_envs * episode_length)
    print('Training hyperparameters:')
    print(f'    total_timesteps: {total_timesteps}')
    print(f'    num_envs: {num_envs}')
    print(f'    episode_length: {episode_length}')
    print(f'    batch_size: {batch_size}')
    print(f'    gradient_steps: {gradient_steps}')
    print(f'    learning_rate_actor: {learning_rate_actor}')
    print(f'    learning_rate_critic: {learning_rate_critic}')
    print(f'    discounting: {discounting}')
    print(f'    tau: {tau}')
    
    # MAIN LOOP.
    total_v_loss = 0.0
    total_p_loss = 0.0
    output_reset = env.reset()
    obs = output_reset['state']
    privileged_obs = output_reset['privileged_state']
    
    evaluation_frequency = 1000
    evaluation_steps = 0
    
    print('Training ...')
    pbar = tqdm.tqdm(total=total_timesteps, desc="Training Progress")
    nb_episodes = 100
    counter_episodes = 0
    for i in range(nb_episodes):
        print(f'Training episode {i} ...')
        ou_noise.reset()
        output_reset = env.reset()
        obs = output_reset['state']
        privileged_obs = output_reset['privileged_state']

        # Training one episode.
        for tt in range(episode_length):
            t = time.time()
            # Collect experience
            if total_steps < warm_up_steps:
                warm_up = True
            else:
                warm_up = False
            obs, privileged_obs, transitions_td = collect_experience(agent,
                                        env,
                                        num_collected_steps=num_collected_steps,
                                        noise=ou_noise,
                                        num_envs=num_envs,
                                        obs=obs,
                                        privileged_obs=privileged_obs,
                                        action_size=action_size,
                                        warm_up=warm_up)

            # Update the number of steps.
            total_steps += transitions_td.obs.shape[0]

            # Store transition.
            memory.add_trajectory_data(transitions_td)
            if memory.current_sz() < batch_size:
                print('Not enough data in memory to train, skipping episode')
                continue

            # Gradient steps with progress bar
            gradient_pbar = tqdm.tqdm(range(gradient_steps), desc="Gradient Steps", leave=False)
            for _ in gradient_pbar:
                # Sample a minibatch of transitions.
                train_data_batch = memory.sample(batch_size=batch_size)
                
                agent.update_normalization(train_data_batch.obs)
                agent.update_normalization_privileged(train_data_batch.privileged_obs)

                v_loss = agent.critic_loss(train_data_batch._asdict())
                critic_optimizer.zero_grad()
                v_loss.backward()
                # Clip critic gradients
                torch.nn.utils.clip_grad_norm_(agent.value_b.parameters(), max_norm=1.0)
                critic_optimizer.step()
                total_v_loss += v_loss.item()

                p_loss = agent.policy_loss(train_data_batch._asdict())
                actor_optimizer.zero_grad()
                p_loss.backward()
                # Clip actor gradients
                torch.nn.utils.clip_grad_norm_(agent.policy_b.parameters(), max_norm=1.0)
                actor_optimizer.step()
                total_p_loss += p_loss.item()

                # Update target networks.
                agent.value_t = soft_update(agent.value_t, agent.value_b, tau)
                agent.policy_t = soft_update(agent.policy_t, agent.policy_b, tau)

        duration = time.time() - t
        denom = episode_length * gradient_steps  # We do gradient_steps updates per episode.
        if denom > 0:
            total_loss = (total_v_loss + total_p_loss) / denom
        sps = num_envs * episode_length / duration if duration > 0 else 0
        
        # Update main progress bar
        pbar.update(num_collected_steps * num_envs)
        pbar.set_postfix({
            'total_steps': total_steps,
            'sps': f'{sps:.2f}',
            'v_loss': f'{total_v_loss:.4f}',
            'p_loss': f'{total_p_loss:.4f}'
        })

        # Run evaluation.
        if total_steps % evaluation_frequency == 0:
            evaluation_steps += 1
            if progress_fn:
                t = time.time()
                with torch.no_grad():
                    episode_count, episode_reward, rollout_envs = eval_unroll(agent, env, episode_length, num_envs)
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
                    'losses/v_loss': total_v_loss,
                    'losses/p_loss': total_p_loss,
                }
                progress_fn(total_steps, progress)

                # Visualize the rollout.
                render_video = True
                if render_video == True:
                    print('Rendering eval video ...')
                    utils_np.generate_video(render_fn=render_fn,
                                            rollout_envs=rollout_envs,
                                            num_envs=num_envs,
                                            ctrl_dt=ctrl_dt,
                                            eval_i=evaluation_steps,
                                            folder_name=ABS_FOLDER_RESUlTS)

                # Save the model.
                print(f'Saving model at {total_steps}')
                torch.save(agent.policy_b.state_dict(), ABS_FOLDER_RESUlTS + f'/ddpg_model_pytorch_{total_steps}.pth')


def progress(num_steps, metrics):
    print(f'steps: {num_steps}, \
            reward: {metrics["eval/episode_reward"]}, \
            eval_sps: {metrics["speed/eval_sps"]}, \
            train_sps: {metrics["speed/sps"]}')
    times.append(datetime.now())
    xdata.append(num_steps)
    ydata.append(metrics['eval/episode_reward'].cpu().numpy())
    loss_v_data.append(metrics['losses/v_loss'])
    loss_p_data.append(metrics['losses/p_loss'])
    eval_sps.append(metrics['speed/eval_sps'])
    train_sps.append(metrics['speed/sps'])
    
    fig, ax = plt.subplots()
    ax.set_xlabel('# environment steps')
    ax.set_ylabel('reward per episode')
    ax.plot(xdata, ydata)
    plt.savefig(ABS_FOLDER_RESUlTS + '/ddpg_training.png')
    
    _, axs = plt.subplots(2, 1)
    axs[0].plot(xdata, loss_v_data, label='critic loss')
    axs[1].plot(xdata, loss_p_data, label='actor loss')
    axs[0].legend()
    axs[1].legend()
    plt.savefig(ABS_FOLDER_RESUlTS + '/ddpg_training_losses.png')

    # save to pandas df
    df = pd.DataFrame({'x': xdata, 'y': ydata, 'eval_sps': eval_sps, 'train_sps': train_sps})
    df.to_csv(ABS_FOLDER_RESUlTS + '/ddpg_training.csv', index=False)


if __name__ == '__main__':
  # temporary fix to cuda memory OOM
  os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
  print('Start training')
  xdata = []
  ydata = []
  loss_v_data = []
  loss_p_data = []
  eval_sps = []
  train_sps = []
  times = [datetime.now()]

  train(progress_fn=progress)

  print(f'time to jit: {times[1] - times[0]}')
  print(f'time to train: {times[-1] - times[1]}')
  print(f'eval steps/sec: {np.mean(eval_sps)}')
  print(f'train steps/sec: {np.mean(train_sps)}')

  train_sps = []