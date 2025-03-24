from datetime import datetime
import functools
import os

from IPython.display import HTML, clear_output

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from brax import envs
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ddpg import train as ddpg

import src.wrapper as wrapper

# Folders.
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

# Set the environment.
env_name = "ant"  # @param ['ant', 'fetch', 'grasp', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'pusher', 'reacher', 'walker2d', 'grasp', 'ur5e']
env = envs.get_environment(env_name=env_name,
                           backend="mjx")
rng = jax.random.PRNGKey(0)
rng, key = jax.random.split(rng)
state = env.reset(rng)

def progress(num_steps, metrics, agent_name):
    print(f'step: {num_steps}, reward: {metrics["eval/episode_reward"]}')
    times.append(datetime.now())
    xdata.append(num_steps)
    ydata.append(metrics['eval/episode_reward'])
    clear_output(wait=True)
    fig, ax = plt.subplots()
    ax.set_xlabel('# environment steps')
    ax.set_ylabel('reward per episode')
    ax.plot(xdata, ydata)
    plt.savefig(f'{ABS_FOLDER_RESUlTS}/{agent_name}.png')

# Train PPO.
AGENT = "PPO"
train_fn = {
  'ant': functools.partial(ppo.train,
                           num_timesteps=50_000_000,
                           num_evals=10,
                           reward_scaling=1,
                           episode_length=1000,
                           normalize_observations=True,
                           action_repeat=1,
                           unroll_length=5,
                           num_minibatches=32,
                           num_updates_per_batch=4,
                           discounting=0.97,
                           learning_rate=3e-4,
                           entropy_cost=1e-2,
                           num_envs=4096,
                           batch_size=2048,
                           seed=1),
}[env_name]

xdata, ydata = [], []
times = [datetime.now()]

make_inference_fn, params, _ = train_fn(environment=env,
                                        progress_fn=functools.partial(progress, agent_name=AGENT))

print(f'time to jit: {times[1] - times[0]}')
print(f'time to train: {times[-1] - times[1]}')

# Train DDPG.
AGENT = "DDPG"
train_fn = functools.partial(ddpg.train,
                             num_timesteps = 50_000_000,
                             episode_length=1000,
                             num_evals=10,
                             num_envs=4096,
                             batch_size=2048,
                             logdir='./logs')

xdata, ydata = [], []
times = [datetime.now()]

train_fn(environment=env,
         progress_fn=functools.partial(progress, agent_name=AGENT),
         )

print(f'time to jit: {times[1] - times[0]}')
print(f'time to train: {times[-1] - times[1]}')