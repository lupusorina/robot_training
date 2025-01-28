import os

import numpy as np

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

from datetime import datetime
import functools
from biped import Biped
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.io import model
from IPython.display import HTML, clear_output
import jax
from jax import numpy as jp
from matplotlib import pyplot as plt
import mediapy as media
import mujoco
import numpy as np
import glfw 

import sys
sys.path.append('../')

from ml_collections import config_dict

jax.config.update("jax_platform_name", "cpu")

current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
folder = 'runs/' + current_time
os.makedirs(folder, exist_ok=True)

if not glfw.init():
  raise Exception("glfw failed to initialize")

brax_ppo_config = config_dict.create(
      num_timesteps=150_000_000,
      num_evals=15,
      reward_scaling=1.0,
      clipping_epsilon=0.2,
      num_resets_per_eval=1,
      episode_length=1000,
      normalize_observations=True,
      action_repeat=1,
      unroll_length=20,
      num_minibatches=32,
      num_updates_per_batch=4,
      discounting=0.97,
      learning_rate=3e-4,
      entropy_cost=0.005,
      num_envs=8192,
      batch_size=256,
      max_grad_norm=1.0,
      network_factory = config_dict.create(
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
      ),
  )

ppo_params = brax_ppo_config

# Environment.
env_name = 'Biped'
env = Biped()

rollout = []


for i in range(10):
  print(i)
  state = env.reset(rng=jax.random.PRNGKey(i))
  rollout.append(state)

render_every = 1
fps = 1.0 / env.ctrl_dt / render_every
print(f"fps: {fps}")
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
    width=640*2,
    height=480,
)

# media.show_video(frames, fps=fps, loop=False)
VIDEO_FOLDER = 'videos'
os.makedirs(VIDEO_FOLDER, exist_ok=True)
media.write_video(f'{VIDEO_FOLDER}/joystick.mp4', frames, fps=fps)

print("video written to ", f'{VIDEO_FOLDER}/joystick.mp4')

# x_data, y_data, y_dataerr = [], [], []
# times = [datetime.now()]

# def progress(num_steps, metrics):
#   clear_output(wait=True)

#   times.append(datetime.now())
#   x_data.append(num_steps)
#   y_data.append(metrics["eval/episode_reward"])
#   y_dataerr.append(metrics["eval/episode_reward_std"])

#   plt.xlim([0, ppo_params["num_timesteps"] * 1.25])
#   plt.xlabel("# environment steps")
#   plt.ylabel("reward per episode")
#   plt.title(f"y={y_data[-1]:.3f}")
#   plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")
#   plt.savefig("ppo_training.png")

#   # display(plt.gcf())
  
# ppo_training_params = dict(ppo_params)
# network_factory = ppo_networks.make_ppo_networks
# if "network_factory" in ppo_params:
#   del ppo_training_params["network_factory"]
#   network_factory = functools.partial(
#       ppo_networks.make_ppo_networks,
#       **ppo_params.network_factory
#   )

# train_fn = functools.partial(
#     ppo.train, **dict(ppo_training_params),
#     network_factory=network_factory,
#     randomization_fn=None,
#     progress_fn=progress
# )

# make_inference_fn, params, metrics = train_fn(
#     environment=env,
#     eval_env=env,
#     wrap_env_fn=wrapper.wrap_for_brax_training,
# )
# print(f"time to jit: {times[1] - times[0]}")
# print(f"time to train: {times[-1] - times[1]}")


