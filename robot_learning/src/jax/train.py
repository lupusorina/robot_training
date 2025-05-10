import os
import subprocess
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=100)

from datetime import datetime
import functools
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from IPython.display import clear_output
from matplotlib import pyplot as plt
import numpy as np
from ml_collections import config_dict
from robot_learning.src.jax.wrapper import wrap_for_brax_training
import jax
from brax.training.agents.ppo import checkpoint as ppo_checkpoint

from robot_learning.src.jax.utils import draw_joystick_command
import time
import robot_learning.src.jax.envs.biped as bb
from etils import epath
import jax
from jax import numpy as jp
import mujoco

from tqdm import tqdm

import mediapy as media

# Set seed
jax.random.PRNGKey(0)

# Folders.
RESULTS = 'results'
if not os.path.exists(RESULTS):
    os.makedirs(RESULTS)
time_now = datetime.now().strftime('%Y%m%d-%H%M%S')
if not os.path.exists(os.path.join(RESULTS, time_now)):
    os.makedirs(os.path.join(RESULTS, time_now))
FOLDER_RESULTS = os.path.join(RESULTS, time_now)
ABS_FOLDER_RESUlTS = os.path.abspath(FOLDER_RESULTS)
print(f"Saving results to {ABS_FOLDER_RESUlTS}")

print("Available devices:", jax.devices())

# FOLDER_RESTORE_CHECKPOINT = os.path.abspath(RESULTS + '/20250429-135159/000035717120')

# Brax PPO config.
brax_ppo_config = config_dict.create(
      num_timesteps=250_000_000,
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
from robot_learning.src.jax.envs.biped import Biped
env = Biped(save_config_folder=ABS_FOLDER_RESUlTS)
eval_env = Biped(save_config_folder=ABS_FOLDER_RESUlTS)

x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]

reward_list = []

def progress(num_steps, metrics):
  clear_output(wait=True)

  times.append(datetime.now())
  x_data.append(num_steps)
  y_data.append(metrics["eval/episode_reward"])
  y_dataerr.append(metrics["eval/episode_reward_std"])

  reward_list.append([num_steps, metrics["eval/episode_reward"]])

  _, ax = plt.subplots()
  ax.set_xlim([0, ppo_params["num_timesteps"] * 1.25])
  ax.set_xlabel("# environment steps")
  ax.set_ylabel("reward per episode")
  ax.set_title(f"y={y_data[-1]:.3f}")
  ax.plot(x_data, y_data)
  ax.fill_between(x_data, np.array(y_data) - np.array(y_dataerr), np.array(y_data) + np.array(y_dataerr), alpha=0.2)
  plt.savefig(f'{ABS_FOLDER_RESUlTS}/reward.pdf')
  plt.savefig(f'{ABS_FOLDER_RESUlTS}/reward.png')
  print("Reward for {} steps: {:.3f}".format(num_steps, y_data[-1]))
  
ppo_training_params = dict(ppo_params)
network_factory = ppo_networks.make_ppo_networks
if "network_factory" in ppo_params:
  del ppo_training_params["network_factory"]
  network_factory = functools.partial(
      ppo_networks.make_ppo_networks,
      **ppo_params.network_factory
  )

from randomize import domain_randomize

train_fn = functools.partial(
    ppo.train, **dict(ppo_training_params),
    network_factory=network_factory,
    progress_fn=progress,
    randomization_fn=domain_randomize,
    save_checkpoint_path=ABS_FOLDER_RESUlTS,
    # restore_checkpoint_path=FOLDER_RESTORE_CHECKPOINT
)

make_inference_fn, params, metrics = train_fn(
    environment=env,
    eval_env=eval_env,
    wrap_env_fn=wrap_for_brax_training,
)
print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")

# Testing: load the latest weights and test the policy.
RESULTS_FOLDER_PATH = os.path.abspath('results')

# Sort by date and get the latest folder.
folders = sorted(os.listdir(RESULTS_FOLDER_PATH))
latest_folder = folders[-1]

# In the latest folder, find the latest folder, ignore the files.
folders = sorted(os.listdir(epath.Path(RESULTS_FOLDER_PATH) / latest_folder))
folders = [f for f in folders if os.path.isdir(epath.Path(RESULTS_FOLDER_PATH) / latest_folder / f)]

policy_fn_list = []
policy_folder_list = []

USE_LATEST_WEIGHTS = True
if USE_LATEST_WEIGHTS:
  latest_weights_folder = folders[-1]
  print(f'Latest weights folder: {latest_weights_folder}')
  policy_fn = ppo_checkpoint.load_policy(epath.Path(RESULTS_FOLDER_PATH) / latest_folder / latest_weights_folder)
  policy_fn_list.append(policy_fn)
  policy_folder_list.append(latest_weights_folder)
else:
  for folder in folders:
    policy_fn = ppo_checkpoint.load_policy(epath.Path(RESULTS_FOLDER_PATH) / latest_folder / folder)
    policy_fn_list.append(policy_fn)
    policy_folder_list.append(folder)

env_name = bb.NAME_ROBOT
print(f'env_name: {env_name}')

for policy_fn, folder in zip(policy_fn_list, policy_folder_list):
  print(f'{folder}')
  eval_env = bb.Biped()

  jit_reset = jax.jit(eval_env.reset)
  jit_step = jax.jit(eval_env.step)
  jit_policy = jax.jit(policy_fn)
  step_fn = jax.jit(eval_env.step)
  rng = jax.random.PRNGKey(1)

  rollout = []
  modify_scene_fns = []

  x_vel = 0.0  #@param {type: "number"}
  y_vel = 0.0  #@param {type: "number"}
  yaw_vel = 0.0  #@param {type: "number"}
  command = jp.array([x_vel, y_vel, yaw_vel])

  phase_dt = 2 * jp.pi * eval_env.ctrl_dt * 1.5
  phase = jp.array([0, jp.pi])

  state = jit_reset(rng)
  state.info["phase_dt"] = phase_dt
  state.info["phase"] = phase

  # create a df to store the state.metrics data
  metrics_list = []
  ctrl_list = []
  state_list = []
  for i in tqdm(range(1400)):
    time_duration = time.time()
    act_rng, rng = jax.random.split(rng)
    ctrl, _ = jit_policy(state.obs, act_rng)
    ctrl_list.append(ctrl)
    state = jit_step(state, ctrl)
    state_list.append(state.obs["state"])
    metrics_list.append(state.metrics)
    if state.done:
      break
    state.info["command"] = command
    rollout.append(state)

    xyz = np.array(state.data.xpos[eval_env._mj_model.body("base_link").id])
    xyz += np.array([0.0, 0.0, 0.0])
    x_axis = state.data.xmat[eval_env._torso_body_id, 0]
    yaw = -np.arctan2(x_axis[1], x_axis[0])
    modify_scene_fns.append(
        functools.partial(
            draw_joystick_command,
            cmd=state.info["command"],
            xyz=xyz,
            theta=yaw,
            scl=np.linalg.norm(state.info["command"]),
        )
    )
    time_diff = time.time() - time_duration

  render_every = 1
  fps = 1.0 / eval_env.ctrl_dt / render_every
  print(f"fps: {fps}")
  traj = rollout[::render_every]
  mod_fns = modify_scene_fns[::render_every]

  scene_option = mujoco.MjvOption()
  scene_option.geomgroup[2] = True
  scene_option.geomgroup[3] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False

  frames = eval_env.render(
      traj,
      camera="track",
      scene_option=scene_option,
      width=640,
      height=480,
      modify_scene_fns=mod_fns,
  )

  # media.show_video(frames, fps=fps, loop=False)
  ABS_FOLDER_RESUlTS = epath.Path(RESULTS_FOLDER_PATH) / latest_folder
  media.write_video(f'{ABS_FOLDER_RESUlTS}/joystick_testing_{folder}_xvel_{x_vel}_yvel_{y_vel}_yawvel_{yaw_vel}.mp4', frames, fps=fps)
  print('Video saved')