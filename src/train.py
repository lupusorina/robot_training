import os
import subprocess

if subprocess.run('nvidia-smi').returncode:
  raise RuntimeError(
      'Cannot communicate with GPU. '
      'Make sure you are using a GPU Colab runtime. '
      'Go to the Runtime menu and select Choose runtime type.'
  )

xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

import numpy as np
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
from ml_collections import config_dict
from wrapper import wrap_for_brax_training


FOLDER_RESULTS = 'results'
TIME_NOW = datetime.now().strftime('%Y%m%d-%H%M%S')
if not os.path.exists(os.path.join(FOLDER_RESULTS, TIME_NOW)):
    os.makedirs(os.path.join(FOLDER_RESULTS, TIME_NOW))
FOLDER_RESULTS = os.path.join(FOLDER_RESULTS, TIME_NOW)
FOLDER_PLOTS = 'plots'
if not os.path.exists(os.path.join(FOLDER_PLOTS, TIME_NOW)):
    os.makedirs(os.path.join(FOLDER_PLOTS, TIME_NOW))
FOLDER_PLOTS = os.path.join(FOLDER_PLOTS, TIME_NOW)
VIDEO_FOLDER = 'videos'
if not os.path.exists(os.path.join(VIDEO_FOLDER, TIME_NOW)):
    os.makedirs(os.path.join(VIDEO_FOLDER, TIME_NOW))
VIDEO_FOLDER = os.path.join(VIDEO_FOLDER, TIME_NOW)

env_name = 'Biped'
env = Biped()

from utils import draw_joystick_command

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

env_name = 'Biped'
env = Biped()

x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]

def progress(num_steps, metrics):
  clear_output(wait=True)

  times.append(datetime.now())
  x_data.append(num_steps)
  y_data.append(metrics["eval/episode_reward"])
  y_dataerr.append(metrics["eval/episode_reward_std"])

  plt.xlim([0, ppo_params["num_timesteps"] * 1.25])
  plt.xlabel("# environment steps")
  plt.ylabel("reward per episode")
  plt.title(f"y={y_data[-1]:.3f}")
  plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")
  plt.savefig(f'{FOLDER_PLOTS}/reward.png')
  # display(plt.gcf())
  
ppo_training_params = dict(ppo_params)
network_factory = ppo_networks.make_ppo_networks
if "network_factory" in ppo_params:
  del ppo_training_params["network_factory"]
  network_factory = functools.partial(
      ppo_networks.make_ppo_networks,
      **ppo_params.network_factory
  )

train_fn = functools.partial(
    ppo.train, **dict(ppo_training_params),
    network_factory=network_factory,
    progress_fn=progress
)

make_inference_fn, params, metrics = train_fn(
    environment=env,
    progress_fn=progress,
    eval_env=env,
    wrap_env_fn=wrap_for_brax_training,
)
print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")

MODEL_NAME = 'model_brax.pkl'
FOLDER_SAVE = 'logs'
if not os.path.exists(FOLDER_SAVE):
    os.makedirs(FOLDER_SAVE)
model.save_params(f'{FOLDER_SAVE}/{MODEL_NAME}', params)

print("Training Complete")

eval_env = env
jit_reset = jax.jit(eval_env.reset)
jit_step = jax.jit(eval_env.step)
jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))
print(f'JITing reset and step')
reset_fn = jax.jit(env.reset)
step_fn = jax.jit(env.step)

rng = jax.random.PRNGKey(1)

rollout = []
modify_scene_fns = []

x_vel = 1.0  #@param {type: "number"}
y_vel = 0.0  #@param {type: "number"}
yaw_vel = 0.0  #@param {type: "number"}
command = jp.array([x_vel, y_vel, yaw_vel])

phase_dt = 2 * jp.pi * eval_env.ctrl_dt * 1.5
phase = jp.array([0, jp.pi])

for j in range(1):
  print(f"episode {j}")
  state = jit_reset(rng)
  state.info["phase_dt"] = phase_dt
  state.info["phase"] = phase
  for i in range(1000):
    act_rng, rng = jax.random.split(rng)
    ctrl, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_step(state, ctrl)
    if state.done:
      break
    state.info["command"] = command
    rollout.append(state)

    xyz = np.array(state.data.xpos[eval_env._mj_model.body("torso").id])
    xyz += np.array([0, 0.0, 0])
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
    width=640*2,
    height=480,
    modify_scene_fns=mod_fns,
)

# media.show_video(frames, fps=fps, loop=False)
media.write_video(f'{VIDEO_FOLDER}/joystick.mp4', frames, fps=fps)