try:
  print('Checking that the installation succeeded:')
  import mujoco

  mujoco.MjModel.from_xml_string('<mujoco/>')
except Exception as e:
  raise e from RuntimeError(
      'Something went wrong during installation. Check the shell output above '
      'for more information.\n'
      'If using a hosted Colab runtime, make sure you enable GPU acceleration '
      'by going to the Runtime menu and selecting "Choose runtime type".'
  )

print('Installation successful.')

import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=100)

import functools
import os
import subprocess

import jax
from jax import numpy as jp
from matplotlib import pyplot as plt
import mediapy as media
import mujoco
import numpy as np

import pandas as pd

from utils import draw_joystick_command
import time
import biped_berkeley as bb

env_name = bb.NAME_ROBOT
print(f'env_name: {env_name}')
env = bb.Biped()
eval_env = env

jit_reset = jax.jit(eval_env.reset)
jit_step = jax.jit(eval_env.step)
print(f'JITing reset and step')
reset_fn = jax.jit(env.reset)
step_fn = jax.jit(env.step)

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
for i in range(1000):
  print(i)
  time_duration = time.time()
  act_rng, rng = jax.random.split(rng)
  ctrl = jp.zeros_like(10)
  state = jit_step(state, ctrl)
  metrics_list.append(state.metrics)
  if state.done:
    break
  state.info["command"] = command
  rollout.append(state)

  xyz = np.array(state.data.xpos[eval_env._mj_model.body("base_link").id])
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
    width=640*2,
    height=480,
    modify_scene_fns=mod_fns,
)

media.show_video(frames, fps=fps, loop=False)
