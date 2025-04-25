# Code started from Brax Ant and modified for the robot learning project.

"""Trains an ant to run in the +x direction."""

from brax import math
from typing import Optional, Dict, Any, Union
import time

import jax
from jax import numpy as jp
import mujoco
from ml_collections import config_dict

from mujoco import mjx
from mujoco.mjx._src import math

# Local imports.
import robot_learning.src.jax.mjx_env as mjx_env

NAME_ROBOT = 'pendulum'
if NAME_ROBOT == 'pendulum':
  import robot_learning.src.assets.pendulum.config as robot_config
else:
  raise ValueError(f'NAME_ROBOT must be "pendulum"')
print('NAME_ROBOT:', NAME_ROBOT)

XML_PATH = robot_config.XML_PATH
ROOT_BODY = robot_config.ROOT_BODY

def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.001,
      exclude_current_positions_from_observation=True,
      reward_config=config_dict.create(
        ctrl_cost_weight=0.001,
        max_speed=8.0,
        max_torque=2.0,
        reset_noise_scale_pos=0.4,
        reset_noise_scale_vel=1.0,
      ),
  )

class Pendulum(mjx_env.MjxEnv):
  def __init__(
      self,
      xml_path: str = XML_PATH,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides)

    self._xml_path = xml_path
    
    # Initialize the model.
    self._mj_model = mujoco.MjModel.from_xml_path(xml_path)
    self._pole_body_id = self._mj_model.body(ROOT_BODY).id
    
    # Set the timesteps.
    self._mj_model.opt.timestep = config.sim_dt
    self.ctrl_dt = config.ctrl_dt
    self._sim_dt = config.sim_dt

    self._ctrl_cost_weight = config.reward_config.ctrl_cost_weight
    self._max_speed = config.reward_config.max_speed
    self._max_torque = config.reward_config.max_torque
    self._reset_noise_scale_pos = config.reward_config.reset_noise_scale_pos
    self._reset_noise_scale_vel = config.reward_config.reset_noise_scale_vel
    self._exclude_current_positions_from_observation = config.exclude_current_positions_from_observation

    # Create the mjx model.
    self._mjx_model = mjx.put_model(self._mj_model)
    
    # Initialize the action space.
    self.nb_joints = self.mj_model.njnt # First joint is freejoint.
    print(f"Number of joints: {self.nb_joints}")
    
    # Control action names.
    self.name_actuators = []
    for i in range(0, self.mj_model.nu):
        self.name_actuators.append(mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i))
    print("Name of actuators:", self.name_actuators)
    
    # Initialize the initial state.
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)

  def reset(self, rng: jax.Array) -> mjx_env.State:
    """Resets the environment to an initial state."""
    # Initialize qpos with zeros and set the initial angle
    qpos = jp.zeros(self.mjx_model.nq)
    qvel = jp.zeros(self.mjx_model.nv)
    
    # Add noise to initial state
    rng, key = jax.random.split(rng)
    qpos = qpos.at[0].set(
        jax.random.uniform(key, (), minval=-self._reset_noise_scale_pos, maxval=self._reset_noise_scale_pos))
    
    rng, key = jax.random.split(rng)
    qvel = qvel.at[0].set(
        jax.random.uniform(key, (), minval=-self._reset_noise_scale_vel, maxval=self._reset_noise_scale_vel))

    # Initialize the data.
    data = mjx.make_data(self.mjx_model)
    if qpos is not None:
      data = data.replace(qpos=qpos)
    if qvel is not None:
      data = data.replace(qvel=qvel)
    data = mjx.forward(self.mjx_model, data)

    # Initialize the observation.
    obs = self._get_obs(data)

    # Initialize the reward and done.
    reward, done = jp.zeros(2)
    
    # Initialize the metrics.
    metrics = {
        'reward_swingup': jp.array(0.0, dtype=jp.float32),
        'reward_ctrl': jp.array(0.0, dtype=jp.float32),
        'angle': jp.array(0.0, dtype=jp.float32),
        'angular_velocity': jp.array(0.0, dtype=jp.float32),
    }

    # Initialize the info.
    info = {
        "last_action": jp.zeros(self.action_size),
        "last_last_action": jp.zeros(self.action_size),
        "qpos": data.qpos,
        "qvel": data.qvel,
    }

    return mjx_env.State(data, obs, reward, done, metrics, info)

  def _wrap_angle(self, angle: jax.Array) -> jax.Array:
    """Wrap angle to be between -pi and pi."""
    return jp.mod(angle + jp.pi, 2 * jp.pi) - jp.pi

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    """Run one timestep of the environment's dynamics."""
    
    # Step the model.
    data = mjx_env.step(
      self.mjx_model, state.data, action, self._n_substeps
    )
    
    state.info["last_last_action"] = state.info["last_action"]
    state.info["last_action"] = action
    state.info["qpos"] = data.qpos
    state.info["qvel"] = data.qvel

    # Get the angle and angular velocity
    angle = self._wrap_angle(data.qpos[0])
    angular_velocity = data.qvel[0]

    # Compute the reward (same as classic control pendulum)
    # The reward is -(theta^2 + 0.1*theta_dt^2 + 0.001*action^2)
    DES_ANGLE = jp.pi
    angle_error = self._wrap_angle(angle - DES_ANGLE)
    swingup_reward = - angle_error - 0.1 * jp.square(angular_velocity)

    ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

    # Get the observation.
    obs = self._get_obs(data)

    # Get the reward and done.
    reward = swingup_reward - ctrl_cost
    done = jp.array(0.0)  # Pendulum never terminates

    # Update the metrics
    metrics = {
        'reward_swingup': swingup_reward,
        'reward_ctrl': -ctrl_cost,
        'angle': angle,
        'angular_velocity': angular_velocity,
    }

    state = state.replace(data=data, obs=obs, reward=reward, done=done, metrics=metrics)
    return state

  def _get_obs(self, data: mjx.Data) -> jax.Array:
    """Observe pendulum angle and angular velocity."""
    angle = self._wrap_angle(data.qpos[0])
    angular_velocity = data.qvel[0]
    
    # Convert angle to cos and sin to avoid angle wrapping issues
    cos_angle = jp.cos(angle)
    sin_angle = jp.sin(angle)
    
    state = jp.array([cos_angle, sin_angle, angular_velocity])

    return {
      "state": state,
      "privileged_state": state,
    }

   # Accessors.
  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self.nb_joints

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model

  @property
  def _n_substeps(self) -> int:
    """Number of sim steps per control step."""
    return int(round(self.ctrl_dt / self._sim_dt))

import mediapy as media
import numpy as np
import os

if __name__ == "__main__":
  parent_dir = os.path.abspath(os.path.join(os.getcwd()))
  xml_path = os.path.join(parent_dir, '../../assets/pendulum/xmls/pendulum.xml')

  eval_env = Pendulum(xml_path=xml_path)
  jit_reset = jax.jit(eval_env.reset)
  jit_step = jax.jit(eval_env.step)
  print(f'JITing reset and step')
  rng = jax.random.PRNGKey(1)

  rollout = []
  state = jit_reset(rng)

  # create a df to store the state.metrics data
  metrics_list = []
  ctrl_list = []
  state_list = []
  for i in range(1400):
    print(i)
    time_duration = time.time()

    # Apply a simple sinusoidal control
    ctrl = jp.array([2.0 * jp.sin(i * 0.1)])
    state = jit_step(state, ctrl)
    state_list.append(state.obs["state"])
    metrics_list.append(state.metrics)
    rollout.append(state)

    time_diff = time.time() - time_duration

  render_every = 1
  fps = 1.0 / eval_env.ctrl_dt / render_every
  print(f"fps: {fps}")
  traj = rollout[::render_every]

  scene_option = mujoco.MjvOption()
  scene_option.geomgroup[2] = True
  scene_option.geomgroup[3] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False

  frames = eval_env.render(
      traj,
      camera="lookat",
      scene_option=scene_option,
      width=640,
      height=480,
  )

  media.write_video(f'{NAME_ROBOT}_swingup_testing.mp4', frames, fps=fps)
  print('Video saved')