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

NAME_ROBOT = 'ant'
if NAME_ROBOT == 'ant':
  import robot_learning.src.assets.ant.config as robot_config
else:
  raise ValueError(f'NAME_ROBOT must be "ant"')
print('NAME_ROBOT:', NAME_ROBOT)

XML_PATH = robot_config.XML_PATH
ROOT_BODY = robot_config.ROOT_BODY

def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.001,
      use_contact_forces=False,
      exclude_current_positions_from_observation=True,
      reward_config=config_dict.create(
        ctrl_cost_weight=0.5,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
      ),
  )


class Ant(mjx_env.MjxEnv):

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
    self._torso_body_id = self._mj_model.body(ROOT_BODY).id
    
    # Set the timesteps.
    self._mj_model.opt.timestep = config.sim_dt
    self.ctrl_dt = config.ctrl_dt
    self._sim_dt = config.sim_dt

    self._ctrl_cost_weight = config.reward_config.ctrl_cost_weight
    self._use_contact_forces = config.use_contact_forces
    self._contact_cost_weight = config.reward_config.contact_cost_weight
    self._healthy_reward = config.reward_config.healthy_reward
    self._terminate_when_unhealthy = config.reward_config.terminate_when_unhealthy
    self._healthy_z_range = config.reward_config.healthy_z_range
    self._contact_force_range = config.reward_config.contact_force_range
    self._reset_noise_scale = config.reward_config.reset_noise_scale
    self._exclude_current_positions_from_observation = config.exclude_current_positions_from_observation

    if self._use_contact_forces:
      raise NotImplementedError('use_contact_forces not implemented.')
  
    # Create the mjx model.
    self._mjx_model = mjx.put_model(self._mj_model)
    
    # Initialize the action space.
    self.nb_joints = self.mj_model.njnt - 1 # First joint is freejoint.
    print(f"Number of joints: {self.nb_joints}")
    
    # Control action names.
    self.name_actuators = []
    for i in range(0, self.mj_model.nu):  # skip root
        self.name_actuators.append(mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i))
    print("Name of actuators:", self.name_actuators)
    
    # Initialize the initial state.
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)


  def reset(self, rng: jax.Array) -> mjx_env.State:
    """Resets the environment to an initial state."""
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)
    
    # x=+U(-0.5, 0.5), y=+U(-0.5, 0.5), yaw=U(-3.14, 3.14).
    rng, key = jax.random.split(rng)
    dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
    quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    new_quat = math.quat_mul(qpos[3:7], quat)
    qpos = qpos.at[3:7].set(new_quat)

    # qpos[7:]=*U(0.5, 1.5)
    low, hi = -self._reset_noise_scale, self._reset_noise_scale
    rng, key = jax.random.split(rng)
    qpos = qpos.at[7:].set(
      qpos[7:] * jax.random.uniform(key, (self.mj_model.nu, ), minval=low, maxval=hi))

    # d(xyzrpy)=U(-0.5, 0.5)
    rng, key = jax.random.split(rng)
    qvel = qvel.at[0:6].set(
      jax.random.uniform(key, (6,), minval=-1.0, maxval=1.0))

    # Initialize the data.
    data = mjx.make_data(self.mjx_model)
    if qpos is not None:
      data = data.replace(qpos=qpos)
    if qvel is not None:
      data = data.replace(qvel=qvel)
    if qpos[7:] is not None:
      data = data.replace(ctrl=jp.zeros(self.mj_model.nu))
    data = mjx.forward(self.mjx_model, data)

    # Initialize the observation.
    obs = self._get_obs(data)

    # Initialize the reward and done.
    reward, done = jp.zeros(2)
    
    # Initialize the metrics.
    metrics = {
        'reward_forward': jp.array(0.0, dtype=jp.float32),
        'reward_survive': jp.array(0.0, dtype=jp.float32),
        'reward_ctrl': jp.array(0.0, dtype=jp.float32),
        'reward_contact': jp.array(0.0, dtype=jp.float32),
        'x_position': jp.array(0.0, dtype=jp.float32),
        'y_position': jp.array(0.0, dtype=jp.float32),
        'distance_from_origin': jp.array(0.0, dtype=jp.float32),
        'x_velocity': jp.array(0.0, dtype=jp.float32),
        'y_velocity': jp.array(0.0, dtype=jp.float32),
    }

    # Initialize the info.
    info = {
        "last_action": jp.zeros(self.action_size),
        "last_last_action": jp.zeros(self.action_size),
        "qpos": data.qpos,
        "qvel": data.qvel,
        "xfrc_applied": data.xfrc_applied,
    }

    return mjx_env.State(data, obs, reward, done, metrics, info)

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
    state.info["xfrc_applied"] = data.xfrc_applied

    # Compute the reward.
    velocity = (data.qpos[0:2] - state.data.qpos[0:2]) / self._sim_dt
    # jax.debug.print("velocity: {}", velocity)
    forward_reward = velocity[0]

    min_z, max_z = self._healthy_z_range
    is_healthy = jp.where(state.data.qpos[2] < min_z, 0.0, 1.0)
    is_healthy = jp.where(state.data.qpos[2] > max_z, 0.0, is_healthy)
    if self._terminate_when_unhealthy:
      healthy_reward = self._healthy_reward
    else:
      healthy_reward = self._healthy_reward * is_healthy
    ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
    contact_cost = 0.0

    # Get the observation.
    obs = self._get_obs(data)

    # Get the reward and done.
    reward = forward_reward + healthy_reward - ctrl_cost - contact_cost
    done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
    done = done.astype(reward.dtype)

    # Update the metrics with scalar values
    metrics = {
        'reward_forward': forward_reward,
        'reward_survive': healthy_reward,
        'reward_ctrl': -ctrl_cost,
        'reward_contact': -contact_cost,
        'x_position': data.qpos[0],
        'y_position': data.qpos[1],
        'distance_from_origin': math.norm(data.qpos[0:2]),
        'x_velocity': velocity[0],
        'y_velocity': velocity[1],
    }

    state = state.replace(data=data, obs=obs, reward=reward, done=done, metrics=metrics)
    return state

  def _get_obs(self, data: mjx.Data) -> jax.Array:
    """Observe ant body position and velocities."""
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()

    if self._exclude_current_positions_from_observation:
      qpos = data.qpos[2:]

    state = jp.hstack([
                qpos,
                qvel
            ])

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
  xml_path = os.path.join(parent_dir, '../../assets/ant/xmls/ant.xml')

  eval_env = Ant(xml_path=xml_path)
  jit_reset = jax.jit(eval_env.reset)
  jit_step = jax.jit(eval_env.step)
  print(f'JITing reset and step')
  rng = jax.random.PRNGKey(1)

  rollout = []
  modify_scene_fns = []

  state = jit_reset(rng)

  # create a df to store the state.metrics data
  metrics_list = []
  ctrl_list = []
  state_list = []
  for i in range(1400):
    print(i)
    time_duration = time.time()

    ctrl = jp.zeros(eval_env.action_size)
    state = jit_step(state, ctrl)
    state_list.append(state.obs["state"])
    metrics_list.append(state.metrics)
    if state.done:
      break
    rollout.append(state)

    xyz = np.array(state.data.xpos[eval_env._mj_model.body("torso").id])
    xyz += np.array([0.0, 0.0, 0.0])
    x_axis = state.data.xmat[eval_env._torso_body_id, 0]
    yaw = -np.arctan2(x_axis[1], x_axis[0])
    # modify_scene_fns.append(
    #     functools.partial(
    #         draw_joystick_command,
    #         xyz=xyz,
    #         theta=yaw,
    #     )
    # )
    time_diff = time.time() - time_duration

  render_every = 1
  fps = 1.0 / eval_env.ctrl_dt / render_every
  print(f"fps: {fps}")
  traj = rollout[::render_every]
  # mod_fns = modify_scene_fns[::render_every]

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
      # modify_scene_fns=mod_fns,
  )

  # media.show_video(frames, fps=fps, loop=False)
  media.write_video(f'{NAME_ROBOT}_joystick_testing.mp4', frames, fps=fps)
  print('Video saved')