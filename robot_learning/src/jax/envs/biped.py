"""
  Joystick task for the Caltech Biped.
  Modified by Sorina Lupu (eslupu@caltech.edu) from the Berkeley biped code from MuJoCo playground
  https://github.com/google-deepmind/mujoco_playground/
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np
import mujoco.viewer

# Local imports.
import robot_learning.src.jax.utils as utils
from robot_learning.src.jax.utils import geoms_colliding
import robot_learning.src.jax.mjx_env as mjx_env

import os
import json

import time

# Constants.
NAME_ROBOT = 'biped'
if NAME_ROBOT == 'biped':
  import robot_learning.src.assets.biped.config as robot_config
else:
  raise ValueError(f'NAME_ROBOT must be "biped"')
print('NAME_ROBOT:', NAME_ROBOT)

XML_PATH = robot_config.XML_PATH
ROOT_BODY = robot_config.ROOT_BODY
FEET_SITES = robot_config.FEET_SITES
FEET_GEOMS = robot_config.FEET_GEOMS
GRAVITY_SENSOR = robot_config.GRAVITY_SENSOR
GLOBAL_LINVEL_SENSOR = robot_config.GLOBAL_LINVEL_SENSOR
GLOBAL_ANGVEL_SENSOR = robot_config.GLOBAL_ANGVEL_SENSOR
LOCAL_LINVEL_SENSOR = robot_config.LOCAL_LINVEL_SENSOR
ACCELEROMETER_SENSOR = robot_config.ACCELEROMETER_SENSOR
GYRO_SENSOR = robot_config.GYRO_SENSOR
IMU_SITE = robot_config.IMU_SITE
HIP_JOINT_NAMES = robot_config.HIP_JOINT_NAMES
DESIRED_HEIGHT = robot_config.DESIRED_HEIGHT

def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.01,
      sim_dt=0.001,
      episode_length=1000,
      action_repeat=1,
      history_len=3,
      action_delay=1,
      soft_joint_pos_limit_factor=0.95,
      noise_config=config_dict.create(
          level=1.0,  # Set to 0.0 to disable noise.
          scales=config_dict.create(
              hip_pos=0.03,  # rad
              kfe_pos=0.05,
              ffe_pos=0.08,
              faa_pos=0.03,
              joint_vel=1.5,  # rad/s
              gravity=0.05,
              linvel=0.1,
              gyro=0.2,  # angvel.
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Tracking related rewards.
              tracking_lin_vel=1.0,
              tracking_ang_vel=0.5,
              # Base related rewards.
              lin_vel_z=0.0,
              ang_vel_xy=-0.15,
              orientation=-1.0,
              base_height=0.0,
              # Energy related rewards.
              torques=-2.5e-4,
              action_rate=-2e-4,
              # Feet related rewards.
              feet_air_time=2.0,
              feet_slip=-0.25,
              feet_height=0.0,
              feet_phase=1.0,
              # Other rewards.
              alive=0.0,
              termination=-1.0,
              # Pose related rewards.
              joint_deviation_knee=-0.1,
              joint_deviation_hip=-0.25,
              dof_pos_limits=-1.0,
              pose=-1.0,
          ),
          max_foot_height=0.15,
          base_height_target=DESIRED_HEIGHT,
      ),
      push_config=config_dict.create(
          enable=True,
          interval_range=[5.0, 10.0],
          magnitude_range=[0.05, 1.0],
      ),
      lin_vel_x=[-0.2, 0.2],
      lin_vel_y=[-0.2, 0.2],
      ang_vel_yaw=[-0.2, 0.2],
  )

class Biped(mjx_env.MjxEnv):
  """Track a joystick command."""

  def __init__(
      self,
      xml_path: str = XML_PATH,
      save_config_folder: str = None,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides)

    # Initialize the model.
    self._mj_model = mujoco.MjModel.from_xml_path(xml_path)

    # Set the timesteps.
    self._mj_model.opt.timestep = config.sim_dt
    self.ctrl_dt = config.ctrl_dt
    self._sim_dt = config.sim_dt

    # Set the rendering size.
    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160

    # Create the mjx model.
    self._mjx_model = mjx.put_model(self._mj_model)
    self._xml_path = xml_path
    
    # Initialize the action space.
    self.idx_actuators_dict = {}
    for i in range(0, self.mj_model.nu):  # skip root
        self.idx_actuators_dict[mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)] = i
    print(f'Model info:')
    print(f'  Name actuators and corresponding index: {self.idx_actuators_dict}')
    print(f'  self.mj_model.nu: {self.mj_model.nu}')

    self.name_joints = []
    for i in range(0, self.mj_model.njnt):
        self.name_joints.append(mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i))
    print(f'  Name joints: {self.name_joints}')

    list_joint_names_to_ignore = ['root', 'L_SPRING_ROLL', 'L_SPRING_PITCH', 'R_SPRING_ROLL', 'R_SPRING_PITCH']

    self.joint_idx_to_ignore_dict = {}
    for name in list_joint_names_to_ignore:
      if mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name) == -1 or \
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name) == 0:
        continue
      self.joint_idx_to_ignore_dict[name] = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name) - 1 # First joint is freejoint.

    self.action_space = jp.zeros(self.action_size)
    print(f'JOINT IDX TO IGNORE: {self.joint_idx_to_ignore_dict}')

    self.actuated_joint_names_to_policy_idx_dict = {
      "L_YAW": 0,
      "L_HAA": 1,
      "L_HFE": 2,
      "L_KFE": 3,
      "R_YAW": 4,
      "R_HAA": 5,
      "R_HFE": 6,
      "R_KFE": 7,
    }

    for name in self.actuated_joint_names_to_policy_idx_dict:
      assert name in self.idx_actuators_dict, f"{name} is not in {self.idx_actuators_dict.keys()}"


    self.joint_names_to_actuator_idx_dict = { name: int(self.mj_model.joint(name).qposadr[0] - 7) \
                                              for name in self.idx_actuators_dict.keys() }

    self.policy_idx_to_actuator_idx_dict = { self.actuated_joint_names_to_policy_idx_dict[name]: self.joint_names_to_actuator_idx_dict[name] 
                                 for name in self.actuated_joint_names_to_policy_idx_dict }

    # Save config files.
    if save_config_folder is not None:
      with open(os.path.join(save_config_folder, 'policy_actuator_mapping.json'), 'w') as f:
        json.dump({
          'actuated_joint_names_to_policy_idx_dict': self.actuated_joint_names_to_policy_idx_dict,
        }, f)

      with open(os.path.join(save_config_folder, 'config.json'), 'w') as f:
        config_dict = self._config.to_dict()
        json.dump(config_dict, f)
 
      with open(os.path.join(save_config_folder, 'idx_actuators_dict.json'), 'w') as f:
        json.dump(self.idx_actuators_dict, f)
    self._post_init()

    dict_initial_qpos = {}
    for i in range(self.mj_model.njnt): # First joint is freejoint.
      name = self.mj_model.joint(i).name
      if name in self.joint_idx_to_ignore_dict.keys():
        continue
      dict_initial_qpos[name] = float(self._mj_model.keyframe("home").qpos[self.mj_model.joint(name).qposadr[0]])

    # Save the initial qpos to a file.
    if save_config_folder is not None:
      with open(os.path.join(save_config_folder, 'initial_qpos.json'), 'w') as f:
        json.dump(dict_initial_qpos, f)

    # Initialize state history buffers
    self._state_history = None
    self._privileged_state_history = None

    self.reset(rng=jax.random.PRNGKey(0))

  def _post_init(self) -> None:
    # Initialize the initial state.
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_q_joints = jp.array(self._mj_model.keyframe("home").qpos[7:])
    self._default_q_joints_without_spring = jp.array([self._default_q_joints[i] for i in range(len(self._default_q_joints)) if i not in self.joint_idx_to_ignore_dict.values()])

    print('model.actuator_gainprm: ', self.mjx_model.actuator_gainprm)
    print('model.actuator_biasprm: ', self.mjx_model.actuator_biasprm)
    # Initialize the soft limits.
    q_j_min, q_j_max = self.mj_model.jnt_range[1:].T # Note: First joint is freejoint.
    c = (q_j_min + q_j_max) / 2
    r = q_j_max - q_j_min
    self._soft_q_j_min = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
    self._soft_q_j_max = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

    # Initialize the noise scale.
    q_j_noise_scale = np.zeros(self.mj_model.nu)

    hip_indices = []
    hip_joint_names = robot_config.HIP_JOINT_NAMES
    if len(hip_joint_names) != 0:
        for side in robot_config.SIDES:
            for joint_name in hip_joint_names:
                hip_indices.append(self._mj_model.joint(f"{side}_{joint_name}").qposadr[0] - 7)
        self._hip_indices = jp.array(hip_indices) # For hip reward deviation.
        q_j_noise_scale[self._hip_indices] = self._config.noise_config.scales.hip_pos

    knee_indices = []
    knee_joint_names = robot_config.KNEE_JOINT_NAMES
    if len(knee_joint_names) != 0:
        for side in robot_config.SIDES:
            for knee_joint_name in knee_joint_names:
                knee_indices.append(self._mj_model.joint(f"{side}_{knee_joint_name}").qposadr[0] - 7)
        self._knee_indices = jp.array(knee_indices) # For knee reward deviation.
        q_j_noise_scale[self._knee_indices] = self._config.noise_config.scales.kfe_pos

    ffe_joint_names = robot_config.ANKLE_FE_JOINT_NAMES
    if len(ffe_joint_names) != 0:
        ffe_indices = []
        for side in robot_config.SIDES:
            for ffe_joint_name in ffe_joint_names:
                ffe_indices.append(self._mj_model.joint(f"{side}_{ffe_joint_name}").qposadr[0] - 7)
        self._ffe_indices = jp.array(ffe_indices) # For ffe reward deviation.
        q_j_noise_scale[self._ffe_indices] = self._config.noise_config.scales.ffe_pos

    faa_joint_names = robot_config.ANKLE_AA_JOINT_NAMES
    if len(faa_joint_names) != 0:
        faa_indices = []
        for side in robot_config.SIDES:
            for faa_joint_name in faa_joint_names:
                faa_indices.append(self._mj_model.joint(f"{side}_{faa_joint_name}").qposadr[0] - 7)
        self._faa_indices = jp.array(faa_indices) # For faa reward deviation.
        q_j_noise_scale[self._faa_indices] = self._config.noise_config.scales.faa_pos
    self._q_j_noise_scale = jp.array(q_j_noise_scale)

    # Initialize the site and geom ids.
    self._imu_site_id = self._mj_model.site(IMU_SITE).id
    self._torso_body_id = self._mj_model.body(ROOT_BODY).id
    self._feet_site_id = np.array([self._mj_model.site(name).id for name in FEET_SITES])
    self._floor_geom_id = self._mj_model.geom("floor").id
    self._feet_geom_id = np.array([self._mj_model.geom(name).id for name in FEET_GEOMS])

    foot_global_linvel_sensor_adr = []
    for site in FEET_SITES:
      sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
      sensor_adr = self._mj_model.sensor_adr[sensor_id]
      sensor_dim = self._mj_model.sensor_dim[sensor_id]
      foot_global_linvel_sensor_adr.append(
          list(range(sensor_adr, sensor_adr + sensor_dim))
      )
    self._foot_global_linvel_sensor_adr = jp.array(foot_global_linvel_sensor_adr)

  def reset(self, rng: jax.Array) -> mjx_env.State:
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
    rng, key = jax.random.split(rng)
    # Create a mask for non-spring joints
    non_spring_mask = jp.ones(self.mj_model.nu)
    for idx in self.joint_idx_to_ignore_dict.values():
        non_spring_mask = non_spring_mask.at[idx].set(0.0)
    # Add noise only to non-spring joints
    noise = jax.random.uniform(key, (self.mj_model.nu,), minval=-0.1, maxval=0.1) * non_spring_mask
    qpos = qpos.at[7:].set(qpos[7:] * (1.0 + noise))

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
      data = data.replace(ctrl=qpos[7:])
    data = mjx.forward(self.mjx_model, data)

    # Phase, freq=U(1.0, 1.5)
    rng, key = jax.random.split(rng)
    gait_freq = jax.random.uniform(key, (1,), minval=1.25, maxval=1.5)
    phase_dt = 2 * jp.pi * self.ctrl_dt * gait_freq
    phase = jp.array([0, jp.pi])

    # Sample the command.
    rng, cmd_rng = jax.random.split(rng)
    cmd = self.sample_command(cmd_rng)

    # Sample push interval.
    rng, push_rng = jax.random.split(rng)
    push_interval = jax.random.uniform(
        push_rng,
        minval=self._config.push_config.interval_range[0],
        maxval=self._config.push_config.interval_range[1],
    )
    push_interval_steps = jp.round(push_interval / self.ctrl_dt).astype(jp.int32)

    info = {
        "rng": rng,
        "step": 0,
        "command": cmd,
        "last_act": jp.zeros(self.action_size),
        "last_last_act": jp.zeros(self.action_size),
        "motor_targets": jp.zeros(self.mjx_model.nu),
        "feet_air_time": jp.zeros(2),
        "last_contact": jp.zeros(2, dtype=bool),
        "swing_peak": jp.zeros(2),
        # Phase related.
        "phase_dt": phase_dt,
        "phase": phase,
        # Push related.
        "push": jp.array([0.0, 0.0]),
        "push_step": 0,
        "push_interval_steps": push_interval_steps,
        "qpos": data.qpos,
        "qvel": data.qvel,
        "xfrc_applied": data.xfrc_applied,
        # Initialize action history
        "action_history": jp.zeros((self._config.action_delay + 1, self.mjx_model.nu)),
    }

    # Initialize the metrics.
    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    metrics["swing_peak"] = jp.zeros(())
    
    # Initialize the contact.
    contact = jp.array([
        geoms_colliding(data, geom_id, self._floor_geom_id)
        for geom_id in self._feet_geom_id
    ])

    # Initialize the observation.
    obs = self._get_obs(data, info, contact, None, None)

    # Initialize the reward and done.
    reward, done = jp.zeros(2)

    # Return the state.
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    state.info["rng"], push1_rng, push2_rng = jax.random.split(state.info["rng"], 3)

    push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
    push_magnitude = jax.random.uniform(
        push2_rng,
        minval=self._config.push_config.magnitude_range[0],
        maxval=self._config.push_config.magnitude_range[1],
    )
    push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
    push *= (
        jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"])
        == 0
    )
    push *= self._config.push_config.enable
    qvel = state.data.qvel
    qvel = qvel.at[:2].set(push * push_magnitude + qvel[:2])
    data = state.data.replace(qvel=qvel)
    state = state.replace(data=data)

    # Step the model.
    action_complete = jp.zeros(self.mjx_model.nu)
    for name, policy_idx in self.actuated_joint_names_to_policy_idx_dict.items():
      if policy_idx is None:
        continue
      action_complete = action_complete.at[self.policy_idx_to_actuator_idx_dict[policy_idx]].set(action[policy_idx])

    motor_targets = self._default_q_joints + action_complete
    state.info["action_history"] = jp.roll(state.info["action_history"], -1, axis=0)
    state.info["action_history"] = state.info["action_history"].at[-1].set(motor_targets)

    data = mjx_env.step(
      self.mjx_model, state.data, state.info["action_history"][0], self._n_substeps
    )
    state.info["motor_targets"] = motor_targets

    contact = jp.array([
        geoms_colliding(data, geom_id, self._floor_geom_id)
        for geom_id in self._feet_geom_id
    ])
    contact_filt = contact | state.info["last_contact"]
    first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
    state.info["feet_air_time"] += self.ctrl_dt
    p_f = data.site_xpos[self._feet_site_id]
    p_fz = p_f[..., -1]
    state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

    obs = self._get_obs(data, state.info, contact, state.obs["state"], state.obs["privileged_state"])
    done = self._get_termination(data)

    # Get the rewards.
    rewards = self._get_reward(
        data, action, state.info, state.metrics, done, first_contact, contact
    )
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = jp.clip(sum(rewards.values()) * self.ctrl_dt, 0.0, 10000.0)

    state.info["push"] = push
    state.info["step"] += 1
    state.info["push_step"] += 1
    phase_tp1 = state.info["phase"] + state.info["phase_dt"]
    state.info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi
    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])
    state.info["command"] = jp.where(
        state.info["step"] > 500,
        self.sample_command(cmd_rng),
        state.info["command"],
    )
    state.info["step"] = jp.where(
        done | (state.info["step"] > 500),
        0,
        state.info["step"],
    )
    state.info["feet_air_time"] *= ~contact
    state.info["last_contact"] = contact
    state.info["swing_peak"] *= ~contact
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])
    state.info["xfrc_applied"] = data.xfrc_applied
    state.info["qpos"] = data.qpos
    state.info["qvel"] = data.qvel

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  def _get_sensor_data(self, data: mjx.Data, sensor_name: str) -> jax.Array:
    """Gets sensor data given sensor name."""
    sensor_id = self.mj_model.sensor(sensor_name).id
    sensor_adr = self.mj_model.sensor_adr[sensor_id]
    sensor_dim = self.mj_model.sensor_dim[sensor_id]
    return data.sensordata[sensor_adr : sensor_adr + sensor_dim]

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    gravity = self._get_sensor_data(data, GRAVITY_SENSOR)
    fall_termination = gravity[-1] < 0.0
    return (
        fall_termination | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    )

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any], contact: jax.Array,
      _state_history_raveled: jax.Array, _privileged_state_history_raveled: jax.Array,
  ) -> mjx_env.Observation:
    gyro = self._get_sensor_data(data, GYRO_SENSOR)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gyro = (
        gyro
        + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gyro
    )

    up_I = jp.array([0, 0, 1])
    R_B_I = math.quat_to_mat(data.qpos[3:7])
    up_B = R_B_I.T @ up_I
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_up_B = (
        up_B
        + (2 * jax.random.uniform(noise_rng, shape=up_B.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gravity
    )

    joint_angles = data.qpos[7:]
    # Create another vector with a smaller shape with the same values
    joint_angles_without_spring = jp.array([joint_angles[i] for i in range(len(joint_angles)) if i not in self.joint_idx_to_ignore_dict.values()])
    q_j_noise_scale_without_spring = jp.array([self._q_j_noise_scale[i] for i in range(len(self._q_j_noise_scale)) if i not in self.joint_idx_to_ignore_dict.values()])
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles_without_spring
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles_without_spring.shape) - 1)
        * self._config.noise_config.level
        * q_j_noise_scale_without_spring
    )

    joint_vel = data.qvel[6:]
    joint_vel_without_spring = jp.array([joint_vel[i] for i in range(len(joint_vel)) if i not in self.joint_idx_to_ignore_dict.values()])
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_vel = (
        joint_vel_without_spring
        + (2 * jax.random.uniform(noise_rng, shape=joint_vel_without_spring.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_vel
    )

    cos = jp.cos(info["phase"])
    sin = jp.sin(info["phase"])
    phase = jp.concatenate([cos, sin])
    
    linvel = self._get_sensor_data(data, LOCAL_LINVEL_SENSOR)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_linvel = (
        linvel
        + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.linvel
    )

    current_state = jp.hstack([
        noisy_linvel,  # 3
        noisy_gyro,  # 3
        noisy_up_B,  # 3
        info["command"],  # 3
        noisy_joint_angles - self._default_q_joints_without_spring,  # 8
        noisy_joint_vel,  # 8
        info["last_act"],  # 8
        phase,
    ])

    accelerometer = self._get_sensor_data(data, ACCELEROMETER_SENSOR)
    global_angvel = self._get_sensor_data(data, GLOBAL_ANGVEL_SENSOR)
    feet_vel_I = data.sensordata[self._foot_global_linvel_sensor_adr].ravel()
    baselink_height_I = data.qpos[2]

    current_privileged_state = jp.hstack([
        current_state,
        gyro,  # 3
        accelerometer,  # 3
        up_B,  # 3
        linvel,  # 3
        global_angvel,  # 3
        joint_angles_without_spring - self._default_q_joints_without_spring,
        joint_vel,
        baselink_height_I,  # 1
        data.actuator_force,  # 10
        contact,  # 2
        feet_vel_I,  # 4*3
        info["feet_air_time"],  # 2
    ])

    # Initialize history buffers if they don't exist
    if _state_history_raveled is None:
        single_state_size = current_state.shape[0]
        _state_history_raveled = jp.zeros((self._config.history_len, single_state_size)).ravel()
    if _privileged_state_history_raveled is None:
        single_privileged_state_size = current_privileged_state.shape[0]
        _privileged_state_history_raveled = jp.zeros((self._config.history_len, single_privileged_state_size)).ravel()

    # Update history buffers
    _state_history = _state_history_raveled.reshape(self._config.history_len, -1)
    _privileged_state_history = _privileged_state_history_raveled.reshape(self._config.history_len, -1)
    new_state_history = jp.roll(_state_history, -1, axis=0)
    new_state_history = new_state_history.at[-1].set(current_state)
    new_privileged_state_history = jp.roll(_privileged_state_history, -1, axis=0)
    new_privileged_state_history = new_privileged_state_history.at[-1].set(current_privileged_state)

    return {
        "state": new_state_history.ravel(),  # Flatten the history
        "privileged_state": new_privileged_state_history.ravel(),  # Flatten the history
    }

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
      first_contact: jax.Array,
      contact: jax.Array,
  ) -> dict[str, jax.Array]:
    del metrics  # Unused.

    return {
        # Tracking rewards.
        "tracking_lin_vel": self._reward_tracking_lin_vel(
            info["command"], self._get_sensor_data(data, LOCAL_LINVEL_SENSOR)
        ),
        "tracking_ang_vel": self._reward_tracking_ang_vel(
            info["command"], self._get_sensor_data(data, GYRO_SENSOR)
        ),
        # Base-related rewards.
        "lin_vel_z": self._cost_lin_vel_z(self._get_sensor_data(data, GLOBAL_LINVEL_SENSOR)),
        "ang_vel_xy": self._cost_ang_vel_xy(self._get_sensor_data(data, GLOBAL_ANGVEL_SENSOR)),
        "orientation": self._cost_orientation(self._get_sensor_data(data, GRAVITY_SENSOR)),
        "base_height": self._cost_base_height(data.qpos[2]),
        # Energy related rewards.
        "torques": self._cost_torques(data.actuator_force),
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        # Feet related rewards.
        "feet_slip": self._cost_feet_slip(data, contact, info),
        "feet_clearance": self._cost_feet_clearance(data, info),
        "feet_height": self._cost_feet_height(
            info["swing_peak"], first_contact, info
        ),
        "feet_air_time": self._reward_feet_air_time(
            info["feet_air_time"], first_contact, info["command"]
        ),
        "feet_phase": self._reward_feet_phase(
            data,
            info["phase"],
            self._config.reward_config.max_foot_height,
            info["command"],
        ),
        # Other rewards.
        "alive": self._reward_alive(),
        "termination": self._cost_termination(done),
        # Pose related rewards.
        "joint_deviation_hip": self._cost_joint_deviation_hip(
            data.qpos[7:], info["command"]
        ),
        "joint_deviation_knee": self._cost_joint_deviation_knee(data.qpos[7:]),
        "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
        "pose": self._cost_joint_angles(data.qpos[7:]),
    }

  # Tracking rewards.

  def _reward_tracking_lin_vel(self, commands: jax.Array, local_vel: jax.Array, ) -> jax.Array:
    lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    return jp.exp(-lin_vel_error)

  def _reward_tracking_ang_vel(self, commands: jax.Array, ang_vel: jax.Array, ) -> jax.Array:
    ang_vel_error = jp.square(commands[2] - ang_vel[2])
    return jp.exp(-ang_vel_error)

  # Base-related rewards.
  def _cost_lin_vel_z(self, global_linvel) -> jax.Array:
    return jp.square(global_linvel[2])

  def _cost_ang_vel_xy(self, global_angvel) -> jax.Array:
    return jp.sum(jp.square(global_angvel[:2]))

  def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
    return jp.sum(jp.square(torso_zaxis[:2]))

  def _cost_base_height(self, base_height: jax.Array) -> jax.Array:
    return jp.square(base_height - self._config.reward_config.base_height_target)

  # Energy related rewards.
  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    return jp.sum(jp.abs(torques))

  def _cost_action_rate(self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array) -> jax.Array:
    del last_last_act  # Unused.
    c1 = jp.sum(jp.square(act - last_act)/dt)
    return c1

  # Other rewards.
  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    out_of_limits = -jp.clip(qpos - self._soft_q_j_min, None, 0.0)
    out_of_limits += jp.clip(qpos - self._soft_q_j_max, 0.0, None)
    return jp.sum(out_of_limits)

  def _cost_termination(self, done: jax.Array) -> jax.Array:
    return done

  def _reward_alive(self) -> jax.Array:
    return jp.array(1.0)

  # Pose-related rewards.
  def _cost_joint_deviation_hip(self, qpos: jax.Array, cmd: jax.Array) -> jax.Array:
    cost = jp.sum(jp.abs(qpos[self._hip_indices] - self._default_q_joints[self._hip_indices]))
    cost *= jp.abs(cmd[1]) > 0.1
    return cost

  def _cost_joint_deviation_knee(self, qpos: jax.Array) -> jax.Array:
    return jp.sum(jp.abs(qpos[self._knee_indices] - self._default_q_joints[self._knee_indices]))

  def _cost_joint_angles(self, q_joints: jax.Array) -> jax.Array:
    weights = jp.array([robot_config.COSTS_JOINT_ANGLES])
    return jp.sum(jp.square(q_joints - self._default_q_joints) * weights)

  # Feet related rewards.
  def _cost_feet_slip(self, data: mjx.Data, contact: jax.Array, info: dict[str, Any]) -> jax.Array:
    del info  # Unused.
    # Get feet velocity in inertial frame
    feet_vel = data.sensordata[self._foot_global_linvel_sensor_adr]
    feet_vel_xy = feet_vel[..., :2] # Get x,y components of feet velocity in inertial frame

    # Calculate slip cost based on feet velocity when in contact
    slip_cost = jp.sum(jp.linalg.norm(feet_vel_xy, axis=-1) * contact)

    return slip_cost

  def _cost_feet_height(
      self,
      swing_peak: jax.Array,
      first_contact: jax.Array,
      info: dict[str, Any],
  ) -> jax.Array:
    del info  # Unused.
    error = swing_peak / self._config.reward_config.max_foot_height - 1.0
    return jp.sum(jp.square(error) * first_contact)

  def _reward_feet_air_time(
      self,
      air_time: jax.Array,
      first_contact: jax.Array,
      commands: jax.Array,
      threshold_min: float = 0.2,
      threshold_max: float = 0.5,
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(commands)
    air_time = (air_time - threshold_min) * first_contact
    air_time = jp.clip(air_time, max=threshold_max - threshold_min)
    reward = jp.sum(air_time)
    reward *= cmd_norm > 0.1  # No reward for zero commands.
    return reward

  def _reward_feet_phase(
      self,
      data: mjx.Data,
      phase: jax.Array,
      foot_height: jax.Array,
      commands: jax.Array,
  ) -> jax.Array:
    # Reward for tracking the desired foot height.
    del commands  # Unused.
    foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    rz = utils.get_rz(phase, swing_height=foot_height)
    error = jp.sum(jp.square(foot_z - rz))
    reward = jp.exp(-error / 0.01)
    # TODO(kevin): Ensure no movement at 0 command.
    # cmd_norm = jp.linalg.norm(commands)
    # reward *= cmd_norm > 0.1  # No reward for zero commands.
    return reward

  def sample_command(self, rng: jax.Array) -> jax.Array:
    rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)

    lin_vel_x = jax.random.uniform(
        rng1, minval=self._config.lin_vel_x[0], maxval=self._config.lin_vel_x[1]
    )
    lin_vel_y = jax.random.uniform(
        rng2, minval=self._config.lin_vel_y[0], maxval=self._config.lin_vel_y[1]
    )
    ang_vel_yaw = jax.random.uniform(
        rng3,
        minval=self._config.ang_vel_yaw[0],
        maxval=self._config.ang_vel_yaw[1],
    )

    # With 10% chance, set everything to zero.
    return jp.where(
        jax.random.bernoulli(rng4, p=0.1),
        jp.zeros(3),
        jp.hstack([lin_vel_x, lin_vel_y, ang_vel_yaw]),
    )

  @property
  def observation_size(self) -> mjx_env.ObservationSize:
    return {
        "state": (self._state_history.size,),
        "privileged_state": (self._privileged_state_history.size,),
    }
  
  # Accessors.
  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    nb_joints = self.mj_model.njnt - 1 # First joint is freejoint.
    return nb_joints - 2 - len(self.joint_idx_to_ignore_dict.keys())

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


from robot_learning.src.jax.utils import draw_joystick_command
import functools
import mediapy as media

def test_jax_biped():

  eval_env = Biped()
  jit_reset = jax.jit(eval_env.reset)
  jit_step = jax.jit(eval_env.step)
  print(f'JITing reset and step')
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
  for i in range(1400):
    print(i)
    time_duration = time.time()

    ctrl = jp.zeros(eval_env.action_size)
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
  media.write_video(f'joystick_testing_xvel_{x_vel}_yvel_{y_vel}_yawvel_{yaw_vel}.mp4', frames, fps=fps)
  print('Video saved')

def test_mujoco_biped():
  paused = False
  def key_callback(keycode):
    if chr(keycode) == ' ':
      nonlocal paused
      paused = not paused

  def read_contact_states(model, data):
    contact_states = {}
    contact_states['R_FOOT'] = False
    contact_states['L_FOOT'] = False

    geom1_list = []
    geom2_list = []
    for i in range(data.ncon):
        contact = data.contact[i]

        name_geom1 = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        name_geom2 = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
        geom1_list.append(name_geom1)
        geom2_list.append(name_geom2)

    if data.ncon != 0:
        if 'L_FOOT' in geom2_list:
            first_entry_idx = geom2_list.index('L_FOOT')
            if geom1_list[first_entry_idx] == 'floor':
                contact_states['L_FOOT'] = True

        if 'R_FOOT' in geom2_list:
            first_entry_idx = geom2_list.index('R_FOOT')
            if geom1_list[first_entry_idx] == 'floor':
                contact_states['R_FOOT'] = True
    return contact_states

  mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
  mj_data = mujoco.MjData(mj_model)

  default_q_joints = jp.array(mj_model.keyframe("home").qpos[7:])
  mj_data.ctrl = default_q_joints.copy()
  viewer = mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_callback)

  dh = 0.0005

  while viewer.is_running():
    mj_data.ctrl = default_q_joints.copy()

    # Slowly move the height down.
    mj_model.eq_data[0][2] -= dh

    # Check for feet contact.
    contact_states = read_contact_states(mj_model, mj_data)
    if contact_states['L_FOOT'] or contact_states['R_FOOT']:
      mj_model.eq_active0 = 0 # Let go of the robot.
      mj_data.eq_active[0] = 0
      dh = 0.0

    if not paused:
      mujoco.mj_step(mj_model, mj_data)
      viewer.sync()

if __name__ == "__main__":
  test_jax_biped()
  # test_mujoco_biped()