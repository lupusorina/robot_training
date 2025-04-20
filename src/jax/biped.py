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

# Local imports.
import src.jax.utils as utils
from src.jax.utils import geoms_colliding
import src.jax.mjx_env as mjx_env

# Constants.
NAME_ROBOT = 'biped'
if NAME_ROBOT == 'biped':
  import src.assets.biped.config as robot_config
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
      ctrl_dt=0.02,
      sim_dt=0.001,
      episode_length=1000,
      action_repeat=1,
      action_scale=0.5,
      history_len=1,
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
              torques=-2.5e-5,
              action_rate=-0.01,
              energy=0.0,
              # Feet related rewards.
              feet_clearance=0.0,
              feet_air_time=2.0,
              feet_slip=-0.25,
              feet_height=0.0,
              feet_phase=1.0,
              # Other rewards.
              stand_still=0.0,
              alive=0.0,
              termination=-1.0,
              # Pose related rewards.
              joint_deviation_knee=-0.1,
              joint_deviation_hip=-0.25,
              dof_pos_limits=-1.0,
              pose=-1.0,
          ),
          tracking_sigma=0.5,
          max_foot_height=0.1,
          base_height_target=DESIRED_HEIGHT,
      ),
      push_config=config_dict.create(
          enable=True,
          interval_range=[5.0, 10.0],
          magnitude_range=[0.1, 2.0],
      ),
      lin_vel_x=[-0.5, 0.5],
      lin_vel_y=[-0.5, 0.5],
      ang_vel_yaw=[-0.5, 0.5],
  )

class Biped(mjx_env.MjxEnv):
  """Track a joystick command."""

  def __init__(
      self,
      xml_path: str = XML_PATH,
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
    self.nb_joints = self.mj_model.njnt - 1 # First joint is freejoint.
    self.action_space = jp.zeros(self.action_size)
    print(f"Number of joints: {self.nb_joints}")

    # Control action names.
    self.name_actuators = []
    for i in range(0, self.mj_model.nu):  # skip root
        self.name_actuators.append(mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i))
    print(self.name_actuators)

    # Used for logging.
    self.state_header = ['noisy_vel_x', 'noisy_vel_y', 'noisy_vel_z', # 3 noisy_linvel
                          'noisy_gyro_x', 'noisy_gyro_y', 'noisy_gyro_z', # 3 noisy_gyro
                          'noisy_gravity_x', 'noisy_gravity_y', 'noisy_gravity_z', # 3 noisy_gravity
                          'command_x', 'command_y', 'command_z', # 3 info["command"]
                          *['res_' + str(self.name_actuators[i]) + '_pos' for i in range(len(self.name_actuators))],
                          *['res_' + str(self.name_actuators[i]) + '_vel' for i in range(len(self.name_actuators))],
                          *['res_last_act_' + str(self.name_actuators[i]) for i in range(len(self.name_actuators))],
                          'phase_1', 'phase_2', 'phase_3', 'phase_4'] # phase TODO: figure out why we have 4 phases

    self.ctrl_header = ['res_' + str(self.name_actuators[i]) for i in range(len(self.name_actuators))]

    self._post_init()

  def _post_init(self) -> None:
    # Initialize the initial state.
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_q_joints = jp.array(self._mj_model.keyframe("home").qpos[7:])

    # Initialize the soft limits.
    q_j_min, q_j_max = self.mj_model.jnt_range[1:].T # Note: First joint is freejoint.
    c = (q_j_min + q_j_max) / 2
    r = q_j_max - q_j_min
    self._soft_q_j_min = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
    self._soft_q_j_max = c + 0.5 * r * self._config.soft_joint_pos_limit_factor
    self.lower_limit_ctrl = np.array(self._soft_q_j_min)
    self.upper_limit_ctrl = np.array(self._soft_q_j_max)

    # Initialize the noise scale.
    q_j_noise_scale = np.zeros(10) # For joint noise.

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

    foot_linvel_sensor_adr = []
    for site in FEET_SITES:
      sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
      sensor_adr = self._mj_model.sensor_adr[sensor_id]
      sensor_dim = self._mj_model.sensor_dim[sensor_id]
      foot_linvel_sensor_adr.append(
          list(range(sensor_adr, sensor_adr + sensor_dim))
      )
    self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

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
    qpos = qpos.at[7:].set(
      qpos[7:] * jax.random.uniform(key, (10,), minval=-0.2, maxval=0.2))

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
    obs = self._get_obs(data, info, contact)
    
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
    motor_targets = self._default_q_joints + action * self._config.action_scale
    data = mjx_env.step(
      self.mjx_model, state.data, motor_targets, self._n_substeps
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

    obs = self._get_obs(data, state.info, contact)
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
      self, data: mjx.Data, info: dict[str, Any], contact: jax.Array
  ) -> mjx_env.Observation:
    gyro = self._get_sensor_data(data, GYRO_SENSOR)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gyro = (
        gyro
        + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gyro
    )

    gravity = data.site_xmat[self._imu_site_id].T @ jp.array([0, 0, -1])
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gravity = (
        gravity
        + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gravity
    )

    joint_angles = data.qpos[7:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.noise_config.level
        * self._q_j_noise_scale
    )

    joint_vel = data.qvel[6:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_vel = (
        joint_vel
        + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
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

    self._state = jp.hstack([
        noisy_linvel,  # 3
        noisy_gyro,  # 3
        noisy_gravity,  # 3
        info["command"],  # 3
        noisy_joint_angles - self._default_q_joints,  # 10
        noisy_joint_vel,  # 10
        info["last_act"],  # 10
        phase,
    ])

    accelerometer = self._get_sensor_data(data, ACCELEROMETER_SENSOR)
    global_angvel = self._get_sensor_data(data, GLOBAL_ANGVEL_SENSOR)
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
    root_height = data.qpos[2]

    self._privileged_state = jp.hstack([
        self._state,
        gyro,  # 3
        accelerometer,  # 3
        gravity,  # 3
        linvel,  # 3
        global_angvel,  # 3
        joint_angles - self._default_q_joints,
        joint_vel,
        root_height,  # 1
        data.actuator_force,  # 10
        contact,  # 2
        feet_vel,  # 4*3
        info["feet_air_time"],  # 2
    ])

    return {
        "state": self._state,
        "privileged_state": self._privileged_state,
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
        "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
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
        "stand_still": self._cost_stand_still(info["command"], data.qpos[7:]),
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
    return jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma)

  def _reward_tracking_ang_vel(self, commands: jax.Array, ang_vel: jax.Array, ) -> jax.Array:
    ang_vel_error = jp.square(commands[2] - ang_vel[2])
    return jp.exp(-ang_vel_error / self._config.reward_config.tracking_sigma)

  # Base-related rewards.
  def _cost_lin_vel_z(self, global_linvel) -> jax.Array:
    return jp.square(global_linvel[2])

  def _cost_ang_vel_xy(self, global_angvel) -> jax.Array:
    return jp.sum(jp.square(global_angvel[:2]))

  def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
    return jp.sum(jp.square(torso_zaxis[:2]))

  def _cost_base_height(self, base_height: jax.Array) -> jax.Array:
    # jax.debug.print(" {x}", x=base_height)
    return jp.square(base_height - self._config.reward_config.base_height_target)

  # Energy related rewards.

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    return jp.sum(jp.abs(torques))

  def _cost_energy(self, qvel: jax.Array, qfrc_actuator: jax.Array) -> jax.Array:
    return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

  def _cost_action_rate(self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array) -> jax.Array:
    del last_last_act  # Unused.
    c1 = jp.sum(jp.square(act - last_act))
    return c1

  # Other rewards.
  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    out_of_limits = -jp.clip(qpos - self._soft_q_j_min, None, 0.0)
    out_of_limits += jp.clip(qpos - self._soft_q_j_max, 0.0, None)
    return jp.sum(out_of_limits)

  def _cost_stand_still(self, commands: jax.Array, qpos: jax.Array, ) -> jax.Array:
    cmd_norm = jp.linalg.norm(commands)
    return jp.sum(jp.abs(qpos - self._default_q_joints)) * (cmd_norm < 0.1)

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
    lin_vel = self._get_sensor_data(data, LOCAL_LINVEL_SENSOR)
    body_vel = lin_vel[:2]
    reward = jp.sum(jp.linalg.norm(body_vel, axis=-1) * contact)
    return reward

  def _cost_feet_clearance(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    del info  # Unused.
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
    foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    delta = jp.abs(foot_z - self._config.reward_config.max_foot_height)
    return jp.sum(delta * vel_norm)

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
        "state": (self._state.size, ),
        "privileged_state": (self._privileged_state.size,),
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


def main():
  env = Biped()
  print(env.state_header)
  print(env.ctrl_header)

if __name__ == "__main__":
  main()