"""
  Joystick task for Berkeley Humanoid using Numpy
  Modified by Sorina Lupu (eslupu@caltech.edu)
"""

from typing import Any, Dict, Optional, Union, List, Callable, Sequence
from ml_collections import config_dict
import mujoco
import gymnasium as gym
from gymnasium import utils as gym_utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import mujoco.viewer

import numpy as np

import utils as utils
from utils import geoms_colliding_np, get_rz_np
import tqdm

import mediapy as media


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.002,
      episode_length=1000,
      action_scale=0.5,
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
          base_height_target=0.5,
      ),
      push_config=config_dict.create(
          enable=True,
          interval_range=[5.0, 10.0],
          magnitude_range=[0.1, 2.0],
      ),
      lin_vel_x=[-1.0, 1.0],
      lin_vel_y=[-1.0, 1.0],
      ang_vel_yaw=[-1.0, 1.0],
  )

NAME_ROBOT = 'biped'
if NAME_ROBOT == 'biped':
    import src.assets.biped.config as robot_config
else:
    raise ValueError(f'NAME_ROBOT must be "biped"')

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


class Biped(gym.Env):
    """Track a joystick command."""

    def __init__(self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
      visualize: bool = False,
      paused: bool = False,
      ):
        # Config.
        self._config = config.lock()
        if config_overrides:
            self._config.update_from_flattened_dict(config_overrides)

        # Initialize Mujoco.
        self.ctrl_dt = config.ctrl_dt
        self.sim_dt = config.sim_dt
        self._mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self._mj_model)
        self._mj_model.opt.timestep = self.sim_dt
        self._nb_joints = self._mj_model.njnt - 1 # First joint is freejoint.
        self.paused = paused
        if self.paused:
            print("Warning: Paused mode is enabled!")

        # Set visualization settings.
        self._mj_model.vis.global_.offwidth = 3840
        self._mj_model.vis.global_.offheight = 2160
        self.visualize_mujoco = visualize
        if self.visualize_mujoco is True:
            self.viewer = mujoco.viewer.launch_passive(self._mj_model, self.data, key_callback=self.key_callback)

        # Info.
        self.info = {}

        # Post init.
        self._post_init()
        self.reset_model()
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=self.observation_size, dtype=np.float64
        )
        
    def key_callback(self, keycode):
        if chr(keycode) == ' ':
            self.paused = not self.paused

    def _post_init(self) -> None:
        self._init_q = self._mj_model.keyframe("home").qpos.copy()
        self._default_q_joints = self._mj_model.keyframe("home").qpos[7:].copy()

        q_j_min, q_j_max = self._mj_model.jnt_range[1:].T # Note: First joint is freejoint.
        c = (q_j_min + q_j_max) / 2
        r = q_j_max - q_j_min
        self._soft_q_j_min = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
        self._soft_q_j_max = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

        # Indices joints.
        q_j_noise_scale = np.zeros(self._nb_joints) # For joint noise.

        hip_indices = []
        hip_joint_names = robot_config.HIP_JOINT_NAMES
        if len(hip_joint_names) != 0:
            for side in robot_config.SIDES:
                for joint_name in hip_joint_names:
                    hip_indices.append(self._mj_model.joint(f"{side}_{joint_name}").qposadr[0] - 7)
            self._hip_indices = np.array(hip_indices) # For hip reward deviation.
            q_j_noise_scale[self._hip_indices] = self._config.noise_config.scales.hip_pos

        knee_indices = []
        knee_joint_names = robot_config.KNEE_JOINT_NAMES
        if len(knee_joint_names) != 0:
            for side in robot_config.SIDES:
                for knee_joint_name in knee_joint_names:
                    knee_indices.append(self._mj_model.joint(f"{side}_{knee_joint_name}").qposadr[0] - 7)
            self._knee_indices = np.array(knee_indices) # For knee reward deviation.
            q_j_noise_scale[self._knee_indices] = self._config.noise_config.scales.kfe_pos

        ffe_joint_names = robot_config.ANKLE_FE_JOINT_NAMES
        if len(ffe_joint_names) != 0:
            ffe_indices = []
            for side in robot_config.SIDES:
                for ffe_joint_name in ffe_joint_names:
                    ffe_indices.append(self._mj_model.joint(f"{side}_{ffe_joint_name}").qposadr[0] - 7)
            self._ffe_indices = np.array(ffe_indices) # For ffe reward deviation.
            q_j_noise_scale[self._ffe_indices] = self._config.noise_config.scales.ffe_pos

        faa_joint_names = robot_config.ANKLE_AA_JOINT_NAMES
        if len(faa_joint_names) != 0:
            faa_indices = []
            for side in robot_config.SIDES:
                for faa_joint_name in faa_joint_names:
                    faa_indices.append(self._mj_model.joint(f"{side}_{faa_joint_name}").qposadr[0] - 7)
            self._faa_indices = np.array(faa_indices) # For faa reward deviation.
            q_j_noise_scale[self._faa_indices] = self._config.noise_config.scales.faa_pos
        self._q_j_noise_scale = np.array(q_j_noise_scale)

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
        self._foot_linvel_sensor_adr = np.array(foot_linvel_sensor_adr)

    def reset_model(self):
        # Setup initial states.
        qpos = self._init_q.copy()
        qvel = np.zeros(self._mj_model.nv)

        # Randomize x=+U(-0.5, 0.5), y=+U(-0.5, 0.5), yaw=U(-3.14, 3.14).
        dxy = np.random.uniform(-0.5, 0.5, 2)
        qpos[0:2] = qpos[0:2] + dxy
        # TODO: add quaternion rotation

        # Randomize joint angles.
        qpos[7:] = qpos[7:] + np.random.uniform(-0.05, 0.05, self._nb_joints)

        # Randomize velocity.
        qvel[0:6] = np.random.uniform(-0.5, 0.5, 6)

        # Initialize data.
        self.data.qpos = qpos
        self.data.qvel = qvel

        # Phase and gait.
        gait_freq = np.random.uniform(1.25, 1.5)
        phase_dt = 2 * np.pi * self.ctrl_dt * gait_freq
        phase = np.array([0.0, np.pi])

        self.info = {
            "step": 0,
            "command": np.zeros(3), # TODO: add command.
            "last_act": np.zeros(self._nb_joints),
            "last_last_act": np.zeros(self._nb_joints),
            "motor_targets": np.zeros(self._nb_joints),
            "feet_air_time": np.zeros(2),
            "last_contact": np.zeros(2, dtype=bool),
            "swing_peak": np.zeros(2),
            # Phase related
            "phase_dt": phase_dt,
            "phase": phase,
            # Push related
            "push": np.zeros(2),
            "push_step": 0,
            # "push_interval_steps": push_interval_steps
        }
        observation = self._get_obs()
        mujoco.mj_step(self._mj_model, self.data)
        return observation
    
    def step(self, action: np.ndarray) -> tuple:
        # Apply action.
        motor_targets = self._default_q_joints + action * self._config.action_scale
        self.info["motor_targets"] = motor_targets
        self.data.ctrl = motor_targets.copy()

        # Step the model.
        if not self.paused:
            mujoco.mj_step(self._mj_model, self.data)

        # Check contacts. TODO: fix this
        # contact = np.array([
        #     geoms_colliding_np(self.data, geom_id, self._floor_geom_id)
        #     for geom_id in self._feet_geom_id
        # ])
        contact = np.zeros(2, dtype=bool)
        contact_filt = contact | self.info["last_contact"]
        first_contact = (self.info["feet_air_time"] > 0.0) * contact_filt
        self.info["feet_air_time"] += self.ctrl_dt
        p_f = self.data.site_xpos[self._feet_site_id]
        p_fz = p_f[..., -1]
        # self.info["swing_peak"] = np.max(self.info["swing_peak"], p_fz)

        # Visualize.
        if self.visualize_mujoco is True:
            if self.viewer.is_running():
                self.viewer.sync()

        # Get obs.
        obs = self._get_obs()
        done = self._get_termination()

        rewards = self._get_rew(done=done,
                    action=action,
                    first_contact=first_contact,
                    contact=contact)

        rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
        }
        reward = np.clip(sum(rewards.values()) * self.ctrl_dt, 0.0, 10000.0)

        # self.info["push"] = push
        self.info["step"] += 1
        self.info["push_step"] += 1
        phase_tp1 = self.info["phase"] + self.info["phase_dt"]
        self.info["phase"] = np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi
        self.info["last_last_act"] = self.info["last_act"]
        self.info["last_act"] = action
        self.info["command"] = np.where(
            self.info["step"] > 500,
            self.sample_command(),
            self.info["command"],
        )
        self.info["step"] = np.where(
            done | (self.info["step"] > 500),
            0,
            self.info["step"],
        )
        self.info["feet_air_time"] *= ~contact
        self.info["last_contact"] = contact
        self.info["swing_peak"] *= ~contact

        # return observation, reward, terminated, False, info
        return obs, reward, done, False, self.info
    
    def _get_obs(self) -> np.ndarray:
        """ Get observation from sensors. """
        # Gyroscope.
        gyro = self.get_sensor_data(GYRO_SENSOR)
        noisy_gyro = gyro.copy() # TODO: add  noise

        # Gravity.
        R_gravity_sensor = self.data.site_xmat[self._imu_site_id].reshape(3, 3).copy()
        gravity = R_gravity_sensor.T @ np.array([0, 0, -1]) # TODO: check this
        noisy_gravity = gravity.copy() # TODO: add noise

        # Joint angles.
        joint_angles = self.data.qpos[7:].copy()
        noisy_joint_angles = joint_angles.copy()

        # Joint velocities.
        joint_vel = self.data.qvel[6:].copy()
        noisy_joint_vel = joint_vel.copy()

        # Phase.
        cos = np.cos(self.info["phase"])
        sin = np.sin(self.info["phase"])
        phase = np.hstack([cos, sin])

        # Linear velocity.        
        linvel = self.get_sensor_data(LOCAL_LINVEL_SENSOR)
        noisy_linvel = linvel.copy()

        # Combine everything into state.
        self._state = np.hstack([
            noisy_linvel,           # 3
            noisy_gyro,             # 3
            noisy_gravity,          # 3
            self.info["command"],   # 3
            noisy_joint_angles - self._default_q_joints,  # 12
            noisy_joint_vel,        # 12
            self.info["last_act"],  # 12
            phase,                  # 4
        ]) # 52

        return self._state

    def sample_command(self, rng=None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()

        # Sample linear and angular velocities
        lin_vel_x = rng.uniform(
            low=self._config.lin_vel_x[0],
            high=self._config.lin_vel_x[1]
        )
        lin_vel_y = rng.uniform(
            low=self._config.lin_vel_y[0],
            high=self._config.lin_vel_y[1]
        )
        ang_vel_yaw = rng.uniform(
            low=self._config.ang_vel_yaw[0],
            high=self._config.ang_vel_yaw[1]
        )

        # With 10% chance, set everything to zero
        if rng.random() < 0.1:
            return np.zeros(3)
        else:
            return np.array([lin_vel_x, lin_vel_y, ang_vel_yaw])

    def _get_rew(self,
                 done: bool,
                 action: np.ndarray,
                 first_contact:np.ndarray,
                 contact: np.ndarray):
        return {
            # Tracking rewards.
            "tracking_lin_vel": self._reward_tracking_lin_vel(
                self.info["command"], self.get_sensor_data(LOCAL_LINVEL_SENSOR)
            ),
            "tracking_ang_vel": self._reward_tracking_ang_vel(
                self.info["command"], self.get_sensor_data(GYRO_SENSOR)
            ),
            # Base-related rewards.
            "lin_vel_z": self._cost_lin_vel_z(self.get_sensor_data(GLOBAL_LINVEL_SENSOR)),
            "ang_vel_xy": self._cost_ang_vel_xy(self.get_sensor_data(GLOBAL_ANGVEL_SENSOR)),
            "orientation": self._cost_orientation(self.get_sensor_data(GRAVITY_SENSOR)),
            "base_height": self._cost_base_height(self.data.qpos[2]),
            # Energy related rewards.
            "torques": self._cost_torques(self.data.actuator_force.copy()),
            "action_rate": self._cost_action_rate(
                action, self.info["last_act"], self.info["last_last_act"]
            ),
            "energy": self._cost_energy(self.data.qvel[6:].copy(), self.data.actuator_force.copy()),
            # Feet related rewards.
            "feet_slip": self._cost_feet_slip(self.data, contact, self.info),
            "feet_clearance": self._cost_feet_clearance(self.data, self.info),
            "feet_height": self._cost_feet_height(
                self.info["swing_peak"], first_contact, self.info
            ),
            "feet_air_time": self._reward_feet_air_time(
                self.info["feet_air_time"], first_contact, self.info["command"]
            ),
            "feet_phase": self._reward_feet_phase(
                self.data,
                self.info["phase"],
                self._config.reward_config.max_foot_height,
                self.info["command"],
            ),
            # Other rewards.
            "alive": self._reward_alive(),
            "termination": self._cost_termination(done),
            "stand_still": self._cost_stand_still(self.info["command"], self.data.qpos[7:].copy()),
            # Pose related rewards.
            "joint_deviation_hip": self._cost_joint_deviation_hip(
                self.data.qpos[7:].copy(), self.info["command"]
            ),
            "joint_deviation_knee": self._cost_joint_deviation_knee(self.data.qpos[7:].copy()),
            "dof_pos_limits": self._cost_joint_pos_limits(self.data.qpos[7:].copy()),
            "pose": self._cost_joint_angles(self.data.qpos[7:].copy()),
        }

    def _get_termination(self):
        gravity = self.get_sensor_data(GRAVITY_SENSOR)
        fall_termination = gravity[-1] < 0.0
        return (
            fall_termination | np.isnan(self.data.qpos).any() | np.isnan(self.data.qvel).any()
        )

    def get_sensor_data(self, sensor_name: str) -> np.ndarray:
      """Gets sensor data given sensor name."""
      sensor_id = self._mj_model.sensor(sensor_name).id
      sensor_adr = self._mj_model.sensor_adr[sensor_id]
      sensor_dim = self._mj_model.sensor_dim[sensor_id]
      return self.data.sensordata[sensor_adr : sensor_adr + sensor_dim].copy()

    @property
    def action_size(self) -> int:
        return self._nb_joints

    @property
    def observation_size(self) -> tuple:
        print(self._state.size)
        return (self._state.size, )


    # Tracking rewards.

    def _reward_tracking_lin_vel(self, commands: np.ndarray, local_vel: np.ndarray, ) -> np.ndarray:
        lin_vel_error = np.sum(np.square(commands[:2] - local_vel[:2]))
        return np.exp(-lin_vel_error / self._config.reward_config.tracking_sigma)

    def _reward_tracking_ang_vel(self, commands: np.ndarray, ang_vel: np.ndarray, ) -> np.ndarray:
        ang_vel_error = np.square(commands[2] - ang_vel[2])
        return np.exp(-ang_vel_error / self._config.reward_config.tracking_sigma)

    # Base-related rewards.
    def _cost_lin_vel_z(self, global_linvel) -> np.ndarray:
        return np.square(global_linvel[2])

    def _cost_ang_vel_xy(self, global_angvel) -> np.ndarray:
        return np.sum(np.square(global_angvel[:2]))

    def _cost_orientation(self, torso_zaxis: np.ndarray) -> np.ndarray:
        return np.sum(np.square(torso_zaxis[:2]))

    def _cost_base_height(self, base_height: np.ndarray) -> np.ndarray:
        return np.square(base_height - self._config.reward_config.base_height_target)

    # Energy related rewards.
    def _cost_torques(self, torques: np.ndarray) -> np.ndarray:
        return np.sum(np.abs(torques))

    def _cost_energy(self, qvel: np.ndarray, qfrc_actuator: np.ndarray) -> np.ndarray:
        return np.sum(np.abs(qvel) * np.abs(qfrc_actuator))

    def _cost_action_rate(self, act: np.ndarray, last_act: np.ndarray, last_last_act: np.ndarray) -> np.ndarray:
        del last_last_act  # Unused.
        c1 = np.sum(np.square(act - last_act))
        return c1

    # Other rewards.
    def _cost_joint_pos_limits(self, qpos: np.ndarray) -> np.ndarray:
        out_of_limits = -np.clip(qpos - self._soft_q_j_min, None, 0.0)
        out_of_limits += np.clip(qpos - self._soft_q_j_max, 0.0, None)
        return np.sum(out_of_limits)

    def _cost_stand_still(self, commands: np.ndarray, qpos: np.ndarray, ) -> np.ndarray:
        cmd_norm = np.linalg.norm(commands)
        return np.sum(np.abs(qpos - self._default_q_joints)) * (cmd_norm < 0.1)

    def _cost_termination(self, done: bool) -> bool:
        return done

    def _reward_alive(self) -> np.ndarray:
        return np.array(1.0)

    # Pose-related rewards.
    def _cost_joint_deviation_hip(self, qpos: np.ndarray, cmd: np.ndarray) -> np.ndarray:
        cost = np.sum(np.abs(qpos[self._hip_indices] - self._default_q_joints[self._hip_indices]))
        cost *= np.abs(cmd[1]) > 0.1
        return cost

    def _cost_joint_deviation_knee(self, qpos: np.ndarray) -> np.ndarray:
        return np.sum(np.abs(qpos[self._knee_indices] - self._default_q_joints[self._knee_indices]))

    def _cost_joint_angles(self, q_joints: np.ndarray) -> np.ndarray:
        weights = np.array([robot_config.COSTS_JOINT_ANGLES])
        return np.sum(np.square(q_joints - self._default_q_joints) * weights)

    # Feet related rewards.
    def _cost_feet_slip(self, data,
                        contact,
                        info: dict[str, Any]) -> np.ndarray:
        del info  # Unused.
        lin_vel = self.get_sensor_data(LOCAL_LINVEL_SENSOR)
        body_vel = lin_vel[:2]
        reward = np.sum(np.linalg.norm(body_vel, axis=-1) * contact)
        return reward

    def _cost_feet_clearance(self,
                             data,
                             info) -> np.ndarray:
        del info  # Unused.
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
        feet_vel = feet_vel.copy() # Add copy here
        vel_xy = feet_vel[..., :2]
        vel_norm = np.sqrt(np.linalg.norm(vel_xy, axis=-1))
        foot_pos = data.site_xpos[self._feet_site_id].copy()
        foot_z = foot_pos[..., -1]
        delta = np.abs(foot_z - self._config.reward_config.max_foot_height)
        return np.sum(delta * vel_norm)

    def _cost_feet_height(
        self,
        swing_peak: np.ndarray,
        first_contact: np.ndarray,
        info: dict[str, Any],
    ) -> np.ndarray:
        del info  # Unused.
        error = swing_peak / self._config.reward_config.max_foot_height - 1.0
        return np.sum(np.square(error) * first_contact)

    def _reward_feet_air_time(
        self,
        air_time: np.ndarray,
        first_contact: np.ndarray,
        commands: np.ndarray,
        threshold_min: float = 0.2,
        threshold_max: float = 0.5,
    ) -> np.ndarray:
        cmd_norm = np.linalg.norm(commands)
        air_time = (air_time - threshold_min) * first_contact
        air_time = np.clip(air_time, a_min=0.0, a_max=threshold_max - threshold_min)
        reward = np.sum(air_time)
        reward *= cmd_norm > 0.1  # No reward for zero commands.
        return reward

    def _reward_feet_phase(
        self,
        data,
        phase: np.ndarray,
        foot_height: np.ndarray,
        commands: np.ndarray,
    ) -> np.ndarray:
        # Reward for tracking the desired foot height.
        del commands  # Unused.
        foot_pos = data.site_xpos[self._feet_site_id].copy()
        foot_z = foot_pos[..., -1]
        rz = utils.get_rz_np(phase, swing_height=foot_height)
        error = np.sum(np.square(foot_z - rz))
        reward = np.exp(-error / 0.01)
        return reward

    def render(
        self,
        trajectory, #: List[State],
        height: int = 240,
        width: int = 320,
        camera: Optional[str] = None,
        scene_option: Optional[mujoco.MjvOption] = None,
        modify_scene_fns: Optional[
            Sequence[Callable[[mujoco.MjvScene], None]]
        ] = None,
    ) -> Sequence[np.ndarray]:
        return render_array(
            self._mj_model,
            trajectory,
            height,
            width,
            camera,
            scene_option=scene_option,
            modify_scene_fns=modify_scene_fns,
        )

def render_array(
    mj_model: mujoco.MjModel,
    trajectory, # Union[List[State], State], # TODO: fix this
    height: int = 480,
    width: int = 640,
    camera: Optional[str] = None,
    scene_option: Optional[mujoco.MjvOption] = None,
    modify_scene_fns: Optional[
        Sequence[Callable[[mujoco.MjvScene], None]]
    ] = None,
    hfield_data = None,
):
  """Renders a trajectory as an array of images."""
  renderer = mujoco.Renderer(mj_model, height=height, width=width)
  camera = camera or -1

  if hfield_data is not None:
    mj_model.hfield_data = hfield_data.reshape(mj_model.hfield_data.shape)
    mujoco.mjr_uploadHField(mj_model, renderer._mjr_context, 0)

  def get_image(state, modify_scn_fn=None) -> np.ndarray:
    d = mujoco.MjData(mj_model)
    d.qpos, d.qvel = state['qpos'], state['qvel']
    d.xfrc_applied = state['xfrc_applied']
    mujoco.mj_forward(mj_model, d)
    renderer.update_scene(d, camera=camera, scene_option=scene_option)
    if modify_scn_fn is not None:
      modify_scn_fn(renderer.scene)
    return renderer.render()

  if isinstance(trajectory, list):
    out = []
    for i, state in enumerate(tqdm.tqdm(trajectory)):
      if modify_scene_fns is not None:
        modify_scene_fn = modify_scene_fns[i]
      else:
        modify_scene_fn = None
      out.append(get_image(state, modify_scene_fn))
  else:
    out = get_image(trajectory)

  renderer.close()
  return out

if __name__ == "__main__":

    env = Biped(visualize=False)
    env.reset_model()

    rollout = []

    for _ in tqdm.tqdm(range(1000)):
        # action = np.random.uniform(-1, 1, env.action_size)
        action = np.zeros(env.action_size)
        obs, rewards, done, _, _ = env.step(action)
        state = {
            'qpos': env.data.qpos.copy(),
            'qvel': env.data.qvel.copy(),
            'xfrc_applied': env.data.xfrc_applied.copy()
        }
        rollout.append(state)

    render_every = 1 # int.
    fps = 1/ env.sim_dt / render_every
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
        width=640,
        height=480,
    )

    # media.show_video(frames, fps=fps, loop=False)
    # ABS_FOLDER_RESUlTS = epath.Path(RESULTS_FOLDER_PATH) / latest_folder
    # NOTE: To make the code run, you need to call: MUJOCO_GL=egl python3 biped_np.py
    media.write_video(f'joystick_testing.mp4', frames, fps=fps)
    print('Video saved')