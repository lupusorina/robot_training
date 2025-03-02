"""
  Joystick task for Berkeley Humanoid using Numpy
  Modified by Sorina Lupu (eslupu@caltech.edu)
"""

# TODO:
#   - make the code compatible between Berkeley biped and ours.

from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from ml_collections import config_dict
import mujoco

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import mujoco.viewer

import numpy as np
import os

# import utils
# from utils import geoms_colliding_np
import tqdm

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
if NAME_ROBOT == 'berkeley_humanoid':
    import assets.berkeley_humanoid.config as robot_config
if NAME_ROBOT == 'biped':
    import assets.biped.config as robot_config
    # raise NotImplementedError

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


class Biped(MujocoEnv, utils.EzPickle):
    """Track a joystick command."""

    def __init__(self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None
      ):
        # Config.
        self._config = config.lock()
        if config_overrides:
            self._config.update_from_flattened_dict(config_overrides)

        # Initialize Mujoco.
        self.ctrl_dt = config.ctrl_dt
        self._sim_dt = config.sim_dt
        self._mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self._mj_model)
        self._mj_model.opt.timestep = self._sim_dt
        self._nb_joints = self._mj_model.njnt - 1 # First joint is freejoint.
        self.paused = True
        print(f"Number of joints: {self._nb_joints}")
        print(f"Nb controls: {self._mj_model.nu}")
        print(self.data.ctrl.shape)

        # Set visualization settings.
        self._mj_model.vis.global_.offwidth = 3840
        self._mj_model.vis.global_.offheight = 2160
        self.visualize_mujoco = True
        if self.visualize_mujoco is True:
            self.viewer = mujoco.viewer.launch_passive(self._mj_model, self.data)

        # Info.
        self.info = {}

        # Post init.
        self._post_init()
        self.reset_model()
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=self.observation_size, dtype=np.float64
        )
        
    def _post_init(self) -> None:
        self._init_q = self._mj_model.keyframe("home").qpos
        self._default_q_joints = self._mj_model.keyframe("home").qpos[7:]

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
        qpos = self._init_q
        qvel = np.zeros(self._mj_model.nv)

        # Randomize x=+U(-0.5, 0.5), y=+U(-0.5, 0.5), yaw=U(-3.14, 3.14).
        dxy = np.random.uniform(-0.5, 0.5, 2)
        qpos[0:2] = qpos[0:2] + dxy
        # TODO: add quaternion rotation

        # Randomize joint angles.
        qpos[7:] = qpos[7:] + np.random.uniform(-0.5, 0.5, self._nb_joints)

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
        self.data.ctrl = motor_targets

        # Step the model.
        if not self.paused:
            mujoco.mj_step(self._mj_model, self.data)
        
        # Visualize.
        if self.visualize_mujoco is True:
            if self.viewer.is_running():
                self.viewer.sync()

        # Get obs.
        obs = self._get_obs()
        done = self._get_termination()

        # rewards, reward_info = self._get_rew()

        # return observation, reward, terminated, False, info
        return obs, None, done, False, self.info
    
    def _get_obs(self) -> np.ndarray:
        """ Get observation from sensors. """
        # Gyroscope.
        gyro = self.get_sensor_data(GYRO_SENSOR)
        noisy_gyro = gyro.copy() # TODO: add  noise

        # Gravity.
        R_gravity_sensor = self.data.site_xmat[self._imu_site_id].reshape(3, 3)
        gravity = R_gravity_sensor.T @ np.array([0, 0, -1]) # TODO: check this
        noisy_gravity = gravity.copy() # TODO: add noise

        # Joint angles.
        joint_angles = self.data.qpos[7:]
        noisy_joint_angles = joint_angles.copy()

        # Joint velocities.
        joint_vel = self.data.qvel[6:]
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

    def _get_rew(self):
        # return reward, reward_info
        pass

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
      return self.data.sensordata[sensor_adr : sensor_adr + sensor_dim]

    @property
    def action_size(self) -> int:
        return self._nb_joints

    @property
    def observation_size(self) -> tuple:
        print(self._state.size)
        return (self._state.size, )


def main():
    biped = Biped()
    biped.reset_model()

    for _ in tqdm.tqdm(range(100000)):
        # action = np.random.uniform(-1, 1, biped.action_size)
        action = np.zeros(biped.action_size)
        obs, rewards, done, _, _ = biped.step(action)
        # if done:
        #     print("Done!")
        #     break

if __name__ == "__main__":
    main()        

