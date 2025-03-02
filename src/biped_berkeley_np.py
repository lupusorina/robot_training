"""
  Joystick task for Berkeley Humanoid using Numpy
  Modified by Sorina Lupu (eslupu@caltech.edu)
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from ml_collections import config_dict
import mujoco


import numpy as np
import os

import utils
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

# global path
parent_dir = os.path.abspath(os.path.join(os.getcwd()))
XML_PATH = os.path.join(parent_dir, 'assets/berkeley_humanoid/xmls/scene_mjx_feetonly_flat_terrain.xml')
ROOT_BODY = "torso"
FEET_SITES = ["l_foot", "r_foot"]
LEFT_FEET_GEOMS = ["l_foot1"]
RIGHT_FEET_GEOMS = ["r_foot1"]
FEET_GEOMS = LEFT_FEET_GEOMS + RIGHT_FEET_GEOMS
GRAVITY_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"
IMU_SITE = "imu"


class Biped():
    """Track a joystick command."""
    def __init__(self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None
      ):
        self._config = config.lock()
        if config_overrides:
            self._config.update_from_flattened_dict(config_overrides)
            
        self.ctrl_dt = config.ctrl_dt
        self._sim_dt = config.sim_dt
        self._mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self._mj_model)

        self._mj_model.opt.timestep = self._sim_dt
        self._mj_model.vis.global_.offwidth = 3840
        self._mj_model.vis.global_.offheight = 2160
        
        self._nb_joints = self._mj_model.njnt - 1 # First joint is freejoint.
        print(f"Number of joints: {self._nb_joints}")
        
        self._post_init()
        
    def _post_init(self) -> None:
        self._init_q = self._mj_model.keyframe("home").qpos
        self._default_q_joints = self._mj_model.keyframe("home").qpos[7:]

        q_j_min, q_j_max = self._mj_model.jnt_range[1:].T # Note: First joint is freejoint.
        c = (q_j_min + q_j_max) / 2
        r = q_j_max - q_j_min
        self._soft_q_j_min = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
        self._soft_q_j_max = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

        # Indices joints.
        hip_indices = []
        hip_joint_names = ["HR", "HAA"]
        for side in ["LL", "LR"]:
            for joint_name in hip_joint_names:
                hip_indices.append(self._mj_model.joint(f"{side}_{joint_name}").qposadr[0] - 7)
        self._hip_indices = np.array(hip_indices) # For hip reward deviation.

        knee_indices = []
        for side in ["LL", "LR"]:
            knee_indices.append(self._mj_model.joint(f"{side}_KFE").qposadr[0] - 7)
        self._knee_indices = np.array(knee_indices) # For knee reward deviation.
        
        ffe_indices = []
        for side in ["LL", "LR"]:
            ffe_indices.append(self._mj_model.joint(f"{side}_FFE").qposadr[0] - 7)
        self._ffe_indices = np.array(ffe_indices) # For ffe reward deviation.

        faa_indices = []
        for side in ["LL", "LR"]:
            faa_indices.append(self._mj_model.joint(f"{side}_FAA").qposadr[0] - 7)
        self._faa_indices = np.array(faa_indices) # For faa reward deviation.
        
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

        q_j_noise_scale = np.zeros(self._nb_joints)
        hip_ids = [0, 1, 2, 6, 7, 8] # TODO fix this hardcoded thing
        q_j_noise_scale[hip_ids] = self._config.noise_config.scales.hip_pos
        q_j_noise_scale[self._knee_indices] = self._config.noise_config.scales.kfe_pos
        q_j_noise_scale[self._ffe_indices] = self._config.noise_config.scales.ffe_pos
        q_j_noise_scale[self._faa_indices] = self._config.noise_config.scales.faa_pos
        self._q_j_noise_scale = np.array(q_j_noise_scale)


    def reset(self):
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
        
        
def main():
    biped = Biped()
    biped.reset()
    print(biped.data.qpos)
    print(biped.data.qvel)
    
if __name__ == "__main__":
    main()        

