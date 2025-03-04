import os

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

HIP_JOINT_NAMES = ["HR", "HAA", "HFE"]
KNEE_JOINT_NAMES = ["KFE"]
ANKLE_FE_JOINT_NAMES = ["FFE"]
# ANKLE_AA_JOINT_NAMES = ["FAA"]
ANKLE_AA_JOINT_NAMES = []

SIDES = ["LL", "LR"]

# COSTS_JOINT_ANGLES = [1.0, 1.0, 0.01, 0.01, 1.0, 1.0,  # left leg.
#                     1.0, 1.0, 0.01, 0.01, 1.0, 1.0]  # right leg.


COSTS_JOINT_ANGLES = [1.0, 1.0, 0.01, 0.01, 1.0,  # left leg.
                      1.0, 1.0, 0.01, 0.01,  1.0]  # right leg.