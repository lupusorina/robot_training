import os

parent_dir = os.path.abspath(os.path.join(os.getcwd()))
XML_PATH = os.path.join(parent_dir, '../assets/biped/xmls/biped_RL.xml')
# XML_PATH = os.path.join(parent_dir, 'robot_learning/src/assets/biped/xmls/biped_RL.xml')

# XML_PATH = os.path.join(parent_dir, '../../assets/biped/xmls/biped_RL.xml')

ROOT_BODY = "base_link"
FEET_SITES = ["l_foot", "r_foot"]
LEFT_FEET_GEOMS = ["L_FOOT"]
RIGHT_FEET_GEOMS = ["R_FOOT"]
FEET_GEOMS = LEFT_FEET_GEOMS + RIGHT_FEET_GEOMS
GRAVITY_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"
IMU_SITE = "imu_location"
DESIRED_HEIGHT = 0.56

HIP_JOINT_NAMES = ["YAW", "HAA", "HFE"]
KNEE_JOINT_NAMES = ["KFE"]
ANKLE_FE_JOINT_NAMES = ["ANKLE"]
# ANKLE_FE_JOINT_NAMES = []
ANKLE_AA_JOINT_NAMES = []

SIDES = ["L", "R"]

                    # L_YAW L_HAA L_HFE L_KFE L_ANKLE
                    # R_YAW R_HAA R_HFE R_KFE R_ANKLE
COSTS_JOINT_ANGLES = [1.0, 1.0, 0.01, 0.01, 1.0,  # left leg.
                      1.0, 1.0, 0.01, 0.01, 1.0]  # right leg.
