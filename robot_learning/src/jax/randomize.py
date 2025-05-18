# Modified from mujoco-playground.
"""Domain randomization for the Caltech's biped environment."""

import jax
from mujoco import mjx
import argparse
import json
import os
from etils import epath
import mujoco

FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 1
TORSO_BODY_NAME = "base_link"

CONFIG_RANDOMIZE = {
  'randomize_floor_friction': False,
  'randomize_link_masses': False,
  'randomize_torso_mass': False,
  'randomize_qpos0': False,
  'randomize_body_ipos': False,
  'randomize_actuator_gainprm': False,
  'randomize_spring_joints': False,
}

parser = argparse.ArgumentParser(description='Domain randomization configuration')
parser.add_argument('--randomize_floor_friction', action='store_true',
                    help='Randomize floor friction')
parser.add_argument('--no_randomize_floor_friction', action='store_true',
                    dest='randomize_floor_friction',
                    help='Disable floor friction randomization')
parser.add_argument('--randomize_link_masses', action='store_true',
                    help='Randomize link masses')
parser.add_argument('--randomize_torso_mass', action='store_true',
                    help='Randomize torso mass')
parser.add_argument('--randomize_qpos0', action='store_true',
                    help='Randomize initial joint positions')
parser.add_argument('--randomize_body_ipos', action='store_true',
                    help='Randomize body center of mass offsets')
parser.add_argument('--randomize_actuator_gainprm', action='store_true',
                    help='Randomize actuator gain parameters')
parser.add_argument('--randomize_spring_joints', action='store_true',
                    help='Randomize spring joints')
args = parser.parse_args()

# Update CONFIG_RANDOMIZE with command line arguments.
CONFIG_RANDOMIZE.update({
    'randomize_floor_friction': args.randomize_floor_friction,
    'randomize_link_masses': args.randomize_link_masses,
    'randomize_torso_mass': args.randomize_torso_mass,
    'randomize_qpos0': args.randomize_qpos0,
    'randomize_body_ipos': args.randomize_body_ipos,
    'randomize_actuator_gainprm': args.randomize_actuator_gainprm,
    'randomize_spring_joints': args.randomize_spring_joints,
})

# Testing: load the latest weights and test the policy.
RESULTS_FOLDER_PATH = os.path.abspath('results')

# Sort by date and get the latest folder.
folders = sorted(os.listdir(RESULTS_FOLDER_PATH))
latest_folder = folders[-1]

# Save the config to a file.
config_path = epath.Path(RESULTS_FOLDER_PATH) / latest_folder / 'config_randomize.json'

# Save the current config.
with open(config_path, 'w') as f:
    json.dump(CONFIG_RANDOMIZE, f, indent=2)

with open(config_path, 'r') as f:
    CONFIG_RANDOMIZE = json.load(f)

folder_path = epath.Path(RESULTS_FOLDER_PATH) / latest_folder
# Read in the actuator mapping.
with open(os.path.join(folder_path, 'idx_actuators_dict.json'), 'r') as f:
    IDX_ACTUATORS_DICT = json.load(f)
    print(f'IDX_ACTUATORS_DICT: {IDX_ACTUATORS_DICT}')

def domain_randomize(model: mjx.Model, rng: jax.Array):
    @jax.vmap
    def rand_dynamics(rng):
        model_updates = {}

        if CONFIG_RANDOMIZE['randomize_floor_friction']:
            # Floor friction: =U(0.4, 1.0).
            rng, key = jax.random.split(rng)
            geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(
                jax.random.uniform(key, minval=0.4, maxval=1.0)
            )
            model_updates["geom_friction"] = geom_friction

        if CONFIG_RANDOMIZE['randomize_link_masses']:
            # Scale all link masses: *U(0.9, 1.1).
            rng, key = jax.random.split(rng)
            dmass = jax.random.uniform(
                key, shape=(model.nbody,), minval=0.9, maxval=1.1
            )
            body_mass = model.body_mass.at[:].set(model.body_mass * dmass)
            model_updates["body_mass"] = body_mass

        if CONFIG_RANDOMIZE['randomize_torso_mass']:
            # Add mass to torso: +U(-1.0, 1.0).
            rng, key = jax.random.split(rng)
            dmass = jax.random.uniform(key, minval=-1.0, maxval=1.0)
            body_mass = model.body_mass.at[TORSO_BODY_ID].set(
                model.body_mass[TORSO_BODY_ID] + dmass
            )
            model_updates["body_mass"] = body_mass

        if CONFIG_RANDOMIZE['randomize_qpos0']:
            # Jitter qpos0: +U(-0.1, 0.1).
            rng, key = jax.random.split(rng)
            qpos0 = model.qpos0
            qpos0 = qpos0.at[7:].set(
                qpos0[7:]
                + jax.random.uniform(key, shape=(10,), minval=-0.1, maxval=0.1)
            )
            model_updates["qpos0"] = qpos0

        if CONFIG_RANDOMIZE['randomize_body_ipos']:
            # Center of mass offset.
            rng, key = jax.random.split(rng)
            com_offset = jax.random.uniform(key, shape=(3,), minval=-0.005, maxval=0.005)
            body_ipos = model.body_ipos
            body_ipos = body_ipos.at[TORSO_BODY_ID].set(
                body_ipos[TORSO_BODY_ID] + com_offset
            )
            model_updates["body_ipos"] = body_ipos

        if CONFIG_RANDOMIZE['randomize_actuator_gainprm']:
            # Kp and Kv for the motors.
            actuator_gainprm = model.actuator_gainprm
            actuator_biasprm = model.actuator_biasprm
            for i in range(model.nu):
                # Get the name of the actuator.
                kp_nominal = model.actuator_gainprm[i][0]
                kd_nominal = model.actuator_biasprm[i][2]
                rng, key = jax.random.split(rng)
                dkp = jax.random.uniform(key, minval=0.9, maxval=1.1)
                dkd = jax.random.uniform(key, minval=0.9, maxval=1.1)
                actuator_gainprm = actuator_gainprm.at[i, 0].set(kp_nominal * dkp)
                actuator_biasprm = actuator_biasprm.at[i, 1].set(-kp_nominal * dkp)
                actuator_biasprm = actuator_biasprm.at[i, 2].set(kd_nominal * dkd)

            model_updates["actuator_gainprm"] = actuator_gainprm
            model_updates["actuator_biasprm"] = actuator_biasprm

        if CONFIG_RANDOMIZE['randomize_spring_joints']:
            # Kp and Kv for the motors.
            SPRING_JOINTS = ['L_SPRING_ROLL', 'L_SPRING_PITCH', 'R_SPRING_ROLL', 'R_SPRING_PITCH']
            IDX_SPRING_JOINTS = [IDX_ACTUATORS_DICT[name] for name in SPRING_JOINTS]
            actuator_gainprm = model.actuator_gainprm
            actuator_biasprm = model.actuator_biasprm
            for i in range(model.nu):
                if i in IDX_SPRING_JOINTS:
                    kp_nominal = model.actuator_gainprm[i][0]
                    kd_nominal = model.actuator_biasprm[i][2]
                    rng, key = jax.random.split(rng)
                    dkp = jax.random.uniform(key, minval=0.9, maxval=1.1)
                    dkd = jax.random.uniform(key, minval=0.9, maxval=1.1)
                    actuator_gainprm = actuator_gainprm.at[i, 0].set(kp_nominal * dkp)
                    actuator_biasprm = actuator_biasprm.at[i, 1].set(-kp_nominal * dkp)
                    actuator_biasprm = actuator_biasprm.at[i, 2].set(kd_nominal * dkd)

        return model_updates

    model_updates = rand_dynamics(rng)

    # Create in_axes mapping for the enabled randomization features.
    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({
        key: 0 for key in model_updates.keys()
    })

    # Update model with the randomized parameters.
    model = model.tree_replace(model_updates)

    return model, in_axes