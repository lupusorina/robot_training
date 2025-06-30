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
  'randomize_floor_friction': {
    'enable': False,
    'minval': 0.4,
    'maxval': 1.0,
  },
  'randomize_link_masses': {
    'enable': False,
    'minval': 0.9, # Percentage of the original mass
    'maxval': 1.1, # Percentage of the original mass
  },
  'randomize_torso_mass': {
    'enable': False,
    'minval': -0.5,
    'maxval': 0.5,
  },
  'randomize_qpos0':
  {
    'enable': False,
    'minval': -0.1,
    'maxval': 0.1,
  },
  'randomize_body_ipos': {
    'enable': False,
    'minval': [-0.05, -0.02, -0.005],
    'maxval': [0.05, 0.02, 0.005],
  },
  'randomize_actuator_gainprm':
  {
    'enable': False,
    'minval': 0.9,
    'maxval': 1.1,
  },
}

# Try to parse command line arguments, but handle Jupyter notebook environments gracefully
try:
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
    args = parser.parse_args()

    # Update CONFIG_RANDOMIZE with command line arguments.
    CONFIG_RANDOMIZE.update({
        'randomize_floor_friction': {
          'enable': args.randomize_floor_friction,
          'minval': 0.4,
          'maxval': 1.0,
        },
        'randomize_link_masses': {
          'enable': args.randomize_link_masses,
          'minval': 0.9,
          'maxval': 1.1,
        },
        'randomize_torso_mass': {
          'enable': args.randomize_torso_mass,
          'minval': -0.5,
          'maxval': 0.5,
        },
        'randomize_qpos0': {
          'enable': args.randomize_qpos0,
          'minval': -0.1,
          'maxval': 0.1,
        },
        'randomize_actuator_gainprm': {
          'enable': args.randomize_actuator_gainprm,
          'minval': 0.9,
          'maxval': 1.1,
        },
        'randomize_body_ipos': {
          'enable': args.randomize_body_ipos,
          'minval': [-0.05, -0.02, -0.005],
          'maxval': [0.05, 0.02, 0.005],
        },
    })
except SystemExit:
    # This happens when argparse fails (e.g., in Jupyter notebooks)
    # Keep the default CONFIG_RANDOMIZE values
    pass

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


def domain_randomize(model: mjx.Model, rng: jax.Array):
    @jax.vmap
    def rand_dynamics(rng):
        model_updates = {}

        if CONFIG_RANDOMIZE['randomize_floor_friction']['enable']:
            # Floor friction: =U(0.4, 1.0).
            rng, key = jax.random.split(rng)
            geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(
                jax.random.uniform(key, minval=CONFIG_RANDOMIZE['randomize_floor_friction']['minval'], maxval=CONFIG_RANDOMIZE['randomize_floor_friction']['maxval'])
            )
            model_updates["geom_friction"] = geom_friction

        if CONFIG_RANDOMIZE['randomize_link_masses']['enable']:
            # Scale all link masses: *U(0.9, 1.1).
            rng, key = jax.random.split(rng)
            dmass = jax.random.uniform(
                key, shape=(model.nbody,), minval=CONFIG_RANDOMIZE['randomize_link_masses']['minval'], maxval=CONFIG_RANDOMIZE['randomize_link_masses']['maxval']
            )
            body_mass = model.body_mass.at[:].set(model.body_mass * dmass)
            model_updates["body_mass"] = body_mass

        if CONFIG_RANDOMIZE['randomize_torso_mass']['enable']:
            # Add mass to torso: +U(-1.0, 1.0).
            rng, key = jax.random.split(rng)
            dmass = jax.random.uniform(key, minval=CONFIG_RANDOMIZE['randomize_torso_mass']['minval'], maxval=CONFIG_RANDOMIZE['randomize_torso_mass']['maxval'])
            body_mass = model.body_mass.at[TORSO_BODY_ID].set(
                model.body_mass[TORSO_BODY_ID] + dmass
            )
            model_updates["body_mass"] = body_mass

        if CONFIG_RANDOMIZE['randomize_qpos0']['enable']:
            # Jitter qpos0: +U(-0.1, 0.1).
            rng, key = jax.random.split(rng)
            qpos0 = model.qpos0
            qpos0 = qpos0.at[7:].set(
                qpos0[7:]
                + jax.random.uniform(key, shape=(10,), minval=CONFIG_RANDOMIZE['randomize_qpos0']['minval'], maxval=CONFIG_RANDOMIZE['randomize_qpos0']['maxval'])
            )
            model_updates["qpos0"] = qpos0

        if CONFIG_RANDOMIZE['randomize_body_ipos']['enable']:
            # Center of mass offset.
            rng, key = jax.random.split(rng)
            com_offset_x = jax.random.uniform(key, shape=(1,), minval=CONFIG_RANDOMIZE['randomize_body_ipos']['minval'][0], maxval=CONFIG_RANDOMIZE['randomize_body_ipos']['maxval'][0])
            com_offset_y = jax.random.uniform(key, shape=(1,), minval=CONFIG_RANDOMIZE['randomize_body_ipos']['minval'][1], maxval=CONFIG_RANDOMIZE['randomize_body_ipos']['maxval'][1])
            com_offset_z = jax.random.uniform(key, shape=(1,), minval=CONFIG_RANDOMIZE['randomize_body_ipos']['minval'][2], maxval=CONFIG_RANDOMIZE['randomize_body_ipos']['maxval'][2])
            com_offset = jax.numpy.array([com_offset_x, com_offset_y, com_offset_z]).reshape(3)
            body_ipos = model.body_ipos

            body_ipos = body_ipos.at[TORSO_BODY_ID].set(
                body_ipos[TORSO_BODY_ID] + com_offset)

            model_updates["body_ipos"] = body_ipos

        if CONFIG_RANDOMIZE['randomize_actuator_gainprm']['enable']:
            # Kp and Kv for the motors.
            actuator_gainprm = model.actuator_gainprm
            actuator_biasprm = model.actuator_biasprm
            for i in range(model.nu):
                # Get the name of the actuator.
                kp_nominal = model.actuator_gainprm[i][0]
                kd_nominal = model.actuator_biasprm[i][2]
                rng, key = jax.random.split(rng)
                dkp = jax.random.uniform(key, minval=CONFIG_RANDOMIZE['randomize_actuator_gainprm']['minval'], maxval=CONFIG_RANDOMIZE['randomize_actuator_gainprm']['maxval'])
                dkd = jax.random.uniform(key, minval=CONFIG_RANDOMIZE['randomize_actuator_gainprm']['minval'], maxval=CONFIG_RANDOMIZE['randomize_actuator_gainprm']['maxval'])
                actuator_gainprm = actuator_gainprm.at[i, 0].set(kp_nominal * dkp)
                actuator_biasprm = actuator_biasprm.at[i, 1].set(-kp_nominal * dkp)
                actuator_biasprm = actuator_biasprm.at[i, 2].set(kd_nominal * dkd)

            model_updates["actuator_gainprm"] = actuator_gainprm
            model_updates["actuator_biasprm"] = actuator_biasprm
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