# Modified from mujoco-playground.
"""Domain randomization for the Caltech's biped environment."""

import jax
from mujoco import mjx
import argparse
import json
import os
from etils import epath

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

args = parser.parse_args()

# Update CONFIG_RANDOMIZE with command line arguments.
CONFIG_RANDOMIZE.update({
    'randomize_floor_friction': args.randomize_floor_friction,
    'randomize_link_masses': args.randomize_link_masses,
    'randomize_torso_mass': args.randomize_torso_mass,
    'randomize_qpos0': args.randomize_qpos0,
    'randomize_body_ipos': args.randomize_body_ipos,
    'randomize_actuator_gainprm': args.randomize_actuator_gainprm,
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

print(CONFIG_RANDOMIZE)

def domain_randomize(model: mjx.Model, rng: jax.Array):
    @jax.vmap
    def rand_dynamics(rng):
        # Floor friction: =U(0.4, 1.0).
        rng, key = jax.random.split(rng)
        geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(
            jax.random.uniform(key, minval=0.4, maxval=1.0)
        )

        # Scale all link masses: *U(0.9, 1.1).
        rng, key = jax.random.split(rng)
        dmass = jax.random.uniform(
            key, shape=(model.nbody,), minval=0.9, maxval=1.1
        )
        body_mass = model.body_mass.at[:].set(model.body_mass * dmass)

        # # Add mass to torso: +U(-1.0, 1.0).
        rng, key = jax.random.split(rng)
        dmass = jax.random.uniform(key, minval=-1.0, maxval=1.0)
        body_mass = body_mass.at[TORSO_BODY_ID].set(
            body_mass[TORSO_BODY_ID] + dmass
        )

        # Jitter qpos0: +U(-0.1, 0.1).
        rng, key = jax.random.split(rng)
        qpos0 = model.qpos0
        qpos0 = qpos0.at[7:].set(
            qpos0[7:]
            + jax.random.uniform(key, shape=(10,), minval=-0.1, maxval=0.1)
        )

        # Center of mass offset.
        rng, key = jax.random.split(rng)
        com_offset = jax.random.uniform(key, shape=(3,), minval=-0.1, maxval=0.1)
        body_ipos = model.body_ipos
        body_ipos = body_ipos.at[TORSO_BODY_ID].set(
            body_ipos[TORSO_BODY_ID] + com_offset
        )

        # Kp and Kv for the motors.
        # Initialize actuator gain parameters
        actuator_gainprm = model.actuator_gainprm

        # Update each actuator's gain parameter
        for i in range(model.nu):
            kp_nominal = model.actuator_gainprm[i][0]
            rng, key = jax.random.split(rng)
            dkp = jax.random.uniform(key, minval=-0.5, maxval=0.5)
            actuator_gainprm = actuator_gainprm.at[i, 0].set(kp_nominal + dkp)

        return (
            geom_friction,
            body_mass,
            qpos0,
            body_ipos,
            actuator_gainprm,
        )

    (
        friction,
        body_mass,
        qpos0,
        body_ipos,
        actuator_gainprm,
    ) = rand_dynamics(rng)

    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({
        "geom_friction": 0,
        "body_mass": 0,
        "qpos0": 0,
        "body_ipos": 0,
        "actuator_gainprm": 0,
    })

    model = model.tree_replace({
        "geom_friction": friction,
        "body_mass": body_mass,
        "qpos0": qpos0,
        "body_ipos": body_ipos,
        "actuator_gainprm": actuator_gainprm,
    })

    return model, in_axes