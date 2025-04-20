# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Domain randomization for the Caltech's biped environment."""

import jax
from mujoco import mjx

FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 1
TORSO_BODY_NAME = "base_link"

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