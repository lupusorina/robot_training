from brax.envs.base import PipelineEnv, State
import jax
import jax.numpy as jnp
from brax.io import mjcf
import numpy as np

XML_MODEL = """
<mujoco model="inverted pendulum">
    <compiler inertiafromgeom="true"/>

    <default>
        <joint armature="0" damping="1"/>
        <geom contype="0" conaffinity="0" friction="1 0.1 0.1"/>
    </default>

    <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <geom name="rail" type="capsule"  size="0.02 1.5" pos="0 0 0" quat="1 0 1 0" rgba="1 1 1 1"/>
        <body name="cart" pos="0 0 0">
            <joint name="slider" type="slide" axis="1 0 0" pos="0 0 0" limited="true" range="-1.5 1.5"/>
            <geom name="cart_geom" pos="0 0 0" quat="1 0 1 0" size="0.1 0.1" type="capsule"/>
            <body name="pole" pos="0 0 0">
                <joint name="hinge" type="hinge" axis="0 1 0" pos="0 0 0"/>
                <geom name="pole_geom" type="capsule" size="0.049 0.3"  fromto="0 0 0 0.001 0 0.6"/>
            </body>
        </body>
    </worldbody>

    <actuator>
        <motor ctrllimited="true" ctrlrange="-3 3" gear="100" joint="slider" name="slide"/>
    </actuator>

</mujoco>
"""

# Environment with JAX.

class CartPoleJax(PipelineEnv):
    """ Environment in JAX for training cart pole balancing """

    def __init__(self, backend: str = 'mjx', **kwargs):
        # Initialize System:
        sys = mjcf.loads(XML_MODEL)
        self.step_dt = 0.02
        n_frames = kwargs.pop('n_frames', int(self.step_dt / sys.opt.timestep))
        self.observation_space = 4
        self.action_space = 1
        super().__init__(sys, backend=backend, n_frames=n_frames)

    def reset(self, rng: jax.Array) -> State:
        key, theta_key, qd_key = jax.random.split(rng, 3)

        theta_init = jax.random.uniform(theta_key, (1,), minval=-0.1, maxval=0.1)[0]

        q_init = jnp.array([0.0, theta_init])
        qd_init = jax.random.uniform(qd_key, (2,), minval=-0.1, maxval=0.1)        
        pipeline_state = self.pipeline_init(q_init, qd_init)
        reward, done = jnp.zeros(2)
        observation = self.get_observation(pipeline_state)

        metrics = {
            'rewards': reward,
            'observation': observation,
        }

        state = State(
            pipeline_state=pipeline_state,
            obs=observation,
            reward=reward,
            done=done,
            metrics=metrics,
        )

        return state

    def step(self, state: State, action: jax.Array) -> State:
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        observation = self.get_observation(pipeline_state)
        x, th = pipeline_state.q

        outside_x = jnp.abs(x) > 1.0
        outside_th = jnp.abs(th) > jnp.pi / 2
        done = outside_x | outside_th
        done = jnp.float32(done)

        reward = jnp.cos(th)

        metrics = {
            'rewards': reward,
            'observation': observation,
        }
        state.metrics.update(metrics)

        state = state.replace(
            pipeline_state=pipeline_state, obs=observation, reward=reward, done=done,
        )
        return state

    def get_observation(self, pipeline_state: State) -> jnp.ndarray:
        # Observation: [x, th, dx, dth]
        return jnp.concatenate([pipeline_state.q, pipeline_state.qd])


# Environment without JAX.

import mujoco as mj
import mujoco.viewer

class CartPole():
    """ Environment for training cart pole balancing. """

    def __init__(self, visualize_mujoco=False):
        self.visualize_mujoco = visualize_mujoco
        self.model = mj.MjModel.from_xml_string(XML_MODEL)
        self.data = mj.MjData(self.model)
        if self.visualize_mujoco is True:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def reset(self):
        random_angle = np.random.uniform(-0.1, 0.1)
        self.data.qpos = np.array([0, random_angle])
        self.data.qvel = np.random.uniform(-0.1, 0.1, 2)
        return self._get_observation()

    def step(self, action):
        self.data.ctrl = np.array([action])
        mj.mj_step(self.model, self.data)
        if self.visualize_mujoco is True:
            if self.viewer.is_running():
                self.viewer.sync()
        return self._get_observation(), self._get_reward(), self._get_done(), {}

    def _get_observation(self):
        return np.concatenate([self.data.qpos, self.data.qvel])

    def _get_reward(self):
        return np.cos(self.data.qpos[1])

    def _get_done(self):
        return np.abs(self.data.qpos[0]) > 1.0 or np.abs(self.data.qpos[1]) > np.pi / 2

def main():
    env = CartPole()
    obs = env.reset()

    for i in range(200):
        action = np.random.uniform(-3, 3)
        obs, _, _, _ = env.step(action)

if __name__ == '__main__':
    main()