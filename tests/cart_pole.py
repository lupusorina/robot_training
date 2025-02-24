from brax.envs.base import PipelineEnv, State
import jax
import jax.numpy as jnp
from brax.io import mjcf, html
import numpy as np

class CartPoleJax(PipelineEnv):
    """ Environment for training Cart Pole balancing """

    def __init__(self, xml_model: str, backend: str = 'mjx', **kwargs):
        # Initialize System:
        sys = mjcf.loads(xml_model)
        self.step_dt = 0.02
        n_frames = kwargs.pop('n_frames', int(self.step_dt / sys.opt.timestep))
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