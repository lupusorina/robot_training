# Tutorial taken from https://deepnote.com/app/jeh15/Brax-Tutorial-a789d37f-631a-455f-a2ad-2f2b88450dd8
# JAX Imports.
import jax
import jax.numpy as jnp

xml_model = """
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

# Brax Imports:
from brax.mjx import pipeline
from brax.io import mjcf, html
import functools
from brax.io import model

import datetime
import os

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.envs.base import PipelineEnv, State

# Load the MJCF model.
sys = mjcf.loads(xml_model)

# Jitting the init and step functions for GPU acceleration.
init_fn = jax.jit(pipeline.init)
step_fn = jax.jit(pipeline.step)
print('JAX JIT Compilation Complete')

# Initializing the state.
state = init_fn(
    sys=sys, q=sys.init_q, qd=jnp.zeros(sys.qd_size()),
)

class CartPole(PipelineEnv):
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


DATE = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
FOLDER_SAVE = f'logs/results'
os.makedirs(f'{FOLDER_SAVE}', exist_ok=True)
print(f'Logs will be saved in {FOLDER_SAVE}')

print("Start training the CartPole Environment")
env = CartPole(xml_model=xml_model, backend='mjx')
eval_env = CartPole(xml_model=xml_model, backend='mjx')

def progress_fn(current_step, metrics):
    if current_step > 0:
        print(f'Step: {current_step} \t Reward: Episode Reward: {metrics["eval/episode_reward"]:.3f}')

make_networks_factory = functools.partial(
    ppo_networks.make_ppo_networks,
    policy_hidden_layer_sizes=(128, 128, 128, 128),
)

train_fn = functools.partial(
    ppo.train,
    num_timesteps=1000,
    num_evals=1,
    episode_length=2,
    num_envs=32,
    num_eval_envs=4,
    batch_size=32,
    num_minibatches=4,
    unroll_length=20,
    num_updates_per_batch=4,
    normalize_observations=True,
    discounting=0.97,
    learning_rate=3.0e-4,
    entropy_cost=1e-2,
    network_factory=make_networks_factory,
    seed=0
)

make_policy_fn, params, _ = train_fn(
    environment=env,
    progress_fn=progress_fn,
    eval_env=eval_env,
)
print("Training Complete")

MODEL_NAME = 'model.pkl'
model.save_params(f'{FOLDER_SAVE}/{MODEL_NAME}', params)


# Inference with JIT.
params = model.load_params(f'{FOLDER_SAVE}/{MODEL_NAME}')

env = CartPole(xml_model=xml_model, backend='mjx')

inference_fn = make_policy_fn(params)
jit_inference_fn = jax.jit(inference_fn)
reset_fn = jax.jit(env.reset)
step_fn = jax.jit(env.step)

key = jax.random.key(42)
state = reset_fn(key)

print("Start Simulation")
state_history = []
num_steps = 200
for i in range(num_steps):
    key, subkey = jax.random.split(key)
    action, _ = jit_inference_fn(state.obs, subkey)
    state = step_fn(state, action)
    state_history.append(state.pipeline_state)
print("Simulation Complete")
    
# MJX Backend Requires Contact Information even if it does not exist.
state_history = list(map(lambda x: x.replace(contact=None), state_history))

# Render the HTML content to a local server.
from flask import Flask, render_template_string
app = Flask(__name__)
@app.route('/')
def home():
    # Render your HTML content.
    raw_html =  html.render(
        sys=env.sys.tree_replace({'opt.timestep': env.dt}),
        states=state_history,
        height=480,
    )

    # Return the rendered HTML content.
    return render_template_string(raw_html)

app.run(debug=False, port=5020)
