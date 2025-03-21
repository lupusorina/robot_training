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
""" Different Wrappers."""

from typing import Any, Callable, Optional, Tuple

from brax.envs.wrappers import training as brax_training
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx

import mjx_env

### For Brax.

class Wrapper(mjx_env.MjxEnv):
  """Wraps an environment to allow modular transformations."""

  def __init__(self, env: Any):  # pylint: disable=super-init-not-called
    self.env = env

  def reset(self, rng: jax.Array) -> mjx_env.State:
    return self.env.reset(rng)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    return self.env.step(state, action)

  @property
  def observation_size(self) -> mjx_env.ObservationSize:
    return self.env.observation_size

  @property
  def action_size(self) -> int:
    return self.env.action_size

  @property
  def unwrapped(self) -> Any:
    return self.env.unwrapped

  def __getattr__(self, name):
    if name == '__setstate__':
      raise AttributeError(name)
    return getattr(self.env, name)

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self.env.mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self.env.mjx_model

  @property
  def xml_path(self) -> str:
    return self.env.xml_path

def wrap_for_brax_training(
    env: mjx_env.MjxEnv,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[
        Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]
    ] = None,
) -> Wrapper:
  """Common wrapper pattern for all brax training agents.

  Args:
    env: environment to be wrapped
    episode_length: length of episode
    action_repeat: how many repeated actions to take per step
    randomization_fn: randomization function that produces a vectorized model
      and in_axes to vmap over

  Returns:
    An environment that is wrapped with Episode and AutoReset wrappers.  If the
    environment did not already have batch dimensions, it is additional Vmap
    wrapped.
  """

  if randomization_fn is None:
    env = brax_training.VmapWrapper(env)  # pytype: disable=wrong-arg-types
  else:
    print("randomization_fn is not None")
    env = BraxDomainRandomizationVmapWrapper(env, randomization_fn)
  env = brax_training.EpisodeWrapper(env, episode_length, action_repeat)
  env = BraxAutoResetWrapper(env)
  return env

class BraxAutoResetWrapper(Wrapper):
  """Automatically resets Brax envs that are done."""

  def reset(self, rng: jax.Array) -> mjx_env.State:
    state = self.env.reset(rng)
    state.info['first_state'] = state.data
    state.info['first_obs'] = state.obs
    return state

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    if 'steps' in state.info:
      steps = state.info['steps']
      steps = jp.where(state.done, jp.zeros_like(steps), steps)
      state.info.update(steps=steps)
    state = state.replace(done=jp.zeros_like(state.done))
    state = self.env.step(state, action)

    def where_done(x, y):
      done = state.done
      if done.shape:
        done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
      return jp.where(done, x, y)

    data = jax.tree.map(where_done, state.info['first_state'], state.data)
    obs = jax.tree.map(where_done, state.info['first_obs'], state.obs)
    return state.replace(data=data, obs=obs)


class BraxDomainRandomizationVmapWrapper(Wrapper):
  """Brax wrapper for domain randomization."""

  def __init__(
      self,
      env: mjx_env.MjxEnv,
      randomization_fn: Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]],
  ):
    super().__init__(env)
    self._mjx_model_v, self._in_axes = randomization_fn(self._mjx_model)

  def _env_fn(self, mjx_model: mjx.Model) -> mjx_env.MjxEnv:
    env = self.env
    env._mjx_model = mjx_model
    return env

  def reset(self, rng: jax.Array) -> mjx_env.State:
    def reset(mjx_model, rng):
      env = self._env_fn(mjx_model=mjx_model)
      return env.reset(rng)

    state = jax.vmap(reset, in_axes=[self._in_axes, 0])(self._mjx_model_v, rng)
    return state

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    def step(mjx_model, s, a):
      env = self._env_fn(mjx_model=mjx_model)
      return env.step(s, a)

    res = jax.vmap(step, in_axes=[self._in_axes, 0, 0])(
        self._mjx_model_v, state, action
    )
    return res

### For Pytorch.
# TODO: too complicated, simplify.
import gymnasium as gym
from gymnasium.vector import utils
from gymnasium import spaces
from typing import ClassVar, Optional
import numpy as np
from brax.envs.base import PipelineEnv

class VectorGymWrapper(gym.vector.VectorEnv):
  """A wrapper that converts batched Brax Env to one that follows Gym VectorEnv API."""

  def __init__(
      self, env: PipelineEnv,
      seed: int = 0,
      backend: Optional[str] = None
  ):
    self._env = env
    self.num_envs = self._env.batch_size
    self.seed(seed)
    self.backend = backend
    self._state = None

    obs_size = self._env.observation_size["state"]
    obs = np.inf * np.ones(obs_size, dtype='float32') # TODO: add obs size in brax.
    obs_space = spaces.Box(-obs, obs, dtype='float32')
    self.observation_space = utils.batch_space(obs_space, self.num_envs)

    action = jax.tree.map(np.array, self._env.sys.actuator.ctrl_range)
    action_space = spaces.Box(action[:, 0], action[:, 1], dtype='float32')
    self.action_space = utils.batch_space(action_space, self.num_envs)

    def reset(key):
        key1, key2 = jax.random.split(key)
        state = self._env.reset(key2)
        return state, state.obs, key1

    self._reset = jax.jit(reset, backend=self.backend)

    def step(state, action):
      # TODO: figure out exactly how/when to use truncated or terminated.
      # looks like PPO does an OR operator between truncated and terminated, but other
      # agents might not.
      state = self._env.step(state, action)
      info = {**state.metrics, **state.info}
      return state, state.obs, state.reward, state.done, state.done, info

    self._step = jax.jit(step, backend=self.backend)

  def reset(self):
    self._state, obs, self._key = self._reset(self._key)
    # We return device arrays for pytorch users.
    return obs

  def step(self, action):
    self._state, obs, reward, terminated, truncated, info = self._step(self._state, action)
    # We return device arrays for pytorch users.
    return self._state, obs, reward, terminated, truncated, info

  def seed(self, seed: int = 0):
    self._key = jax.random.PRNGKey(seed)

from brax.io import torch

class TorchWrapper(gym.Wrapper):
  """Wrapper that converts Jax tensors to PyTorch tensors."""

  def __init__(self, env, device: Optional[torch.Device] = None):
    """Creates a gym Env to one that outputs PyTorch tensors."""
    self.env = env
    self.device = device
    self.action_space = env.action_space
    self.observation_space = env.observation_space

  def reset(self):
    obs = self.env.reset()
    return torch.jax_to_torch(obs, device=self.device)

  def step(self, action):
    action = torch.torch_to_jax(action)
    state, obs, reward, terminated, truncated, info = self.env.step(action)
    obs = state.obs
    reward = state.reward
    done = state.done
    info = state.info
    obs = torch.jax_to_torch(obs, device=self.device)
    reward = torch.jax_to_torch(reward, device=self.device)
    done = torch.jax_to_torch(done, device=self.device)
    info = torch.jax_to_torch(info, device=self.device)
    return obs, reward, done, done, info

## Test for VectorGymWrapper
def test_VectorGymWrapper():

  # Biped
  episode_length = 1000

  from biped_berkeley import Biped
  action_repeat = True
  batch_size = 100
  auto_reset = True
  env = Biped()

  if episode_length is not None:
    env = brax_training.EpisodeWrapper(env, episode_length, action_repeat) # Maintains episode step count and sets done at episode end.
  if batch_size:
    env = brax_training.VmapWrapper(env, batch_size) # Vectorizes Brax env.
  if auto_reset:
    env = brax_training.AutoResetWrapper(env) # Automatically resets Brax envs that are done.

  env = VectorGymWrapper(env, seed=0, backend=None)
  env = TorchWrapper(env, device='cuda')

if __name__ == "__main__":
  test_VectorGymWrapper()
