
### For Pytorch.
# TODO: too complicated, simplify.
import gymnasium as gym
from gymnasium.vector import utils
from gymnasium import spaces
from typing import ClassVar, Optional
import numpy as np
from brax.envs.base import PipelineEnv
import jax

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

    obs_size = self._env.observation_size
    obs = np.inf * np.ones(obs_size, dtype='float32') # TODO: add obs size in brax.
    obs_space = spaces.Box(-obs, obs, dtype='float32')
    self.observation_space = utils.batch_space(obs_space, self.num_envs)

    action = jax.tree.map(np.array, self._env.ctrl_range)
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

  from jax.biped import Biped
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
