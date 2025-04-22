from typing import Union

import numpy as np
import mujoco

from typing import Any, Dict, Optional, Tuple, Union

import time
import os
import sys
import random
import torch
import gymnasium as gym

from gymnasium.vector.utils import batch_space

from brax.io import torch as brax_torch

def get_rz_np(
    phi: Union[np.ndarray, float],
    swing_height: Union[np.ndarray, float] = 0.08
) -> np.ndarray:
  def cubic_bezier_interpolation(y_start, y_end, x):
    y_diff = y_end - y_start
    bezier = x**3 + 3 * (x**2 * (1 - x))
    return y_start + y_diff * bezier

  x = (phi + np.pi) / (2 * np.pi)
  stance = cubic_bezier_interpolation(0, swing_height, 2 * x)
  swing = cubic_bezier_interpolation(swing_height, 0, 2 * x - 1)
  return np.where(x <= 0.5, stance, swing)

def get_collision_info_np(
    contact: Any, geom1: int, geom2: int
) -> Tuple[np.ndarray, np.ndarray]:
  """Get the distance and normal of the collision between two geoms."""
  mask = (np.array([geom1, geom2]) == contact.geom).all(axis=1)
  mask |= (np.array([geom2, geom1]) == contact.geom).all(axis=1)

  # If no contacts found, return a large distance and zero normal
  if not np.any(mask):
    return np.array(1e4), np.zeros(3)

  idx = np.where(mask, contact.dist, 1e4).argmin()
  dist = contact.dist[idx] * mask[idx]
  normal = (dist < 0) * contact.frame[idx, :3]
  return dist, normal

def geoms_colliding_np(state,
                       geom1: int, geom2: int) -> bool:
  """Return True if the two geoms are colliding."""
  return get_collision_info_np(state.contact, geom1, geom2)[0] < 0

def draw_joystick_command(
    scn,
    cmd,
    xyz,
    theta,
    rgba=None,
    radius=0.02,
    scl=1.0,
):
  if rgba is None:
    rgba = [0.2, 0.2, 0.6, 0.3]
  scn.ngeom += 1
  scn.geoms[scn.ngeom - 1].category = mujoco.mjtCatBit.mjCAT_DECOR

  vx, vy, vtheta = cmd

  angle = theta + vtheta
  rotation_matrix = np.array(
      [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
  )

  arrow_from = xyz
  rotated_velocity = rotation_matrix @ np.array([vx, vy])
  to = np.asarray([rotated_velocity[0], rotated_velocity[1], 0])
  to = to / (np.linalg.norm(to) + 1e-6)
  arrow_to = arrow_from + to * scl

  mujoco.mjv_initGeom(
      geom=scn.geoms[scn.ngeom - 1],
      type=mujoco.mjtGeom.mjGEOM_ARROW.value,
      size=np.zeros(3),
      pos=np.zeros(3),
      mat=np.zeros(9),
      rgba=np.asarray(rgba).astype(np.float32),
  )
  mujoco.mjv_connector(
      geom=scn.geoms[scn.ngeom - 1],
      type=mujoco.mjtGeom.mjGEOM_ARROW.value,
      width=radius,
      from_=arrow_from,
      to=arrow_to,
  )

def set_seed(seed: Optional[int] = None) -> int:
  ''' Taken from skrl '''
  # generate a random seed
  if seed is None:
    try:
      seed = int.from_bytes(os.urandom(4), byteorder=sys.byteorder)
    except NotImplementedError:
      seed = int(time.time() * 1000)
    seed %= 2**31  # NumPy's legacy seeding seed must be between 0 and 2**32 - 1
  seed = int(seed)

  # numpy
  random.seed(seed)
  np.random.seed(seed)

  # torch
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

  return seed

from brax.envs.base import Env, State, Wrapper
import jax
import jax.numpy as jp

class EpisodeWrapper(Wrapper):
  """Maintains episode step count and sets done at episode end."""

  def __init__(self, env: Env, episode_length: int, action_repeat: int):
    super().__init__(env)
    self.episode_length = episode_length
    self.action_repeat = action_repeat

  def reset(self, rng: jax.Array) -> State:
    state = self.env.reset(rng)
    state.info['steps'] = jp.zeros(rng.shape[:-1])
    state.info['truncation'] = jp.zeros(rng.shape[:-1])
    # Keep separate record of episode done as state.info['done'] can be erased
    # by AutoResetWrapper
    state.info['episode_done'] = jp.zeros(rng.shape[:-1])
    episode_metrics = dict()
    episode_metrics['sum_reward'] = jp.zeros(rng.shape[:-1])
    episode_metrics['length'] = jp.zeros(rng.shape[:-1])
    for metric_name in state.metrics.keys():
      episode_metrics[metric_name] = jp.zeros(rng.shape[:-1])
    state.info['episode_metrics'] = episode_metrics
    return state

  def step(self, state: State, action: jax.Array) -> State:
    def f(state, _):
      nstate = self.env.step(state, action)
      return nstate, nstate.reward

    state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
    state = state.replace(reward=jp.sum(rewards, axis=0))
    steps = state.info['steps'] + self.action_repeat
    one = jp.ones_like(state.done)
    zero = jp.zeros_like(state.done)
    episode_length = jp.array(self.episode_length, dtype=jp.int32)
    done = jp.where(steps >= episode_length, one, state.done)
    state.info['truncation'] = jp.where(
        steps >= episode_length, 1 - state.done, zero
    )
    state.info['steps'] = steps

    # Aggregate state metrics into episode metrics
    prev_done = state.info['episode_done']
    state.info['episode_metrics']['sum_reward'] += jp.sum(rewards, axis=0)
    state.info['episode_metrics']['sum_reward'] *= (1 - prev_done)
    state.info['episode_metrics']['length'] += self.action_repeat
    state.info['episode_metrics']['length'] *= (1 - prev_done)
    for metric_name in state.metrics.keys():
      if metric_name != 'reward':
        state.info['episode_metrics'][metric_name] += state.metrics[metric_name]
        state.info['episode_metrics'][metric_name] *= (1 - prev_done)
    state.info['episode_done'] = done
    return state.replace(done=done)

from typing import ClassVar
from gymnasium import spaces
from gymnasium import utils
from brax.io import image
import jax

class VectorGymWrapper(gym.vector.VectorEnv):
  """A wrapper that converts batched Brax Env to one that follows Gym VectorEnv API."""

  # Flag that prevents `gym.register` from misinterpreting the `_step` and
  # `_reset` as signs of a deprecated gym Env API.
  _gym_disable_underscore_compat: ClassVar[bool] = True

  def __init__(
      self, env, seed: int = 0, backend: Optional[str] = None
  ):
    self._env = env
    
    # Reset the env.
    self._env.reset(jax.random.PRNGKey(seed))
    
    self.metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 1 / self._env.ctrl_dt,
    }
    if not hasattr(self._env, 'batch_size'):
      raise ValueError('underlying env must be batched')

    self.num_envs = self._env.batch_size
    self.seed(seed)
    self.backend = backend
    self._state = None

    obs = np.inf * np.ones(self._env.observation_size["state"][0], dtype='float32')
    obs_space = spaces.Box(-obs, obs, dtype='float32')
    self.observation_space = batch_space(obs_space, self.num_envs)

    action = np.inf * np.ones(self._env.action_size, dtype='float32')
    action_space = spaces.Box(-action, action, dtype='float32')
    self.action_space = batch_space(action_space, self.num_envs)

    def reset(key):
      key1, key2 = jax.random.split(key)
      state = self._env.reset(key2)
      return state, state.obs, key1

    self._reset = jax.jit(reset, backend=self.backend)

    def step(state, action):
      state = self._env.step(state, action)
      info = {**state.metrics, **state.info}
      return state, state.obs, state.reward, state.done, info

    self._step = jax.jit(step, backend=self.backend)

  def reset(self):
    self._state, obs, self._key = self._reset(self._key)
    return obs

  def step(self, action):
    self._state, obs, reward, terminated, info = self._step(self._state, action)
    # truncated = jax.numpy.zeros_like(terminated) # No truncation in the env.
    return obs, reward, terminated, None, info

  def seed(self, seed: int = 0):
    self._key = jax.random.PRNGKey(seed)


class TorchWrapper(gym.vector.VectorEnv):
  """Wrapper that converts Jax tensors to PyTorch tensors."""

  def __init__(self, env: gym.vector.VectorEnv, device: Optional[torch.device] = None):
    """Creates a gym Env to one that outputs PyTorch tensors."""
    super().__init__()
    self.env = env
    self.device = device

  def reset(self):
    obs = self.env.reset()
    return brax_torch.jax_to_torch(obs, device=self.device)

  def step(self, action):
    action = brax_torch.torch_to_jax(action)
    obs, reward, terminated, truncated, info = self.env.step(action)
    obs = brax_torch.jax_to_torch(obs, device=self.device)
    reward = brax_torch.jax_to_torch(reward, device=self.device)
    terminated = brax_torch.jax_to_torch(terminated, device=self.device)
    truncated = brax_torch.jax_to_torch(truncated, device=self.device)
    info = brax_torch.jax_to_torch(info, device=self.device)
    return obs, reward, terminated, truncated, info

  def seed(self, seed: Optional[int] = None):
    return self.env.seed(seed)

  def render(self, mode: str = "human"):
    return self.env.render(mode)


class VmapWrapper(Wrapper):
  """Vectorizes Brax env."""

  def __init__(self, env: Env, batch_size: Optional[int] = None):
    super().__init__(env)
    self.batch_size = batch_size

  def reset(self, rng: jax.Array) -> State:
    if self.batch_size is not None:
      rng = jax.random.split(rng, self.batch_size)
    return jax.vmap(self.env.reset)(rng)

  def step(self, state: State, action: jax.Array) -> State:
    return jax.vmap(self.env.step)(state, action)


class AutoResetWrapper(Wrapper):
  """Automatically resets Brax envs that are done."""

  def reset(self, rng: jax.Array) -> State:
    state = self.env.reset(rng)
    state.info['first_data'] = state.data
    state.info['first_obs'] = state.obs
    return state

  def step(self, state: State, action: jax.Array) -> State:
    if 'steps' in state.info:
      steps = state.info['steps']
      steps = jp.where(state.done, jp.zeros_like(steps), steps)
      state.info.update(steps=steps)
    state = state.replace(done=jp.zeros_like(state.done))
    state = self.env.step(state, action)

    def where_done(x, y):
      done = state.done
      if done.shape:
        done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
      return jp.where(done, x, y) # Returns x if the environment is done, y if it's not

    data = jax.tree.map(
        where_done, state.info['first_data'], state.data
    )
    obs = jax.tree.map(where_done, state.info['first_obs'], state.obs)
    return state.replace(data=data, obs=obs)

