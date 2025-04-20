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
