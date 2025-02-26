import sys
sys.path.append('../')
from src.utils import get_rz
import jax.numpy as jp
import jax

_SEED = 1
DT = 0.002

rng = jax.random.PRNGKey(_SEED)
rng, key = jax.random.split(rng)

# Phase, freq = U(1.0, 1.5)
gait_freq = jax.random.uniform(key, (1,), minval=1.25, maxval=1.5)
phase_dt = 2 * jp.pi * DT * gait_freq
phase = jp.array([0, jp.pi])

out = get_rz(phase, 0.08)

print(out)
