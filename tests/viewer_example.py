"""An example integration of MJX with the MuJoCo viewer."""
""" to run: >> mjpython viewer_example.py """

import logging
import time
from typing import Sequence

from absl import app
from absl import flags
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
import mujoco.viewer


_JIT = flags.DEFINE_bool('jit', True, 'To jit or not to jit.')

NAME_ROBOT = 'biped'
if NAME_ROBOT == 'berkeley_humanoid':
    import assets.berkeley_humanoid.config as robot_config
if NAME_ROBOT == 'biped':
    import assets.biped.config as robot_config
    # raise NotImplementedError

print('NAME_ROBOT:', NAME_ROBOT)

_MODEL_PATH = robot_config.XML_PATH

_VIEWER_GLOBAL_STATE = {
    'running': True,
}

def key_callback(key: int) -> None:
    if key == 32:  # Space bar
        _VIEWER_GLOBAL_STATE['running'] = not _VIEWER_GLOBAL_STATE['running']
        logging.info('RUNNING = %s', _VIEWER_GLOBAL_STATE['running'])

def _main(argv: Sequence[str]) -> None:
    """Launches MuJoCo passive viewer fed by MJX."""
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    jax.config.update('jax_debug_nans', True)

    print(f'Loading model from: {_MODEL_PATH}.')

    m = mujoco.MjModel.from_xml_path(_MODEL_PATH)
    d = mujoco.MjData(m)
    mx = mjx.put_model(m)
    dx = mjx.put_data(m, d)

    print(f'Default backend: {jax.default_backend()}')
    step_fn = mjx.step
    if _JIT.value:
        print('JIT-compiling the model physics step...')
        start = time.time()
        step_fn = jax.jit(step_fn).lower(mx, dx).compile()
        elapsed = time.time() - start
        print(f'Compilation took {elapsed}s.')

    viewer = mujoco.viewer.launch_passive(m, d, key_callback=key_callback)
    with viewer:
        while True:
            start = time.time()

            # TODO(robotics-simulation): recompile when changing disable flags, etc.
            dx = dx.replace(
                ctrl=jp.array(d.ctrl),
                act=jp.array(d.act),
                xfrc_applied=jp.array(d.xfrc_applied),
            )
            dx = dx.replace(
                qpos=jp.array(d.qpos), qvel=jp.array(d.qvel), time=jp.array(d.time)
            )  # handle resets
            mx = mx.tree_replace({
                'opt.gravity': m.opt.gravity,
                'opt.tolerance': m.opt.tolerance,
                'opt.ls_tolerance': m.opt.ls_tolerance,
                'opt.timestep': m.opt.timestep,
            })

            if _VIEWER_GLOBAL_STATE['running']:
                dx = step_fn(mx, dx)

            mjx.get_data_into(d, m, dx)
            viewer.sync()

            elapsed = time.time() - start
            if elapsed < m.opt.timestep:
                time.sleep(m.opt.timestep - elapsed)

def main():
  app.run(_main)

if __name__ == '__main__':
  main()