import os
import subprocess

if subprocess.run('nvidia-smi').returncode:
  raise RuntimeError(
      'Cannot communicate with GPU. '
      'Make sure you are using a GPU Colab runtime. '
      'Go to the Runtime menu and select Choose runtime type.'
  )

# Add an ICD config so that glvnd can pick up the Nvidia EGL driver.
# This is usually installed as part of an Nvidia driver package, but the Colab
# kernel doesn't install its driver via APT, and as a result the ICD is missing.
# (https://github.com/NVIDIA/libglvnd/blob/master/src/EGL/icd_enumeration.md)
NVIDIA_ICD_CONFIG_PATH = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
if not os.path.exists(NVIDIA_ICD_CONFIG_PATH):
  with open(NVIDIA_ICD_CONFIG_PATH, 'w') as f:
    f.write("""{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
""")

xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=100)

from datetime import datetime
import functools
from biped_berkeley import Biped
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from IPython.display import clear_output
from matplotlib import pyplot as plt
import numpy as np
from ml_collections import config_dict
from wrapper import wrap_for_brax_training

# Folders.
FOLDER_RESULTS = 'results'
time_now = datetime.now().strftime('%Y%m%d-%H%M%S')
if not os.path.exists(os.path.join(FOLDER_RESULTS, time_now)):
    os.makedirs(os.path.join(FOLDER_RESULTS, time_now))
FOLDER_RESULTS = os.path.join(FOLDER_RESULTS, time_now)
ABS_FOLDER_RESUlTS = os.path.abspath(FOLDER_RESULTS)
print(f"Saving results to {ABS_FOLDER_RESUlTS}")

FOLDER_PLOTS = 'plots'
if not os.path.exists(os.path.join(FOLDER_PLOTS, time_now)):
    os.makedirs(os.path.join(FOLDER_PLOTS, time_now))
FOLDER_PLOTS = os.path.join(FOLDER_PLOTS, time_now)

VIDEO_FOLDER = 'videos'
if not os.path.exists(os.path.join(VIDEO_FOLDER, time_now)):
    os.makedirs(os.path.join(VIDEO_FOLDER, time_now))
VIDEO_FOLDER = os.path.join(VIDEO_FOLDER, time_now)


# Brax PPO config.
brax_ppo_config = config_dict.create(
      num_timesteps=150_000_000,
      num_evals=15,
      reward_scaling=1.0,
      clipping_epsilon=0.2,
      num_resets_per_eval=1,
      episode_length=1000,
      normalize_observations=True,
      action_repeat=1,
      unroll_length=20,
      num_minibatches=32,
      num_updates_per_batch=4,
      discounting=0.97,
      learning_rate=3e-4,
      entropy_cost=0.005,
      num_envs=8192,
      batch_size=256,
      max_grad_norm=1.0,
      network_factory = config_dict.create(
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
      ),
  )
ppo_params = brax_ppo_config

# Environment.
env_name = 'Berkeley Biped'
print(f"Starting training for {env_name}")
env = Biped()

x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]

def progress(num_steps, metrics):
  clear_output(wait=True)

  times.append(datetime.now())
  x_data.append(num_steps)
  y_data.append(metrics["eval/episode_reward"])
  y_dataerr.append(metrics["eval/episode_reward_std"])

  plt.xlim([0, ppo_params["num_timesteps"] * 1.25])
  plt.xlabel("# environment steps")
  plt.ylabel("reward per episode")
  plt.title(f"y={y_data[-1]:.3f}")
  plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")
  plt.savefig(f'{FOLDER_PLOTS}/reward.png')
  # display(plt.gcf())
  
ppo_training_params = dict(ppo_params)
network_factory = ppo_networks.make_ppo_networks
if "network_factory" in ppo_params:
  del ppo_training_params["network_factory"]
  network_factory = functools.partial(
      ppo_networks.make_ppo_networks,
      **ppo_params.network_factory
  )

train_fn = functools.partial(
    ppo.train, **dict(ppo_training_params),
    network_factory=network_factory,
    progress_fn=progress,
    save_checkpoint_path=ABS_FOLDER_RESUlTS,
)

make_inference_fn, params, metrics = train_fn(
    environment=env,
    progress_fn=progress,
    eval_env=env,
    wrap_env_fn=wrap_for_brax_training,
)
print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")
