from brax.training.agents.ppo import checkpoint as ppo_checkpoint


import jax
from jax import numpy as jp

import numpy as np
import mediapy as media

import mujoco

from robot_learning.src.pytorch.envs.biped_np import Biped
from tqdm import tqdm
import tqdm

import os

path = '/home/sorinal/biped_mujoco/lightweight_biped_training/src/jax/results/20250422-093202/000035717120'
policy_fn = ppo_checkpoint.load_policy(path)
jit_policy = jax.jit(policy_fn)
rng = jax.random.PRNGKey(1)

env = Biped(visualize=False)
output = env.reset()

phase_dt = 2 * jp.pi * env.ctrl_dt * 1.5
phase = jp.array([0, jp.pi])

env.info["phase_dt"] = phase_dt
env.info["phase"] = phase

priv_obs = env.get_privileged_observation()
obs = env.get_observation()

obs_dict = {
    'privileged_observation': priv_obs,
    'observation': obs,
}

rollout = []
ctrl_list = []

info_phase_list = []
time_list = []
time_counter = 0

for _ in tqdm.tqdm(range(1000)):

    obs = env.get_observation()
    priv_obs = env.get_privileged_observation()
    obs_dict = {
        'privileged_state': np.zeros(priv_obs.shape),
        'state': obs,
    }
    act_rng, rng = jax.random.split(rng)

    action_ppo, _ = jit_policy(obs_dict, act_rng)
    action_ppo_np = np.array(action_ppo)
    ctrl_list.append(action_ppo_np)

    priviliged_obs, rewards, done, _, _ = env.step(action_ppo_np)
    state = {
        'qpos': env.data.qpos.copy(),
        'qvel': env.data.qvel.copy(),
        'xfrc_applied': env.data.xfrc_applied.copy()
    }
    rollout.append(state)

    # Keep track of time.
    time_counter += env.ctrl_dt

    # Save data.
    time_list.append(time_counter)
    info_phase_list.append(env.info["phase"])

FOLDER = 'evaluation_results'
if not os.path.exists(FOLDER):
    os.makedirs(FOLDER)

    
render_video = True
if render_video:
    render_every = 1 # int.
    fps = 1/ env.ctrl_dt / render_every

    traj = rollout[::render_every]

    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[2] = True
    scene_option.geomgroup[3] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False

    frames = env.render(
        traj,
        camera="track",
        scene_option=scene_option,
        width=640,
        height=480,
    )

    # media.show_video(frames, fps=fps, loop=False)
    # ABS_FOLDER_RESUlTS = epath.Path(RESULTS_FOLDER_PATH) / latest_folder
    # NOTE: To make the code run, you need to call: MUJOCO_GL=egl python3 test_ppo_from_brax.py
    media.write_video(f'{FOLDER}/joystick_testing.mp4', frames, fps=fps)
    print('Video saved')


import matplotlib.pyplot as plt
import pandas as pd
ctrl_df = pd.DataFrame(ctrl_list)
ctrl_df.to_csv(f'{FOLDER}/ctrl.csv')

col_names = ['L_YAW', 'L_HAA', 'L_HFE', 'L_KFE', 'L_ANKLE',
             'R_YAW', 'R_HAA', 'R_HFE', 'R_KFE', 'R_ANKLE']
df = pd.read_csv(f'{FOLDER}/ctrl.csv', names=col_names)

fig, ax = plt.subplots(5, 2, figsize=(15, 10), sharex=True, sharey=True)

# First half of the col names.
for i in range(len(df.columns) // 2):
    ax[i, 0].plot(df[df.columns[i]], linewidth=1.0, label=df.columns[i], color='#1f77b4')
    ax[i, 0].set_title(df.columns[i])
    ax[i, 0].legend(loc='upper right')

# Second half of the col names.
for i in range(len(df.columns) // 2):
    ax[i, 1].plot(df[df.columns[i + len(df.columns) // 2]], linewidth=1.0, label=df.columns[i + len(df.columns) // 2], color='#ff7f0e')
    ax[i, 1].set_title(df.columns[i + len(df.columns) // 2])
    ax[i, 1].legend(loc='upper right')

plt.tight_layout()
plt.legend(df.columns, loc='upper right')
plt.savefig(f'{FOLDER}/ctrl.png')

info_phase_list = np.array(info_phase_list)

print(info_phase_list.shape)
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
plt.plot(time_list, info_phase_list[:, 0])
plt.plot(time_list, info_phase_list[:, 1])
ax.set_xlabel('Time step')
ax.set_ylabel('Phase')
plt.savefig(f'{FOLDER}/phase.png')
