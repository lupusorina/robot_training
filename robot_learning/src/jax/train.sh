#! /bin/bash


#MUJOCO_GL=egl python3 train.py
#echo "Script 1. MUJOCO_GL=egl python3 train.py Done."

MUJOCO_GL=egl python3 train.py --randomize_body_ipos --randomize_torso_mass --randomize_link_masses
