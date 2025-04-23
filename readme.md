## Install requirements

```
pip3 install -r requirements.txt
```

Code running with Python3.12.

## Install Brax
(I use the master branch in Brax)

First, clone the Brax repo, then install it:
```
cd brax
pip3 install -e .
```

## Install the repo

```
pip3 install -e .
```

## Run biped training code in Jax

```
cd src/jax
python3 train.py
```
Inference: test.ipynb



## File structure

```
robot_learning
    └── src
            └── jax
                └── biped.py                   (Biped in Jax)
                └── train.py                   (Train PPO on Biped)
                └── test.ipynb                 (Jupyter notebook for testing)
                └── mjx_env.py                 (file taken from mujoco-playground and modified)
                └── wrapper.py                 (file taken from mujoco-playground and modified)
                └── randomize.py               (domain randomization)
                └── utils.py                   (utils)

            └── pytorch
                └── biped_np.py                (Biped in PyTorch)
                └── train.py                   (Train PPO on Biped)
                └── test.ipynb                 (Jupyter notebook for testing)
                └── utils_np.py                (utils)

            └── assets
                └── biped                      (biped)

    └── tests
```