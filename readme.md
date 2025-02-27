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


## Run biped training code

```
cd src/
python3 train.py
```

Inference: test.ipynb

## File structure

```
src/
    └── biped.py                   (Biped in Jax)
    └── train.py                   (Train PPO on Biped)
    └── mjx_env.py                 (file taken from mujoco-playground)
    └── wrapper.py                 (file taken from mujoco-playground)
    └── assets
        └── berkeley_humanoid       (Berkeley Biped model)
        └── biped                   (biped)

tests/
    └── cart_pole.py                (Cart Pole for MuJoCo (both JAX and NP))
    └── train_brax.py               (Train PPO on Brax) - envs in parallel
    └── train_pytorch.py            (Train PPO on PyTorch) - envs in parallel
    └── ppo.py                      (PPO implementation)
    └── test_cart_pole_pytorch.py   (Test PPO on Cart Pole for PyTorch)
```