## Install requirements

```
pip3 install -r requirements.txt
```

Code running with python 3.12.

## Run biped training code

```
cd src/
python3 train.py
```

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