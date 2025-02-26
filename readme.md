## Install requirements

```
pip3 install -r requirements.txt
```

Code running with python 3.12.

## File structure

```
src/
└── short_range/
tests/
└── cart_pole.py                (Cart Pole for MuJoCo (both JAX and NP))
└── train_brax.py               (Train PPO on Brax) - envs in parallel
└── train_pytorch.py            (Train PPO on PyTorch) - envs in parallel
└── ppo.py                      (PPO implementation)
└── test_cart_pole_pytorch.py   (Test PPO on Cart Pole for PyTorch)
```