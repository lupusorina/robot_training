from setuptools import setup, find_packages
import os

setup(
    name='robot_learning',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'torch>=2.0.0',
        'gymnasium>=0.28.1',
        'jax>=0.4.0',
        'jaxlib>=0.4.0',
        'mujoco>=2.3.0',
        'mediapy>=1.1.0',
        'brax>=0.9.0',
        'matplotlib>=3.5.0',
        'pandas>=1.3.0',
        'tqdm>=4.65.0',
    ],
    python_requires='>=3.12',
    author='Sorina Lupu',
    author_email='lupusorina@yahoo.com',
    description='Robot training package with PyTorch and JAX support',
    long_description=open('readme.md').read() if os.path.exists('readme.md') else '',
    long_description_content_type='text/markdown',
)
