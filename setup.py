from setuptools import setup, find_packages
import os

setup(
    name='robot_learning',
    version='0.1',
    packages=find_packages(),
    python_requires='>=3.12',
    author='Sorina Lupu',
    author_email='lupusorina@yahoo.com',
    description='Robot training package with PyTorch and JAX support',
    long_description=open('readme.md').read() if os.path.exists('readme.md') else '',
    long_description_content_type='text/markdown',
)
