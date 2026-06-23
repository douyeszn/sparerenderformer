#!/usr/bin/env python
"""Setup script for GP-Sparse Attention / RenderFormer integration."""

from setuptools import setup, find_packages

setup(
    name='renderformer-sparse',
    version='0.1.0',
    description='GP-Sparse Attention for RenderFormer neural rendering',
    author='Victor Miene',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
        'einops>=0.7.0',
        'numpy<2.0',
    ],
    extras_require={
        'train': [
            'objaverse',
            'trimesh',
            'h5py',
            'tqdm',
        ],
        'dev': [
            'pytest',
        ],
    },
)
