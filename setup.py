from setuptools import setup, find_packages
description = """
This package implements the observer-based neuro-adaptive optimized control (NAOC) method proposed by Li et al. in their paper of
Observer-Based Neuro-Adaptive Optimized Control of Strict-Feedback Nonlinear Systems With State Constraints
"""

setup(
    name='naocnp',
    version='0.0.1',
    description=description,
    author='Hossein ZahakiMansoor',
    author_email='mrzahaki@gmail.com',
    url='https://github.com/MrZahaki/naoc',
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'numpy'
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
    ],
    python_requires='>=3.6',
)
