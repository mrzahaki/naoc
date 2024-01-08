# NAOC: Observer-Based Neuro-Adaptive Optimized Control

This package implements the observer-based neuro-adaptive optimized control (NAOC) method proposed by Li et al. in their paper [Observer-Based Neuro-Adaptive Optimized Control of Strict-Feedback Nonlinear Systems With State Constraints](https://ieeexplore.ieee.org/document/9336267).

## Introduction

The NAOC method is a novel adaptive neural network output feedback optimized control strategy for a class of strict-feedback nonlinear systems subject to unknown internal dynamics, state constraints, and immeasurable states. The method uses radial basis function neural networks to approximate the unknown system functions, an adaptive state observer to estimate the immeasurable states, and a barrier type of optimal cost function and the actor–critic architecture to construct the optimal virtual and actual controllers.

The main features of the NAOC method are:

- It can handle the state constraints and achieve the optimal control objective simultaneously.
- It does not require the knowledge of the system dynamics or the state information for control design.
- It can guarantee the uniform ultimate boundedness of all the closed-loop signals and the nonviolation of the state constraints.


x Under Developement
