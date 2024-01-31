# NAOC: Observer-Based Neuro-Adaptive Optimized Control

This package implements the observer-based neuro-adaptive optimized control (NAOC) method proposed by Li et al. in their paper [Observer-Based Neuro-Adaptive Optimized Control of Strict-Feedback Nonlinear Systems With State Constraints](https://ieeexplore.ieee.org/document/9336267).

## Introduction

The NAOC method is a novel adaptive neural network output feedback optimized control strategy for a class of strict-feedback nonlinear systems subject to unknown internal dynamics, state constraints, and immeasurable states. The method uses radial basis function neural networks to approximate the unknown system functions, an adaptive state observer to estimate the immeasurable states, and a barrier type of optimal cost function and the actor–critic architecture to construct the optimal virtual and actual controllers.

The main features of the NAOC method are:

- It can handle the state constraints and achieve the optimal control objective simultaneously.
- It does not require the knowledge of the system dynamics or the state information for control design.
- It can guarantee the uniform ultimate boundedness of all the closed-loop signals and the nonviolation of the state constraints.


#### Characteristics of the Program:

1. **Modular Structure:** A modular structure has been employed, with multiple classes organized to serve distinct purposes, such as `OnlineSolver`, `Observer`, `CriticNN`, `ActorNN`, `Controller`, and `Sim1`. This organization enhances code readability and maintainability.

2. **Object-Oriented Design:** Object-oriented programming (OOP) principles have been applied through the definition of classes and their methods. This approach facilitates the encapsulation of related functionalities within classes, contributing to improved code organization and reusability.

3. **TensorFlow Usage:** The code utilizes TensorFlow for the definition and manipulation of tensors, construction of neural networks (e.g., RBFNN), and implementation of online solvers.

4. **OnlineSolver:** The `OnlineSolver` class is designed as a generic class for updating states based on a state reducer function, demonstrating its utility for various iterative algorithms.

5. **Observer:** An observer for a control system is implemented in the `Observer` class. It encompasses an observer state reducer, an execution method, and other relevant functionalities.

6. **CriticNN and ActorNN:** These classes are components of an actor-critic architecture, commonly utilized in reinforcement learning. They encompass the definition of neural networks, learning rates, and other parameters.

7. **Controller:** The `Controller` class represents a controller with methods for predicting control actions based on state variables.

8. **Simx:** `Simx` class is formulated for simulating a system with a specific state reducer. It involves the updating of the state based on a control input and the subsequent return of the state and output.

9. **Configuration Parameters:** Parameters such as learning rates, forgetting rates, and other coefficients have been introduced, enabling flexibility and customization of the control system.

The code also includes a simulation scenario using the NAOC (Nonlinear Adaptive Observer and Controller) framework. It utilizes various classes and parameters to simulate a control system, providing visualizations of system states, observer estimates, and other relevant metrics over time. The simulation involves iterative steps, updating the control system based on observed states and feedback mechanisms.
