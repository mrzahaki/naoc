"""
Filename: observer.py
Author: Hossein ZahakiMansoor
Description: 
    This module implements an Observer class in TensorFlow, designed for state estimation
    in dynamical systems. The observer incorporates a Radial Basis Function Neural Network (RBFNN)
    and an online solver for updating the state estimation. It includes an update law for adapting
    the RBFNN weights. The module also provides methods for executing the observer, updating weights,
    and computing the error surface.

Dependencies:
    - TensorFlow (tf)
    - naocnp.onlinesolver.OnlineSolver
    - naocnp.rbfnn.RBFNN

Class Overview:
    - Observer: Implements the state observer with an RBFNN for state estimation.
      Methods include observer_state_reducer, exec, critic_normalizer, updatelaw_state_reducer,
      update, and error_surface.

Usage Example:
    observer = Observer(L, C, Bn, learning_rate, forgetting_rate, ...)
    x_hat, y_hat = observer(y, u)

License:
    Copyright 2024 Hossein Zahaki

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

import tensorflow as tf
# from naocnp.rbfnn import RBFNN
from naocnp.onlinesolver import OnlineSolver
from typing import Any
from naocnp.rbfnn import RBFNN

# A: nxn
# x_hat: nx1
# L: nx1
# y: scaler
# RBFNN: input 1xn (calculations nx1)
# Bn: nx1
# C: 1xn
class Observer(tf.Module):
    
    def __init__(
            self, 
            L,
            C,
            Bn,
            learning_rate,
            forgetting_rate,
            rbfnn_input_dim_vector:float|tuple|list|tf.Tensor,
            rbfnn_output_dim_vector:float|tuple|list|tf.Tensor,
            rbfnn_num_centers_vector:float|tuple|list|tf.Tensor,
            rbfnn_sigma_vector:float|tuple|list|tf.Tensor,
            rbfnn_input_intervals_vector: tuple|list|tf.Tensor,
            x_init: tuple|list|tf.Tensor=None,
            solver_step=1e-3,
            dtype = tf.float32,
            ):
        """
        Initializes the Observer module.

        Args:
        - L (list|tuple|tf.Tensor): System matrix.
        - C (list|tuple|tf.Tensor): Output matrix.
        - Bn (list|tuple|tf.Tensor): Control matrix.
        - learning_rate (float): Learning rate for weight updates.
        - forgetting_rate (float): Forgetting rate for update law.
        - rbfnn_input_dim_vector (float|tuple|list|tf.Tensor): Input dimensions for RBFNN.
        - rbfnn_output_dim_vector (float|tuple|list|tf.Tensor): Output dimensions for RBFNN.
        - rbfnn_num_centers_vector (float|tuple|list|tf.Tensor): Number of centers for RBFNN.
        - rbfnn_sigma_vector (float|tuple|list|tf.Tensor): Sigma values for RBFNN.
        - rbfnn_input_intervals_vector (tuple|list|tf.Tensor): Input intervals for RBFNN.
        - x_init (tuple|list|tf.Tensor): Initial state. Default is None.
        - solver_step (float): Solver step size. Default is 1e-3.
        - dtype (tf.Dtype): Data type for tensors. Default is tf.float32.
        """

        self.Bn = tf.constant(Bn, dtype=dtype, shape=(len(Bn), 1))
        self.L = tf.constant(L, dtype=dtype, shape=(len(L), 1))
        self.C = tf.constant(C, dtype=dtype, shape=(1, len(C)))

        rbfnn_vector = RBFNN.generator(
                            input_dim_vector=rbfnn_input_dim_vector,
                            output_dim_vector=rbfnn_output_dim_vector,
                            num_centers_vector=rbfnn_num_centers_vector,
                            sigma_vector=rbfnn_sigma_vector,
                            input_intervals_vector=rbfnn_input_intervals_vector,
                        )
        rbfnn_vector = enumerate(rbfnn_vector)
        self.rbfnn_vector = [(idx + 1, obj) for idx, obj in rbfnn_vector]


        L = tf.transpose(self.L)
        n = max(L.shape)

        self.x_hat = tf.Variable(tf.zeros((n, 1), dtype=dtype))
        if x_init :
            self.x_hat = tf.constant(x_init, shape=(n, 1), dtype=dtype)

        # Create matrix A
        A = tf.Variable(tf.zeros((n, n), dtype=dtype))
        A[:-1, 1:].assign(tf.eye(n-1, dtype=dtype))
        A[:, 0].assign(-L)
        self.A = A

        self.rbfnn_data = tf.Variable(tf.zeros((n, 1), dtype=dtype))

        self.solver_step = solver_step
        self.observer_solver = OnlineSolver(
            state_reducer=self.observer_state_reducer,
            solver_step=solver_step,
            )
        self.updatelaw_solver = OnlineSolver(
            state_reducer=self.updatelaw_state_reducer,
            solver_step=solver_step,
            )
        
        self.lrate = learning_rate
        self.frate = forgetting_rate


    def observer_state_reducer(self, t, x_hat, y, u):
        """
        Computes the state derivative for the observer.

        Args:
        - t: Time parameter (not used in this implementation).
        - x_hat: Current estimated state.
        - y: Observed output.
        - u: Control input.

        Returns:
        - x_hat_dot: State derivative for the observer.
        """
        x_hat_dot = tf.linalg.matmul(self.A, x_hat)
        x_hat_dot = x_hat_dot + self.L * y

        # rbfnn_data n x 1
        rbfnn_data = [rbfnn(x_hat[:idx]) for idx, rbfnn in self.rbfnn_vector]
        self.rbfnn_data.assign(rbfnn_data)
        x_hat_dot = x_hat_dot + self.rbfnn_data + self.Bn * u

        return x_hat_dot


    def exec(self, y, u):
        """
        Executes the observer for a given time step.

        Args:
        - y: Observed output.
        - u: Control input.

        Returns:
        - x_hat: Updated estimated state.
        - y_hat: Estimated output.
        """
        x_hat = self.observer_solver.update(self.x_hat, y, u)
        self.x_hat = x_hat
        y_hat = tf.linalg.matmul(self.C, x_hat)[0, 0]
        self.update(y, y_hat)
        return x_hat, y_hat
    

    def critic_normalizer(self, y, y_hat)->tf.Tensor:
        """
        Computes the critic normalizer for the error surface.

        Args:
        - y: Observed output.
        - y_hat: Estimated output.

        Returns:
        - tf.Tensor: Critic normalizer tensor. nx1
        """
        return self.rbfnn_data + self.L * (y - y_hat)


    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """
        Callable method for the Observer module.

        Returns:
        - Any: Result of the exec method.
        """
        return self.exec(*args, **kwds)
    

    def updatelaw_state_reducer(self, t, w_hat, e, lterm=0):
        """
        Computes the state derivative for the update law.

        Args:
        - t: Time parameter (not used in this implementation).
        - w_hat: Update law state.
        - e: Error (difference between observed and estimated output).
        - lterm: Additional term for the update law.

        Returns:
        - w_hat_dot: State derivative for the update law.
        """
        w_hat_dot = lterm - self.frate * tf.abs(e) * w_hat
        return w_hat_dot


    def update(self, y, y_hat):
        """
        Updates the RBFNN weights using the update law.

        Args:
        - y: Observed output.
        - y_hat: Estimated output.
        """
        length = len(self.rbfnn_vector) - 1
        e = y-y_hat
        lterm = 0
        for idx, rbfnn in self.rbfnn_vector:
            if idx == length:
                lterm = -self.lrate * (y - y_hat) * rbfnn.phi / self.L[-1]
            w_hat = self.updatelaw_solver.update(rbfnn.weights, e, lterm)
            rbfnn.weights.assign(w_hat)        


    def error_surface(self, alpha:tf.Tensor)->tf.Tensor:
        """
        Computes the error surface.

        Args:
        - alpha: Virtual-actual controller tensor. nx1

        Returns:
        - tf.Tensor: Error surface tensor. nx1
        """
        alphavar = tf.Variable(tf.roll(alpha, shift=1, axis=0))
        alphavar[0, :].assign(0)
        return self.x_hat - alphavar