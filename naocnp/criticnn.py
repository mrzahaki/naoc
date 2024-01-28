"""
Filename: criticnn.py
Author: Hossein ZahakiMansoor
Description: 
    This module implements the CriticNN class in TensorFlow, serving as a critic for 
    reinforcement learning applications. The CriticNN utilizes Radial Basis Function Neural Networks (RBFNN)
    and an online solver for updating weights based on error variables, controller efforts, and observer terms.
    It supports compact set parameters, and importance weights for state errors and control efforts.
    The module also includes methods for prediction and weight updates.

Dependencies:
    - TensorFlow (tf)
    - naocnp.rbfnn.RBFNN
    - naocnp.onlinesolver.OnlineSolver
    - naocnp.observer.Observer

Class Overview:
    - CriticNN: Implements the critic for reinforcement learning with RBFNN.
      Methods include predict, update, and phi_normalizer.

Usage Example:
    critic = CriticNN(q, r, kb, learning_rate, observer, ...)
    critic.update(s, controller_vector, y, y_hat)
    critic_estimation = critic(s)

License:
    Copyright 2024 Hossein ZahakiMansoor

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


from typing import Any
import tensorflow as tf
from naocnp.rbfnn import RBFNN
from naocnp.onlinesolver import OnlineSolver
# from naocnp.actornn import ActorNN
# from naocnp.controller import Controller
from naocnp.observer import Observer
from naocnp.rbfnn import RBFNN
import numpy as np

# q tensor  nx1, importance of the state error
# r tensor nx1, importance of the control effort
# kb tensor nx1,  $\mathcal{A}_{s_1} = \{ s_1 : |s_1| < k_{b1} \}$ is a compact set containing origin.
# learning_rate tensor nx1, eta_
class CriticNN(tf.Module):

    def __init__(
            self,
            q: tuple|list|tf.Tensor,
            r: tuple|list|tf.Tensor,
            kb: tuple|list|tf.Tensor,
            learning_rate: tuple|list|tf.Tensor,
            observer: Observer,
            rbfnn_num_centers_vector: tuple|list|tf.Tensor,
            eta: tuple|list|tf.Tensor,
            eta_bar: tuple|list|tf.Tensor,
            actornn=None,
            rbfnn_input_intervals_vector: tuple|list|tf.Tensor=None,
            rbfnn_sigma_vector:float|tuple|list|tf.Tensor=1.0,
            dtype:tf.DType = tf.float32,
            solver_step:float=1e-3,
            ):
        """
        Initializes the CriticNN module.

        Args:
        - q (tuple|list|tf.Tensor): Importance of the state error.
        - r (tuple|list|tf.Tensor): Importance of the control effort.
        - kb (tuple|list|tf.Tensor): Set containing the origin.
        - learning_rate (tuple|list|tf.Tensor): Learning rates for weight updates.
        - observer (Observer): Instance of the Observer module.
        - rbfnn_num_centers_vector (tuple|list|tf.Tensor): Number of centers for RBFNN.
        - eta (tuple|list|tf.Tensor): Eta values for the critic.
        - eta_bar (tuple|list|tf.Tensor): Eta bar values for the critic.
        - actornn: Instance of ActorNN (optional).
        - rbfnn_input_intervals_vector (tuple|list|tf.Tensor): Input intervals for RBFNN (optional).
        - rbfnn_sigma_vector (float|tuple|list|tf.Tensor): Sigma values for RBFNN. Default is 1.0.
        - dtype (tf.DType): Data type for tensors. Default is tf.float32.
        - solver_step (float): Solver step size. Default is 1e-3.
        """
        self.observer = observer
        self.rbfnn_vector = RBFNN.generator(
                input_dim_vector=1,
                output_dim_vector=1,
                num_centers_vector=rbfnn_num_centers_vector,
                input_intervals_vector=rbfnn_input_intervals_vector,
                sigma_vector=rbfnn_sigma_vector,
                )
        self.nstages = len(self.rbfnn_vector)
        basis_shape = self.nstages, 1
        self.q = tf.constant(q, dtype=dtype, shape=basis_shape)
        self.r = tf.constant(r, dtype=dtype, shape=basis_shape)
        self.kb = tf.constant(kb, dtype=dtype, shape=basis_shape)
        self.eta = tf.constant(eta, dtype=dtype, shape=basis_shape)
        self.eta_bar = tf.constant(eta_bar, dtype=dtype, shape=basis_shape)
        self.learning_rate = tf.constant(learning_rate, dtype=dtype, shape=basis_shape)
        self.actornn = actornn
        self.enum_rbfnn_vector = list(enumerate(self.rbfnn_vector))
        self.criticdata = tf.Variable(tf.zeros(basis_shape, dtype=dtype))

        self.solver_step = solver_step
        self.actor_solver = OnlineSolver(
            state_reducer=self.critic_state_reducer,
            solver_step=solver_step,
            )
        self.normalization_list = [None for i in range(self.nstages)]


    def set_actornn(self, actornn):
        """
        Sets the ActorNN instance for the CriticNN module.

        Args:
        - actornn: Instance of ActorNN.
        """
        self.actornn = actornn
    

    # s tensor nx1: error variable(surface)
    def predict(self, s: tf.Tensor):
        """
        Predicts the critic values for given error variables.

        Args:
        - s (tf.Tensor): Error variables.

        Returns:
        - tf.Tensor: Critic values.
        """
        criticdata = [rbfnn(si) for rbfnn, si in zip(self.rbfnn_vector,  tf.reshape(s, (-1)))]
        self.criticdata.assign(criticdata)
        return self.criticdata


    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """
        Callable method for the CriticNN module.

        Returns:
        - Any: Result of the predict method.
        """
        return self.predict(*args, **kwds)


    def critic_state_reducer(
            self, 
            t: float, 
            wc: tf.Tensor, 
            phic_norm: tf.Tensor, 
            actor: RBFNN, 
            observer_term: tf.Tensor, 
            controller: tf.Tensor,
            si:tf.Tensor,
            learning_rate:tf.Tensor,
            kbi:tf.Tensor,
            ri:tf.Tensor,
            etai:tf.Tensor,
            eta_bari:tf.Tensor,
            qi:tf.Tensor,
            ):
        """
        Computes the state derivative for the critic.

        Args:
        - t: Time parameter (not used in this implementation).
        - wc: Critic weights.
        - phic_norm: Normalized RBFNN output.
        - actor: RBFNN instance for the actor.
        - observer_term: Observer term for normalization.
        - controller: Control effort term for normalization.
        - si: Error variable (surface).
        - learning_rate: Learning rate for weight updates.
        - kbi: Compact set parameter.
        - ri: Importance of the control effort.
        - etai: Eta values for the critic.
        - eta_bari: Eta bar values for the critic.
        - qi: Importance of the state error.

        Returns:
        - tf.Tensor: Updated critic weights.
        """
        # print('wc: ', wc)
        # print('phic_norm: ', phic_norm)
        # print('actor: ', actor)
        # print('observer_term: ', observer_term)
        # print('controller: ', controller)
        # print('si: ', si)
        # print('learning_rate: ', learning_rate)
        # print('kbi: ', kbi)
        # print('ri: ', ri)
        # print('etai: ', etai)
        # print('eta_bari: ', eta_bari)
        # print('qi: ', qi)

        phic_norm_trs = tf.transpose(phic_norm)
        wc_dot = -learning_rate / (phic_norm_trs @ phic_norm + 1)
        wc_dot = wc_dot * phic_norm
        
        kbns = kbi**2 - si**2
        eta_s_kbns = etai * si / kbns + eta_bari * si

        cost = qi * np.log10(kbi**2 / kbns)
        cost = cost - eta_s_kbns**2 / ri
        cost = cost + actor(si)**2 / (4 * ri)
        cost = cost + (phic_norm_trs @ wc)[0]
        cost = cost + 2 * eta_s_kbns * (observer_term - controller)
        wc_dot = wc_dot * cost

        return wc_dot


    # s tensor nx1: error variable(surface)
    # controllers tensor nx1: control effort
    def update(self, s: tf.Tensor, controller_vector: tf.Tensor, y, y_hat):
         """
         Updates the critic weights based on error variables, controller efforts, and observer terms.

         Args:
         - s (tf.Tensor): Error variables (surface).
         - controller_vector (tf.Tensor): Control efforts.
         - y: Actual system output.
         - y_hat: Predicted system output (from observer).
         """
         # observer_vector nx1
         observer_term = self.observer.critic_normalizer(y, y_hat)
         for idx, critic in self.enum_rbfnn_vector:

            si = s[idx, 0]
            observer = observer_term[idx, 0]
            controller = controller_vector[idx, 0]

            phi_norm = CriticNN.phi_normalizer(
                critic.calculate_phi(si),
                controller,
                observer
                )
            
            controller = controller_vector[idx - 1, 0] if idx > 0 else 0
            weights = self.actor_solver.update(
                critic.weights, 
                phi_norm,
                self.actornn.rbfnn_vector[idx],
                observer,
                controller,
                si,
                learning_rate=self.learning_rate[idx, 0],
                kbi = self.kb[idx, 0],
                ri = self.r[idx, 0],
                etai = self.eta[idx, 0],
                eta_bari = self.eta_bar[idx, 0],
                qi = self.q[idx, 0],
                )
            critic.weights.assign(weights)

    # normalization list   
    @staticmethod
    def phi_normalizer(phi, controller, observer_term):
        """
        Computes the normalized phi value.

        Args:
        - phi: RBFNN output.
        - controller: Control effort.
        - observer_term: Observer term.

        Returns:
        - tf.Tensor: Normalized phi value.
        """
        return phi * (controller + observer_term)