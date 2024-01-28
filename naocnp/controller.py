"""
Filename: controller.py
Author: Hossein ZahakiMansoor
Description: 
    This module implements a Controller class in TensorFlow, designed for generating control inputs
    based on the error variable in a dynamical system. The controller incorporates an Actor Neural Network (ActorNN)
    and parameters such as eta, eta_bar, r, and kb for defining the control objective and constraints. The predict
    method is provided to generate control inputs based on the error variable.

Dependencies:
    - TensorFlow (tf)
    - naocnp.actornn.ActorNN

Class Overview:
    - Controller: Implements a controller with an Actor Neural Network for generating control inputs.
      Methods include predict.

Usage Example:
    controller = Controller(actornn, eta, eta_bar, r, kb, ...)
    alpha = controller(s)

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
# from naocnp.actornn import ActorNN

# -> actors (n, ), actornn vector
# -> eta nx1, The parameter $ \eta_1 $ determines the weight of the barrier function in the cost function
# -> eta_bar nx1, The parameter $ \bar{\eta}_1 $ is used to penalize the square of the state variable $ s_1 $
#    in the cost function, which reflects the control objective of minimizing the tracking error.
# -> r nx1, the control effort
# -> kb nx1, $\mathcal{A}_{s_1} = \{ s_1 : |s_1| < k_{b1} \}$ is a compact set containing origin.
class Controller:
    def __init__(
            self,
            actornn,
            eta: tuple|list|tf.Tensor,
            eta_bar: tuple|list|tf.Tensor,
            r: tuple|list|tf.Tensor,
            kb: tuple|list|tf.Tensor,
            dtype: tf.DType=tf.float32,
            ) -> None:
        """
        Initializes the Controller module.

        Args:
        - actornn: Instance of ActorNN.
        - eta (tuple|list|tf.Tensor): Eta parameter for the barrier function.
        - eta_bar (tuple|list|tf.Tensor): Eta bar parameter for penalizing state variable square.
        - r (tuple|list|tf.Tensor): Control effort.
        - kb (tuple|list|tf.Tensor): Compact set parameter.
        - dtype (tf.DType): Data type for tensors. Default is tf.float32.
        """
        basis_shape = (max(len(eta), len(eta_bar), len(r), len(kb)), 1)

        eta = tf.constant(eta, shape=basis_shape, dtype=dtype)
        eta_bar = tf.constant(eta_bar, shape=basis_shape, dtype=dtype)
        r = tf.constant(r, shape=basis_shape, dtype=dtype)
        kb = tf.constant(kb, shape=basis_shape, dtype=dtype)
        
        self.actornn = actornn
        self.eta = eta
        self.eta_bar = eta_bar
        self.r = r
        self.kb = kb
        self.random_controller = tf.random.uniform(shape=basis_shape, dtype=dtype)


    # s nx1, error variable(surface)
    # retval tensor nx1
    def predict(self, s:tf.Tensor)->tf.Tensor:
        """
        Predicts the control input based on the error variable (surface).

        Args:
        - s (tf.Tensor): Error variable (surface).

        Returns:
        - tf.Tensor: Predicted control input.
        """
        if s is None:
            return self.random_controller
        
        alpha = -(self.eta * s) / (self.r * (self.kb**2 - s**2))
        alpha = alpha - (self.eta_bar * s) / self.r
        alpha = alpha - self.actornn(s) / (2 * self.r)
        return alpha


    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """
        Callable method for the Controller module.

        Returns:
        - Any: Result of the predict method.
        """
        return self.predict(*args, **kwds) 