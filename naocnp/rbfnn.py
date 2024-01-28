"""
Filename: rbfnn.py
Author: Hossein ZahakiMansoor
Description:
    This module implements a Radial Basis Function Neural Network (RBFNN) in TensorFlow. 
    The RBFNN is designed for regression tasks with configurable input dimensions, output dimensions, 
    number of centers, input intervals, and sigma values. It provides methods for initialization, 
    calculating RBF activations, predicting outputs, and generating multiple instances based on vectors.

Class Overview:
    - RBFNN: Implements the Radial Basis Function Neural Network with methods for initialization, prediction,
      and generating multiple instances based on input vectors. Includes a generator method for creating
      multiple RBFNN instances.

Dependencies:
    - TensorFlow (tf)
    - numpy (np)

Usage Example:
    rbfnn = RBFNN(input_dim, output_dim, num_centers, input_intervals, sigma)
    prediction = rbfnn(input_data)

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
import numpy as np

class RBFNN(tf.Module):
    
    def __init__(self, input_dim, output_dim, num_centers, input_intervals=None, sigma=1.0):
        """
        Initializes the Radial Basis Function Neural Network (RBFNN) module.

        Args:
        - input_dim (int): Number of input dimensions.
        - output_dim (int): Number of output dimensions.
        - num_centers (int): Number of RBF centers.
        - input_intervals (list of tuples): List of input value intervals. Default is None.
        - sigma (float): Width parameter for RBF functions. Default is 1.0.
        """
        self.output_dim = output_dim
        self.num_centers = num_centers
        if input_intervals is None:
            input_intervals = [(0, 1)] * input_dim
        self.centers = self.initialize_centers(input_dim, num_centers, input_intervals)
        self.weights = tf.Variable(tf.random.normal([num_centers, output_dim]))
        self.sigma = sigma
        self.phi = None


    def initialize_centers(self, input_dim, num_centers, input_intervals):
        """
        Initializes the RBF centers based on input intervals.

        Args:
        - input_dim (int): Number of input dimensions.
        - num_centers (int): Number of RBF centers.
        - input_intervals (list of tuples): List of input value intervals.

        Returns:
        - tf.Tensor: Tensor containing the initialized RBF centers.
        """
        centers = []
        for interval in input_intervals:
            center_values = tf.random.uniform((num_centers,), interval[0], interval[1])
            centers.append(center_values)
        return tf.constant(np.asarray(centers), dtype=tf.float32)


    def calculate_phi(self, x):
        """
        Calculates the RBF activation values (phi) for the given input.

        Args:
        - x (tf.Tensor): Input tensor of shape (input_dim, 1).

        Returns:
        - tf.Tensor: RBF activation values (phi) tensor of shape (num_centers, output_dim).
        """
        distances = tf.norm(x - self.centers, axis=0)
        phi = tf.exp(-0.5 * (distances / self.sigma) ** 2)
        return tf.reshape(phi, (self.num_centers, self.output_dim))


    def predict(self, x):
        """
        Predicts the output for the given input using the RBFNN model.

        Args:
        - x (tf.Tensor): Input tensor of shape (input_dim, 1).

        Returns:
        - tf.Tensor: Predicted output tensor.
        """
        phi = self.calculate_phi(x)
        self.phi = phi
        retval = tf.linalg.matmul(tf.transpose(self.weights), phi)        
        return tf.reshape(retval, (-1,))


    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """
        Callable method for the RBFNN module. Calls the predict method.

        Returns:
        - tf.Tensor: Predicted output tensor.
        """
        return self.predict(*args, **kwds)
    

    @staticmethod
    def generator(
            input_dim_vector: list|tuple|tf.Tensor, 
            output_dim_vector: list|tuple|tf.Tensor, 
            num_centers_vector: list|tuple|tf.Tensor, 
            input_intervals_vector: list|tuple|tf.Tensor, 
            sigma_vector: list|tuple|tf.Tensor,
            )->tuple:
        """
        Generates a tuple of RBFNN instances based on the provided vectors.

        Args:
        - input_dim_vector (list|tuple|tf.Tensor): List of input dimensions.
        - output_dim_vector (list|tuple|tf.Tensor): List of output dimensions.
        - num_centers_vector (list|tuple|tf.Tensor): List of numbers of RBF centers.
        - input_intervals_vector (list|tuple|tf.Tensor): List of input intervals. Can be None.
        - sigma_vector (list|tuple|tf.Tensor): List of sigma values.

        Returns:
        - tuple: Tuple of RBFNN instances.
        """
        rbfnns = []
        nstages = 1e4
        try:
            nstages = min(nstages, len(input_dim_vector))
        except:
            pass
        try:
            nstages = min(nstages, len(output_dim_vector))
        except:
            pass
        try:
            nstages = min(nstages, len(num_centers_vector))
        except:
            pass
        try:
            nstages = min(nstages, len(input_intervals_vector))
        except:
            pass
        try:
            nstages = min(nstages, len(sigma_vector))
        except:
            pass

        inprange = tuple(range(nstages))
        ttype = type(input_dim_vector)
        if ttype is int or ttype is float:
            input_dim_vector = [input_dim_vector for _ in inprange]
        ttype = type(output_dim_vector)
        if ttype is int or ttype is float:
            output_dim_vector = [output_dim_vector for _ in inprange]
        ttype = type(num_centers_vector)
        if ttype is int or ttype is float:
            num_centers_vector = [num_centers_vector for _ in inprange]
        ttype = type(sigma_vector)
        if ttype is int or ttype is float:
            sigma_vector = [sigma_vector for _ in inprange]
        if input_intervals_vector is None:
            input_intervals_vector = [None for _ in inprange]

        for (input_dim, 
             output_dim, 
             num_centers, 
             input_intervals, 
             sigma) in zip(
            input_dim_vector,
            output_dim_vector,
            num_centers_vector,
            input_intervals_vector,
            sigma_vector,
        ):
            rbfnn = RBFNN(
                input_dim, 
                output_dim, 
                num_centers, 
                input_intervals, 
                sigma
            )
            rbfnns.append(rbfnn)
        return tuple(rbfnns)
