"""
Filename: onlinesolver.py
Author: Hossein ZahakiMansoor
Description: 
    This module provides the OnlineSolver class, a utility for numerical simulations
    with an online state update mechanism. It supports both internal state updates and 
    external state updates based on a specified solver step and state reducer function.

Dependencies:
    - numpy (np)

Class Overview:
    - OnlineSolver: Implements an online solver with methods for updating internal and external states.
      Methods include solver_update_int_state and solver_update_ext_state.

Usage Example:
    solver = OnlineSolver(state_reducer, solver_step, state_init)
    updated_state = solver.update(*args, **kwargs)

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

import numpy as np # for numerical operations

class OnlineSolver:
    def __init__(
            self, 
            state_reducer: callable, 
            solver_step: float=1e-4,
            state_init: np.ndarray=None, 
        ) -> None:
        """
        Initializes the OnlineSolver.

        Args:
        - state_reducer (callable): Function to reduce state at each step.
        - solver_step (float): Step size for the solver. Default is 1e-4.
        - state_init (np.ndarray): Initial state. Default is None.
        """
        self._solver_cs = state_init  # current state
        self._solver_ct = 0.0 # current time
        self._solver_ss = solver_step # solver step
        self._solver_sr = state_reducer # state reducer
        self.update = self.solver_update_int_state # update function
        if state_init is None:
            self.update = self.solver_update_ext_state # update function


    def solver_update_int_state(self, *args, **kwargs)->np.ndarray:
        """
        Updates the internal state based on the solver step and state reducer.

        Args:
        - args: Additional positional arguments.
        - kwargs: Additional keyword arguments.

        Returns:
        - np.ndarray: Updated internal state.
        """
        self._solver_cs = self._solver_cs + self._solver_sr(self._solver_ct, self._solver_cs, *args, **kwargs) * self._solver_ss
        self._solver_ct = self._solver_ct + self._solver_ss
        return self._solver_cs
    
    
    def solver_update_ext_state(self, state: np.ndarray, *args, **kwargs)->np.ndarray:
        """
        Updates the external state based on the solver step and state reducer.

        Args:
        - state (np.ndarray): External state to be updated.
        - args: Additional positional arguments.
        - kwargs: Additional keyword arguments.

        Returns:
        - np.ndarray: Updated external state.
        """
        state =  state + self._solver_sr(self._solver_ct, state, *args, **kwargs) * self._solver_ss
        self._solver_ct = self._solver_ct + self._solver_ss
        return state
