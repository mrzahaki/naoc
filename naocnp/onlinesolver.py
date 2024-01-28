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
