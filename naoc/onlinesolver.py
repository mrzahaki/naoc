import numpy as np # for numerical operations

class OnlineSolver:
    def __init__(
            self, 
            state_reducer:callable, 
            solver_step:float=1e-4,
            state_init:np.ndarray=None, 
        ) -> None:

        self._solver_cs = state_init  # current state
        self._solver_ct = 0.0 # current time
        self._solver_ss = solver_step # solver step
        self._solver_sr = state_reducer # state reducer
        self.update = self.solver_update_int_state # update function
        if state_init is None:
            self.update = self.solver_update_ext_state # update function

    def solver_update_int_state(self, *args, **kwargs)->np.ndarray:
        self._solver_cs += self._solver_sr(self._solver_ct, self._solver_cs, *args, **kwargs) * self._solver_ss
        self._solver_ct += self._solver_ss
        return self._solver_cs
    
    def solver_update_ext_state(self, state:np.ndarray, *args, **kwargs)->np.ndarray:
        state += self._solver_sr(self._solver_ct, state, *args, **kwargs) * self._solver_ss
        self._solver_ct += self._solver_ss
        return state
    