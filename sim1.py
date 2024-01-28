from typing import Any
import tensorflow as tf
from naocnp.onlinesolver import OnlineSolver

class Sim1:

    def __init__(
            self,
            solver_step:float=1e-3,
            x_int: tuple|list|tf.Tensor=None,
            dtype:tf.DType = tf.float32,
            nstages:int = 2
            ) -> None:
        """
        Initializes the Sim1 class.

        Args:
        - solver_step (float): Solver step for the OnlineSolver. Default is 1e-3.
        - x_int (tuple | list | tf.Tensor): Initial state. Default is None.
        - dtype (tf.DType): Data type for tensors. Default is tf.float32.
        - nstages (int): Number of stages. Default is 2.
        """
        self.solver_step = solver_step
        self.sim_solver = OnlineSolver(
            state_reducer=self.sim_state_reducer,
            solver_step=solver_step,
            )
        self.x = tf.Variable(tf.zeros((nstages, 1), dtype=dtype))
        if x_int :
            self.x = tf.constant(x_int, shape=(nstages, 1), dtype=dtype)
        self.x_dot_data = tf.Variable(tf.zeros((nstages, 1), dtype=dtype))


    def sim_state_reducer(self, t, x, u):
        """
        State reducer function for the Sim1 class.

        Args:
        - t (float): Time.
        - x (tf.Tensor): State vector.
        - u (float): Control input.

        Returns:
        - tf.Tensor: State derivative vector.
        """
        x1, x2 = tf.reshape(x, (-1))
        x1_dot = -tf.sin(2 * x1)**2 + x2
        x2_dot = (-1 - tf.sin(x1) * tf.cos(x2))**2 + u
        
        self.x_dot_data.assign([[x1_dot], [x2_dot]])
        return self.x_dot_data


    def exec(self, u):
        """
        Executes the simulation for one time step.

        Args:
        - u (float): Control input.

        Returns:
        - Tuple[tf.Tensor, float]: State vector and output.
        """
        self.x = self.sim_solver.update(self.x, u)
        y = self.x[0, 0]
        return self.x, y
    

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """
        Callable method for the Sim1 class.

        Returns:
        - Any: Result of the exec method.
        """
        self.exec(*args, **kwds)