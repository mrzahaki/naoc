from naoc.rbfnn import RBFNN
from naoc.onlinesolver import OnlineSolver
import tensorflow as tf
import numpy as np


class Observer(RBFNN):
    def __init__(
            self, 
            rbf_units:int, 
            observer_gain:float,
            solver_step=1e-4,  
            learning_rate=0.01,
            **kwargs
            ):
        
        self._observer_gain = observer_gain

        super(self.__class__, self).__init__(
            rbf_units,
            learning_rate=learning_rate,
            **kwargs
            )
            
        self._main_solver = OnlineSolver(
            state_reducer=self.state_reducer,
            solver_step=solver_step
            )

    def call_rbfnn(self, state):
        tensor = tf.convert_to_tensor(state.reshape(-1, 1))
        state = self.call(tensor)
        return state.numpy().reshape(-1)

    def state_reducer(self, _, __, feedback, controller, error):
        state_dot =  self.call_rbfnn(feedback)
        state_dot += controller
        state_dot += self._observer_gain * error
        return state_dot

    def update_weights(self, output, target, forgetting_rate=1.0):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(target - output * forgetting_rate))
            gradients = tape.gradient(loss, self.trainable_variables)
            self.rbfnn_optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    # state is the vector of stage i states in strict-feedback system as xi
    def estimate(self, state, controller, output, target, forgetting_rate=1.0):
        output_error = target - output
        # print(state.reshape(-1, 1))
        self.update_weights(output, target, forgetting_rate=forgetting_rate)
        state[-1] = self._main_solver.update(state[-1], state, controller, output_error)
        return state

  