from typing import Any
import tensorflow as tf
from naocnp.rbfnn import RBFNN
from naocnp.onlinesolver import OnlineSolver
from naocnp.criticnn import CriticNN
from naocnp.observer import Observer


# q tensor  nx1, importance of the state error
# r tensor nx1, importance of the control effort
# kb tensor nx1,  $\mathcal{A}_{s_1} = \{ s_1 : |s_1| < k_{b1} \}$ is a compact set containing origin.
# learning_rate tensor nx1, eta_
class ActorNN(tf.Module):

    def __init__(
            self,
            q: tuple|list|tf.Tensor,
            r: tuple|list|tf.Tensor,
            kb: tuple|list|tf.Tensor,
            learning_rate: tuple|list|tf.Tensor,
            rbfnn_num_centers_vector: tuple|list|tf.Tensor,
            observer:Observer,
            rbfnn_input_intervals_vector: tuple|list|tf.Tensor=None,
            rbfnn_sigma_vector:float|tuple|list|tf.Tensor=1.0,
            criticnn:CriticNN=None,
            dtype:tf.DType = tf.float32,
            solver_step:float=1e-3,
            ):
        """
        Initializes the ActorNN module.

        Args:
        - q (tuple|list|tf.Tensor): Importance of the state error.
        - r (tuple|list|tf.Tensor): Importance of the control effort.
        - kb (tuple|list|tf.Tensor): Compact set parameter.
        - learning_rate (tuple|list|tf.Tensor): Eta parameter for learning rate.
        - rbfnn_num_centers_vector (tuple|list|tf.Tensor): Number of centers for RBFNN.
        - observer: Instance of Observer.
        - rbfnn_input_intervals_vector (tuple|list|tf.Tensor): Input intervals for RBFNN.
        - rbfnn_sigma_vector (float|tuple|list|tf.Tensor): Sigma parameter for RBFNN.
        - criticnn: Instance of CriticNN.
        - dtype (tf.DType): Data type for tensors. Default is tf.float32.
        - solver_step (float): Solver step for OnlineSolver. Default is 1e-3.
        """
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
        self.learning_rate = tf.constant(learning_rate, dtype=dtype, shape=basis_shape)
        self.criticnn = criticnn
        self.observer = observer
        self.actordata = tf.Variable(tf.zeros(basis_shape, dtype=dtype))
        self.enum_rbfnn_vector = list(enumerate(self.rbfnn_vector))

        self.solver_step = solver_step
        self.actor_solver = OnlineSolver(
            state_reducer=self.actor_state_reducer,
            solver_step=solver_step,
            )
        

    def set_criticnn(self, criticnn:CriticNN):
        """
        Sets the CriticNN instance for the ActorNN.

        Args:
        - criticnn: Instance of CriticNN.
        """
        self.criticnn = criticnn
    

    # s tensor nx1: error variable(surface)
    def predict(self, s: tf.Tensor):
        """
        Predicts the actor output based on the error variable (surface).

        Args:
        - s (tf.Tensor): Error variable (surface).

        Returns:
        - tf.Tensor: Predicted actor output.
        """
        actordata = [rbfnn(si) for rbfnn, si in zip(self.rbfnn_vector, tf.reshape(s, (-1)))]
        self.actordata.assign(actordata)
        return self.actordata


    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """
        Callable method for the ActorNN module.

        Returns:
        - Any: Result of the predict method.
        """
        return self.predict(*args, **kwds)


    def actor_state_reducer(
            self, 
            t: float, 
            wa: tf.Tensor, 
            phia: tf.Tensor, 
            wc: tf.Tensor, 
            phic: tf.Tensor, 
            si: tf.Tensor,
            lra: tf.Tensor,
            lrc: tf.Tensor,
            kbi: tf.Tensor,
            ri: tf.Tensor,
            qi: tf.Tensor,
            ):
        """
        State reducer function for the ActorNN module.

        Args:
        - t (float): Time.
        - wa (tf.Tensor): Actor weights.
        - phia (tf.Tensor): Actor phi.
        - wc (tf.Tensor): Critic weights.
        - phic (tf.Tensor): Critic phi.
        - si (tf.Tensor): Error variable.
        - lra (tf.Tensor): Learning rate for actor.
        - lrc (tf.Tensor): Learning rate for critic.
        - kbi (tf.Tensor): Compact set parameter.
        - ri (tf.Tensor): Control effort.
        - qi (tf.Tensor): Importance parameter.

        Returns:
        - tf.Tensor: Actor weights derivative.
        """
        # print('wa: ', wa)
        # print('phia: ', phia)
        # print('wc: ', wc)
        # print('phic: ', phic)
        # print('si: ', si)
        # print('lra: ', lra)
        # print('lrc: ', lrc)
        # print('kbi: ', kbi)
        # print('ri: ', ri)
        # print('qi: ', qi)
        
        phi2phitr2wa = phia @ tf.transpose(phia) @ wa
        kbns = kbi**2 - si**2
        wa_dot = qi / (2 * ri * kbns) * phia * si
        wa_dot = wa_dot - lra * phi2phitr2wa
        critic_term = lrc / (4 * ri * (tf.transpose(phic) @ phic + 1))
        critic_term = critic_term * phi2phitr2wa @ tf.transpose(phic) @ wc

        # print('wa_dot: ', wa_dot)
        # print('critic_term: ', critic_term)
        wa_dot = wa_dot + critic_term
        # print('wa_dot: ', wa_dot)
        return wa_dot


    # s tensor nx1: error variable(surface)
    # controllers tensor nx1: control effort
    def update(self, s: tf.Tensor, controller_vector: tf.Tensor, y, y_hat):
         """
        Updates the ActorNN weights.

        Args:
        - s (tf.Tensor): Error variable (surface).
        - controller_vector (tf.Tensor): Control effort.
        - y: True output.
        - y_hat: Predicted output.
        """
        # observer_vector nx1
         observer_term = self.observer.critic_normalizer(y, y_hat)
         critic = self.criticnn.rbfnn_vector
         for idx, actor in self.enum_rbfnn_vector:

            si = s[idx, 0]
            controller = controller_vector[idx, 0]

            critic_phi_norm = CriticNN.phi_normalizer(
                critic[idx].calculate_phi(si),
                controller,
                observer_term[idx, 0]
                )
            
            weights = self.actor_solver.update(
                actor.weights,
                actor.phi, 
                critic[idx].weights,
                critic_phi_norm,
                si,
                self.learning_rate[idx, 0],
                self.criticnn.learning_rate[idx, 0],
                kbi = self.kb[idx, 0],
                ri = self.r[idx, 0],
                qi = self.q[idx, 0],
                )
            actor.weights.assign(weights)          
         