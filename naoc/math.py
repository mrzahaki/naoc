import numpy as np

# the Gaussian basis function
def rbf_wrapper(center, width):
  def rbf(x):
    return np.exp(-(x - center).T @ (x - center) / width**2)
  return rbf
  

def numerical_derivative(func, t, h=1e-6):
  # Central difference formula (4th order)
  return (func(t - 2*h) - 8*func(t - h) + 8*func(t + h) - func(t + 2*h)) / (12 * h)
