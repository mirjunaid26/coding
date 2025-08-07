# Tensors are a special type of data structure very similar to arrays and matrices.
# PyTorch uses tensors to encode the inputs and outputs of a model, as well as the network parameters.
# Tensors are similar to numpy arrays, except that tensors can run on GPUSs or other hardware accelerators.

import torch
import numpy as np

# Initialize a tensor directly from data
data = [1, 2, 3, 4]
x_data = torch.tensor(data)
print(data)
print(x_data)
print("----------")

# From a numpy array
np_array = np.array([[1, 2], [3, 4]])
x_np = torch.from_numpy(np_array)
print(np_array)
print(x_np)
print("----------")

new_array = np.array([1, 2, 3], ndmin=2)
print(new_array)
print("----------")

# From another tensor
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random tensor: \n {x_rand} \n")

