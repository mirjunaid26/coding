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

# shape
shape = (2, 3)

rand_tensor = torch.rand(shape)
print(f"Random Tensor: \n {rand_tensor} \n")
print("----------")

ones_tensor = torch.ones(shape)
print(f"Ones Tensor: \n {ones_tensor} \n")
print("----------")

zeros_tensor = torch.zeros(shape)
print(f"Zeros Tensor: \n {zeros_tensor}")
print("----------")

flat_tensor = ones_tensor.view(-1) # flatten the tensor / only applicable to tensors
print(flat_tensor)
print("----------")

#flat_tensor2 = np_array.view(-1) # cannot flatten a numpy array
#print(flat_tensor2)

flat_tensor2 = np_array.reshape(-1) # flatten the numpy array
print(flat_tensor2)

# Attributes of a tensor
tensor = torch.rand(3, 4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Tensor is stored on: {tensor.device}")
print("----------")

# Operations on tensors
if torch.cuda.is_available():
    tensor = tensor.to(torch.cuda.current_device())
    print(f"Tensor is now stored on: {tensor.device}")
else:
    print("CUDA is not available, tensor remains on CPU.")
    
# Standard numpy-like indexing and slicing
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First Column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:, 1] = 0 # change the second column to zero
tensor[:, 2] = 0 # change the third column to zero
print("Modified tensor:", tensor)
#print(tensor)
print("----------")

# Joining tensors
t1 = torch.cat([tensor], dim=1)
print(t1)
print("----------")
t2 = t1.view(-1)
print(t2)
print("----------")
t3 = t2.view(4, 4)
print(t3)

t4 = torch.rand(8, 8)
print(t4)
t5 = t4.view(-1)
print(t5)

t6 = torch.cat([t2, t5], dim=0)
print(t6)
t6_shape = t6.shape
print("Shape of t6:", t6_shape)

