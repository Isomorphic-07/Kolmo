import torch
import torch.nn as nn

#Download and MIT introduction to DL package

import mitdeeplearning as mdl

import numpy as np
import matplotlib.pyplot as plt


#Tensors are data structures (think of them as multi-dimensionl arrays), represented 
# as n-dimensional arrays of base datatypes (provides a way to generalize vectors and matrices to
# higher dimensions) PyTorch provides the ability to perform computation on these tensors, define
# neural networks and train them efficiently

"""
The shape of a PyTorch Tensor defines its number of dimensions and size of each dimension
The ndim or dim of a Pytorch tensor provides number of dimensions (n-dimesnions)
"""
integer = torch.tensor(1234)
decimal = torch.tensor(3.14159265359)


print(f"'integer' is a {integer.ndim}-d Tensor: {integer}")
print(f"`decimal` is a {decimal.ndim}-d Tensor: {decimal}")