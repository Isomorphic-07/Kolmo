import torch
import torch.nn as nn

#Download and MIT introduction to DL package

#import mitdeeplearning as mdl

import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

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

#Vectors and lists are used to create 1-d tensors
fibonacci = torch.tensor([1, 1, 2, 3, 5, 8])
count_to_100 = torch.tensor(range(100))

print(f"`fibonacci` is a {fibonacci.ndim}-d Tensor with shape: {fibonacci.shape}")
print(f"`count_to_100` is a {count_to_100.ndim}-d Tensor with shape: {count_to_100.shape}")

#Next, let’s create 2-d (i.e., matrices) and higher-rank tensors. In image processing and 
# computer vision, we will use 4-d Tensors with dimensions corresponding to batch size, 
# number of color channels, image height, and image width.

### Defining higher-order Tensors ###

'''TODO: Define a 2-d Tensor'''
matrix = torch.tensor([[1, 2, 3],[4, 5, 6]])

assert isinstance(matrix, torch.Tensor), "matrix must be a torch Tensor object"
assert matrix.ndim == 2

'''TODO: Define a 4-d Tensor.'''
# Use torch.zeros to initialize a 4-d Tensor of zeros with size 10 x 3 x 256 x 256.
#   You can think of this as 10 images where each image is RGB 256 x 256.
images = torch.zeros(10, 3, 256, 256)

assert isinstance(images, torch.Tensor), "images must be a torch Tensor object"
assert images.ndim == 4, "images must have 4 dimensions"
assert images.shape == (10, 3, 256, 256), "images is incorrect shape"
print(f"images is a {images.ndim}-d Tensor with shape: {images.shape}")

#we can use slicing to access subtensors within a higher rank tensor
row_vector = matrix[1]
column_vector = matrix[:, 1]
scalar = matrix[0, 1]


print(f"`row_vector`: {row_vector}")
print(f"`column_vector`: {column_vector}")
print(f"`scalar`: {scalar}")

#Addition arithmetic
a = torch.tensor(15)
b = torch.tensor(61)

c1 = torch.add(a, b)
c2 = a + b #PyTorch overrides the "+" operation so that it is able to act on Tensors
print(f"c1: {c1}")
print(f"c2: {c2}")

### Defining a dense layer ###

# num_inputs: number of input nodes
# num_outputs: number of output nodes
# x: input to the layer
# torch.nn.Module serves as a base class for all neural network modules

class OurDenseLayer(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(OurDenseLayer, self).__init__()
        # Define and initialize parameters: a weight matrix W and bias b
        # Note that the parameter initialize is random!
        self.W = torch.nn.Parameter(torch.randn(num_inputs, num_outputs))
        self.bias = torch.nn.Parameter(torch.randn(num_outputs))
        #Weight and bias parameters are generated randomly ~N(0,1)

    def forward(self, x):
        '''TODO: define the operation for z (hint: use torch.matmul).'''
        z = torch.matmul(x, self.W) + self.bias

        '''TODO: define the operation for out (hint: use torch.sigmoid).'''
        y = torch.sigmoid(z)
        return y
    
# Define a layer and test the output!
num_inputs = 2
num_outputs = 3
layer = OurDenseLayer(num_inputs, num_outputs)
x_input = torch.tensor([[1, 2.]])
y = layer(x_input) #this is simply pytorch syntax

print(f"input shape: {x_input.shape}")
print(f"output shape: {y.shape}")
print(f"output result: {y}")

### Defining a neural network using the PyTorch Sequential API ###

# define the number of inputs and outputs
n_input_nodes = 2
n_output_nodes = 3
# Define the model
'''TODO: Use the Sequential API to define a neural network with a
    single linear (dense!) layer, followed by non-linearity to compute z'''
model = nn.Sequential(
    nn.Linear(in_features= n_input_nodes, out_features = n_output_nodes), #perform affine transformation z = xW + b
    nn.Sigmoid()
)

# Test the model with example input
x_input = torch.tensor([[1, 2.]])
model_output = model(x_input)
print(f"input shape: {x_input.shape}")
print(f"output shape: {y.shape}")
print(f"output result: {y}")

### Defining a model using subclassing ###
#allows for flexibility to define custom layers, custom training lops, custom activation functions.
# custom models

class LinearWithSigmoidActivation(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearWithSigmoidActivation, self).__init__()
        
        self.linear = nn.Linear(in_features= num_inputs, out_features= num_outputs)
        self.activation = nn.Sigmoid()
        
    def forward(self, inputs):
        linear_output = self.linear(inputs)
        output = self.activation(linear_output)
        return output

n_input_nodes = 2
n_output_nodes = 3
model = LinearWithSigmoidActivation(n_input_nodes, n_output_nodes)
x_input = torch.tensor([[1, 2.]])
y = model(x_input)
print(f"input shape: {x_input.shape}")
print(f"output shape: {y.shape}")
print(f"output result: {y}")

### Custom behavior with subclassing nn.Module ###

class LinearButSometimesIdentity(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearButSometimesIdentity, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    '''TODO: Implement the behavior where the network outputs the input, unchanged,
        under control of the isidentity argument.'''
    def forward(self, inputs, isidentity=False): #we must pass through ididentity = True to
        #return the output as input
      ''' TODO '''
      if isidentity:
          return inputs
      else:
          return self.linear(inputs)
      

# Test the IdentityModel
model = LinearButSometimesIdentity(num_inputs=2, num_outputs=3)
x_input = torch.tensor([[1, 2.]])

'''TODO: pass the input into the model and call with and without the input identity option.'''
out_with_linear = model(x_input)

out_with_identity = model(x_input, True)

print(f"input: {x_input}")
print("Network linear output: {}; network identity output: {}".format(out_with_linear, out_with_identity))

#Automatic Differentiation: requires_grad attribute controls whether autograd should record operations
# on that tensor, when a forward pass is made through the network, Pytorch build a computational graph dynamically

### Gradient computation ###

# y = x^2
# Example: x = 3.0
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
y.backward()  # Compute the gradient

dy_dx = x.grad
print("dy_dx of y=x^2 at x=3.0 is: ", dy_dx)
assert dy_dx == 6.0

#Stochastic Gradient Descent: We will implement a Loss of L = (x - x_f)^2 where x_f is 
# a desired value. we will use SGD for optimization to minimize this loss

### Function minimization with autograd and gradient descent ###
# Initialize a random value for our intial x
x = torch.randn(1)
print(f"Initializing x={x.item()}")

learning_rate = 1e-2  # Learning rate
history = []
x_f = 4  # Target value

# We will run gradient descent for a number of iterations. At each iteration, we compute the loss,
#   compute the derivative of the loss with respect to x, and perform the update.
for i in range(500):
    x = torch.tensor([x], requires_grad=True)

    # TODO: Compute the loss as the square of the difference between x and x_f
    loss = (x - x_f)**2

    # Backpropagate through the loss to compute gradients
    loss.backward()

    # Update x with gradient descent
    x = x.item() - learning_rate * x.grad

    history.append(x.item())

# Plot the evolution of x as we optimize toward x_f!
plt.plot(history)
plt.plot([0, 500], [x_f, x_f])
plt.legend(('Predicted', 'True'))
plt.xlabel('Iteration')
plt.ylabel('x value')
plt.show()