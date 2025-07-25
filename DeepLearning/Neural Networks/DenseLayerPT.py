import torch
import torch.nn as nn
class MyDenseLayer (nn.Module): #note, our custom dense layer is a subclass of torch.nn.Module which
    #is the base class for all Pytorch models and layers
    def __init__(self, input_dim, output_dim):
        super(MyDenseLayer, self).__init__() #initialises the Keras machinery
        
        # initialize weights and bias
        self.W = nn.Parameter(torch.randn(input_dim, output_dim, requires_grad = True)) #samples the weights from N(0,1)
        #updated via backpropagation
        self.b = nn.Parameter(torch.randn(1, output_dim, requires_grad = True))
        #gradients comupted via automatic differentiation of the loss function
        
    def forward(self, inputs):
        
        #forward propagate the inputs
        z = torch.matmul(inputs, self.W) + self.b #gives us the linear layer
        
        #feed through a non-linear activation
        output = torch.sigmoid(z)
        
        return output #gives us the activation value \in [0, 1]
    
#PT implementation:
"""
m #the amount of inputs

layer = nn.Linear(in_features = m, out_features = 2)

#MultiOutput Perceptron (Deep Neural Network)
model = nn.Sequential( #creates container foor layers, executed in order
    nn.Linear(m, n1), #the hidden layer is size = n (input layer)
    nn.ReLU(), #ReLU activation element -wize, ReLU(z) = max(0, z), introduce non-linearity
    .
    .
    .
    nn.ReLU(),
    nn.Linear(nK, 2) 
)

"""