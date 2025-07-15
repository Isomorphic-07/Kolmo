class MyDenseLayer (nn.Module): #note, our custom dense layer is a subclass of torch.nn.Module which
    #is the base class for all Pytorch models and layers
    def __init__(self, input_dim, output_dim):
        super(MyDenseLayer, self).__init__() #initialises the Keras machinery
        
        # initialize weights and bias
        self.W = nn.Parameter(torch.randn(input_dim, output_dim, requires_grad = True)) #
        #updated via backpropagation
        self.b = nn.Parameter(torch.randn(1, output_dim, requires_grad = True))
        
    def forward(self, inputs):
        
        #forward propagate the inputs
        z = torch.matmul(inputs, self.W) + self.b #gives us the linear layer
        
        #feed through a non-linear activation
        output = torch.sigmoid(z)
        
        return output #gives us the activation value \in [0, 1]