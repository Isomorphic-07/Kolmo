import tensorflow as tf
class MyDenseLayer (tf.keras.layers.Layer): #note, our custom dense layer is a subclass of tf.kera.layers.Layer
    def __init__(self, input_dim, output_dim):
        super(MyDenseLayer, self).__init__() #initialises the Keras machinery
        
        # initialize weights and bias
        self.W = self.add_weight([input_dim, output_dim]) #self.add_weight registors this tensor as a trainable parameter
        #updated via backpropagation, 
        self.b = self.add_weight([1, output_dim])
        
    def call(self, inputs):
        
        #forward propagate the inputs
        z = tf.matmul(inputs, self.W) + self.b #gives us the linear layer
        
        #feed through a non-linear activation
        output = tf.math.sigmoid(z)
        
        return output #gives us the activation value \in [0, 1]
        