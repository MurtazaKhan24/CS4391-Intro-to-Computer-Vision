import numpy as np
from classifier import Classifier
from layers import fc_forward, fc_backward, relu_forward, relu_backward


class TwoLayerNet(Classifier):
    """
    A neural network with two layers, using a ReLU nonlinearity on its one
    hidden layer. That is, the architecture should be:

    input -> FC layer -> ReLU layer -> FC layer -> scores
    """
    def __init__(self, input_dim=3072, num_classes=10, hidden_dim=512,
                 weight_scale=1e-2):
        """
        Initialize a new two layer network.

        Inputs:
        - input_dim: The number of dimensions in the input.
        - num_classes: The number of classes over which to classify
        - hidden_dim: The size of the hidden layer
        - weight_scale: The weight matrices of the model will be initialized
          from a Gaussian distribution with standard deviation equal to
          weight_scale. The bias vectors of the model will always be
          initialized to zero.
        """
        self.W1 = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.b2 = np.zeros(num_classes)

    def parameters(self):
        params = {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
        }
        return params

    def forward(self, X):
        a1, cache_fc1 = fc_forward(X, self.W1, self.b1)   
        h1, cache_relu = relu_forward(a1)              
        scores, cache_fc2 = fc_forward(h1, self.W2, self.b2)  
        cache = (cache_fc1, cache_relu, cache_fc2)
        return scores, cache

    def backward(self, grad_scores, cache):
        cache_fc1, cache_relu, cache_fc2 = cache
        dh1, dW2, db2 = fc_backward(grad_scores, cache_fc2)
        da1 = relu_backward(dh1, cache_relu)
        dX, dW1, db1 = fc_backward(da1, cache_fc1)

        grads = {
            'W1': dW1,
            'b1': db1,
            'W2': dW2,
            'b2': db2,
        }
        return grads