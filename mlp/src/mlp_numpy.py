"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP

        TODO:
        Implement initialization of the network.
        """
        if len(n_hidden) == 0:
            self.input_layer = LinearModule(n_inputs, n_classes)
        else:
            self.input_layer = LinearModule(n_inputs, n_hidden[0])

            modules = []
            for i in range(1, len(n_hidden)):
                modules.append(LinearModule(n_hidden[i - 1], n_hidden[i]))

            self.hidden_layers = modules
            self.output_layer  = LinearModule(n_hidden[-1], n_classes)

        self.n_hidden  = n_hidden
        self.n_inputs  = n_inputs
        self.n_classes = n_classes
        self.relu      = ReLUModule()
        self.softmax   = SoftMaxModule()

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """

        if len(self.n_hidden) == 0:
            pass

        out = self.input_layer.forward(x)
        out = self.relu.forward(out)

        for layer in self.hidden_layers:
            out = layer.forward(out)
            out = self.relu.forward(out)

        out = self.output_layer.forward(out)
        out = self.softmax.forward(out)

        return out

    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss

        TODO:
        Implement backward pass of the network.
        """

        dout = self.softmax.backward(dout)
        dout = self.output_layer.backward(dout)
        for layer in reversed(self.hidden_layers[:-1]):
            dout = self.relu.backward(dout)
            dout = layer.backward(dout)

        dout = self.relu.backward(dout)
        dout = self.input_layer.backward(dout)

        return

    def step(self, learning_rate):

        self.input_layer.params['weight'] -= learning_rate * self.input_layer.grads['weight']
        self.input_layer.params['bias']   -= learning_rate * self.input_layer.grads['bias']

        for layer in self.hidden_layers:
            layer.params['weight'] -= learning_rate * layer.grads['weight']
            layer.params['bias']   -= learning_rate * layer.grads['bias']

        self.output_layer.params['weight'] -= learning_rate * self.output_layer.grads['weight']
        self.output_layer.params['bias']   -= learning_rate * self.output_layer.grads['bias']

        return
