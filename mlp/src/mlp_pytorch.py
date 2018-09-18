"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
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
        """
        super(MLP, self).__init__()

        if len(n_hidden) == 0:
            self.input_layer = nn.Linear(n_inputs, n_classes)
        else:
            self.input_layer = nn.Linear(n_inputs, n_hidden[0])

            modules = []
            for i in range(1, len(n_hidden)):
                modules.append(nn.Linear(n_hidden[i - 1], n_hidden[i]))

            self.hidden_layers = nn.ModuleList(modules)
            self.output_layer  = nn.Linear(n_hidden[-1], n_classes)

        self.n_hidden  = n_hidden
        self.n_inputs  = n_inputs
        self.n_classes = n_classes
        self.relu      = nn.ReLU()
        self.softmax   = nn.Softmax()

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

        out = self.relu(self.input_layer(x))

        for layer in self.hidden_layers:
            out = self.relu(layer(out))

        out = self.output_layer(out)

        return out
