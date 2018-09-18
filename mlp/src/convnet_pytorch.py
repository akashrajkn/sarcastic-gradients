"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn


class ConvNet(nn.Module):
    """
    This class implements a Convolutional Neural Network in PyTorch.
    It handles the different vgg and parameters of the model.
    Once initialized an ConvNet object can perform forward.
    """

    def __init__(self, n_channels, n_classes):
        """
        Initializes ConvNet object.

        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem
        """
        super(ConvNet, self).__init__()

        # Define the layers for the VGG network
        # TODO: Add batchnorm layers.
        vgg = []

        # conv1
        vgg.append(nn.Conv2d(3, 64, 3, 1, 1))
        vgg.append(nn.ReLU())

        # maxpool1
        vgg.append(nn.MaxPool2d(3, 2, 1))

        # conv2
        vgg.append(nn.Conv2d(64, 128, 3, 1, 1))
        vgg.append(nn.ReLU())

        # maxpool2
        vgg.append(nn.MaxPool2d(3, 2, 1))

        # conv3_a
        vgg.append(nn.Conv2d(128, 256, 3, 1, 1))
        # conv3_b
        vgg.append(nn.Conv2d(256, 256, 3, 1, 1))
        vgg.append(nn.ReLU())

        # maxpool3
        vgg.append(nn.MaxPool2d(3, 2, 1))
        3

        # conv4_a
        vgg.append(nn.Conv2d(256, 512, 3, 1, 1))
        # conv4_b
        vgg.append(nn.Conv2d(512, 512, 3, 1, 1))
        vgg.append(nn.ReLU())

        # maxpool4
        vgg.append(nn.MaxPool2d(3, 2, 1))

        # conv5_a
        vgg.append(nn.Conv2d(512, 512, 3, 1, 1))
        # conv5_b
        vgg.append(nn.Conv2d(512, 512, 3, 1, 1))
        vgg.append(nn.ReLU())

        # maxpool5
        vgg.append(nn.MaxPool2d(3, 2, 1))

        # avgpool
        vgg.append(nn.AvgPool2d(1, 1, 0))

        self.sequential = nn.Sequential(*vgg)

        # linear
        self.linear = nn.Linear(512, n_classes)

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.
        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        """

        out = self.linear(self.sequential(x).reshape(x.shape[0], -1))

        return out
