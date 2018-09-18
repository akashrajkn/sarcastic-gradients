"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample

        """

        self.params = {'weight': np.random.normal(0, 0.0001, [in_features, out_features]),
                       'bias'  : np.zeros(out_features)}
        self.grads  = {'weight': np.zeros([in_features, out_features]),
                       'bias'  : np.zeros(out_features)}

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
        """
        self.x = x
        out    = np.einsum("io,bi->bo", self.params['weight'], x) + self.params['bias']

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        self.grads['weight'] = self.x.T @ dout
        dx                   = dout @ self.params['weight'].T

        return dx


class ReLUModule(object):
    """
    ReLU activation module.
    """

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
        """

        self.out = np.maximum(np.zeros(x.shape), x)

        return self.out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        dx = dout * (self.out > 0)

        return dx


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def softmax_batch(self, x):

        y       = np.exp(x.T - x.max(axis=1)).T
        ret_val = (y.T / y.sum(axis=1)).T
        return ret_val

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
        """

        self.sigma = self.softmax_batch(x)
        out = self.sigma

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        b, siz = self.sigma.shape
        M      = np.dstack([self.sigma] * siz)
        I      = np.moveaxis(np.dstack([np.eye(siz)] * b), -1, 0)
        dx     = np.einsum("bi,bij->bj", dout, M * (I - np.swapaxes(M, 1, 2)))

        return dx


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.

        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """

        l_l = -np.log(x[range(y.shape[0]), y.argmax(axis=1)])
        out = np.sum(l_l) / y.shape[0]

        return out

    def backward(self, x, y):
        """
        Backward pass.

        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """

        dx = - (y / x) / x.shape[0]

        return dx
