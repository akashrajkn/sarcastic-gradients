import numpy as np
import torch
import torch.nn as nn

"""
The modules/function here implement custom versions of batch normalization in PyTorch.
In contrast to more advanced implementations no use of a running mean/variance is made.
You should fill in code into indicated sections.
"""


######################################################################################
# Code for Question 3.1
######################################################################################

class CustomBatchNormAutograd(nn.Module):
    """
    This nn.module implements a custom version of the batch norm operation for MLPs.
    The operations called in self.forward track the history if the input tensors have the
    flag requires_grad set to True. The backward pass does not need to be implemented, it
    is dealt with by the automatic differentiation provided by PyTorch.
    """

    def __init__(self, n_neurons, eps=1e-5):
        """
        Initializes CustomBatchNormAutograd object.

        Args:
          n_neurons: int specifying the number of neurons
          eps: small float to be added to the variance for stability
        """
        super(CustomBatchNormAutograd, self).__init__()

        self.n_neurons = n_neurons
        self.eps       = eps

        self.gamma     = nn.Parameter(torch.ones(n_neurons))
        self.beta      = nn.Parameter(torch.zeros(n_neurons))

    def forward(self, input):
        """
        Compute the batch normalization

        Args:
          input: input tensor of shape (n_batch, n_neurons)
        Returns:
          out: batch-normalized tensor
        """
        assert input.shape[1] == self.n_neurons, "Input has incorrect shape"

        mu  = torch.mean(input, 0)
        var = torch.var(input, dim=0, unbiased=False)
        x   = (input - mu) / ((var + self.eps) ** (0.5))
        out = x * self.gamma + self.beta

        return out


######################################################################################
# Code for Question 3.2 b)
######################################################################################


class CustomBatchNormManualFunction(torch.autograd.Function):
    """
    This torch.autograd.Function implements a functional custom version of the batch norm operation for MLPs.
    Using torch.autograd.Function allows you to write a custom backward function.
    The function will be called from the nn.Module CustomBatchNormManualModule
    Inside forward the tensors are (automatically) not recorded for automatic differentiation since the backward
    pass is done via the backward method.
    The forward pass is not called directly but via the apply() method. This makes sure that the context objects
    are dealt with correctly. Example:
      my_bn_fct = CustomBatchNormManualFunction()
      normalized = fct.apply(input, gamma, beta, eps)
    """

    @staticmethod
    def forward(ctx, input, gamma, beta, eps=1e-5):
        """
        Compute the batch normalization

        Args:
          ctx: context object handling storing and retrival of tensors and constants and specifying
               whether tensors need gradients in backward pass
          input: input tensor of shape (n_batch, n_neurons)
          gamma: variance scaling tensor, applied per neuron, shpae (n_neurons)
          beta: mean bias tensor, applied per neuron, shpae (n_neurons)
          eps: small float added to the variance for stability
        Returns:
          out: batch-normalized tensor
        """
        mu    = torch.mean(input, 0)
        var   = torch.var(input, dim=0, unbiased=False)
        x_hat = (input - mu) / ((var + eps) ** (0.5))
        out   = x_hat * gamma + beta

        ctx.save_for_backward(input, x_hat, out, gamma, beta, mu, var)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute backward pass of the batch normalization.

        Args:
          ctx: context object handling storing and retrival of tensors and constants and specifying
               whether tensors need gradients in backward pass
        Returns:
          out: tuple containing gradients for all input arguments
        """
        x, x_hat, y, gamma, beta, mu, var = ctx.saved_tensors

        grad_gamma = torch.sum(grad_output * x_hat, 0) if ctx.needs_input_grad[1] else None
        grad_beta  = torch.sum(grad_output, 0)         if ctx.needs_input_grad[2] else None
        grad_input = (gamma * (1 / torch.sqrt(var)) / x.shape[0]) * \
                     (x.shape[0] * grad_output - x_hat * grad_gamma - grad_beta) \
                                                       if ctx.needs_input_grad[0] else None

        # return gradients of the three tensor inputs and None for the constant eps
        return grad_input, grad_gamma, grad_beta, None


######################################################################################
# Code for Question 3.2 c)
######################################################################################

class CustomBatchNormManualModule(nn.Module):
    """
    This nn.module implements a custom version of the batch norm operation for MLPs.
    In self.forward the functional version CustomBatchNormManualFunction.forward is called.
    The automatic differentiation of PyTorch calls the backward method of this function in the backward pass.
    """

    def __init__(self, n_neurons, eps=1e-5):
        """
        Initializes CustomBatchNormManualModule object.

        Args:
          n_neurons: int specifying the number of neurons
          eps: small float to be added to the variance for stability
        """
        super(CustomBatchNormManualModule, self).__init__()

        self.n_neurons = n_neurons
        self.eps       = eps
        self.gamma     = nn.Parameter(torch.ones(self.n_neurons))
        self.beta      = nn.Parameter(torch.zeros(self.n_neurons))

    def forward(self, input):
        """
        Compute the batch normalization via CustomBatchNormManualFunction

        Args:
          input: input tensor of shape (n_batch, n_neurons)
        Returns:
          out: batch-normalized tensor
        """
        assert input.shape[1] == self.n_neurons, "Input has incorrect shape"

        out = CustomBatchNormManualFunction().apply(input, self.gamma, self.beta, self.eps)

        return out
