################################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()

        self.seq_length  = seq_length
        self.input_dim   = input_dim
        self.num_hidden  = num_hidden
        self.num_classes = num_classes
        self.batch_size  = batch_size

        # Network
        self.W_hx = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.num_hidden, self.input_dim)))
        self.W_hh = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.num_hidden, self.num_hidden)))
        self.W_ph = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.num_classes, self.num_hidden)))
        self.b_h  = nn.Parameter(torch.zeros(num_hidden, batch_size))
        self.b_p  = nn.Parameter(torch.zeros(num_classes, batch_size))

    def forward(self, x):
        # h_init
        h_t = torch.zeros(self.num_hidden, self.batch_size)
        x   = x.t()

        for t in range(self.seq_length - 1):
            x_t = x[t, :].view(self.input_dim, self.batch_size)
            h_t = torch.tanh(torch.matmul(self.W_hx, x_t) + torch.matmul(self.W_hh, h_t) + self.b_h)

        return (torch.matmul(self.W_ph, h_t) + self.b_p).t()
