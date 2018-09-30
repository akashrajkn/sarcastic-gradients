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

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()

        self.seq_length  = seq_length
        self.input_dim   = input_dim
        self.num_hidden  = num_hidden
        self.num_classes = num_classes
        self.batch_size  = batch_size

        # Network
        self.W_xg = nn.Parameter(nn.init.xavier_uniform_(torch.empty(input_dim, num_hidden)))
        self.W_hg = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_hidden, num_hidden)))

        self.W_xi = nn.Parameter(nn.init.xavier_uniform_(torch.empty(input_dim, num_hidden)))
        self.W_hi = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_hidden, num_hidden)))

        self.W_xf = nn.Parameter(nn.init.xavier_uniform_(torch.empty(input_dim, num_hidden)))
        self.W_hf = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_hidden, num_hidden)))

        self.W_xo = nn.Parameter(nn.init.xavier_uniform_(torch.empty(input_dim, num_hidden)))
        self.W_ho = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_hidden, num_hidden)))

        self.W_hp = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_hidden, num_classes)))

        self.b_g = nn.Parameter(torch.zeros(num_hidden))
        self.b_i = nn.Parameter(torch.zeros(num_hidden))
        self.b_f = nn.Parameter(torch.zeros(num_hidden))
        self.b_o = nn.Parameter(torch.zeros(num_hidden))
        self.b_p = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x):

        # h_init, c_init
        h = torch.zeros(self.batch_size, self.num_hidden)
        c = torch.zeros(self.batch_size, self.num_hidden)

        for t in range(self.seq_length):
            # convert input at time (t) to one-hot vector. only makes sense
            # of input_dim == num_classes
            x_t_emb = x[:, t].view(-1, 1).type(torch.long)
            x_t = torch.zeros(self.batch_size, self.input_dim)
            # print(x_t_emb)
            x_t.scatter_(1, x_t_emb, 1)

            # modulation gate
            g = torch.matmul(x_t, self.W_xg) + torch.matmul(h, self.W_hg) + self.b_g
            g = torch.tanh(g)

            # input gate
            i = torch.matmul(x_t, self.W_xi) + torch.matmul(h, self.W_hi) + self.b_i
            i = torch.sigmoid(i)

            # forget gate
            f = torch.matmul(x_t, self.W_xf) + torch.matmul(h, self.W_hf) + self.b_f
            f = torch.sigmoid(f)

            # output gate
            o = torch.matmul(x_t, self.W_xo) + torch.matmul(h, self.W_ho) + self.b_o
            o = torch.sigmoid(o)

            # new cell state
            c = torch.mul(g, i) + torch.mul(c, f)
            h = torch.mul(torch.tanh(c), o)

        return torch.matmul(h, self.W_hp) + self.b_p
