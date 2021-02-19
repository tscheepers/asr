#!/usr/bin/env python
# ----------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2017 Sean Naren
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------------------

import torch
from sequence_wise import SequenceWise


class BatchLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_norm=True):
        super(BatchLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_norm = SequenceWise(torch.nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, bias=True)

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x, h0=None, c0=None, n_timesteps=None):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        if n_timesteps is not None:
            x = torch.nn.utils.rnn.pack_padded_sequence(x, n_timesteps)
        h = (h0, c0) if h0 is not None and c0 is not None else None
        x, (hn, cn) = self.rnn(x, h)
        if n_timesteps is not None:
            x, _ = torch.nn.utils.rnn.pad_packed_sequence(x)
        return x, hn, cn