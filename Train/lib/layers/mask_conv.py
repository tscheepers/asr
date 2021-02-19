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


class MaskConv(torch.nn.Module):
    def __init__(self, seq_module):
        """
        Adds padding to the output of the module based on the given lengths. This is to ensure that the
        results of the model do not change when batch sizes change during inference.
        Input needs to be in the shape of (Batch, Channel, Feature, Time)
        :param seq_module: The sequential module containing the conv stack.
        """
        super(MaskConv, self).__init__()
        self.seq_module = seq_module

    def forward(self, x, n_timesteps=None):
        """
        :param x: The input of size (Batch, Channel, Feature, Time)
        :param n_timesteps: The actual length of each sequence in the batch
        :return: (x, n) Masked output from the module and the timestep lengths
        """
        n = None if n_timesteps is None else n_timesteps.cpu().int()
        for module in self.seq_module:
            x = module(x)

            if n is None:  # If no inputs should be masked upto a specific length we do not apply padding
                continue

            # Calculate next n down the convolutional layer, because after a conv the number of output timesteps changes
            if type(module) == torch.nn.modules.conv.Conv2d:
                n = ((n + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1) //
                     module.stride[1] + 1)

            # Apply the mask
            mask = torch.BoolTensor(x.size()).fill_(0)
            if x.is_cuda:
                mask = mask.cuda()
            for i, length in enumerate(n):
                length = length.item()
                if (mask[i].size(2) - length) > 0:
                    mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
            x = x.masked_fill(mask, 0)

        return x, n
