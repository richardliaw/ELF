# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from rlpytorch import Model, ActorCritic

import torch
import torch.nn as nn

class VisionNetwork(nn.Module):
    """Generic vision network"""

    def __init__(self, inputs, num_outputs):
        """TF visionnet in PyTorch.

        Params:
            inputs (tuple): (channels, rows/height, cols/width)
            num_outputs (int): logits size
        """
        filters = options.get("conv_filters", [
            [16, [8, 8], 4],
            [32, [4, 4], 2],
            [512, [10, 10], 1],
        ])
        layers = []
        in_channels, in_size = inputs[0], inputs[1:]

        for out_channels, kernel, stride in filters[:-1]:
            padding, out_size = valid_padding(in_size, kernel,
                                              [stride, stride])
            layers.append(
                SlimConv2d(in_channels, out_channels, kernel, stride, padding))
            in_channels = out_channels
            in_size = out_size

        out_channels, kernel, stride = filters[-1]
        layers.append(
            SlimConv2d(in_channels, out_channels, kernel, stride, None))
        self._convs = nn.Sequential(*layers)

        self.logits = SlimFC(
            out_channels, num_outputs, initializer=nn.init.xavier_uniform_)
        self.value_branch = SlimFC(
            out_channels, 1)

    def hidden_layers(self, obs):
        """ Internal method - pass in torch tensors, not numpy arrays

        args:
            obs: observations and features"""
        res = self._convs(obs)
        res = res.squeeze(3)
        res = res.squeeze(2)
        return res

    def forward(self, obs):
        """Internal method. Implements the

        Args:
            obs (PyTorch): observations and features

        Return:
            logits (PyTorch): logits to be sampled from for each state
            value (PyTorch): value function for each state"""
        res = self.hidden_layers(obs)
        logits = self.logits(res)
        value = self.value_branch(res)
        return logits, value




class Model_ActorCritic(Model):
    def __init__(self, args):
        super(Model_ActorCritic, self).__init__(args)

        params = args.params

        self.linear_dim = 1920
        relu_func = lambda : nn.LeakyReLU(0.1)
        # relu_func = nn.ReLU

        self.trunk = nn.Sequential(
            nn.Conv2d(3 * params["hist_len"], 32, 5, padding = 2),
            relu_func(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, 5, padding = 2),
            relu_func(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding = 1),
            relu_func(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, padding = 1),
            relu_func(),
            nn.MaxPool2d(2, 2),
        )

        self.conv2fc = nn.Sequential(
            nn.Linear(self.linear_dim, 512),
            nn.PReLU()
        )

        self.policy_branch = nn.Linear(512, params["num_action"])
        self.value_branch = nn.Linear(512, 1)
        self.softmax = nn.Softmax()

    def forward(self, x):
        # Get the last hist_len frames.
        s = self._var(x["s"])
        # print("input size = " + str(s.size()))
        rep = self.trunk(s)
        # print("trunk size = " + str(rep.size()))
        rep = self.conv2fc(rep.view(-1, self.linear_dim))
        policy = self.softmax(self.policy_branch(rep))
        value = self.value_branch(rep)
        return dict(pi=policy, V=value)

# Format: key, [model, method]
Models = {
    "actor_critic" : [Model_ActorCritic, ActorCritic]
}


class SlimConv2d(nn.Module):
    """Simple mock of tf.slim Conv2d"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel,
                 stride,
                 padding,
                 initializer=nn.init.xavier_uniform,
                 activation_fn=nn.ReLU,
                 bias_init=0):
        super(SlimConv2d, self).__init__()
        layers = []
        if padding:
            layers.append(nn.ZeroPad2d(padding))
        conv = nn.Conv2d(in_channels, out_channels, kernel, stride)
        if initializer:
            initializer(conv.weight)
        nn.init.constant_(conv.bias, bias_init)

        layers.append(conv)
        if activation_fn:
            layers.append(activation_fn())
        self._model = nn.Sequential(*layers)

    def forward(self, x):
        return self._model(x)


class SlimFC(nn.Module):
    """Simple PyTorch of `linear` function"""

    def __init__(self,
                 in_size,
                 out_size,
                 initializer=None,
                 activation_fn=None,
                 bias_init=0):
        super(SlimFC, self).__init__()
        layers = []
        linear = nn.Linear(in_size, out_size)
        if initializer:
            initializer(linear.weight)
        nn.init.constant_(linear.bias, bias_init)
        layers.append(linear)
        if activation_fn:
            layers.append(activation_fn())
        self._model = nn.Sequential(*layers)

    def forward(self, x):
        return self._model(x)


def valid_padding(in_size, filter_size, stride_size):
    """Note: Padding is added to match TF conv2d `same` padding. See
    www.tensorflow.org/versions/r0.12/api_docs/python/nn/convolution

    Params:
        in_size (tuple): Rows (Height), Column (Width) for input
        stride_size (tuple): Rows (Height), Column (Width) for stride
        filter_size (tuple): Rows (Height), Column (Width) for filter

    Output:
        padding (tuple): For input into torch.nn.ZeroPad2d
        output (tuple): Output shape after padding and convolution
    """
    in_height, in_width = in_size
    filter_height, filter_width = filter_size
    stride_height, stride_width = stride_size

    out_height = np.ceil(float(in_height) / float(stride_height))
    out_width = np.ceil(float(in_width) / float(stride_width))

    pad_along_height = int(
        ((out_height - 1) * stride_height + filter_height - in_height))
    pad_along_width = int(
        ((out_width - 1) * stride_width + filter_width - in_width))
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    output = (out_height, out_width)
    return padding, output
