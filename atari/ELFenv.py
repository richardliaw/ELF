from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import queue
import threading
import argparse

from game import Loader
from rlpytorch import ArgsProvider

# class VectorEnv(object):
#     @classmethod
#     def wrap(self, make_env=None, existing_envs=[]):
#         return _VectorizedGymEnv(make_env, existing_envs)

#     def vector_reset(self, vector_width):
#         raise NotImplementedError

#     def reset_at(self, index):
#         raise NotImplementedError

#     def vector_step(self, actions):
#         raise NotImplementedError

#     def first_env(self):
#         raise NotImplementedError



class ELFPongEnv():
    def __init__(self, *args):
        parser = argparse.ArgumentParser()
        cmd_line = "--num_games 64 --batchsize 16 --hist_len 1 --frame_skip 4 --actor_only".split(" ")
        loader = Loader()
        args = ArgsProvider.Load(parser, [loader], cmd_line=cmd_line)
        # args = ArgsProvider.Load(parser, [loader], cmd_line=cmd_line)
        self.GCwrapped = loader.initialize()
        self.GCwrapped.reg_callback("actor", lambda x: 1)
        self.GCwrapped.start()
        # infos = self.GCwrapped.GC.Wait(0)
        # batch = self.GCwrapped.inputs[infos.gid].first_k(infos.batchsize)
        # self.init_batch = batch.to_numpy()

    def vector_reset(self, vector_width):
        return self.init_batch['s']

    def reset_at(self, index):
        return self.init_batch['s'][index]

    def vector_step(self, actions):
        sel_reply = self.GCwrapped.replies[infos.gid].first_k(batchsize)
        # TODO: build out a reply
        if isinstance(reply, dict) and sel_reply is not None:

            # Current we only support reply to the most recent history.
            batch_key = "%s-%d" % (self.GCwrapped.idx2name[infos.gid], infos.gid)
            sel_reply.copy_from(reply, batch_key=batch_key)
        infos = self.GCwrapped.GC.Wait(0)
        batch = self.GCwrapped.inputs[infos.gid].first_k(infos.batchsize)
        numpy_batch = batch.to_numpy()
        return numpy_batch['s'], numpy_batch['r'], numpy_batch['last_terminal'], None

    def first_env(self):
        pass

import torch
from torch import nn
from model import SlimConv2d, SlimFC, valid_padding

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


if __name__ == '__main__':
    env = ELFPongEnv()
    start = env.vector_reset(0)

    steps = env.vector_step(actions)
