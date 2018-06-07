from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch import nn
import queue
import threading
import argparse

from game import Loader
from rlpytorch import ArgsProvider
from ray.rllib.utils.vector_env import VectorEnv
import gym

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



class ELFPongEnv(VectorEnv):
    unwrapped = None
    observation_space = gym.spaces.Box(0, 1, (80, 80, 3))
    action_space = gym.spaces.Discrete(6)
    def __init__(self, cfg=dict()):
        parser = argparse.ArgumentParser()
        cmd_line = "--num_games {games} --batchsize {batch}".format(
            games=cfg.get("games", 64),
            batch=cfg.get("batch", 16))
        cmd_line += " --hist_len 1 --frame_skip 4 --actor_only"
        cmd_line = cmd_line.split(" ")
        loader = Loader()
        args = ArgsProvider.Load(parser, [loader], cmd_line=cmd_line)
        self.GCwrapped = loader.initialize()
        self.GCwrapped.reg_callback("actor", lambda x: 1)
        self.GCwrapped.Start()
        self._last_infos = infos = self.GCwrapped.GC.Wait(0)
        self._bsize = infos.batchsize
        self._last_batch = self.GCwrapped.inputs[infos.gid].first_k(infos.batchsize)
        self.init_batch = self.process_batch(self._last_batch)

    def process_batch(self, batch):
        # import ipdb; ipdb.set_trace()
        npbatch = batch.to_numpy()
        new_batch = {}
        new_batch['s'] = process_states(npbatch['s'])
        new_batch['last_terminal'] = npbatch['last_terminal'].squeeze()
        new_batch['last_r'] = npbatch['last_r'].squeeze()
        return new_batch

    def vector_reset(self, vector_width):
        return self.init_batch['s']

    def reset_at(self, index):
        return self.init_batch['s'][index, :, :, :]

    def vector_step(self, actions):
        sel_reply = self.GCwrapped.replies[self._last_infos.gid].first_k(self._bsize)
        reply = {
            'a': torch.from_numpy(actions)
        }
        # TODO: build out a reply
        if isinstance(reply, dict) and sel_reply is not None:
            # Current we only support reply to the most recent history.
            batch_key = "%s-%d" % (self.GCwrapped.idx2name[self._last_infos.gid], self._last_infos.gid)
            sel_reply.copy_from(reply, batch_key=batch_key)
        self.GCwrapped.GC.Steps(self._last_infos)
        self._last_infos = infos = self.GCwrapped.GC.Wait(0)
        self._last_batch = self.GCwrapped.inputs[self._last_infos.gid].first_k(infos.batchsize)
        numpy_batch = self.process_batch(self._last_batch)
        return numpy_batch['s'], numpy_batch['last_r'], numpy_batch['last_terminal'], None

    def first_env(self):
        pass

def process_states(batch):
    batch = batch.squeeze(0)
    batch = batch[:, :, :80, :]
    batch = np.transpose(batch, (0, 2, 3, 1))
    return batch

if __name__ == '__main__':
    env = ELFPongEnv({})
    start = (env.vector_reset(0))
    env.reset_at(2)
    for i in range(50):
        steps = env.vector_step(np.zeros(16))
