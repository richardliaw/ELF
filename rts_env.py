from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch import nn
import queue
import threading
import argparse

#from game import Loader
from rlpytorch import *
import os
from ray.rllib.utils.vector_env import VectorEnv
import gym


class ELFRTSEnv(VectorEnv):
    unwrapped = None
    observation_space = gym.spaces.Box(0, 1, (20, ))
    action_space = gym.spaces.Discrete(9)

    def __init__(self, cfg=dict()):
        trainer = Trainer()
        runner = SingleProcessRun()
        os.environ["game"] = "./rts/game_MC/game"
        os.environ["model"] = "actor_critic"
        os.environ["model_file"] = "./rts/game_MC/model"
        defaults = {
            'T': 20,
            'additional_labels': 'id,last_terminal',
            'batchsize': cfg.get("batch", 64),
            'freq_update': 1,
            'num_games': cfg.get("games", 64),
            'players': 'type=AI_NN,fs=50,args=backup/AI_SIMPLE|start/500|decay/0.99;type=AI_SIMPLE,fs=20',
            'trainer_stats': 'winrate'}
        env, all_args = load_env(os.environ, trainer=trainer, runner=runner, overrides=defaults)
        self.GCwrapped = env["game"].initialize()
        self.GCwrapped.reg_callback("actor", lambda x: 1)
        self.GCwrapped.reg_callback("train", lambda x: 1)
        
        self.GCwrapped.Start()
        self._last_infos = infos = self.GCwrapped.GC.Wait(0)
        self._bsize = cfg.get("batch", 64)
        self._last_batch = self.GCwrapped.inputs[infos.gid].first_k(infos.batchsize)
        self.init_batch = self.process_batch(self._last_batch)

    def process_batch(self, batch):
        npbatch = batch.to_numpy()
        new_batch = {}
        new_batch['s'] = process_states(npbatch['s'])
        new_batch['last_terminal'] = npbatch['last_terminal'][-1, :]
        new_batch['last_r'] = npbatch['last_r'][-1, :]
        return new_batch

    def vector_reset(self, vector_width):
        return self.init_batch['s']

    def reset_at(self, index):
        return self.init_batch['s'][index]

    def vector_step(self, actions):
        new_reply = self.GCwrapped.replies[self._last_infos.gid]
        if new_reply is not None:
            sel_reply = new_reply.first_k(self._bsize)
        else:
            sel_reply = None
        reply = {
            'a': torch.LongTensor(actions),
            'pi': torch.FloatTensor(np.ones((self._bsize, 9)) * 0.01),
            'V': torch.FloatTensor(np.ones((self._bsize, 1)) * -0.1026)
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

def process_states(npbatch):
    npbatch = npbatch.sum(2).sum(3)[-1, :, :]
    npbatch /= 20 * 20
    return npbatch

if __name__ == '__main__':
    env = ELFRTSEnv({})
    start = (env.vector_reset(0))
    env.reset_at(2)
    for i in range(50):
        steps = env.vector_step(np.ones(64))
        import ipdb; ipdb.set_trace()
