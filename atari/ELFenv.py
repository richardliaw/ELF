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

    def vector_reset(self, vector_width):
        pass

    def reset_at(self, index):
        pass

    def vector_step(self, actions):
        infos = self.GCwrapped.GC.Wait(0)
        batch = self.GCwrapped.inputs[infos.gid].first_k(infos.batchsize)
        sel_reply = self.GCwrapped.replies[infos.gid].first_k(batchsize)
        if isinstance(reply, dict) and sel_reply is not None:
            # Current we only support reply to the most recent history.
            batch_key = "%s-%d" % (self.GCwrapped.idx2name[infos.gid], infos.gid)
            sel_reply.copy_from(reply, batch_key=batch_key)
        # import ipdb; ipdb.set_trace()
        # res = GC._call(batch)
        numpy_batch = batch.to_numpy()

        pass

    def first_env(self):
        pass

if __name__ == '__main__':
    ELFPongEnv()
