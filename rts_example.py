# Copyright (c) 2017-present, Facebook, Inc.

# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime
import sys
import os

from rlpytorch import *

if __name__ == '__main__':
    trainer = Trainer()
    runner = SingleProcessRun()
    os.environ["game"] = "./rts/game_MC/game"
    os.environ["model"] = "actor_critic"
    os.environ["model_file"] = "./rts/game_MC/model"
    defaults = {
        'T': 20,
        'additional_labels': 'id,last_terminal',
        'batchsize': 128,
        'freq_update': 1,
        'num_games': 1024,
        'players': 'type=AI_NN,fs=50,args=backup/AI_SIMPLE|start/500|decay/0.99;type=AI_SIMPLE,fs=20',
        'trainer_stats': 'winrate'}
    env, all_args = load_env(os.environ, trainer=trainer, runner=runner, overrides=defaults)
    GC = env["game"].initialize()

    # model = env["model_loaders"][0].load_model(GC.params)
    # env["mi"].add_model("model", model, opt=True)
    # env["mi"].add_model("actor", model, copy=True, cuda=all_args.gpu is not None, gpu_id=all_args.gpu)
    import ipdb; ipdb.set_trace()

    #trainer.setup(sampler=env["sampler"], mi=env["mi"], rl_method=env["method"])

    GC.reg_callback("train", lambda x: x)
    GC.reg_callback("actor", lambda y: y)
    GC.Start()
    print("STARTED")
    elapsed_wait_only = 0
    for _ in range(50):
         info = GC.GC.Wait(0)
         batch = GC.inputs[info.gid].first_k(info.batchsize)
         # import ipdb; ipdb.set_trace()
         # res = GC._call(batch)
         numpy_batch = batch.to_numpy()
         print("iter", _)
    #runner.setup(GC, episode_summary=trainer.episode_summary,
    #            episode_start=trainer.episode_start)

    #runner.run()

