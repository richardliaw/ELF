from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import time

import ray
from ray.rllib.pg.pg_policy_graph import PGPolicyGraph
from ray.rllib.utils.actors import TaskPool
from ray.rllib.utils.common_policy_evaluator import CommonPolicyEvaluator
#from ray.rllib.utils.gpu import roundrobin_assign_gpu


def benchmark_gym(env_name):
    print("--- benchmarking gym env {} ---".format(env_name))
    env = gym.make(env_name)
    action = env.action_space.sample()
    start = time.time()
    i = 0
    env.reset()
    while time.time() - start < 10:
        while True:
            obs, rew, done, info = env.step(action)
            i += 1
            if done:
                env.reset()
                print("\rFrames per second", i / (time.time() - start), end="")
                break
    print()

def benchmark_elf(vector_width=64, games=64):
    print("--- benchmarking ELFevaluator  ---")
    from atari_env import ELFPongEnv
    config = {
        "gamma": 0.99,
        "model": {
            "fcnet_hiddens": [64, 64],
        }
    }
    elf_env = ELFPongEnv
    ev = CommonPolicyEvaluator(
        lambda cfg: elf_env(cfg),
        policy_graph=PGPolicyGraph,
        batch_steps=2000,
        batch_mode="pack_episodes",
        vector_width=vector_width,
        sample_async=True,
        compress_observations=False,
        model_config=config["model"],
        env_config={"games": games, "batch": vector_width},
        policy_config=config)

    total = 0
    start = time.time()
    print("Warming up...")
    while time.time() - start < 3:
        batch = ev.sample()
        total += batch.count
    print("Actions per second", total / (time.time() - start))
    print("Evaluating...")
    start = time.time()
    total = 0
    while time.time() - start < 5:
        batch = ev.sample()
        total += batch.count
    print("Actions per second", total / (time.time() - start))
    print()


def benchmark_single_core(env_name, vector_width=1):
    print("--- benchmarking one evaluator {} ---".format(env_name))
    config = {
        "gamma": 0.99,
        "model": {
            "fcnet_hiddens": [64, 64],
        }
    }
    ev = CommonPolicyEvaluator(
        lambda _: gym.make(env_name),
        policy_graph=PGPolicyGraph,
        batch_steps=2000,
        batch_mode="pack_episodes",
        vector_width=vector_width,
        sample_async=True,
        compress_observations=False,
        model_config=config["model"],
        policy_config=config)
    total = 0
    start = time.time()
    print("Warming up...")
    while time.time() - start < 3:
        batch = ev.sample()
        total += batch.count
    print("Actions per second", total / (time.time() - start))
    print("Evaluating...")
    start = time.time()
    total = 0
    while time.time() - start < 5:
        batch = ev.sample()
        total += batch.count
    print("Actions per second", total / (time.time() - start))
    print()


@ray.remote
class Collector(object):
    def __init__(self, evaluators):
        self.evaluators = evaluators
        self.tasks = TaskPool()
        for ev in self.evaluators:
            self.tasks.add(ev, ev.sample.remote())
            self.tasks.add(ev, ev.sample.remote())
            self.tasks.add(ev, ev.sample.remote())
            self.tasks.add(ev, ev.sample.remote())

    def do_barrier(self):
        ray.get([ev.apply.remote(lambda x: 0) for ev in self.evaluators])

    def do_collect(self, timeout):
        start = time.time()
        n = 0
        count = 0
        while True:
            for ev, batch in self.tasks.completed(prefetch_data=True):
                if time.time() - start > timeout:
                    return n, count
                n += 1
                self.tasks.add(ev, ev.sample.remote())
                batch = ray.get(batch)
                count += batch.count


def benchmark_distributed(
        env_name, num_collectors, evaluators_per_collector, vector_width,
        gpu=False, num_cpus=1, compress=False):
    print("--- benchmarking distributed sampling {} ---".format(env_name))
    config = {
        "gamma": 0.99,
        "model": {
            "fcnet_hiddens": [64, 64],
        }
    }
    collectors = []
    remote_cls = CommonPolicyEvaluator.as_remote(
        num_gpus=0, num_cpus=num_cpus)
    all_evs = []
    for _ in range(num_collectors):
        evs = []
        for _ in range(evaluators_per_collector):
            ev = remote_cls.remote(
                lambda _: gym.make(env_name),
                policy_graph=PGPolicyGraph,
                batch_steps=1000,
                batch_mode="pack_episodes",
                vector_width=vector_width,
                compress_observations=compress,
                model_config=config["model"],
                policy_config=config,
                sample_async=False,
                init_hook=roundrobin_assign_gpu if gpu else None)
            evs.append(ev)
        all_evs.extend(evs)
        collectors.append(Collector.remote(evs))

    total = 0
    start = time.time()
    print("Warming up...")
    futures = [c.do_collect.remote(3) for c in collectors]
    batches = 0
    for (n, count) in ray.get(futures):
        total += count
        batches += n
    print("Actions per second", total / (time.time() - start))
    print("Batches", n)
    print("Evaluating...")
    total = 0
    start = time.time()
    for _ in range(10):
        batches = 0
        futures = [c.do_collect.remote(10) for c in collectors]
        for (n, count) in ray.get(futures):
            total += count
            batches += n
        print("Actions per second", total / (time.time() - start))
        print("Batches", n)
        print()
#        ray.get([c.do_barrier.remote() for c in collectors])


if __name__ == "__main__":
    ray.init(redis_address="localhost:6379")
    benchmark_elf()
#    benchmark_gym("CartPole-v0")
#    benchmark_gym("PongNoFrameskip-v4")
#    benchmark_single_core("CartPole-v0", vector_width=128)
#    benchmark_single_core("PongNoFrameskip-v4", vector_width=1)
    #benchmark_distributed("Pendulum-v0", 4, 32, 64, gpu=False, compress=False)
#    benchmark_distributed("PongNoFrameskip-v4", 4, 8, 32, gpu=False, compress=True)
