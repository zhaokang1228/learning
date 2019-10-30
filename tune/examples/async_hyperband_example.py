#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import random

import numpy as np

import ray
from ray.tune import Trainable, run, sample_from
from ray.tune.schedulers import AsyncHyperBandScheduler


class MyTrainableClass(Trainable):
    """Example agent whose learning curve is a random sigmoid.

    The dummy hyperparameters "width" and "height" determine the slope and
    maximum reward value reached.
    """

    def _setup(self, config):
        self.timestep = 0

    def _train(self):
        self.timestep += 1
        # v = tan(timestep/width)*height
        v = np.tanh(float(self.timestep) / self.config["width"])
        v *= self.config["height"]

        # Here we use `episode_reward_mean`, but you can also report other
        # objectives such as loss or accuracy.
        return {"episode_reward_mean": v}

    def _save(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, "checkpoint")
        with open(path, "w") as f:
            f.write(json.dumps({"timestep": self.timestep}))
        return path

    def _restore(self, checkpoint_path):
        with open(checkpoint_path) as f:
            self.timestep = json.loads(f.read())["timestep"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    args, _ = parser.parse_known_args()
    ray.init()
    """
    asynchronous hyperband early stopping, configured with
    `episode_reward_mean` as the
    objective and `training_iteration` as the time unit,
    which is automatically filled by Tune.
    
    time_attr 时间基元;               reward_attr　目标属性;
    grace_period　至少的时间（次数）;   max_t 最大的时间（次数）;
    
    
    AsyncHyperBandScheduler调度参数：
    
    time_attr (str) – 用于比较时间的训练结果。注意，您可以传递一些非时间的东西，比如
        training_iteration作为进度的度量，惟一的要求是属性应该单调地增加。
    reward_attr (str) – 训练结果目标值属性。与time_attr一样，这可以引用任何客观值。停止过程将使用此属性。
    max_t (float) – 每次训练的最大时间单位。max_t时间单位(由time_attr决定)通过后，将停止试验。
        此参数相当于tune.run()中的参数stop的作用。
    grace_period (float) – 至少在这个时候停止试验。这些单元与time_attr所命名的属性相同。
    reduction_factor (float) –用于设定减半率和用量。这只是一个无单位标量。
    brackets (int) –试验块数量。 每个试验块具有不同的减半率，由减少系数指定。
    
    """
    ahb = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        reward_attr="episode_reward_mean",
        grace_period=15,
        reduction_factor=2,
        max_t=100)

    run(MyTrainableClass,
        name="asynchyperband_test",
        scheduler=ahb,
        **{
            # 停止条件　training_iteration=99999
            "stop": {
                "training_iteration": 1 if args.smoke_test else 99999
            },
            # 抽样30次,　即通过AsyncHyperBandScheduler调度　对MyTrainableClass进行30次训练。
            "num_samples": 30,
            # 机器资源　测个试验用一个CPU资源。
            "resources_per_trial": {
                "cpu": 1,
                "gpu": 0
            },
            "config": {
                "width": sample_from(
                    lambda spec: 10 + int(90 * random.random())),
                "height": sample_from(lambda spec: int(100 * random.random())),
            },
        })
