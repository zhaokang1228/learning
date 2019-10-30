#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import random
import time

import ray
from ray.tune import Trainable, run
from ray.tune.schedulers import PopulationBasedTraining


class MyTrainableClass(Trainable):
    """Fake agent whose learning rate is determined by dummy factors."""

    def _setup(self, config):
        self.timestep = 0
        self.current_value = 0.0

    def _train(self):
        time.sleep(0.1)

        # Reward increase is parabolic as a function of factor_2, with a
        # maxima around factor_1=10.0.
        self.current_value += max(
            0.0, random.gauss(5.0 - (self.config["factor_1"] - 10.0)**2, 2.0))

        # Flat increase by factor_2
        self.current_value += random.gauss(self.config["factor_2"], 1.0)

        # Here we use `episode_reward_mean`, but you can also report other
        # objectives such as loss or accuracy.
        return {"episode_reward_mean": self.current_value}

    def _save(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, "checkpoint")
        with open(path, "w") as f:
            f.write(
                json.dumps({
                    "timestep": self.timestep,
                    "value": self.current_value
                }))
        return path

    def _restore(self, checkpoint_path):
        with open(checkpoint_path) as f:
            data = json.loads(f.read())
            self.timestep = data["timestep"]
            self.current_value = data["value"]

    def reset_config(self, new_config):
        self.config = new_config
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    args, _ = parser.parse_known_args()
    if args.smoke_test:
        ray.init(num_cpus=4)  # force pausing to happen for test
    else:
        ray.init(num_cpus=2)
    """
    PopulationBasedTraining调度需要检查点，故一般使用基于类的Trainable进行训练。
    
    time_attr (str) –训练结果用的比较时间。注意，您可以传递一些非时间的东西，比如training_iteration
        作为进度的度量，惟一的要求是属性应该单调地增加。
    reward_attr (str) – 训练结果目标值属性。与time_attr一样，这可以引用任何客观值。停止过程将使用此属性。
    perturbation_interval (float) – 模型将考虑在这个 time_attr时间间隔内的扰动。注意，扰动会导致检查
        点开销，因此不应该将此设置为太频繁。此处一般只的是时间，即多久进行一次扰动。
    hyperparam_mutations (dict) –Hyperparams变异。格式如下:对于每个键，可以提供列表或函数。列表
        指定一组允许的分类值。函数指定连续参数的分布。您必须指定至少一个hyperparam_variables或custom_explore_fn。
    resample_probability (float) – 当应用hyperparam_mutation时，从原始分布重新采样的概率。如果不
        进行重采样，则连续的值会受到1.2或0.8倍的扰动，离散的值则会变为相邻值。
    custom_explore_fn (func) –您还可以指定一个定制的探测函数。在应用了 hyperparam_mutations 的内
        置扰动之后，这个函数被调用为f(config)，并且应该返回根据需要更新的配置。您必须指定至少一个
        hyperparam_mutations 或 custom_explore_fn。
    log_config (bool) – 是否在每个训练运行过程时将每个模型的ray配置记录到local_dir。
        允许重建调度的配置信息。
        
    """
    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        reward_attr="episode_reward_mean",
        perturbation_interval=10,
        hyperparam_mutations={
            # Allow for scaling-based perturbations, with a uniform backing
            # distribution for resampling.
            "factor_1": lambda: random.uniform(0.0, 20.0),
            # Allow perturbations within this set of categorical values.
            "factor_2": [1, 2],
        })

    # Try to find the best factor 1 and factor gai2
    run(MyTrainableClass,
        name="pbt_test",
        scheduler=pbt,
        # reuse_actors (bool) –是否在不同的试验之间重用actor。这可以极大地加快经常启动
        # 和停止参与者(actors)的实验(例如，在时间复用模式下的PBT)。这要求试验具有相同的资源需求。
        reuse_actors=False,
        # verbose   为0时, 控制台不打印实验的状态和结果
        #           为1时, 控制台只打印状态
        #           为2时, 控制台打印状态和结果
        verbose=2,
        # verbose=1,
        **{
            "stop": {
                "training_iteration": 20 if args.smoke_test else 100
            },
            "num_samples": 20,
            "config": {
                "factor_1": 4.0,
                "factor_2": 1.0,
            },
        })
