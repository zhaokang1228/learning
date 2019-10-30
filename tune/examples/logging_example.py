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
from ray import tune
from ray.tune import Trainable, run, Experiment
from ray.tune.logger import  DEFAULT_LOGGERS

# 日志类
class TestLogger(tune.logger.Logger):
    def on_result(self, result):
        print("TestLogger", result)

# 可训练命名函数
# trial.trainable_name  调用训练器的名字
# trial.trial_id　　　   训练器的ID
def trial_str_creator(trial):
    return "{}_{}_123".format(trial.trainable_name, trial.trial_id)


class MyTrainableClass(Trainable):
    """Example agent whose learning curve is a random sigmoid.

    The dummy hyperparameters "width" and "height" determine the slope and
    maximum reward value reached.
    """

    def _setup(self, config):
        self.timestep = 0

    def _train(self):
        self.timestep += 1
        v = np.tanh(float(self.timestep) / self.config["width"])
        v *= self.config["height"]

        # Here we use `episode_reward_mean`, but you can also report other
        # objectives such as loss or accuracy.
        # 这里我们使用`episode_reward_mean`，也可以报告其他目标，例如loss或accuracy。
        # 可以选择　episode_reward_mean, mean_loss, mean_accuracy　和　timesteps_this_iter .
        return {"episode_reward_mean": v}

    # 保存训练的检查点到　checkpoint_dir　文件下生成一个json文件。
    # 默认情况下是在 ~/ray_results/实验名　下（此代码运行后是在　~/ray_results/hyperband_test　下）
    def _save(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, "checkpoint")
        with open(path, "w") as f:
            f.write(json.dumps({"timestep": self.timestep}))
        return path

    # 从给定的检查点恢复（训练出错）
    def _restore(self, checkpoint_path):
        with open(checkpoint_path) as f:
            self.timestep = json.loads(f.read())["timestep"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    args, _ = parser.parse_known_args()
    ray.init()
    exp = Experiment(
        name="hyperband_test",
        run=MyTrainableClass,
        #  num_samples 从超参数空间抽样两次，此代码即训练两次可训练函数MyTrainableClass
        num_samples=10,
        # 可训练函数重命名
        # trial_name_creator=tune.function(trial_str_creator),
        # 打印训练过程中每次试验的状态信息。
        # loggers=[TestLogger],
        stop={"training_iteration": 1 if args.smoke_test else 999},
        # random.random()生成0和1之间的随机浮点数float
        # config参数的含义是　 width取值空间为[10, 100], height　取值空间为　[0, 100]
        config={
            "width": tune.sample_from(lambda spec: 10 + int(90 * random.random())),
            "height": tune.sample_from(lambda spec: int(100 * random.random()))
        })

    trials = run(exp)
