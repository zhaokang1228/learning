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
from ray.tune import Trainable, run, Experiment, sample_from
from ray.tune.schedulers import HyperBandScheduler


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

    # Hyperband early stopping, configured with `episode_reward_mean` as the
    # objective and `training_iteration` as the time unit,
    # which is automatically filled by Tune.
    # time_attr 时间基元;               reward_attr　停止条件;
    # max_t 最大的时间（次数）;
    """
    HyperBandScheduler调度是要支持检查点的。

    time_attr (str) – 训练结果用于比较时间。注意，您可以传递一些非时间的东西，比如
        training_iteration作为进度的度量，惟一的要求是属性应该单调地增加。
    reward_attr (str) – 训练结果目标值属性。与time_attr一样，这可以引用任何客观值。停止过程将使用此属性。
    max_t (int) –每次试验的最大时间单位。max_t时间单位(由time_attr决定)通过后，将停止试验。调度
        程序将在此时间通过后终止试验。注意，这与原始HyperBand论文中提到的max_t的语义不同。
    
    控制台打印信息
        Round　 大试验块，里边最多有五个试验块（Bracket），最多１４３个试验样本。　
                运行的时候是按大实验块的编号顺序执行，每次最多运行５个试验块
        Bracket 试验块　里边从第一个实验块开始，依次存5,8,15,34,81 个试验（样本），
                其中样本数量按照前开始依次存入，只有最后一个试验快可以试验样本不填充满。
                具体　试验块（Bracket）个数(ｎ+1)是由`(5/(n+1))*3**n >= num_samples`,取ｎ最小满足此公式的整数，
                这个括号内可以容纳的训练个数为`(5/(n+1))*3**n`个（计算使用进一法，可以不容纳满即最后一个括号可以不容纳满）。
                当试验块（Bracket）个数大于５后，ｎ=n-1，重新开始计.
                注 每个Round中的  第一个Bracket的5个试验样本全部运行max_t次迭代
                                第二个Bracket的8个试验首先有一部分运行m2（低于max_t）的迭代次数，剩余部分中的一部分再运行m2+的迭代次数
                                    直至最后一次吧剩余部分全部运行max_t
                                第三个Bracket和第二个一样，只是刚开始迭代的次数要比上次一的要小。直至第五次同样的方式。
                    其中初始迭代次数为max_t*3**n(n为Bracket的个数减1)
                Max Size (n)　表示当前试验块（取样）中没有运行结束的试验个数（当此试验块没有运行时，此值代表此试验块最大能容纳多少试验），
                    注：当此试验块里的试验运行结束后，此值代表此实验块最后运行结束的试验个数。
                Milestone (r)　表示试验将在下一次减半之前运行的迭代次数。最后依次增加达到max_t的大小。
                completed   表示近似进度指标。  
        Bracket(试验块)后边的大括号内代表当前试验块中试验的状态（RUNNING　正在运行　PAUSED　等待　TERMINATED　运行终止）和对应的个数
    """
    hyperband = HyperBandScheduler(
        time_attr="training_iteration",
        reward_attr="episode_reward_mean",
        max_t=100)

    exp = Experiment(
        name="hyperband_test",
        run=MyTrainableClass,
        num_samples=20,
        stop={"training_iteration": 1 if args.smoke_test else 99999},
        config={
            "width": sample_from(lambda spec: 10 + int(90 * random.random())),
            "height": sample_from(lambda spec: int(100 * random.random()))
        })

    run(exp, scheduler=hyperband)
