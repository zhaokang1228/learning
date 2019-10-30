"""This test checks that GeneticSearch is functional.

It also checks that it is usable with a separate scheduler.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
from ray.tune import run
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.automl import GeneticSearch
from ray.tune.automl import ContinuousSpace, DiscreteSpace, SearchSpace

import time



def michalewicz_function(config, reporter):
    """f(x) = -sum{sin(xi) * [sin(i*xi^gai2 / pi)]^(2m)}"""
    import numpy as np
    x = np.array(
        [config['x1'], config['x2'], config['x3'], config['x4'], config['x5']])
    sin_x = np.sin(x)
    z = (np.arange(1, 6) / np.pi * (x * x))
    sin_z = np.power(np.sin(z), 20)  # let m = 20
    y = np.dot(sin_x, sin_z)
    time.sleep(0.01)
    # Negate y since we want to minimize y value
    reporter(timesteps_total=2, neg_mean_loss=-y)


if __name__ == "__main__":
    start =time.time()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    args, _ = parser.parse_known_args()
    ray.init()
    """
     ContinuousSpace('x1', 0, 4, 100)　把0到4分成100等分，即形成１００个数字，命名为x1 从中随机抽取
      就是一个np.linspace(start, end, num)函数中选一个
     DiscreteSpace('x5', [-1, 0, 1, gai2, 3])　从-1,0,1,gai2,3　中随机抽取一个
    """
    space = SearchSpace({
        ContinuousSpace('x1', 0, 4, 100),
        ContinuousSpace('x2', -2, 2, 100),
        ContinuousSpace('x3', 1, 5, 100),
        ContinuousSpace('x4', -3, 3, 100),
        DiscreteSpace('x5', [-1, 0, 1, 2, 3]),
    })

    config = {"stop": {"training_iteration": 10}}
    """
    遗传优化
    
    
    search_space 样本空间
                连续空间，ContinuousSpace('x1', 0, 4, 100)　把0到4分成100等分，即形成１００个数字，命名为x1 从中随机抽取
                离散空间，DiscreteSpace('x5', [-1, 0, 1, gai2, 3])　从-1,0,1,gai2,3　中随机抽取一个
    reward_attr　目标属性
    max_generation 是trail的迭代次数　（最大时间基元）
    population_size为初始的trail个数（初始基数大小）
    其中默认参数        population_decay=0.95,  基数衰减率
                      keep_top_ratio=0.gai2,     训练的最佳表现率　
                      selection_bound=0.4,    训练的选择率
                      crossover_bound=0.4     训练的交叉率
    也可以自己设置。
    总共进行max_generation次优化(此代码为１０次),每次trail个数为上一次的population_decay倍取整（此代码为０．９５）
    即：第一次运行50个训练，第二次会运行50*0.95 = 47个训练，第三次会运行47*0.95=44个训练 ...... 第１０次运行２８个训练
        共３８０个
    """
    algo = GeneticSearch(
        space,
        reward_attr="neg_mean_loss",
        max_generation=2 if args.smoke_test else 10,
        population_size=10 if args.smoke_test else 50)
    scheduler = AsyncHyperBandScheduler(reward_attr="neg_mean_loss")
    run(michalewicz_function,
        name="my_exp",
        search_alg=algo,
        scheduler=scheduler,
        **config)
    print(time.time()-start)