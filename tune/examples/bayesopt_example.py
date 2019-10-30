"""This test checks that BayesOpt is functional.

It also checks that it is usable with a separate scheduler.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
from ray.tune import run
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch


def easy_objective(config, reporter):
    import time
    time.sleep(0.2)
    for i in range(config["iterations"]):
        reporter(
            timesteps_total=i,
            neg_mean_loss=-(config["height"] - 14)**2 +
            abs(config["width"] - 3))
        time.sleep(0.02)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    args, _ = parser.parse_known_args()
    ray.init()

    space = {'width': (0, 20), 'height': (-100, 100)}

    config = {
        "num_samples": 10 if args.smoke_test else 100,
        "config": {
            "iterations": 100,
        },
        "stop": {
            "timesteps_total": 100
        }
    }
    """
    BayesOptSearch优化参数：
    
    space (dict) –连续的搜索空间。参数将从该空间取样，用于运行试验。
    max_concurrent (int) –最大同时试验次数。默认为10。
    reward_attr (str) –训练结果目标值属性。这指的是一个递增的值。
    utility_kwargs (dict) – 参数来定义实用函数。必须为键 kind、kappa和xi提供值。
            其中　kind 只有三个选项：　ucb,ei,poi　
                      其中ucb和xi没有关系。
                      ucb:  mean + kappa * std,
                      ei:   (mean - y_max - xi) * norm.cdf(z)/std) + std * norm.pdf(z),其中z = (mean - y_max - xi
                      poi:  (mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z)
    random_state (int) –用于初始化BayesOpt。
    verbose (int) – 设置BayesOpt包的复杂级别。

    """
    algo = BayesOptSearch(
        space,
        max_concurrent=8,
        reward_attr="neg_mean_loss",
        utility_kwargs={
            "kind": "ucb",
            "kappa": 2.5,
            "xi": 0.0
        },
        verbose=1
        )
    scheduler = AsyncHyperBandScheduler(reward_attr="neg_mean_loss",brackets=3)
    run(easy_objective,
        name="my_exp",
        search_alg=algo,
        scheduler=scheduler,
        **config)
