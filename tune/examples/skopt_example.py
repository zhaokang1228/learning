"""This test checks that Skopt is functional.

It also checks that it is usable with a separate scheduler.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
from ray.tune import run
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.skopt import SkOptSearch


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
    from skopt import Optimizer

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    args, _ = parser.parse_known_args()
    ray.init()

    config = {
        "num_samples": 10 if args.smoke_test else 50,
        "config": {
            "iterations": 100,
        },
        "stop": {
            "timesteps_total": 100
        },
    }
    optimizer = Optimizer([(0, 20), (-100, 100)])
    previously_run_params = [[10, 0], [15, -20]]
    known_rewards = [-189, -1144]

    """
    SkOptSearch参数：

    optimizer (skopt.optimizer.Optimizer) –优化器提供从skopt。
    parameter_names (list) – 参数名称列表。应该匹配优化器输出的维度。
    max_concurrent (int) –最大同时试验次数。默认为10。
    reward_attr (str) –训练结果目标值属性。这指的是一个递增的值。
    points_to_evaluate (list of lists) –在从优化器中采样之前，首先要运行的点的列表，例如，这些可以是您
        已经知道的参数配置，可以很好地帮助优化器选择合适的值。每个点都是使用parameter_names给出
        的顺序定义的参数列表。
    evaluated_rewards (list) –如果您以前评估过作为points_to_evaluate传递进来的参数，那么您可以通过将
        奖励属性作为列表传递进来，从而避免重新运行那些试验，这样优化器就可以被告知结果，而不需要
        重新计算试验。必须与points_to_evaluate相同的长度。

    """
    # algo = SkOptSearch(
    #     optimizer, ["width", "height"],
    #     max_concurrent=4,
    #     reward_attr="neg_mean_loss",
    #     points_to_evaluate=previously_run_params,
    #     evaluated_rewards=known_rewards)
    # scheduler = AsyncHyperBandScheduler(reward_attr="neg_mean_loss")
    # run(easy_objective,
    #     name="skopt_exp_with_warmstart",
    #     search_alg=algo,
    #     scheduler=scheduler,
    #     verbose=gai2,
    #     **config)

    # # Now run the experiment without known rewards

    algo = SkOptSearch(
        optimizer, ["width", "height"],
        max_concurrent=4,
        reward_attr="neg_mean_loss",
        points_to_evaluate=previously_run_params)
    scheduler = AsyncHyperBandScheduler(reward_attr="neg_mean_loss")
    run(easy_objective,
        name="skopt_exp",
        search_alg=algo,
        scheduler=scheduler,
        **config)
