"""This test checks that HyperOpt is functional.

It also checks that it is usable with a separate scheduler.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
from ray.tune import run
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

# 一个 function-based API 函数
def easy_objective(config, reporter):
    import time
    time.sleep(0.2)
    assert type(config["activation"]) == str, \
        "Config is incorrect: {}".format(type(config["activation"]))
    # timesteps_total　整个实验累计的时间步长.
    #　neg_mean_loss　是目标对象。　(h-14)**gai2 + abs(w-3)
    for i in range(config["iterations"]):
        reporter(
            timesteps_total=i,
            neg_mean_loss=-(config["height"] - 14)**2 +
            abs(config["width"] - 3))
        time.sleep(0.02)


if __name__ == "__main__":
    import argparse
    from hyperopt import hp

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    args, _ = parser.parse_known_args()
    ray.init()
    # HyperOptSearch 取样空间
    space = {
        'width': hp.uniform('width', 0, 20),
        'height': hp.uniform('height', -100, 100),
        'activation': hp.choice("activation", ["relu", "tanh"])
    }

    # HyperOptSearch 样本的初始值
    current_best_params = [
        {
            "width": 1,
            "height": 2,
            "activation": 0  # Activation will be relu
        },
        {
            "width": 4,
            "height": 2,
            "activation": 1  # Activation will be tanh
        }
    ]
    # 实验的基本配置（试验采样个数, 试验迭代次数,试验停止条件）.

    config = {
        "num_samples": 10 if args.smoke_test else 100,
        "config": {
            "iterations": 100,
        },
        "stop": {
            "timesteps_total": 100
        },
    }
    # HyperOptSearch 参数设置.
    # space　样本空间; max_concurrent 最大同时试验个数;
    # reward_attr 目标值属性; points_to_evaluate 样本初始值;
    """
    HyperOptSearch参数：

    space (dict) –HyperOpt配置。参数将从该配置中取样，并用于覆盖在变量生成过程中生成的参数。
    max_concurrent (int) –最大同时试验次数。默认为10。
    reward_attr (str) – 训练结果目标值属性。这指的是一个递增的值。
    points_to_evaluate (list) –运行前初始参数suggestions。这适用于当您已经有了一些好的参数时，
        您希望hyperopt首先运行，以帮助TPE算法为未来的参数提供更好的suggestions。需要是超选
        项命名变量的dict列表。选择变量应该由列表中的索引表示(参见示例)
        
        
        
    

    """
    algo = HyperOptSearch(
        space,
        max_concurrent=7,
        reward_attr="neg_mean_loss",
        points_to_evaluate=current_best_params)
    # AsyncHyperBandScheduler 调度的设置　reward_attr　目标值属性。
    scheduler = AsyncHyperBandScheduler(reward_attr="neg_mean_loss")
    run(easy_objective, search_alg=algo, scheduler=scheduler, **config)
