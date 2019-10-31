一下是tune中的一些例子的代码实例和运行结果，有个人不部分代码说明，实例也可以在`example`文件中找到对应的例子。

## 一、 一般的例子：
### １．自定义日志记录器和自定义试验名命名的示例。

 **ray/python/ray/tune/examples/logging_example.py** 
 
源码：
```
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

# 日志函数
class TestLogger(tune.logger.Logger):
    def on_result(self, result):
        print("TestLogger", result)

# 实验命名函数
# trial.trainable_name  调用训练器的名字
# trial.trial_id　　　   训练器的ID
def trial_str_creator(trial):
    return "{}_{}_123".format(trial.trainable_name, trial.trial_id)


class MyTrainableClass(Trainable):
    """Example agent whose learning curve is a random sigmoid.

    The dummy hyperparameters "width" and "height" determine the slope and
    maximum reward value reached.
    """
	# 初始化
    def _setup(self, config):
        self.timestep = 0
	# 训练函数
    def _train(self):
        self.timestep += 1
        v = np.tanh(float(self.timestep) / self.config["width"])
        v *= self.config["height"]


        # 这里我们使用`episode_reward_mean`，也可以报告其他目标，例如loss或accuracy。
        # 可以选择　episode_reward_mean, mean_loss, mean_accuracy 和 timesteps_this_iter .
        return {"episode_reward_mean": v}

    # 保存训练的检查点到　checkpoint_dir　文件下生成一个json文件。　
    # 路径也可以自定义，tune.run()中的参数local_dir可以设置位置
    # 默认情况下是在 '~/ray_results/实验名'下（此代码运行后是在'~/ray_results/hyperband_test'下）
    def _save(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, "checkpoint")
        with open(path, "w") as f:
            f.write(json.dumps({"timestep": self.timestep}))
        return path

    # 从给定的检查点恢复函数（训练出错时调用）
    def _restore(self, checkpoint_path):
        with open(checkpoint_path) as f:
            self.timestep = json.loads(f.read())["timestep"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()　#　命令行解析函数
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    args, _ = parser.parse_known_args()
    #	ray启动，如果是在ray的分布式集群上使用，启动需要加上对应主节点的ip和端口好　例如：redis_address="192.168.10.1:6379"
    ray.init()		
    exp = Experiment(
        name="hyperband_test", 	# 实验名
        run=MyTrainableClass,	# 试验（trial）名
        #  num_samples 从超参数空间抽样两次，此代码即训练两次可训练函数MyTrainableClass
        num_samples=2,
        # 可训练函数重命名
        trial_name_creator=tune.function(trial_str_creator),
        loggers=[TestLogger], # 调用自定义日志函数
        #试验终止条件，此处不是试验早期停止条件
        stop={"training_iteration": 1 if args.smoke_test else 999},
        # random.random()生成0和1之间的随机浮点数float
        # config　参数可以理解为参数空间	，试验训练过程自定义的参数头可以通过此参数传到试验中
        #　参数的含义是　 width取值空间为[10, 100], height　取值空间为　[0, 100]
        config={
            "width": tune.sample_from(lambda spec: 10 + int(90 * random.random())),
            "height": tune.sample_from(lambda spec: int(100 * random.random()))
        })

    trials = run(exp)

```
实验结果：

```
/usr/bin/python3.5 /home/kangkang/PycharmProjects/ray/python/ray/tune/examples/logging_example.py
2019-04-21 16:54:51,571	INFO node.py:423 -- Process STDOUT and STDERR is being redirected to /tmp/ray/session_2019-04-21_16-54-51_20983/logs.
2019-04-21 16:54:51,674	INFO services.py:363 -- Waiting for redis server at 127.0.0.1:13169 to respond...
2019-04-21 16:54:51,800	INFO services.py:363 -- Waiting for redis server at 127.0.0.1:33321 to respond...
2019-04-21 16:54:51,801	INFO services.py:760 -- Starting Redis shard with 3.35 GB max memory.
2019-04-21 16:54:51,819	INFO services.py:1384 -- Starting the Plasma object store with 5.03 GB memory using /dev/shm.
2019-04-21 16:54:51,898	INFO tune.py:60 -- Tip: to resume incomplete experiments, pass resume='prompt' or resume=True to run()
2019-04-21 16:54:51,898	INFO tune.py:211 -- Starting a new experiment.
== Status ==
Using FIFO scheduling algorithm.
Resources requested: 0/8 CPUs, 0/0 GPUs
Memory usage on this node: 3.9/16.8 GB

2019-04-21 16:54:52,556	WARNING util.py:62 -- The `start_trial` operation took 0.6490788459777832 seconds to complete, which may be a performance bottleneck.
== Status ==
Using FIFO scheduling algorithm.
Resources requested: 1/8 CPUs, 0/0 GPUs
Memory usage on this node: 4.3/16.8 GB
Result logdir: /home/kangkang/ray_results/hyperband_test
Number of trials: 2 ({'RUNNING': 1, 'PENDING': 1})
PENDING trials:
 - MyTrainableClass_245fefff_123:	PENDING
RUNNING trials:
 - MyTrainableClass_4969aace_123:	RUNNING

Result for MyTrainableClass_245fefff_123:
  date: 2019-04-21_16-54-53
  done: false
  episode_reward_mean: 0.19046719267795137
  experiment_id: 7263a08cbc554d5a9256fef77c290c34
  hostname: kangkang-1994
  iterations_since_restore: 1
  node_ip: 192.168.4.102
  pid: 21022
  time_since_restore: 5.173683166503906e-05
  time_this_iter_s: 5.173683166503906e-05
  time_total_s: 5.173683166503906e-05
  timestamp: 1555836893
  timesteps_since_restore: 0
  training_iteration: 1
  
TestLogger {'node_ip': '192.168.4.102', 'timestamp': 1555836893, 'config': {'height': 16, 'width': 84}, 'experiment_id': '7263a08cbc554d5a9256fef77c290c34', 'training_iteration': 1, 'time_total_s': 5.173683166503906e-05, 'done': False, 'iterations_since_restore': 1, 'episodes_total': None, 'time_since_restore': 5.173683166503906e-05, 'pid': 21022, 'time_this_iter_s': 5.173683166503906e-05, 'hostname': 'kangkang-1994', 'timesteps_since_restore': 0, 'date': '2019-04-21_16-54-53', 'timesteps_total': None, 'episode_reward_mean': 0.19046719267795137}
Result for MyTrainableClass_4969aace_123:
  date: 2019-04-21_16-54-53
  done: false
  episode_reward_mean: 1.7676692568367371
  experiment_id: 512eb0621175451c84ea143d724f840f
  hostname: kangkang-1994
  iterations_since_restore: 1
  node_ip: 192.168.4.102
  pid: 21015
  time_since_restore: 2.86102294921875e-05
  time_this_iter_s: 2.86102294921875e-05
  time_total_s: 2.86102294921875e-05
  timestamp: 1555836893
  timesteps_since_restore: 0
  training_iteration: 1
  
TestLogger {'node_ip': '192.168.4.102', 'timestamp': 1555836893, 'experiment_id': '512eb0621175451c84ea143d724f840f', 'pid': 21015, 'date': '2019-04-21_16-54-53', 'time_total_s': 2.86102294921875e-05, 'config': {'height': 99, 'width': 56}, 'iterations_since_restore': 1, 'episode_reward_mean': 1.7676692568367371, 'episodes_total': None, 'time_since_restore': 2.86102294921875e-05, 'time_this_iter_s': 2.86102294921875e-05, 'hostname': 'kangkang-1994', 'timesteps_since_restore': 0, 'done': False, 'timesteps_total': None, 'training_iteration': 1}
TestLogger ............
.......
.......
TestLogger {'node_ip': '192.168.4.102', 'training_iteration': 998, 'hostname': 'kangkang-1994', 'timesteps_total': None, 'time_this_iter_s': 1.3589859008789062e-05, 'iterations_since_restore': 998, 'timestamp': 1555837061, 'done': False, 'date': '2019-04-21_16-57-41', 'pid': 21247, 'time_since_restore': 0.018785715103149414, 'experiment_id': '1c364c8bf6fa43c5b0f8aae7c231b098', 'time_total_s': 0.018785715103149414, 'config': {'height': 79, 'width': 53}, 'episode_reward_mean': 78.99999999999999, 'timesteps_since_restore': 0, 'episodes_total': None}
2019-04-21 16:57:41,062	INFO ray_trial_executor.py:178 -- Destroying actor for trial MyTrainableClass_2ef074ed_123. If your trainable is slow to initialize, consider setting reuse_actors=True to reduce actor creation overheads.
Result for MyTrainableClass_2ef074ed_123:
  date: 2019-04-21_16-57-41
  done: true
  episode_reward_mean: 78.99999999999999
  experiment_id: 1c364c8bf6fa43c5b0f8aae7c231b098
  hostname: kangkang-1994
  iterations_since_restore: 999
  node_ip: 192.168.4.102
  pid: 21247
  time_since_restore: 0.018810272216796875
  time_this_iter_s: 2.4557113647460938e-05
  time_total_s: 0.018810272216796875
  timestamp: 1555837061
  timesteps_since_restore: 0
  training_iteration: 999
  
TestLogger {'node_ip': '192.168.4.102', 'training_iteration': 999, 'hostname': 'kangkang-1994', 'timesteps_total': None, 'time_this_iter_s': 2.4557113647460938e-05, 'iterations_since_restore': 999, 'timestamp': 1555837061, 'done': True, 'date': '2019-04-21_16-57-41', 'pid': 21247, 'time_since_restore': 0.018810272216796875, 'experiment_id': '1c364c8bf6fa43c5b0f8aae7c231b098', 'time_total_s': 0.018810272216796875, 'config': {'height': 79, 'width': 53}, 'episode_reward_mean': 78.99999999999999, 'timesteps_since_restore': 0, 'episodes_total': None}
== Status ==
Using FIFO scheduling algorithm.
Resources requested: 0/8 CPUs, 0/0 GPUs
Memory usage on this node: 4.5/16.8 GB
Result logdir: /home/kangkang/ray_results/hyperband_test
Number of trials: 2 ({'TERMINATED': 2})
TERMINATED trials:
 - MyTrainableClass_fb55cbbf_123:	TERMINATED, [1 CPUs, 0 GPUs], [pid=21243], 0 s, 999 iter, 51 rew
 - MyTrainableClass_2ef074ed_123:	TERMINATED, [1 CPUs, 0 GPUs], [pid=21247], 0 s, 999 iter, 79 rew


Process finished with exit code 0
```

此示例中日志主要是通过自定义一个`TestLogger(tune.logger.Logger)`，然后在实验中添加一个配置信息`loggers=[TestLogger]`,最后在控制台上打印出每个迭代试验的状态信息。即下面的所述：

```
TestLogger {'node_ip': '192.168.4.102', 'timestamp': 1555836893, 'experiment_id': '512eb0621175451c84ea143d724f840f', 'pid': 21015, 'date': '2019-04-21_16-54-53', 'time_total_s': 2.86102294921875e-05, 'config': {'height': 99, 'width': 56}, 'iterations_since_restore': 1, 'episode_reward_mean': 1.7676692568367371, 'episodes_total': None, 'time_since_restore': 2.86102294921875e-05, 'time_this_iter_s': 2.86102294921875e-05, 'hostname': 'kangkang-1994', 'timesteps_since_restore': 0, 'done': False, 'timesteps_total': None, 'training_iteration': 1}
TestLogger ............
.......
.......
TestLogger {'node_ip': '192.168.4.102', 'training_iteration': 998, 'hostname': 'kangkang-1994', 'timesteps_total': None, 'time_this_iter_s': 1.3589859008789062e-05, 'iterations_since_restore': 998, 'timestamp': 1555837061, 'done': False, 'date': '2019-04-21_16-57-41', 'pid': 21247, 'time_since_restore': 0.018785715103149414, 'experiment_id': '1c364c8bf6fa43c5b0f8aae7c231b098', 'time_total_s': 0.018785715103149414, 'config': {'height': 79, 'width': 53}, 'episode_reward_mean': 78.99999999999999, 'timesteps_since_restore': 0, 'episodes_total': None}
```

自定义可训练函数名：
首先通过可训练命名函数　`trial_str_creator(trial)`　把可训练函数转换成字符串形式，然后通过设置实验参数 `trial_name_creator=tune.function(trial_str_creator)`　完成命名。其中参数`trial_str_creator`可自定义。
本代码通过`trial_str_creator(trial)`函数把可训练函数转换为为`{训练名}_{训练ＩＤ}_123`。

### ２．使用带有AsyncHyperBandScheduler的Trainable类的示例。

 **ray/python/ray/tune/examples/async_hyperband_example.py** 
源码：

```
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

    # asynchronous hyperband early stopping, configured with
    # `episode_reward_mean` as the
    # objective and `training_iteration` as the time unit,
    # which is automatically filled by Tune.
    # time_attr 时间基元;               reward_attr　目标条件;
    # grace_period　至少的时间（次数）;   max_t 最大的时间（次数）
    # 调度算法的为异步超带　AsyncHyperBandScheduler　。默认是的先进先出（FIFO）
    ahb = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        reward_attr="episode_reward_mean",
        grace_period=5,
        max_t=100)

    run(MyTrainableClass,
        name="asynchyperband_test",
        scheduler=ahb, 	# 引用调度算
        **{
            # 停止条件　training_iteration=99999
            "stop": {
                "training_iteration": 1 if args.smoke_test else 99999
            },
            # 抽样30次,　即通过AsyncHyperBandScheduler调度　对MyTrainableClass进行30次训练。
            "num_samples": 30,
            # 机器资源　每个试验用一个CPU和GPU资源，此处cpu资源为主机cpu核心数。
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

```
结果：

```
/usr/bin/python3.5 /home/kangkang/PycharmProjects/ray/python/ray/tune/examples/async_hyperband_example.py
2019-04-21 21:59:12,417	INFO node.py:423 -- Process STDOUT and STDERR is being redirected to /tmp/ray/session_2019-04-21_21-59-12_19182/logs.
2019-04-21 21:59:12,520	INFO services.py:363 -- Waiting for redis server at 127.0.0.1:32072 to respond...
2019-04-21 21:59:12,640	INFO services.py:363 -- Waiting for redis server at 127.0.0.1:47740 to respond...
2019-04-21 21:59:12,643	INFO services.py:760 -- Starting Redis shard with 3.35 GB max memory.
2019-04-21 21:59:12,662	INFO services.py:1384 -- Starting the Plasma object store with 5.03 GB memory using /dev/shm.
2019-04-21 21:59:12,766	INFO tune.py:60 -- Tip: to resume incomplete experiments, pass resume='prompt' or resume=True to run()
2019-04-21 21:59:12,766	INFO tune.py:211 -- Starting a new experiment.
== Status ==
Using AsyncHyperBand: num_stopped=0
Bracket: Iter 45.000: None | Iter 15.000: None | Iter 5.000: None
Bracket: Iter 45.000: None | Iter 15.000: None
Bracket: Iter 45.000: None
Resources requested: 0/8 CPUs, 0/0 GPUs
Memory usage on this node: 4.3/16.8 GB

2019-04-21 21:59:14,132	WARNING util.py:62 -- The `start_trial` operation took 1.1828277111053467 seconds to complete, which may be a performance bottleneck.
== Status ==
Using AsyncHyperBand: num_stopped=0
Bracket: Iter 45.000: None | Iter 15.000: None | Iter 5.000: None
Bracket: Iter 45.000: None | Iter 15.000: None
Bracket: Iter 45.000: None
Resources requested: 1/8 CPUs, 0/0 GPUs
Memory usage on this node: 4.8/16.8 GB
Result logdir: /home/kangkang/ray_results/asynchyperband_test
Number of trials: 30 ({'RUNNING': 1, 'PENDING': 29})
PENDING trials:
 - MyTrainableClass_1_height=7,width=57:	PENDING
 - MyTrainableClass_2_height=77,width=25:	PENDING
 - MyTrainableClass_3_height=24,width=71:	PENDING
 - MyTrainableClass_4_height=4,width=19:	PENDING
 - MyTrainableClass_5_height=98,width=55:	PENDING
 - MyTrainableClass_6_height=36,width=99:	PENDING
 - MyTrainableClass_7_height=57,width=44:	PENDING
 - MyTrainableClass_8_height=94,width=99:	PENDING
 - MyTrainableClass_9_height=76,width=80:	PENDING
  ... 11 not shown
 - MyTrainableClass_21_height=7,width=76:	PENDING
 - MyTrainableClass_22_height=97,width=74:	PENDING
 - MyTrainableClass_23_height=14,width=53:	PENDING
 - MyTrainableClass_24_height=6,width=17:	PENDING
 - MyTrainableClass_25_height=69,width=21:	PENDING
 - MyTrainableClass_26_height=83,width=55:	PENDING
 - MyTrainableClass_27_height=19,width=78:	PENDING
 - MyTrainableClass_28_height=34,width=85:	PENDING
 - MyTrainableClass_29_height=35,width=57:	PENDING
RUNNING trials:
 - MyTrainableClass_0_height=54,width=90:	RUNNING

Result for MyTrainableClass_0_height=54,width=90:
  date: 2019-04-21_21-59-14
  done: false
  episode_reward_mean: 0.5999753098612407
  experiment_id: dbb12a9d42ec4107bceca3feb43d783f
  hostname: kangkang-1994
  iterations_since_restore: 1
  node_ip: 192.168.4.102
  pid: 19219
  time_since_restore: 3.0040740966796875e-05
  time_this_iter_s: 3.0040740966796875e-05
  time_total_s: 3.0040740966796875e-05
  timestamp: 1555855154
  timesteps_since_restore: 0
  training_iteration: 1
  
Result for MyTrainableClass_1_height=7,width=57:
......
......
......
2019-04-21 21:59:22,944	INFO ray_trial_executor.py:178 -- Destroying actor for trial MyTrainableClass_26_height=83,width=55. If your trainable is slow to initialize, consider setting reuse_actors=True to reduce actor creation overheads.
Result for MyTrainableClass_25_height=69,width=21:
  date: 2019-04-21_21-59-23
  done: true
  episode_reward_mean: 68.98991422160057
  experiment_id: 2554df5c37ed477fab6390a2f3a9aec5
  hostname: kangkang-1994
  iterations_since_restore: 100
  node_ip: 192.168.4.102
  pid: 19605
  time_since_restore: 0.0022301673889160156
  time_this_iter_s: 1.239776611328125e-05
  time_total_s: 0.0022301673889160156
  timestamp: 1555855163
  timesteps_since_restore: 0
  training_iteration: 100
  
2019-04-21 21:59:23,202	INFO ray_trial_executor.py:178 -- Destroying actor for trial MyTrainableClass_25_height=69,width=21. If your trainable is slow to initialize, consider setting reuse_actors=True to reduce actor creation overheads.
== Status ==
Using AsyncHyperBand: num_stopped=30
Bracket: Iter 45.000: 42.57614813705181 | Iter 15.000: 24.071527675803438 | Iter 5.000: 4.667534561014101
Bracket: Iter 45.000: 66.05974774234495 | Iter 15.000: 19.528068736719277
Bracket: Iter 45.000: 3.9304791553474514
Resources requested: 0/8 CPUs, 0/0 GPUs
Memory usage on this node: 4.7/16.8 GB
Result logdir: /home/kangkang/ray_results/asynchyperband_test
Number of trials: 30 ({'TERMINATED': 30})
TERMINATED trials:
 - MyTrainableClass_0_height=54,width=90:	TERMINATED, [1 CPUs, 0 GPUs], [pid=19219], 0 s, 100 iter, 43.4 rew
 - MyTrainableClass_1_height=7,width=57:	TERMINATED, [1 CPUs, 0 GPUs], [pid=19218], 0 s, 5 iter, 0.612 rew
 - MyTrainableClass_2_height=77,width=25:	TERMINATED, [1 CPUs, 0 GPUs], [pid=19221], 0 s, 100 iter, 76.9 rew
 - MyTrainableClass_3_height=24,width=71:	TERMINATED, [1 CPUs, 0 GPUs], [pid=19217], 0 s, 5 iter, 1.69 rew
 - MyTrainableClass_4_height=4,width=19:	TERMINATED, [1 CPUs, 0 GPUs], [pid=19215], 0 s, 100 iter, 4 rew
 - MyTrainableClass_5_height=98,width=55:	TERMINATED, [1 CPUs, 0 GPUs], [pid=19216], 0 s, 100 iter, 93 rew
 - MyTrainableClass_6_height=36,width=99:	TERMINATED, [1 CPUs, 0 GPUs], [pid=19220], 0 s, 5 iter, 1.82 rew
 - MyTrainableClass_7_height=57,width=44:	TERMINATED, [1 CPUs, 0 GPUs], [pid=19222], 0 s, 15 iter, 18.7 rew
 - MyTrainableClass_8_height=94,width=99:	TERMINATED, [1 CPUs, 0 GPUs], [pid=19365], 0 s, 100 iter, 72 rew
 - MyTrainableClass_9_height=76,width=80:	TERMINATED, [1 CPUs, 0 GPUs], [pid=19362], 0 s, 100 iter, 64.5 rew
 - MyTrainableClass_10_height=80,width=31:	TERMINATED, [1 CPUs, 0 GPUs], [pid=19364], 0 s, 100 iter, 79.7 rew
 - MyTrainableClass_11_height=24,width=27:	TERMINATED, [1 CPUs, 0 GPUs], [pid=19366], 0 s, 15 iter, 12.1 rew
 - MyTrainableClass_12_height=58,width=48:	TERMINATED, [1 CPUs, 0 GPUs], [pid=19435], 0 s, 100 iter, 56.2 rew
 - MyTrainableClass_13_height=14,width=49:	TERMINATED, [1 CPUs, 0 GPUs], [pid=19430], 0 s, 15 iter, 4.16 rew
 - MyTrainableClass_14_height=8,width=78:	TERMINATED, [1 CPUs, 0 GPUs], [pid=19374], 0 s, 15 iter, 1.52 rew
 - MyTrainableClass_15_height=80,width=60:	TERMINATED, [1 CPUs, 0 GPUs], [pid=19360], 0 s, 15 iter, 19.6 rew
 - MyTrainableClass_16_height=33,width=55:	TERMINATED, [1 CPUs, 0 GPUs], [pid=19361], 0 s, 5 iter, 2.99 rew
 - MyTrainableClass_17_height=42,width=23:	TERMINATED, [1 CPUs, 0 GPUs], [pid=19445], 0 s, 45 iter, 40.4 rew
 - MyTrainableClass_18_height=64,width=69:	TERMINATED, [1 CPUs, 0 GPUs], [pid=19363], 0 s, 5 iter, 4.63 rew
 - MyTrainableClass_19_height=15,width=83:	TERMINATED, [1 CPUs, 0 GPUs], [pid=19536], 0 s, 15 iter, 2.68 rew
 - MyTrainableClass_20_height=37,width=78:	TERMINATED, [1 CPUs, 0 GPUs], [pid=19516], 0 s, 5 iter, 2.37 rew
 - MyTrainableClass_21_height=7,width=76:	TERMINATED, [1 CPUs, 0 GPUs], [pid=19514], 0 s, 5 iter, 0.46 rew
 - MyTrainableClass_22_height=97,width=74:	TERMINATED, [1 CPUs, 0 GPUs], [pid=19515], 0 s, 45 iter, 52.7 rew
 - MyTrainableClass_23_height=14,width=53:	TERMINATED, [1 CPUs, 0 GPUs], [pid=19535], 0 s, 15 iter, 3.86 rew
 - MyTrainableClass_24_height=6,width=17:	TERMINATED, [1 CPUs, 0 GPUs], [pid=19518], 0 s, 5 iter, 1.72 rew
 - MyTrainableClass_25_height=69,width=21:	TERMINATED, [1 CPUs, 0 GPUs], [pid=19605], 0 s, 100 iter, 69 rew
 - MyTrainableClass_26_height=83,width=55:	TERMINATED, [1 CPUs, 0 GPUs], [pid=19600], 0 s, 45 iter, 55.9 rew
 - MyTrainableClass_27_height=19,width=78:	TERMINATED, [1 CPUs, 0 GPUs], [pid=19597], 0 s, 5 iter, 1.22 rew
 - MyTrainableClass_28_height=34,width=85:	TERMINATED, [1 CPUs, 0 GPUs], [pid=19594], 0 s, 5 iter, 2 rew
 - MyTrainableClass_29_height=35,width=57:	TERMINATED, [1 CPUs, 0 GPUs], [pid=19590], 0 s, 5 iter, 3.06 rew


Process finished with exit code 0

```
本示例主要是使用AsyncHyperBandScheduler调度算法进行试验训练，次调度共有五个参数`time_attr` 时间基元;               `reward_attr`　停止条件参数，即目标属性;　`grace_period`　至少的时间（时间基元次数）;   `max_t` 最大的时间（时间基元次数）;　reduction_factor (float) –用于设定减半率和用量。这只是一个无单位标量；brackets (int) –括号数量。 每个括号具有不同的减半率，由减少系数指定。更多详情请参考https://openreview.net/forum?id=S1Y7OOlRZ。
此示例中

```
 ahb = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        reward_attr="episode_reward_mean",
        grace_period=5,
        max_t=100)
```
`  time_attr="training_iteration`　以`training_iteration`为时间基元，　`  reward_attr="episode_reward_mean"`以`episode_reward_mean`为终止目标，` grace_period=5`指定一个训练至少运行５次基元，`max_t=100`指定一个训练运行基元次数不能超过100次。

结果中有一段信息:

```
Using AsyncHyperBand: num_stopped=0
Bracket: Iter 45.000: None | Iter 15.000: None | Iter 5.000: None
Bracket: Iter 45.000: None | Iter 15.000: None
Bracket: Iter 45.000: None
```
AsyncHyperBandScheduler调度算法生成３个括号，括号数是由参数`brackets`决定的默认是３个 ，迭代（此代码中是迭代数，也可以是时间值）。最小数是由参数`grace_period`决定的，最大值是由`max_t`决定的。训练后每次后边的值为括号内的均值，即目标值。

```
Using AsyncHyperBand: num_stopped=30
Bracket: Iter 45.000: 42.57614813705181 | Iter 15.000: 24.071527675803438 | Iter 5.000: 4.667534561014101
Bracket: Iter 45.000: 66.05974774234495 | Iter 15.000: 19.528068736719277
Bracket: Iter 45.000: 3.9304791553474514
```

### ３．使用具有HyperBandScheduler的Trainable类的示例
 **ray/python/ray/tune/examples/hyperband_example.py** 
源码：

```
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
    # 调度算法是HyperBandScheduler
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

```
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

结果：

```
/usr/bin/python3.5 /home/kangkang/PycharmProjects/ray/python/ray/tune/examples/hyperband_example.py
2019-04-22 10:56:09,896	INFO node.py:423 -- Process STDOUT and STDERR is being redirected to /tmp/ray/session_2019-04-22_10-56-09_30444/logs.
2019-04-22 10:56:09,999	INFO services.py:363 -- Waiting for redis server at 127.0.0.1:31625 to respond...
2019-04-22 10:56:10,119	INFO services.py:363 -- Waiting for redis server at 127.0.0.1:25220 to respond...
2019-04-22 10:56:10,120	INFO services.py:760 -- Starting Redis shard with 3.35 GB max memory.
2019-04-22 10:56:10,134	INFO services.py:1384 -- Starting the Plasma object store with 5.03 GB memory using /dev/shm.
2019-04-22 10:56:10,249	INFO tune.py:60 -- Tip: to resume incomplete experiments, pass resume='prompt' or resume=True to run()
2019-04-22 10:56:10,250	INFO tune.py:211 -- Starting a new experiment.
== Status ==
Using HyperBand: num_stopped=0 total_brackets=0
Round #0:
Resources requested: 0/8 CPUs, 0/0 GPUs
Memory usage on this node: 4.5/16.8 GB

2019-04-22 10:56:11,618	WARNING util.py:62 -- The `start_trial` operation took 1.2681245803833008 seconds to complete, which may be a performance bottleneck.
== Status ==
Using HyperBand: num_stopped=0 total_brackets=3
Round #0:
  Bracket(Max Size (n)=5, Milestone (r)=100, completed=0.0%): {PENDING: 4, RUNNING: 1} 
  Bracket(Max Size (n)=8, Milestone (r)=33, completed=0.0%): {PENDING: 8} 
  Bracket(Max Size (n)=15, Milestone (r)=11, completed=0.0%): {PENDING: 7} 
Resources requested: 1/8 CPUs, 0/0 GPUs
Memory usage on this node: 5.0/16.8 GB
Result logdir: /home/kangkang/ray_results/hyperband_test
Number of trials: 20 ({'RUNNING': 1, 'PENDING': 19})
PENDING trials:
 - MyTrainableClass_0_height=77,width=28:	PENDING
 - MyTrainableClass_1_height=62,width=47:	PENDING
 - MyTrainableClass_2_height=40,width=38:	PENDING
 - MyTrainableClass_4_height=1,width=28:	PENDING
 - MyTrainableClass_5_height=30,width=32:	PENDING
 - MyTrainableClass_6_height=10,width=49:	PENDING
 - MyTrainableClass_7_height=48,width=64:	PENDING
 - MyTrainableClass_8_height=67,width=90:	PENDING
 - MyTrainableClass_9_height=68,width=39:	PENDING
 - MyTrainableClass_10_height=56,width=13:	PENDING
 - MyTrainableClass_11_height=16,width=37:	PENDING
 - MyTrainableClass_12_height=34,width=66:	PENDING
 - MyTrainableClass_13_height=15,width=17:	PENDING
 - MyTrainableClass_14_height=2,width=18:	PENDING
 - MyTrainableClass_15_height=11,width=76:	PENDING
 - MyTrainableClass_16_height=43,width=44:	PENDING
 - MyTrainableClass_17_height=37,width=34:	PENDING
 - MyTrainableClass_18_height=89,width=62:	PENDING
 - MyTrainableClass_19_height=62,width=93:	PENDING
RUNNING trials:
 - MyTrainableClass_3_height=25,width=75:	RUNNING

Result for MyTrainableClass_0_height=77,width=28:
  date: 2019-04-22_10-56-11
  done: false
  episode_reward_mean: 2.7488313785447405
  experiment_id: f19bcff2fea14722a5cf2b85291eecd7
  hostname: kangkang-1994
  iterations_since_restore: 1
  node_ip: 192.168.4.102
  pid: 30474
  time_since_restore: 3.9577484130859375e-05
  time_this_iter_s: 3.9577484130859375e-05
  time_total_s: 3.9577484130859375e-05
  timestamp: 1555901771
  timesteps_since_restore: 0
  training_iteration: 1
  
Result for MyTrainableClass_1_height=62,width=47:
......
......
......
2019-04-22 10:56:16,247	INFO ray_trial_executor.py:178 -- Destroying actor for trial MyTrainableClass_1_height=62,width=47. If your trainable is slow to initialize, consider setting reuse_actors=True to reduce actor creation overheads.
== Status ==
Using HyperBand: num_stopped=2 total_brackets=3
Round #0:
  Bracket(Max Size (n)=5, Milestone (r)=100, completed=100.0%): {TERMINATED: 5} 
  Bracket(Max Size (n)=8, Milestone (r)=33, completed=34.2%): {PAUSED: 3, PENDING: 2, RUNNING: 3} 
  Bracket(Max Size (n)=5, Milestone (r)=44, completed=22.6%): {RUNNING: 5, TERMINATED: 2} 
Resources requested: 8/8 CPUs, 0/0 GPUs
Memory usage on this node: 5.3/16.8 GB
Result logdir: /home/kangkang/ray_results/hyperband_test
Number of trials: 20 ({'PAUSED': 3, 'TERMINATED': 7, 'RUNNING': 8, 'PENDING': 2})
PAUSED trials:
 - MyTrainableClass_8_height=67,width=90:	PAUSED, [1 CPUs, 0 GPUs], [pid=30475], 0 s, 33 iter, 23.5 rew
 - MyTrainableClass_9_height=68,width=39:	PAUSED, [1 CPUs, 0 GPUs], [pid=30477], 0 s, 33 iter, 46.9 rew
 - MyTrainableClass_12_height=34,width=66:	PAUSED, [1 CPUs, 0 GPUs], [pid=30479], 0 s, 33 iter, 15.7 rew
PENDING trials:
 - MyTrainableClass_10_height=56,width=13:	PENDING
 - MyTrainableClass_11_height=16,width=37:	PENDING
RUNNING trials:
 - MyTrainableClass_5_height=30,width=32:	RUNNING, [1 CPUs, 0 GPUs], [pid=30620], 0 s, 24 iter, 19.1 rew
 - MyTrainableClass_6_height=10,width=49:	RUNNING, [1 CPUs, 0 GPUs], [pid=30624], 0 s, 23 iter, 4.38 rew
 - MyTrainableClass_7_height=48,width=64:	RUNNING, [1 CPUs, 0 GPUs], [pid=30692], 0 s, 13 iter, 9.62 rew
 - MyTrainableClass_13_height=15,width=17:	RUNNING, [1 CPUs, 0 GPUs], [pid=30695], 0 s, 11 iter, 8.55 rew
 - MyTrainableClass_16_height=43,width=44:	RUNNING, [1 CPUs, 0 GPUs], [pid=30719], 0 s, 30 iter, 25.5 rew
 - MyTrainableClass_17_height=37,width=34:	RUNNING, [1 CPUs, 0 GPUs], [pid=30632], 0 s, 21 iter, 20.3 rew
 - MyTrainableClass_18_height=89,width=62:	RUNNING, [1 CPUs, 0 GPUs], [pid=30623], 0 s, 20 iter, 27.8 rew
 - MyTrainableClass_19_height=62,width=93:	RUNNING, [1 CPUs, 0 GPUs], [pid=30699], 0 s, 11 iter, 7.3 rew
TERMINATED trials:
 - MyTrainableClass_0_height=77,width=28:	TERMINATED, [1 CPUs, 0 GPUs], [pid=30474], 0 s, 100 iter, 76.9 rew
 - MyTrainableClass_1_height=62,width=47:	TERMINATED, [1 CPUs, 0 GPUs], [pid=30478], 0 s, 100 iter, 60.3 rew
 - MyTrainableClass_2_height=40,width=38:	TERMINATED, [1 CPUs, 0 GPUs], [pid=30481], 0 s, 100 iter, 39.6 rew
 - MyTrainableClass_3_height=25,width=75:	TERMINATED, [1 CPUs, 0 GPUs], [pid=30476], 0 s, 100 iter, 21.8 rew
 - MyTrainableClass_4_height=1,width=28:	TERMINATED, [1 CPUs, 0 GPUs], [pid=30480], 0 s, 100 iter, 0.998 rew
 - MyTrainableClass_14_height=2,width=18:	TERMINATED, [1 CPUs, 0 GPUs], [pid=30702], 0 s, 11 iter, 1.09 rew
 - MyTrainableClass_15_height=11,width=76:	TERMINATED, [1 CPUs, 0 GPUs], [pid=30621], 0 s, 11 iter, 1.58 rew
......
......
......
== Status ==
Using HyperBand: num_stopped=10 total_brackets=3
Round #0:
  Bracket(Max Size (n)=5, Milestone (r)=100, completed=100.0%): {TERMINATED: 5} 
  Bracket(Max Size (n)=3, Milestone (r)=100, completed=100.0%): {TERMINATED: 8} 
  Bracket(Max Size (n)=2, Milestone (r)=100, completed=100.0%): {TERMINATED: 7} 
Resources requested: 0/8 CPUs, 0/0 GPUs
Memory usage on this node: 5.0/16.8 GB
Result logdir: /home/kangkang/ray_results/hyperband_test
Number of trials: 20 ({'TERMINATED': 20})
TERMINATED trials:
 - MyTrainableClass_0_height=77,width=28:	TERMINATED, [1 CPUs, 0 GPUs], [pid=30474], 0 s, 100 iter, 76.9 rew
 - MyTrainableClass_1_height=62,width=47:	TERMINATED, [1 CPUs, 0 GPUs], [pid=30478], 0 s, 100 iter, 60.3 rew
 - MyTrainableClass_2_height=40,width=38:	TERMINATED, [1 CPUs, 0 GPUs], [pid=30481], 0 s, 100 iter, 39.6 rew
 - MyTrainableClass_3_height=25,width=75:	TERMINATED, [1 CPUs, 0 GPUs], [pid=30476], 0 s, 100 iter, 21.8 rew
 - MyTrainableClass_4_height=1,width=28:	TERMINATED, [1 CPUs, 0 GPUs], [pid=30480], 0 s, 100 iter, 0.998 rew
 - MyTrainableClass_5_height=30,width=32:	TERMINATED, [1 CPUs, 0 GPUs], [pid=30620], 0 s, 33 iter, 23.2 rew
 - MyTrainableClass_6_height=10,width=49:	TERMINATED, [1 CPUs, 0 GPUs], [pid=30624], 0 s, 33 iter, 5.87 rew
 - MyTrainableClass_7_height=48,width=64:	TERMINATED, [1 CPUs, 0 GPUs], [pid=30692], 0 s, 33 iter, 22.8 rew
 - MyTrainableClass_8_height=67,width=90:	TERMINATED, [1 CPUs, 0 GPUs], [pid=30801], 0 s, 100 iter, 53.9 rew
 - MyTrainableClass_9_height=68,width=39:	TERMINATED, [1 CPUs, 0 GPUs], [pid=30865], 0 s, 100 iter, 67.2 rew
 - MyTrainableClass_10_height=56,width=13:	TERMINATED, [1 CPUs, 0 GPUs], [pid=30818], 0 s, 100 iter, 56 rew
 - MyTrainableClass_11_height=16,width=37:	TERMINATED, [1 CPUs, 0 GPUs], [pid=30798], 0 s, 33 iter, 11.4 rew
 - MyTrainableClass_12_height=34,width=66:	TERMINATED, [1 CPUs, 0 GPUs], [pid=30479], 0 s, 33 iter, 15.7 rew
 - MyTrainableClass_13_height=15,width=17:	TERMINATED, [1 CPUs, 0 GPUs], [pid=30797], 0 s, 44 iter, 14.8 rew
 - MyTrainableClass_14_height=2,width=18:	TERMINATED, [1 CPUs, 0 GPUs], [pid=30702], 0 s, 11 iter, 1.09 rew
 - MyTrainableClass_15_height=11,width=76:	TERMINATED, [1 CPUs, 0 GPUs], [pid=30621], 0 s, 11 iter, 1.58 rew
 - MyTrainableClass_16_height=43,width=44:	TERMINATED, [1 CPUs, 0 GPUs], [pid=30799], 0 s, 100 iter, 42.1 rew
 - MyTrainableClass_17_height=37,width=34:	TERMINATED, [1 CPUs, 0 GPUs], [pid=30632], 0 s, 44 iter, 31.8 rew
 - MyTrainableClass_18_height=89,width=62:	TERMINATED, [1 CPUs, 0 GPUs], [pid=30810], 0 s, 100 iter, 82.2 rew
 - MyTrainableClass_19_height=62,width=93:	TERMINATED, [1 CPUs, 0 GPUs], [pid=30622], 0 s, 44 iter, 27.3 rew


Process finished with exit code 0
```
此实例代码主要是HyperBand调度算法进行训练调度。HyperBandScheduler 早期停止使用 HyperBand 优化算法的试验。它将试验放到多个容器（Round）中，每个容器中有5个括号，括号里边的试验是一次填充的。它将试验分为不同大小的括号，并定期在每个括号内提前停止性能较差的试验。要使用HyperBand ，所需要做的就是指定一个试验可以运行max_t的最大时间长度、时间单位time_attr和报告的目标值reward_attr的名称。我们根据给定的值自动确定其他超带参数的合理值。
结果中的：

```
实例１
== Status ==
Using HyperBand: num_stopped=0 total_brackets=3
Round #0:
  Bracket(Max Size (n)=5, Milestone (r)=100, completed=0.0%): {PENDING: 4, RUNNING: 1} 
  Bracket(Max Size (n)=8, Milestone (r)=33, completed=0.0%): {PENDING: 8} 
  Bracket(Max Size (n)=15, Milestone (r)=11, completed=0.0%): {PENDING: 7} 
```
其中“Max Size”表示根据 Hyperband 算法设置的待定/运行实验的最大数量。“Milestone” 表示此括号内的试验将有迭代Milestone次后停止。“Completed”表示一个近似的进度度量，有些括号，比如未填充的，不会达到100。此实验中生成了3个括号，以第一个括号为例，其中共有5个待定/运行状态的实验（4个待定和1个运行），迭代100次后有试验停止，完成有化进度为0。后边的`{PENDING: 4, RUNNING: 1}` 　是试验个数和状态。　第三个括号可以容纳15个实验，但是只填充7个。
	具体**括号个数(ｎ+1)** 是由`(5/(n+1))*3**n >= num_samples`,取ｎ最小满足此公式的整数，这个括号内可以容纳的训练个数为`(5/(n+1))*3**n`个（可以不容纳满即最后一个括号可以不容纳满）。当括号个数大于５后，ｎ=n-1，重新开始计数，如下面代码中的实例２。实例１解释：因为num_samples=30,通过公式`(5/(n+1))*3**n >= num_samples`计算得ｎ＝２，故括号是３个。
```
实例２
Using HyperBand: num_stopped=0 total_brackets=9
Round #0:
  Bracket(Max Size (n)=5, Milestone (r)=100, completed=0.0%): {PENDING: 4, RUNNING: 1} 
  Bracket(Max Size (n)=8, Milestone (r)=33, completed=0.0%): {PENDING: 8} 
  Bracket(Max Size (n)=15, Milestone (r)=11, completed=0.0%): {PENDING: 15} 
  Bracket(Max Size (n)=34, Milestone (r)=3, completed=0.0%): {PENDING: 34} 
  Bracket(Max Size (n)=81, Milestone (r)=1, completed=0.0%): {PENDING: 81} 
Round #1:
  Bracket(Max Size (n)=5, Milestone (r)=100, completed=0.0%): {PENDING: 5} 
  Bracket(Max Size (n)=8, Milestone (r)=33, completed=0.0%): {PENDING: 8} 
  Bracket(Max Size (n)=15, Milestone (r)=11, completed=0.0%): {PENDING: 15} 
  Bracket(Max Size (n)=34, Milestone (r)=3, completed=0.0%): {PENDING: 29} 
```


### ４．使用基于函数的API和HyperOptSearch（HyperOpt TPE的SearchAlgorithm包装器）优化基本函数。 还使用AsyncHyperBandScheduler。
 **ray/python/ray/tune/examples/hyperopt_example.py** 

源码：

```
"""This test checks that HyperOpt is functional.

It also checks that it is usable with a separate scheduler.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
from ray.tune import run
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest import HyperOptSearch

# 一个 function-based API 函数
def easy_objective(config, reporter):
    import time
    time.sleep(0.2)
    assert type(config["activation"]) == str, \
        "Config is incorrect: {}".format(type(config["activation"]))
    # timesteps_total　整个实验累计的时间步长.
    #　neg_mean_loss　是目标对象。　(h-14)**2 + abs(w-3)
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
    algo = HyperOptSearch(
        space,
        max_concurrent=７,
        reward_attr="neg_mean_loss",
        points_to_evaluate=current_best_params)
    # AsyncHyperBandScheduler 调度的设置　reward_attr　目标值属性。
    scheduler = AsyncHyperBandScheduler(reward_attr="neg_mean_loss")
    run(easy_objective, search_alg=algo, scheduler=scheduler, **config)

```

结果：

```
/usr/bin/python3.5 /home/kangkang/PycharmProjects/ray/python/ray/tune/examples/hyperopt_example.py
2019-04-22 20:23:16,657	INFO node.py:423 -- Process STDOUT and STDERR is being redirected to /tmp/ray/session_2019-04-22_20-23-16_8308/logs.
2019-04-22 20:23:16,760	INFO services.py:363 -- Waiting for redis server at 127.0.0.1:40977 to respond...
2019-04-22 20:23:16,882	INFO services.py:363 -- Waiting for redis server at 127.0.0.1:30259 to respond...
2019-04-22 20:23:16,884	INFO services.py:760 -- Starting Redis shard with 3.35 GB max memory.
2019-04-22 20:23:16,903	INFO services.py:1384 -- Starting the Plasma object store with 5.03 GB memory using /dev/shm.
2019-04-22 20:23:17,022	INFO tune.py:60 -- Tip: to resume incomplete experiments, pass resume='prompt' or resume=True to run()
2019-04-22 20:23:17,022	INFO tune.py:211 -- Starting a new experiment.
== Status ==
Using AsyncHyperBand: num_stopped=0
Bracket: Iter 90.000: None | Iter 30.000: None | Iter 10.000: None
Bracket: Iter 90.000: None | Iter 30.000: None
Bracket: Iter 90.000: None
Resources requested: 0/8 CPUs, 0/0 GPUs
Memory usage on this node: 4.6/16.8 GB

2019-04-22 20:23:18,345	WARNING util.py:62 -- The `start_trial` operation took 1.2118103504180908 seconds to complete, which may be a performance bottleneck.
== Status ==
Using AsyncHyperBand: num_stopped=0
Bracket: Iter 90.000: None | Iter 30.000: None | Iter 10.000: None
Bracket: Iter 90.000: None | Iter 30.000: None
Bracket: Iter 90.000: None
Resources requested: 1/8 CPUs, 0/0 GPUs
Memory usage on this node: 5.1/16.8 GB
Result logdir: /home/kangkang/ray_results/easy_objective
Number of trials: 7 ({'PENDING': 6, 'RUNNING': 1})
PENDING trials:
 - easy_objective_2_activation=relu,height=2.0,iterations=100,width=1.0:	PENDING
 - easy_objective_3_activation=tanh,height=22.039,iterations=100,width=9.738:	PENDING
 - easy_objective_4_activation=relu,height=-42.895,iterations=100,width=14.567:	PENDING
 - easy_objective_5_activation=relu,height=-97.671,iterations=100,width=9.2559:	PENDING
 - easy_objective_6_activation=relu,height=-91.837,iterations=100,width=17.422:	PENDING
 - easy_objective_7_activation=relu,height=-24.74,iterations=100,width=7.8179:	PENDING
RUNNING trials:
 - easy_objective_1_activation=tanh,height=2.0,iterations=100,width=4.0:	RUNNING

Result for easy_objective_1_activation=tanh,height=2.0,iterations=100,width=4.0:
  date: 2019-04-22_20-23-18
  done: false
  experiment_id: d8546e2d1af6421081c16d3ecf7890f8
  hostname: kangkang-1994
  iterations_since_restore: 1
  neg_mean_loss: -143.0
  node_ip: 192.168.4.102
  pid: 8346
  time_since_restore: 0.20081138610839844
  time_this_iter_s: 0.20081138610839844
  time_total_s: 0.20081138610839844
  timestamp: 1555935798
  timesteps_since_restore: 0
  timesteps_total: 0
  training_iteration: 1
  
Result for easy_objective_2_activation=relu,height=2.0,iterations=100,width=1.0:
......
......
......
== Status ==
Using AsyncHyperBand: num_stopped=15
Bracket: Iter 90.000: -113.96190914029448 | Iter 30.000: -113.96190914029448 | Iter 10.000: -274.0867642439374
Bracket: Iter 90.000: -1495.9444016686882 | Iter 30.000: -2668.405977022503
Bracket: Iter 90.000: -2035.7743775394906
Resources requested: 6/8 CPUs, 0/0 GPUs
Memory usage on this node: 5.3/16.8 GB
Result logdir: /home/kangkang/ray_results/easy_objective
Number of trials: 21 ({'RUNNING': 6, 'TERMINATED': 15})
RUNNING trials:
 - easy_objective_10_activation=tanh,height=-43.103,iterations=100,width=9.1504:	RUNNING, [1 CPUs, 0 GPUs], [pid=8505], 2 s, 79 iter, 78 ts
 - easy_objective_16_activation=tanh,height=11.77,iterations=100,width=12.675:	RUNNING, [1 CPUs, 0 GPUs], [pid=8495], 0 s, 10 iter, 9 ts
 - easy_objective_18_activation=relu,height=57.345,iterations=100,width=0.77405:	RUNNING
 - easy_objective_19_activation=relu,height=-69.197,iterations=100,width=5.766:	RUNNING
 - easy_objective_20_activation=relu,height=-85.805,iterations=100,width=12.582:	RUNNING
 - easy_objective_21_activation=relu,height=2.6658,iterations=100,width=2.1419:	RUNNING
TERMINATED trials:
 - easy_objective_1_activation=tanh,height=2.0,iterations=100,width=4.0:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8346], 3 s, 100 iter, 99 ts
 - easy_objective_2_activation=relu,height=2.0,iterations=100,width=1.0:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8350], 3 s, 100 iter, 99 ts
 - easy_objective_3_activation=tanh,height=22.039,iterations=100,width=9.738:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8349], 3 s, 100 iter, 99 ts
 - easy_objective_4_activation=relu,height=-42.895,iterations=100,width=14.567:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8344], 0 s, 10 iter, 9 ts
 - easy_objective_5_activation=relu,height=-97.671,iterations=100,width=9.2559:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8343], 0 s, 10 iter, 9 ts
 - easy_objective_6_activation=relu,height=-91.837,iterations=100,width=17.422:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8347], 1 s, 30 iter, 29 ts
 - easy_objective_7_activation=relu,height=-24.74,iterations=100,width=7.8179:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8348], 3 s, 100 iter, 99 ts
  ... 1 not shown
 - easy_objective_9_activation=relu,height=31.004,iterations=100,width=18.059:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8492], 0 s, 10 iter, 9 ts
 - easy_objective_11_activation=relu,height=-62.645,iterations=100,width=9.9918:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8506], 0 s, 10 iter, 9 ts
 - easy_objective_12_activation=tanh,height=-42.239,iterations=100,width=0.57173:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8563], 0 s, 10 iter, 9 ts
 - easy_objective_13_activation=tanh,height=82.56,iterations=100,width=16.152:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8507], 0 s, 10 iter, 9 ts
 - easy_objective_14_activation=relu,height=-61.933,iterations=100,width=10.29:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8494], 0 s, 10 iter, 9 ts
 - easy_objective_15_activation=relu,height=-99.896,iterations=100,width=8.0056:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8504], 0 s, 10 iter, 9 ts
 - easy_objective_17_activation=relu,height=-77.97,iterations=100,width=4.3232:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8493], 0 s, 10 iter, 9 ts
......
......
......
== Status ==
Using AsyncHyperBand: num_stopped=100
Bracket: Iter 90.000: -16.017010699846495 | Iter 30.000: -122.40616806732639 | Iter 10.000: -1262.8869120182433
Bracket: Iter 90.000: -127.60673438392044 | Iter 30.000: -558.7796649878293
Bracket: Iter 90.000: -1196.9048713540942
Resources requested: 0/8 CPUs, 0/0 GPUs
Memory usage on this node: 5.1/16.8 GB
Result logdir: /home/kangkang/ray_results/easy_objective
Number of trials: 100 ({'TERMINATED': 100})
TERMINATED trials:
 - easy_objective_1_activation=tanh,height=2.0,iterations=100,width=4.0:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8346], 3 s, 100 iter, 99 ts
 - easy_objective_2_activation=relu,height=2.0,iterations=100,width=1.0:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8350], 3 s, 100 iter, 99 ts
 - easy_objective_3_activation=tanh,height=22.039,iterations=100,width=9.738:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8349], 3 s, 100 iter, 99 ts
 - easy_objective_4_activation=relu,height=-42.895,iterations=100,width=14.567:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8344], 0 s, 10 iter, 9 ts
 - easy_objective_5_activation=relu,height=-97.671,iterations=100,width=9.2559:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8343], 0 s, 10 iter, 9 ts
 - easy_objective_6_activation=relu,height=-91.837,iterations=100,width=17.422:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8347], 1 s, 30 iter, 29 ts
 - easy_objective_7_activation=relu,height=-24.74,iterations=100,width=7.8179:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8348], 3 s, 100 iter, 99 ts
 - easy_objective_8_activation=relu,height=59.236,iterations=100,width=13.549:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8345], 3 s, 100 iter, 99 ts
 - easy_objective_9_activation=relu,height=31.004,iterations=100,width=18.059:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8492], 0 s, 10 iter, 9 ts
 - easy_objective_10_activation=tanh,height=-43.103,iterations=100,width=9.1504:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8505], 3 s, 90 iter, 89 ts
 - easy_objective_11_activation=relu,height=-62.645,iterations=100,width=9.9918:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8506], 0 s, 10 iter, 9 ts
 - easy_objective_12_activation=tanh,height=-42.239,iterations=100,width=0.57173:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8563], 0 s, 10 iter, 9 ts
 - easy_objective_13_activation=tanh,height=82.56,iterations=100,width=16.152:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8507], 0 s, 10 iter, 9 ts
 - easy_objective_14_activation=relu,height=-61.933,iterations=100,width=10.29:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8494], 0 s, 10 iter, 9 ts
 - easy_objective_15_activation=relu,height=-99.896,iterations=100,width=8.0056:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8504], 0 s, 10 iter, 9 ts
 - easy_objective_16_activation=tanh,height=11.77,iterations=100,width=12.675:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8495], 4 s, 100 iter, 99 ts
 - easy_objective_17_activation=relu,height=-77.97,iterations=100,width=4.3232:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8493], 0 s, 10 iter, 9 ts
 - easy_objective_18_activation=relu,height=57.345,iterations=100,width=0.77405:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8632], 0 s, 10 iter, 9 ts
 - easy_objective_19_activation=relu,height=-69.197,iterations=100,width=5.766:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8653], 0 s, 10 iter, 9 ts
 - easy_objective_20_activation=relu,height=-85.805,iterations=100,width=12.582:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8634], 1 s, 30 iter, 29 ts
 - easy_objective_21_activation=relu,height=2.6658,iterations=100,width=2.1419:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8633], 4 s, 100 iter, 99 ts
 - easy_objective_22_activation=tanh,height=-62.539,iterations=100,width=15.516:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8635], 0 s, 10 iter, 9 ts
 - easy_objective_23_activation=relu,height=-54.651,iterations=100,width=19.78:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8644], 0 s, 10 iter, 9 ts
 - easy_objective_24_activation=relu,height=31.777,iterations=100,width=4.8282:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8713], 1 s, 30 iter, 29 ts
 - easy_objective_25_activation=tanh,height=87.221,iterations=100,width=13.24:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8707], 1 s, 30 iter, 29 ts
 - easy_objective_26_activation=tanh,height=-26.199,iterations=100,width=15.387:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8715], 0 s, 10 iter, 9 ts
 - easy_objective_27_activation=tanh,height=28.882,iterations=100,width=10.723:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8709], 1 s, 30 iter, 29 ts
 - easy_objective_28_activation=tanh,height=-11.364,iterations=100,width=15.472:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8718], 4 s, 100 iter, 99 ts
 - easy_objective_29_activation=relu,height=-76.104,iterations=100,width=0.29346:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8636], 0 s, 10 iter, 9 ts
 - easy_objective_30_activation=tanh,height=18.403,iterations=100,width=19.242:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8645], 4 s, 100 iter, 99 ts
 - easy_objective_31_activation=relu,height=98.954,iterations=100,width=17.299:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8827], 0 s, 10 iter, 9 ts
 - easy_objective_32_activation=tanh,height=-71.268,iterations=100,width=2.9988:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8828], 0 s, 10 iter, 9 ts
 - easy_objective_33_activation=tanh,height=85.757,iterations=100,width=4.5031:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8846], 0 s, 10 iter, 9 ts
 - easy_objective_34_activation=tanh,height=-93.095,iterations=100,width=12.713:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8826], 0 s, 10 iter, 9 ts
 - easy_objective_35_activation=relu,height=-41.012,iterations=100,width=9.2997:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8901], 1 s, 30 iter, 29 ts
 - easy_objective_36_activation=relu,height=41.72,iterations=100,width=12.577:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8845], 4 s, 100 iter, 99 ts
 - easy_objective_37_activation=relu,height=23.447,iterations=100,width=0.037218:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8844], 4 s, 100 iter, 99 ts
 - easy_objective_38_activation=tanh,height=-15.482,iterations=100,width=17.64:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8847], 1 s, 30 iter, 29 ts
 - easy_objective_39_activation=relu,height=28.733,iterations=100,width=11.508:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8836], 1 s, 30 iter, 29 ts
 - easy_objective_40_activation=relu,height=41.587,iterations=100,width=7.7579:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8965], 4 s, 90 iter, 89 ts
 - easy_objective_41_activation=relu,height=-74.526,iterations=100,width=4.5107:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8966], 0 s, 10 iter, 9 ts
 - easy_objective_42_activation=tanh,height=-10.126,iterations=100,width=5.4023:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9034], 1 s, 30 iter, 29 ts
 - easy_objective_43_activation=tanh,height=-29.027,iterations=100,width=5.7968:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8967], 4 s, 90 iter, 89 ts
 - easy_objective_44_activation=tanh,height=-20.773,iterations=100,width=15.253:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8986], 5 s, 100 iter, 99 ts
 - easy_objective_45_activation=relu,height=-37.923,iterations=100,width=19.04:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8984], 4 s, 90 iter, 89 ts
 - easy_objective_46_activation=relu,height=-55.618,iterations=100,width=15.684:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8975], 2 s, 30 iter, 29 ts
 - easy_objective_47_activation=relu,height=-81.815,iterations=100,width=13.326:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8983], 0 s, 10 iter, 9 ts
 - easy_objective_48_activation=relu,height=-16.775,iterations=100,width=9.9054:	TERMINATED, [1 CPUs, 0 GPUs], [pid=8985], 0 s, 10 iter, 9 ts
 - easy_objective_49_activation=relu,height=-16.656,iterations=100,width=17.078:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9100], 5 s, 90 iter, 89 ts
 - easy_objective_50_activation=tanh,height=-12.626,iterations=100,width=0.56666:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9114], 1 s, 30 iter, 29 ts
 - easy_objective_51_activation=relu,height=-59.508,iterations=100,width=7.9994:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9113], 0 s, 10 iter, 9 ts
 - easy_objective_52_activation=relu,height=-82.268,iterations=100,width=15.803:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9101], 0 s, 10 iter, 9 ts
 - easy_objective_53_activation=tanh,height=-42.702,iterations=100,width=7.8688:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9179], 5 s, 90 iter, 89 ts
 - easy_objective_54_activation=relu,height=82.824,iterations=100,width=2.4541:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9175], 1 s, 30 iter, 29 ts
 - easy_objective_55_activation=tanh,height=-60.813,iterations=100,width=7.381:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9169], 1 s, 30 iter, 29 ts
 - easy_objective_56_activation=relu,height=30.025,iterations=100,width=4.0291:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9110], 2 s, 30 iter, 29 ts
 - easy_objective_57_activation=tanh,height=-77.924,iterations=100,width=18.636:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9109], 0 s, 10 iter, 9 ts
 - easy_objective_58_activation=relu,height=-23.747,iterations=100,width=2.4056:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9111], 0 s, 10 iter, 9 ts
 - easy_objective_59_activation=tanh,height=-70.226,iterations=100,width=11.475:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9112], 5 s, 90 iter, 89 ts
 - easy_objective_60_activation=tanh,height=-83.233,iterations=100,width=5.4006:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9264], 1 s, 30 iter, 29 ts
 - easy_objective_61_activation=relu,height=60.864,iterations=100,width=11.823:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9274], 0 s, 10 iter, 9 ts
 - easy_objective_62_activation=relu,height=4.3718,iterations=100,width=7.933:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9272], 6 s, 100 iter, 99 ts
 - easy_objective_63_activation=tanh,height=43.548,iterations=100,width=11.334:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9285], 1 s, 30 iter, 29 ts
 - easy_objective_64_activation=tanh,height=-11.167,iterations=100,width=9.0313:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9342], 5 s, 90 iter, 89 ts
 - easy_objective_65_activation=relu,height=97.036,iterations=100,width=7.3837:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9344], 0 s, 10 iter, 9 ts
 - easy_objective_66_activation=relu,height=-89.846,iterations=100,width=9.3844:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9336], 0 s, 10 iter, 9 ts
 - easy_objective_67_activation=relu,height=28.911,iterations=100,width=11.531:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9273], 6 s, 100 iter, 99 ts
 - easy_objective_68_activation=relu,height=34.76,iterations=100,width=12.296:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9277], 5 s, 90 iter, 89 ts
 - easy_objective_69_activation=relu,height=5.4938,iterations=100,width=13.39:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9276], 7 s, 100 iter, 99 ts
 - easy_objective_70_activation=tanh,height=-55.228,iterations=100,width=5.8144:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9275], 0 s, 10 iter, 9 ts
 - easy_objective_71_activation=relu,height=-61.194,iterations=100,width=3.1344:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9430], 0 s, 10 iter, 9 ts
 - easy_objective_72_activation=tanh,height=44.231,iterations=100,width=6.6059:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9462], 1 s, 30 iter, 29 ts
 - easy_objective_73_activation=tanh,height=4.857,iterations=100,width=3.3751:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9499], 6 s, 90 iter, 89 ts
 - easy_objective_74_activation=tanh,height=-16.763,iterations=100,width=4.1779:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9439], 2 s, 30 iter, 29 ts
 - easy_objective_75_activation=tanh,height=58.71,iterations=100,width=2.951:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9442], 2 s, 30 iter, 29 ts
 - easy_objective_76_activation=tanh,height=-95.809,iterations=100,width=15.238:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9441], 7 s, 90 iter, 89 ts
 - easy_objective_77_activation=relu,height=-3.6177,iterations=100,width=4.676:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9465], 8 s, 100 iter, 99 ts
 - easy_objective_78_activation=tanh,height=79.994,iterations=100,width=18.282:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9440], 1 s, 10 iter, 9 ts
 - easy_objective_79_activation=relu,height=-76.902,iterations=100,width=0.071296:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9438], 0 s, 10 iter, 9 ts
 - easy_objective_80_activation=tanh,height=-60.449,iterations=100,width=17.423:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9564], 0 s, 10 iter, 9 ts
 - easy_objective_81_activation=relu,height=-2.0585,iterations=100,width=6.5233:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9600], 1 s, 30 iter, 29 ts
 - easy_objective_82_activation=relu,height=85.786,iterations=100,width=2.2052:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9592], 2 s, 30 iter, 29 ts
 - easy_objective_83_activation=relu,height=-89.93,iterations=100,width=7.9265:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9631], 0 s, 10 iter, 9 ts
 - easy_objective_84_activation=tanh,height=15.722,iterations=100,width=17.204:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9565], 7 s, 100 iter, 99 ts
 - easy_objective_85_activation=tanh,height=-38.696,iterations=100,width=6.4104:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9591], 1 s, 10 iter, 9 ts
 - easy_objective_86_activation=relu,height=20.501,iterations=100,width=11.198:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9583], 8 s, 100 iter, 99 ts
 - easy_objective_87_activation=relu,height=-90.48,iterations=100,width=16.536:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9575], 1 s, 10 iter, 9 ts
 - easy_objective_88_activation=relu,height=67.222,iterations=100,width=4.1735:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9569], 1 s, 30 iter, 29 ts
 - easy_objective_89_activation=relu,height=21.282,iterations=100,width=14.251:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9700], 6 s, 90 iter, 89 ts
 - easy_objective_90_activation=relu,height=30.697,iterations=100,width=19.791:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9721], 7 s, 90 iter, 89 ts
 - easy_objective_91_activation=relu,height=-72.994,iterations=100,width=11.6:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9701], 0 s, 10 iter, 9 ts
 - easy_objective_92_activation=relu,height=99.173,iterations=100,width=2.224:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9741], 0 s, 10 iter, 9 ts
 - easy_objective_93_activation=tanh,height=66.591,iterations=100,width=7.9354:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9717], 0 s, 10 iter, 9 ts
 - easy_objective_94_activation=relu,height=2.8147,iterations=100,width=6.0621:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9718], 6 s, 100 iter, 99 ts
 - easy_objective_95_activation=relu,height=-68.509,iterations=100,width=15.28:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9704], 3 s, 30 iter, 29 ts
 - easy_objective_96_activation=relu,height=59.767,iterations=100,width=15.689:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9769], 0 s, 10 iter, 9 ts
 - easy_objective_97_activation=tanh,height=-46.522,iterations=100,width=5.7094:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9699], 1 s, 10 iter, 9 ts
 - easy_objective_98_activation=relu,height=43.732,iterations=100,width=3.6313:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9837], 1 s, 30 iter, 29 ts
 - easy_objective_99_activation=relu,height=9.6149,iterations=100,width=7.876:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9845], 3 s, 100 iter, 99 ts
 - easy_objective_100_activation=relu,height=2.3878,iterations=100,width=5.025:	TERMINATED, [1 CPUs, 0 GPUs], [pid=9863], 3 s, 90 iter, 89 ts


Process finished with exit code 0
```
本示例主要基础训练模型用的是function-based API 函数，通过AsyncHyperBandScheduler调度算法调度和搜索算法HyperOpt进行参数优化。function-based API 函数技术示例中的`easy_objective(config, reporter)`函数模型，此函数模型和Class-baesd API 类模型功能相同，前者训练模型不能进行检查点存储，后者可以进行检查点存储和读取。
搜索算法HyperOpt是一种由HyperOpt支持的搜索算法，故使用前需要安装hyperopt模块（`pip install hyperopt`）,此搜索算法主要有四个参数，
 - space (dict) –HyperOpt配置。参数将从该配置中取样，并用于覆盖在变量生成过程中生成的参数。
 - max_concurrent (int) –最大同时试验次数。默认为10。 	
 - reward_attr (str) –   训练结果目标值属性。这指的是一个递增的值。 	
 - points_to_evaluate (list)   –运行前初始参数suggestions。这适用于当您已经有了一些好的参数时，您希望hyperopt首先运行，以帮助TPE算法为未来的参数提供更好的suggestions。需要是超选项命名变量的dict列表。选择变量应该由列表中的索引表示。
本示例中space 中用了两个激励函数`'activation': hp.choice("activation", ["relu", "tanh"])`进行参数优化， `max_concurrent=7,`注意max_concurrent参数是同事并行个数，不能超过及其资源总数。
注意此代码中的调度算法`AsyncHyperBandScheduler(reward_attr="neg_mean_loss")`　的目标对象`neg_mean_loss`是在function-based API 函数中认为设置的不是tune默认的输出标签。

### ５．使用带有PopulationBasedTraining调度程序的Trainable类的示例。
 **ray/python/ray/tune/examples/pbt_example.py** 

```
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
        ray.init()
    # time_attr 时间基元,  reward_attr 目标属性
    #　perturbation_interval　扰动间隔, hyperparam_mutations 样本空间
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

    # Try to find the best factor 1 and factor 2
    run(MyTrainableClass,
        name="pbt_test",
        scheduler=pbt,
        # reuse_actors (bool) –是否在不同的试验之间重用actor。这可以极大地加快经常启动
        # 和停止参与者(actors)的实验(例如，在时间复用模式下的PBT)。这要求试验具有相同的资源需求。
        reuse_actors=True,
        # verbose   为0时, 控制台不打印实验的状态和结果
        #           为1时, 控制台只打印状态
        #           为2时, 控制台打印状态和结果
        verbose=0,
        **{
            "stop": {
                "training_iteration": 20 if args.smoke_test else 100
            },
            "num_samples": 10,
            "config": {
                "factor_1": 4.0,
                "factor_2": 1.0,
            },
        })

```
结果：

```
/usr/bin/python3.5 /home/kangkang/PycharmProjects/ray/python/ray/tune/examples/pbt_example.py
2019-04-23 21:13:05,643	INFO node.py:423 -- Process STDOUT and STDERR is being redirected to /tmp/ray/session_2019-04-23_21-13-05_19856/logs.
2019-04-23 21:13:05,746	INFO services.py:363 -- Waiting for redis server at 127.0.0.1:13566 to respond...
2019-04-23 21:13:05,868	INFO services.py:363 -- Waiting for redis server at 127.0.0.1:15165 to respond...
2019-04-23 21:13:05,869	INFO services.py:760 -- Starting Redis shard with 3.35 GB max memory.
2019-04-23 21:13:05,883	INFO services.py:1384 -- Starting the Plasma object store with 5.03 GB memory using /dev/shm.
2019-04-23 21:13:05,962	INFO tune.py:60 -- Tip: to resume incomplete experiments, pass resume='prompt' or resume=True to run()
2019-04-23 21:13:05,963	INFO tune.py:211 -- Starting a new experiment.
2019-04-23 21:13:07,375	WARNING util.py:62 -- The `start_trial` operation took 1.372983694076538 seconds to complete, which may be a performance bottleneck.
2019-04-23 21:13:08,821	INFO pbt.py:81 -- [explore] perturbed config from {'factor_2': 1.0, 'factor_1': 4.0} -> {'factor_2': 1, 'factor_1': 4.8}
2019-04-23 21:13:08,821	INFO pbt.py:230 -- [exploit] transferring weights from trial MyTrainableClass_0 (score 11.266456744839445) -> MyTrainableClass_5 (score 3.720406248567462)
2019-04-23 21:13:10,859	INFO pbt.py:81 -- [explore] perturbed config from {'factor_2': 1.0, 'factor_1': 4.0} -> {'factor_2': 2, 'factor_1': 3.2}
2019-04-23 21:13:10,859	INFO pbt.py:230 -- [exploit] transferring weights from trial MyTrainableClass_3 (score 24.863309484263233) -> MyTrainableClass_9 (score 9.706690342388455)
2019-04-23 21:13:10,873	INFO pbt.py:81 -- [explore] perturbed config from {'factor_2': 1.0, 'factor_1': 4.0} -> {'factor_2': 2, 'factor_1': 11.97176005813976}
2019-04-23 21:13:10,873	INFO pbt.py:230 -- [exploit] transferring weights from trial MyTrainableClass_3 (score 24.863309484263233) -> MyTrainableClass_8 (score 7.875286178654381)
2019-04-23 21:13:12,091	INFO pbt.py:81 -- [explore] perturbed config from {'factor_2': 1.0, 'factor_1': 4.0} -> {'factor_2': 1, 'factor_1': 3.2}
2019-04-23 21:13:12,092	INFO pbt.py:230 -- [exploit] transferring weights from trial MyTrainableClass_3 (score 40.44066879698752) -> MyTrainableClass_7 (score 21.43433066954473)
2019-04-23 21:13:13,048	INFO pbt.py:81 -- [explore] perturbed config from {'factor_2': 1.0, 'factor_1': 4.0} -> {'factor_2': 1, 'factor_1': 3.2}
2019-04-23 21:13:13,048	INFO pbt.py:230 -- [exploit] transferring weights from trial MyTrainableClass_3 (score 53.99122596246579) -> MyTrainableClass_4 (score 33.75459630359976)
2019-04-23 21:13:14,089	INFO pbt.py:81 -- [explore] perturbed config from {'factor_2': 1.0, 'factor_1': 4.0} -> {'factor_2': 1, 'factor_1': 3.2}
2019-04-23 21:13:14,089	INFO pbt.py:230 -- [exploit] transferring weights from trial MyTrainableClass_0 (score 63.96900207095729) -> MyTrainableClass_2 (score 43.60729315734055)
2019-04-23 21:13:15,249	INFO pbt.py:81 -- [explore] perturbed config from {'factor_2': 2, 'factor_1': 11.97176005813976} -> {'factor_2': 2, 'factor_1': 14.366112069767711}
2019-04-23 21:13:15,249	INFO pbt.py:230 -- [exploit] transferring weights from trial MyTrainableClass_8@perturbed[factor_1=11.972,factor_2=2] (score 133.78819909630343) -> MyTrainableClass_5@perturbed[factor_1=4.8,factor_2=1] (score 57.1389666408087)
2019-04-23 21:13:17,292	INFO pbt.py:81 -- [explore] perturbed config from {'factor_2': 2, 'factor_1': 3.2} -> {'factor_2': 2, 'factor_1': 3.84}
2019-04-23 21:13:17,293	INFO pbt.py:230 -- [exploit] transferring weights from trial MyTrainableClass_9@perturbed[factor_1=3.2,factor_2=2] (score 104.63047422773649) -> MyTrainableClass_7@perturbed[factor_1=3.2,factor_2=1] (score 81.15242003458089)
2019-04-23 21:13:18,164	INFO pbt.py:81 -- [explore] perturbed config from {'factor_2': 2, 'factor_1': 14.366112069767711} -> {'factor_2': 2, 'factor_1': 11.492889655814169}
2019-04-23 21:13:18,164	INFO pbt.py:230 -- [exploit] transferring weights from trial MyTrainableClass_5@perturbed[factor_1=14.366,factor_2=2] (score 181.0357966877706) -> MyTrainableClass_1 (score 76.88064621865908)
2019-04-23 21:13:19,407	INFO pbt.py:81 -- [explore] perturbed config from {'factor_2': 2, 'factor_1': 14.366112069767711} -> {'factor_2': 1, 'factor_1': 17.23933448372125}
2019-04-23 21:13:19,407	INFO pbt.py:230 -- [exploit] transferring weights from trial MyTrainableClass_5@perturbed[factor_1=14.366,factor_2=2] (score 195.41526413397534) -> MyTrainableClass_4@perturbed[factor_1=3.2,factor_2=1] (score 102.64373177056822)
2019-04-23 21:13:20,247	INFO pbt.py:81 -- [explore] perturbed config from {'factor_2': 2, 'factor_1': 14.366112069767711} -> {'factor_2': 1, 'factor_1': 19.275903394473413}
2019-04-23 21:13:20,247	INFO pbt.py:230 -- [exploit] transferring weights from trial MyTrainableClass_5@perturbed[factor_1=14.366,factor_2=2] (score 211.50598372809048) -> MyTrainableClass_6 (score 99.01680285641353)
2019-04-23 21:13:21,304	INFO ray_trial_executor.py:178 -- Destroying actor for trial MyTrainableClass_3. If your trainable is slow to initialize, consider setting reuse_actors=True to reduce actor creation overheads.
2019-04-23 21:13:21,413	INFO ray_trial_executor.py:178 -- Destroying actor for trial MyTrainableClass_5@perturbed[factor_1=14.366,factor_2=2]. If your trainable is slow to initialize, consider setting reuse_actors=True to reduce actor creation overheads.
2019-04-23 21:13:21,491	INFO ray_trial_executor.py:178 -- Destroying actor for trial MyTrainableClass_6@perturbed[factor_1=19.276,factor_2=1]. If your trainable is slow to initialize, consider setting reuse_actors=True to reduce actor creation overheads.
2019-04-23 21:13:21,499	INFO ray_trial_executor.py:178 -- Destroying actor for trial MyTrainableClass_8@perturbed[factor_1=11.972,factor_2=2]. If your trainable is slow to initialize, consider setting reuse_actors=True to reduce actor creation overheads.
2019-04-23 21:13:21,503	INFO ray_trial_executor.py:178 -- Destroying actor for trial MyTrainableClass_4@perturbed[factor_1=17.239,factor_2=1]. If your trainable is slow to initialize, consider setting reuse_actors=True to reduce actor creation overheads.
2019-04-23 21:13:21,509	INFO ray_trial_executor.py:178 -- Destroying actor for trial MyTrainableClass_9@perturbed[factor_1=3.2,factor_2=2]. If your trainable is slow to initialize, consider setting reuse_actors=True to reduce actor creation overheads.
2019-04-23 21:13:21,520	INFO ray_trial_executor.py:178 -- Destroying actor for trial MyTrainableClass_7@perturbed[factor_1=3.84,factor_2=2]. If your trainable is slow to initialize, consider setting reuse_actors=True to reduce actor creation overheads.

Process finished with exit code 0

 
```

本示例主要是用PopulationBasedTraining (PBT)调度进行训练。PBT并行地训练一组模型(或代理)。性能较差的模型会周期性地克隆性能最好的模型的状态，并对它们的超参数进行随机突变，希望能够超过当前性能最好的模型。如果试验的数量超过了集群的能力，将对试验进行时间复用，以平衡整个实验的训练进展。
PopulationBasedTraining 调度算法共有一下几个参数：
```
time_attr (str) –训练结果用的比较时间。注意，您可以传递一些非时间的东西，比如training_iteration
	作为进度的度量，惟一的要求是属性应该单调地增加。
reward_attr (str) – 训练结果目标值属性。与time_attr一样，这可以引用任何客观值。停止过程将使用	
	此属性。
perturbation_interval (float) – 模型将考虑在这个 time_attr时间间隔内的扰动。注意，扰动会导致检查
	点开销，因此不应该将此设置为太频繁。
hyperparam_mutations (dict) –Hyperparams变异。格式如下:对于每个键，可以提供列表或函数。列表
	指定一组允许的分类值。函数指定连续参数的分布。您必须指定至少一个hyperparam_variables或
	custom_explore_fn。
resample_probability (float) – 当应用hyperparam_mutation时，从原始分布重新采样的概率。如果不
	进行重采样，则连续的值会受到1.2或0.8倍的扰动，离散的值则会变为相邻值。
custom_explore_fn (func) –您还可以指定一个定制的探测函数。在应用了hyperparam_variables的内
	置扰动之后，这个函数被调用为f(config)，并且应该返回根据需要更新的配置。您必须指定至少一个
	hyperparam_variables或custom_explore_fn。

```
本示例中`perturbation_interval＝10`,是每１０个  time_attr（即training_iteration），进行一次参数克隆，即低性能的实验克隆一次高性能的实验。`{'factor_2': 1.0, 'factor_1': 4.0} 克隆 {'factor_2': 1, 'factor_1': 4.8}`　，试验`MyTrainableClass_0(score 11.266456744839445)`克隆`MyTrainableClass_5 (score 3.720406248567462)`，控制台显示如下：

```
2019-04-23 21:13:08,821	INFO pbt.py:81 -- [explore] perturbed config from {'factor_2': 1.0, 'factor_1': 4.0} -> {'factor_2': 1, 'factor_1': 4.8}
2019-04-23 21:13:08,821	INFO pbt.py:230 -- [exploit] transferring weights from trial MyTrainableClass_0 (score 11.266456744839445) -> MyTrainableClass_5 (score 3.720406248567462)
```
### ６．使用PopulationBasedTraining调度程序优化分布式RLlib算法（PPO）的示例
 **ray/python/ray/tune/examples/pbt_ppo_example.py** 

源码：

```
#!/usr/bin/env python
"""Example of using PBT with RLlib.

Note that this requires a cluster with at least 8 GPUs in order for all trials
to run concurrently, otherwise PBT will round-robin train the trials which
is less efficient (or you can set {"gpu": 0} to use CPUs for SGD instead).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import ray
from ray.tune import run, sample_from
from ray.tune.schedulers import PopulationBasedTraining

if __name__ == "__main__":

    # Postprocess the perturbed config to ensure it's still valid
    def explore(config):
        # ensure we collect enough timesteps to do sgd
        if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
            config["train_batch_size"] = config["sgd_minibatch_size"] * 2
        # ensure we run at least one sgd iter
        if config["num_sgd_iter"] < 1:
            config["num_sgd_iter"] = 1
        return config

    pbt = PopulationBasedTraining(
        time_attr="time_total_s",
        reward_attr="episode_reward_mean",
        perturbation_interval=120,
        resample_probability=0.25,
        # Specifies the mutations of these hyperparams
        hyperparam_mutations={
            "lambda": lambda: random.uniform(0.9, 1.0),
            "clip_param": lambda: random.uniform(0.01, 0.5),
            "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            "num_sgd_iter": lambda: random.randint(1, 30),
            "sgd_minibatch_size": lambda: random.randint(128, 16384),
            "train_batch_size": lambda: random.randint(2000, 160000),
        },
        custom_explore_fn=explore)

    ray.init()
    run(
        "PPO",
        name="pbt_humanoid_test",
        scheduler=pbt,
        **{
            "env": "Humanoid-v1",
            "num_samples": 8,
            "config": {
                "kl_coeff": 1.0,
                "num_workers": 8,
                # "num_gpus": 1,
                "num_gpus": 8,
                "model": {
                    "free_log_std": True
                },
                # These params are tuned from a fixed starting value.
                "lambda": 0.95,
                "clip_param": 0.2,
                "lr": 1e-4,
                # These params start off randomly drawn from a set.
                "num_sgd_iter": sample_from(
                    lambda spec: random.choice([10, 20, 30])),
                "sgd_minibatch_size": sample_from(
                    lambda spec: random.choice([128, 512, 2048])),
                "train_batch_size": sample_from(
                    lambda spec: random.choice([10000, 20000, 40000]))
            },
        })

```

代码中暂无环境函数故不能运行。

主要是用PopulationBasedTraining调度来运行强化学习的PPO算法。



##  二、Keras 示例

### tune_mnist_keras
转换Keras MNIST示例以使用Tune与基于函数的API和Keras回调，另外展示了命令行(argparse)调用tune.
代码：

```
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler

# keras 的回调函数
#    on_epoch_begin: 在每轮开始时被调用。
#    on_epoch_end: 在每轮结束时被调用。
#    on_batch_begin: 在每批开始时被调用。
#    on_batch_end: 在每批结束时被调用。
#    on_train_begin: 在模型训练开始时被调用。
#    on_train_end: 在模型训练结束时被调用。

class TuneCallback(keras.callbacks.Callback):
    def __init__(self, reporter, logs={}):
        self.reporter = reporter
        self.iteration = 0

    def on_train_end(self, epoch, logs={}):
        self.reporter(
            timesteps_total=self.iteration, done=1, mean_accuracy=logs["acc"])

    def on_batch_end(self, batch, logs={}):
        self.iteration += 1
        self.reporter(
            timesteps_total=self.iteration, mean_accuracy=logs["acc"])

# 基于功能的函数（ the function-based API）
def train_mnist(args, cfg, reporter):
    # We set threads here to avoid contention, as Keras
    # is heavily parallelized across multiple cores.
    # 我们在这里设置线程以避免争用，因为Keras在多个核心上大量并行化。
    K.set_session(
        K.tf.Session(
            config=K.tf.ConfigProto(
                intra_op_parallelism_threads=args.threads,
                inter_op_parallelism_threads=args.threads)))
    vars(args).update(cfg)
    batch_size = 128
    num_classes = 10
    epochs = 12

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    # 将类向量转换为二进制类矩阵
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(
        Conv2D(
            32,
            kernel_size=(args.kernel1, args.kernel1),
            activation='relu',
            input_shape=input_shape))
    model.add(Conv2D(64, (args.kernel2, args.kernel2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(args.poolsize, args.poolsize)))
    model.add(Dropout(args.dropout1))
    model.add(Flatten())
    model.add(Dense(args.hidden, activation='relu'))
    model.add(Dropout(args.dropout2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.SGD(lr=args.lr, momentum=args.momentum),
        metrics=['accuracy'])

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        validation_data=(x_test, y_test),
        callbacks=[TuneCallback(reporter)])


def create_parser():
    parser = argparse.ArgumentParser(description='Keras MNIST Example')
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    parser.add_argument(
        "--use-gpu", action="store_true", help="Use GPU in training.")
    parser.add_argument(
        '--jobs',
        type=int,
        default=1,
        help='number of jobs to run concurrently (default: 1)')
    parser.add_argument(
        '--threads',
        type=int,
        default=2,
        help='threads used in operations (default: 2)')
    parser.add_argument(
        '--steps',
        type=float,
        default=0.01,
        metavar='LR',
        help='learning rate (default: 0.01)')
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        metavar='LR',
        help='learning rate (default: 0.01)')
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.5,
        metavar='M',
        help='SGD momentum (default: 0.5)')
    parser.add_argument(
        '--kernel1',
        type=int,
        default=3,
        help='Size of first kernel (default: 3)')
    parser.add_argument(
        '--kernel2',
        type=int,
        default=3,
        help='Size of second kernel (default: 3)')
    parser.add_argument(
        '--poolsize', type=int, default=2, help='Size of Pooling (default: 2)')
    parser.add_argument(
        '--dropout1',
        type=float,
        default=0.25,
        help='Size of first kernel (default: 0.25)')
    parser.add_argument(
        '--hidden',
        type=int,
        default=128,
        help='Size of Hidden Layer (default: 128)')
    parser.add_argument(
        '--dropout2',
        type=float,
        default=0.5,
        help='Size of first kernel (default: 0.5)')
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    mnist.load_data()  # we do this because it's not threadsafe

    ray.init()
    # time_attr 时间基元;               reward_attr　目标属性;
    # grace_period　至少的时间（次数）;   max_t 最大的时间（次数）;
    sched = AsyncHyperBandScheduler(
        time_attr="timesteps_total",
        reward_attr="mean_accuracy",
        max_t=400,
        grace_period=20)

    # 基于功能的函数（ the function-based API）进行注册
    tune.register_trainable(
        "TRAIN_FN",
        lambda config, reporter: train_mnist(args, config, reporter))
    tune.run(
        "TRAIN_FN",
        name="exp",
        scheduler=sched,
        **{
            "stop": {
                "mean_accuracy": 0.99,
                "timesteps_total": 10 if args.smoke_test else 300
            },
            "num_samples": 1 if args.smoke_test else 10,
            "resources_per_trial": {
                "cpu": args.threads,
                "gpu": 0.5 if args.use_gpu else 0
            },
            # 训练的参数空间
            "config": {
                "lr": tune.sample_from(
                    lambda spec: np.random.uniform(0.001, 0.1)),
                "momentum": tune.sample_from(
                    lambda spec: np.random.uniform(0.1, 0.9)),
                "hidden": tune.sample_from(
                    lambda spec: np.random.randint(32, 512)),
                "dropout1": tune.sample_from(
                    lambda spec: np.random.uniform(0.2, 0.8)),
            }
        })

```
结果：

```
/usr/bin/python3.5 /home/kangkang/PycharmProjects/ray/python/ray/tune/examples/tune_mnist_keras.py
Using TensorFlow backend.
2019-04-24 15:17:33,018	INFO node.py:423 -- Process STDOUT and STDERR is being redirected to /tmp/ray/session_2019-04-24_15-17-33_10366/logs.
2019-04-24 15:17:33,124	INFO services.py:363 -- Waiting for redis server at 127.0.0.1:40302 to respond...
2019-04-24 15:17:33,249	INFO services.py:363 -- Waiting for redis server at 127.0.0.1:37945 to respond...
2019-04-24 15:17:33,250	INFO services.py:760 -- Starting Redis shard with 3.35 GB max memory.
2019-04-24 15:17:33,275	INFO services.py:1384 -- Starting the Plasma object store with 5.03 GB memory using /dev/shm.
2019-04-24 15:17:33,420	INFO tune.py:60 -- Tip: to resume incomplete experiments, pass resume='prompt' or resume=True to run()
2019-04-24 15:17:33,420	INFO tune.py:211 -- Starting a new experiment.
== Status ==
Using AsyncHyperBand: num_stopped=0
Bracket: Iter 180.000: None | Iter 60.000: None | Iter 20.000: None
Bracket: Iter 180.000: None | Iter 60.000: None
Bracket: Iter 180.000: None
Resources requested: 0/8 CPUs, 0/0 GPUs
Memory usage on this node: 5.2/16.8 GB

2019-04-24 15:17:34,065	WARNING util.py:62 -- The `start_trial` operation took 0.582388162612915 seconds to complete, which may be a performance bottleneck.
== Status ==
Using AsyncHyperBand: num_stopped=0
Bracket: Iter 180.000: None | Iter 60.000: None | Iter 20.000: None
Bracket: Iter 180.000: None | Iter 60.000: None
Bracket: Iter 180.000: None
Resources requested: 2/8 CPUs, 0/0 GPUs
Memory usage on this node: 5.5/16.8 GB
Result logdir: /home/kangkang/ray_results/exp
Number of trials: 10 ({'RUNNING': 1, 'PENDING': 9})
PENDING trials:
 - TRAIN_FN_1_dropout1=0.70157,hidden=355,lr=0.015667,momentum=0.64191:	PENDING
 - TRAIN_FN_2_dropout1=0.53226,hidden=386,lr=0.089037,momentum=0.51701:	PENDING
 - TRAIN_FN_3_dropout1=0.67211,hidden=353,lr=0.072741,momentum=0.38509:	PENDING
 - TRAIN_FN_4_dropout1=0.41722,hidden=388,lr=0.035141,momentum=0.60679:	PENDING
 - TRAIN_FN_5_dropout1=0.3269,hidden=152,lr=0.080855,momentum=0.37858:	PENDING
 - TRAIN_FN_6_dropout1=0.56712,hidden=325,lr=0.067953,momentum=0.73784:	PENDING
 - TRAIN_FN_7_dropout1=0.38755,hidden=292,lr=0.068198,momentum=0.35942:	PENDING
 - TRAIN_FN_8_dropout1=0.59125,hidden=336,lr=0.011336,momentum=0.28552:	PENDING
 - TRAIN_FN_9_dropout1=0.30037,hidden=278,lr=0.012657,momentum=0.2521:	PENDING
RUNNING trials:
 - TRAIN_FN_0_dropout1=0.51614,hidden=79,lr=0.020046,momentum=0.37763:	RUNNING

(pid=10401) Using TensorFlow backend.
(pid=10406) Using TensorFlow backend.
(pid=10402) Using TensorFlow backend.
(pid=10407) Using TensorFlow backend.
(pid=10401) 2019-04-24 15:17:35.537444: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
(pid=10401) 2019-04-24 15:17:35.542481: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 3591380000 Hz
(pid=10401) 2019-04-24 15:17:35.542832: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x7fad8434a360 executing computations on platform Host. Devices:
(pid=10401) 2019-04-24 15:17:35.542852: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): <undefined>, <undefined>
(pid=10406) 2019-04-24 15:17:35.576839: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
(pid=10406) 2019-04-24 15:17:35.580205: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 3591380000 Hz
(pid=10406) 2019-04-24 15:17:35.580545: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x7f9d4c34a360 executing computations on platform Host. Devices:
(pid=10406) 2019-04-24 15:17:35.580567: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): <undefined>, <undefined>
(pid=10402) 2019-04-24 15:17:35.606809: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
(pid=10402) 2019-04-24 15:17:35.610057: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 3591380000 Hz
(pid=10402) 2019-04-24 15:17:35.610417: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x7f946834a360 executing computations on platform Host. Devices:
(pid=10402) 2019-04-24 15:17:35.610438: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): <undefined>, <undefined>
(pid=10407) 2019-04-24 15:17:35.616004: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
(pid=10407) 2019-04-24 15:17:35.619619: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 3591380000 Hz
(pid=10407) 2019-04-24 15:17:35.619809: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x7f4aa034a360 executing computations on platform Host. Devices:
(pid=10407) 2019-04-24 15:17:35.619823: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): <undefined>, <undefined>
(pid=10401) WARNING:tensorflow:From /usr/local/lib/python3.5/dist-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
(pid=10401) Instructions for updating:
(pid=10401) Colocations handled automatically by placer.
(pid=10401) x_train shape: (60000, 28, 28, 1)
(pid=10401) 60000 train samples
(pid=10401) 10000 test samples
......
......
......
  
== Status ==
Using AsyncHyperBand: num_stopped=1
Bracket: Iter 180.000: None | Iter 60.000: 0.8072916666666667 | Iter 20.000: 0.5859375
Bracket: Iter 180.000: None | Iter 60.000: 0.734375
Bracket: Iter 180.000: None
Resources requested: 8/8 CPUs, 0/0 GPUs
Memory usage on this node: 8.4/16.8 GB
Result logdir: /home/kangkang/ray_results/exp
Number of trials: 10 ({'TERMINATED': 1, 'RUNNING': 4, 'PENDING': 5})
PENDING trials:
 - TRAIN_FN_5_dropout1=0.49894,hidden=123,lr=0.042594,momentum=0.26115:	PENDING
 - TRAIN_FN_6_dropout1=0.23476,hidden=308,lr=0.028059,momentum=0.32336:	PENDING
 - TRAIN_FN_7_dropout1=0.6067,hidden=443,lr=0.097536,momentum=0.41231:	PENDING
 - TRAIN_FN_8_dropout1=0.31372,hidden=330,lr=0.082998,momentum=0.45492:	PENDING
 - TRAIN_FN_9_dropout1=0.64405,hidden=259,lr=0.021086,momentum=0.17926:	PENDING
RUNNING trials:
 - TRAIN_FN_1_dropout1=0.58749,hidden=161,lr=0.030451,momentum=0.59908:	RUNNING, [2 CPUs, 0 GPUs], [pid=13970], 34 s, 66 iter, 66 ts, 0.828 acc
 - TRAIN_FN_2_dropout1=0.60399,hidden=188,lr=0.058388,momentum=0.61773:	RUNNING, [2 CPUs, 0 GPUs], [pid=13967], 34 s, 63 iter, 63 ts, 0.812 acc
 - TRAIN_FN_3_dropout1=0.46442,hidden=51,lr=0.067125,momentum=0.68395:	RUNNING, [2 CPUs, 0 GPUs], [pid=13971], 34 s, 68 iter, 68 ts, 0.766 acc
 - TRAIN_FN_4_dropout1=0.52176,hidden=145,lr=0.084363,momentum=0.73998:	RUNNING, [2 CPUs, 0 GPUs], [pid=14086], 21 s, 34 iter, 34 ts, 0.703 acc
TERMINATED trials:
 - TRAIN_FN_0_dropout1=0.32133,hidden=309,lr=0.074049,momentum=0.13397:	TERMINATED, [2 CPUs, 0 GPUs], [pid=13964], 10 s, 20 iter, 20 ts, 0.531 acc

Result for TRAIN_FN_4_dropout1=0.52176,hidden=145,lr=0.084363,momentum=0.73998:
  date: 2019-04-24_16-08-32
  done: false
  experiment_id: 8e7c986d9133454e90466962e175b4fa
  hostname: kangkang-1994
  iterations_since_restore: 38
  mean_accuracy: 0.8125
  node_ip: 192.168.4.102
  pid: 14086
  time_since_restore: 23.817898988723755
  time_this_iter_s: 0.6790781021118164
  time_total_s: 23.817898988723755
  timestamp: 1556093312
  timesteps_since_restore: 0
  timesteps_total: 38
  training_iteration: 38
  
Result for TRAIN_FN_3_dropout1=0.46442,hidden=51,lr=0.067125,momentum=0.68395:
  date: 2019-04-24_16-08-33
  done: false
  experiment_id: c367c5c2a8a1452cb211a9d07a60e23b
  hostname: kangkang-1994
  iterations_since_restore: 74
  mean_accuracy: 0.8046875
  node_ip: 192.168.4.102
  pid: 13971
  time_since_restore: 37.91959357261658
  time_this_iter_s: 0.8096158504486084
  time_total_s: 37.91959357261658
  timestamp: 1556093313
  timesteps_since_restore: 0
  timesteps_total: 74
  training_iteration: 74
  ......
  ......
  ......
  Result for TRAIN_FN_7_dropout1=0.6067,hidden=443,lr=0.097536,momentum=0.41231:
  date: 2019-04-24_16-12-06
  done: false
  experiment_id: 6cb68d0bd8b841febd1b7b69b16c338e
  hostname: kangkang-1994
  iterations_since_restore: 293
  mean_accuracy: 0.953125
  node_ip: 192.168.4.102
  pid: 14080
  time_since_restore: 90.88926076889038
  time_this_iter_s: 0.2632167339324951
  time_total_s: 90.88926076889038
  timestamp: 1556093526
  timesteps_since_restore: 0
  timesteps_total: 293
  training_iteration: 293
  
Result for TRAIN_FN_7_dropout1=0.6067,hidden=443,lr=0.097536,momentum=0.41231:
2019-04-24 16:12:08,435	INFO ray_trial_executor.py:178 -- Destroying actor for trial TRAIN_FN_7_dropout1=0.6067,hidden=443,lr=0.097536,momentum=0.41231. If your trainable is slow to initialize, consider setting reuse_actors=True to reduce actor creation overheads.
  date: 2019-04-24_16-12-08
  done: true
  experiment_id: 6cb68d0bd8b841febd1b7b69b16c338e
  hostname: kangkang-1994
  iterations_since_restore: 300
  mean_accuracy: 0.96875
  node_ip: 192.168.4.102
  pid: 14080
  time_since_restore: 92.59368801116943
  time_this_iter_s: 0.23879218101501465
  time_total_s: 92.59368801116943
  timestamp: 1556093528
  timesteps_since_restore: 0
  timesteps_total: 300
  training_iteration: 300
  
== Status ==
Using AsyncHyperBand: num_stopped=5
Bracket: Iter 180.000: 0.9244791666666667 | Iter 60.000: 0.8359375 | Iter 20.000: 0.6848958333333334
Bracket: Iter 180.000: 0.890625 | Iter 60.000: 0.7578125
Bracket: Iter 180.000: None
Resources requested: 0/8 CPUs, 0/0 GPUs
Memory usage on this node: 6.1/16.8 GB
Result logdir: /home/kangkang/ray_results/exp
Number of trials: 10 ({'TERMINATED': 10})
TERMINATED trials:
 - TRAIN_FN_0_dropout1=0.32133,hidden=309,lr=0.074049,momentum=0.13397:	TERMINATED, [2 CPUs, 0 GPUs], [pid=13964], 10 s, 20 iter, 20 ts, 0.531 acc
 - TRAIN_FN_1_dropout1=0.58749,hidden=161,lr=0.030451,momentum=0.59908:	TERMINATED, [2 CPUs, 0 GPUs], [pid=13970], 155 s, 300 iter, 300 ts, 0.883 acc
 - TRAIN_FN_2_dropout1=0.60399,hidden=188,lr=0.058388,momentum=0.61773:	TERMINATED, [2 CPUs, 0 GPUs], [pid=13967], 158 s, 300 iter, 300 ts, 0.945 acc
 - TRAIN_FN_3_dropout1=0.46442,hidden=51,lr=0.067125,momentum=0.68395:	TERMINATED, [2 CPUs, 0 GPUs], [pid=13971], 148 s, 300 iter, 300 ts, 0.875 acc
 - TRAIN_FN_4_dropout1=0.52176,hidden=145,lr=0.084363,momentum=0.73998:	TERMINATED, [2 CPUs, 0 GPUs], [pid=14086], 155 s, 300 iter, 300 ts, 0.945 acc
 - TRAIN_FN_5_dropout1=0.49894,hidden=123,lr=0.042594,momentum=0.26115:	TERMINATED, [2 CPUs, 0 GPUs], [pid=14088], 11 s, 20 iter, 20 ts, 0.406 acc
 - TRAIN_FN_6_dropout1=0.23476,hidden=308,lr=0.028059,momentum=0.32336:	TERMINATED, [2 CPUs, 0 GPUs], [pid=14078], 31 s, 60 iter, 60 ts, 0.758 acc
 - TRAIN_FN_7_dropout1=0.6067,hidden=443,lr=0.097536,momentum=0.41231:	TERMINATED, [2 CPUs, 0 GPUs], [pid=14080], 92 s, 300 iter, 300 ts, 0.969 acc
 - TRAIN_FN_8_dropout1=0.31372,hidden=330,lr=0.082998,momentum=0.45492:	TERMINATED, [2 CPUs, 0 GPUs], [pid=13965], 31 s, 60 iter, 60 ts, 0.836 acc
 - TRAIN_FN_9_dropout1=0.64405,hidden=259,lr=0.021086,momentum=0.17926:	TERMINATED, [2 CPUs, 0 GPUs], [pid=13969], 28 s, 60 iter, 60 ts, 0.484 acc


Process finished with exit code 0
```
此代码主要用的是基于功能的函数( the function-based API )作为训练基础，其中用到了keras(库)及回调函数，另外用到基于tune的AsyncHyperBandScheduler调度算法，（此调度前边已经介绍过，此处不再介绍）还有argparse的参数解析。此处主要根据代码及实验结果简单介绍一下基于功能的函数( the function-based API )函数、回调函数和argparse的命令行解析。
本示例中的基于功能的函数是 `train_mnist(args, cfg, reporter)`　，参数的报告主要通过调用回调函数`TuneCallback(keras.callbacks.Callback)`进行报告，运行是要对功能函数进行注册`tune.register_trainable( "TRAIN_FN",     lambda config, reporter: train_mnist(args, config, reporter))` 其中` "TRAIN_FN"`是功能函数的名,`config, reporter`　为 `train_mnist(args, config, reporter)`函数的参数。此处的config主要是训练的参数配置信息，reporter是指要报告的信息（提供给tune实现处理如终止训练），最后在 tune.run(  "TRAIN_FN",  name="exp",...)运行实验。
keras中回调函数主要有六个函数：
 - on_epoch_begin: 在每轮开始时被调用。
 -  on_epoch_end: 在每轮结束时被调用。  
 - on_batch_begin:   在每批开始时被调用。
 -  on_batch_end: 在每批结束时被调用。     
 -  on_train_begin: 在模型训练开始时被调用。         
 -  on_train_end: 在模型训练结束时被调用。
此代码中的回调函数主要是报告训练的状态信息。
argparse的命令行解析，主要是argparse库的应该。本代码通过创建`create_parser()`进行命令行解析。
主要例子而格式如下：
```
def create_parser():
    parser = argparse.ArgumentParser(description='解析描述')
    ＃  parser.add_argument("--smoke-test", action="store_true", help="Finish quickly for testing")
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    parser.add_argument(
        "--use-gpu", action="store_true", help="Use GPU in training.")
    parser.add_argument(
        '--jobs',
        type=int,
        default=1,
        help='number of jobs to run concurrently (default: 1)')
    ......
 return parser
        
```
此代码中主要用的是`add_argument（）`进行参数添加。以

    parser.add_argument(
            "--smoke-test", action="store_true", help="Finish quickly for testing")

进行解释。　 `"--smoke-test"`　是自定义名；`help`是命令注释；`action="store_true"`　运行对应的命令时为真，不运行为假。例如在命令行中运行：

```
root@kangkang-1994:/home/kangkang# python3 /home/kangkang/PycharmProjects/ray/python/ray/tune/examples/tune_mnist_keras.py --smoke-test
```
此时`smoke-test`为真代码中  `"num_samples": 1 if args.smoke_test else 10`　的"num_samples＝１．
help的作用主要是在命令行中对运行命令进行解析，例如：运行 `python3 /home/kangkang/PycharmProjects/ray/python/ray/tune/examples/tune_mnist_keras.py -h`　命令，其中最后应该是跟 `--help`或者`-h`

```
root@kangkang-1994:/home/kangkang# python3 /home/kangkang/PycharmProjects/ray/python/ray/tune/examples/tune_mnist_keras.py -h
Using TensorFlow backend.
usage: tune_mnist_keras.py [-h] [--smoke-test] [--use-gpu] [--jobs JOBS]
                           [--threads THREADS] [--steps LR] [--lr LR]
                           [--momentum M] [--kernel1 KERNEL1]
                           [--kernel2 KERNEL2] [--poolsize POOLSIZE]
                           [--dropout1 DROPOUT1] [--hidden HIDDEN]
                           [--dropout2 DROPOUT2]

Keras MNIST Example

optional arguments:
  -h, --help           show this help message and exit
  --smoke-test         Finish quickly for testing
  --use-gpu            Use GPU in training.
  --jobs JOBS          number of jobs to run concurrently (default: 1)
  --threads THREADS    threads used in operations (default: 2)
  --steps LR           learning rate (default: 0.01)
  --lr LR              learning rate (default: 0.01)
  --momentum M         SGD momentum (default: 0.5)
  --kernel1 KERNEL1    Size of first kernel (default: 3)
  --kernel2 KERNEL2    Size of second kernel (default: 3)
  --poolsize POOLSIZE  Size of Pooling (default: 2)
  --dropout1 DROPOUT1  Size of first kernel (default: 0.25)
  --hidden HIDDEN      Size of Hidden Layer (default: 128)
  --dropout2 DROPOUT2  Size of first kernel (default: 0.5)
root@kangkang-1994:/home/kangkang# python3 /home/kangkang/PycharmProjects/ray/python/ray/tune/examples/tune_mnist_keras.py -help
Using TensorFlow backend.
usage: tune_mnist_keras.py [-h] [--smoke-test] [--use-gpu] [--jobs JOBS]
                           [--threads THREADS] [--steps LR] [--lr LR]
                           [--momentum M] [--kernel1 KERNEL1]
                           [--kernel2 KERNEL2] [--poolsize POOLSIZE]
                           [--dropout1 DROPOUT1] [--hidden HIDDEN]
                           [--dropout2 DROPOUT2]
tune_mnist_keras.py: error: argument -h/--help: ignored explicit argument 'elp'
root@kangkang-1994:/home/kangkang# python3 /home/kangkang/PycharmProjects/ray/python/ray/tune/examples/tune_mnist_keras.py --help
Using TensorFlow backend.
usage: tune_mnist_keras.py [-h] [--smoke-test] [--use-gpu] [--jobs JOBS]
                           [--threads THREADS] [--steps LR] [--lr LR]
                           [--momentum M] [--kernel1 KERNEL1]
                           [--kernel2 KERNEL2] [--poolsize POOLSIZE]
                           [--dropout1 DROPOUT1] [--hidden HIDDEN]
                           [--dropout2 DROPOUT2]

Keras MNIST Example

optional arguments:
  -h, --help           show this help message and exit
  --smoke-test         Finish quickly for testing
  --use-gpu            Use GPU in training.
  --jobs JOBS          number of jobs to run concurrently (default: 1)
  --threads THREADS    threads used in operations (default: 2)
  --steps LR           learning rate (default: 0.01)
  --lr LR              learning rate (default: 0.01)
  --momentum M         SGD momentum (default: 0.5)
  --kernel1 KERNEL1    Size of first kernel (default: 3)
  --kernel2 KERNEL2    Size of second kernel (default: 3)
  --poolsize POOLSIZE  Size of Pooling (default: 2)
  --dropout1 DROPOUT1  Size of first kernel (default: 0.25)
  --hidden HIDDEN      Size of Hidden Layer (default: 128)
  --dropout2 DROPOUT2  Size of first kernel (default: 0.5)

```

## 三、PyTorch　实例

### mnist_pytorch
转换PyTorch MNIST示例以使用Tune和基于函数的API。另外展示了argparse命令行解析来使用Tune。
代码：

```
# Original Code here:
# https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument(
    '--batch-size',
    type=int,
    default=64,
    metavar='N',
    help='input batch size for training (default: 64)')
parser.add_argument(
    '--test-batch-size',
    type=int,
    default=1000,
    metavar='N',
    help='input batch size for testing (default: 1000)')
parser.add_argument(
    '--epochs',
    type=int,
    default=1,
    metavar='N',
    help='number of epochs to train (default: 1)')
parser.add_argument(
    '--lr',
    type=float,
    default=0.01,
    metavar='LR',
    help='learning rate (default: 0.01)')
parser.add_argument(
    '--momentum',
    type=float,
    default=0.5,
    metavar='M',
    help='SGD momentum (default: 0.5)')
parser.add_argument(
    '--no-cuda',
    action='store_true',
    default=False,
    help='disables CUDA training')
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)')
parser.add_argument(
    '--smoke-test', action="store_true", help="Finish quickly for testing")


def train_mnist(args, config, reporter):
    vars(args).update(config)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    #为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.manual_seed(args.seed)
    # 为当前GPU设置随机种子；如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子。
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '~/data',
            train=True,
            download=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))
            ])),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '~/data',
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))
            ])),
        batch_size=args.test_batch_size,
        shuffle=True,
        **kwargs)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

    model = Net()
    if args.cuda:
        model.cuda()

    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum)

    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

    def test():
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                # sum up batch loss
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                # get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(
                    target.data.view_as(pred)).long().cpu().sum()

        test_loss = test_loss / len(test_loader.dataset)
        accuracy = correct.item() / len(test_loader.dataset)
        reporter(mean_loss=test_loss, mean_accuracy=accuracy)

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test()


if __name__ == "__main__":
    datasets.MNIST('~/data', train=True, download=True)
    args = parser.parse_args()

    import numpy as np
    import ray
    from ray import tune
    from ray.tune.schedulers import AsyncHyperBandScheduler

    ray.init()
    # time_attr 时间基元;               reward_attr　目标属性;
    # grace_period　至少的时间（次数）;   max_t 最大的时间（次数）;
    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        reward_attr="neg_mean_loss",
        max_t=400,
        grace_period=20)
    # 基于功能的函数（ the function-based API）进行注册
    tune.register_trainable(
        "TRAIN_FN",
        lambda config, reporter: train_mnist(args, config, reporter))
    tune.run(
        "TRAIN_FN",
        name="exp",
        scheduler=sched,
        **{
            "stop": {
                "mean_accuracy": 0.98,
                "training_iteration": 1 if args.smoke_test else 20
            },
            "resources_per_trial": {
                "cpu": 3,
                # "gpu": int(not args.no_cuda)
            },
            "num_samples": 1 if args.smoke_test else 10,
            "config": {
                "lr": tune.sample_from(
                    lambda spec: np.random.uniform(0.001, 0.1)),
                "momentum": tune.sample_from(
                    lambda spec: np.random.uniform(0.1, 0.9)),
            }
        })

```
运行结果（使用cpu进行实验）：

```
/usr/bin/python3.5 /home/kangkang/PycharmProjects/ray/python/ray/tune/examples/mnist_pytorch.py
2019-04-24 19:37:11,892	INFO node.py:423 -- Process STDOUT and STDERR is being redirected to /tmp/ray/session_2019-04-24_19-37-11_5077/logs.
2019-04-24 19:37:11,997	INFO services.py:363 -- Waiting for redis server at 127.0.0.1:37534 to respond...
2019-04-24 19:37:12,113	INFO services.py:363 -- Waiting for redis server at 127.0.0.1:59931 to respond...
2019-04-24 19:37:12,114	INFO services.py:760 -- Starting Redis shard with 3.35 GB max memory.
2019-04-24 19:37:12,133	INFO services.py:1384 -- Starting the Plasma object store with 5.03 GB memory using /dev/shm.
2019-04-24 19:37:12,280	INFO tune.py:60 -- Tip: to resume incomplete experiments, pass resume='prompt' or resume=True to run()
2019-04-24 19:37:12,281	INFO tune.py:211 -- Starting a new experiment.
== Status ==
Using AsyncHyperBand: num_stopped=0
Bracket: Iter 180.000: None | Iter 60.000: None | Iter 20.000: None
Bracket: Iter 180.000: None | Iter 60.000: None
Bracket: Iter 180.000: None
Resources requested: 0/8 CPUs, 0/0 GPUs
Memory usage on this node: 5.5/16.8 GB

2019-04-24 19:37:13,646	WARNING util.py:62 -- The `start_trial` operation took 1.2476763725280762 seconds to complete, which may be a performance bottleneck.
== Status ==
Using AsyncHyperBand: num_stopped=0
Bracket: Iter 180.000: None | Iter 60.000: None | Iter 20.000: None
Bracket: Iter 180.000: None | Iter 60.000: None
Bracket: Iter 180.000: None
Resources requested: 3/8 CPUs, 0/0 GPUs
Memory usage on this node: 5.7/16.8 GB
Result logdir: /home/kangkang/ray_results/exp
Number of trials: 10 ({'RUNNING': 1, 'PENDING': 9})
PENDING trials:
 - TRAIN_FN_1_lr=0.022625,momentum=0.34345:	PENDING
 - TRAIN_FN_2_lr=0.065667,momentum=0.17842:	PENDING
 - TRAIN_FN_3_lr=0.0046286,momentum=0.82255:	PENDING
 - TRAIN_FN_4_lr=0.05494,momentum=0.88637:	PENDING
 - TRAIN_FN_5_lr=0.037683,momentum=0.49114:	PENDING
 - TRAIN_FN_6_lr=0.0049163,momentum=0.13329:	PENDING
 - TRAIN_FN_7_lr=0.012478,momentum=0.11843:	PENDING
 - TRAIN_FN_8_lr=0.032357,momentum=0.61504:	PENDING
 - TRAIN_FN_9_lr=0.079978,momentum=0.83846:	PENDING
RUNNING trials:
 - TRAIN_FN_0_lr=0.043764,momentum=0.67148:	RUNNING

Result for TRAIN_FN_0_lr=0.043764,momentum=0.67148:
  date: 2019-04-24_19-37-37
  done: false
  experiment_id: 12e59bae093942da87a144b50257e54f
  hostname: kangkang-1994
  iterations_since_restore: 1
  mean_accuracy: 0.9713
  mean_loss: 0.09200096435546876
  neg_mean_loss: -0.09200096435546876
  node_ip: 192.168.4.102
  pid: 5110
  time_since_restore: 23.326541423797607
  time_this_iter_s: 23.326541423797607
  time_total_s: 23.326541423797607
  timestamp: 1556105857
  timesteps_since_restore: 0
  training_iteration: 1
  ,,,,,,
  ,,,,,,
  ,,,,,,
    
2019-04-24 19:39:12,712	INFO ray_trial_executor.py:178 -- Destroying actor for trial TRAIN_FN_9_lr=0.079978,momentum=0.83846. If your trainable is slow to initialize, consider setting reuse_actors=True to reduce actor creation overheads.
2019-04-24 19:39:12,715	INFO ray_trial_executor.py:178 -- Destroying actor for trial TRAIN_FN_8_lr=0.032357,momentum=0.61504. If your trainable is slow to initialize, consider setting reuse_actors=True to reduce actor creation overheads.
== Status ==
Using AsyncHyperBand: num_stopped=0
Bracket: Iter 180.000: None | Iter 60.000: None | Iter 20.000: None
Bracket: Iter 180.000: None | Iter 60.000: None
Bracket: Iter 180.000: None
Resources requested: 0/8 CPUs, 0/0 GPUs
Memory usage on this node: 5.7/16.8 GB
Result logdir: /home/kangkang/ray_results/exp
Number of trials: 10 ({'TERMINATED': 10})
TERMINATED trials:
 - TRAIN_FN_0_lr=0.043764,momentum=0.67148:	TERMINATED, [3 CPUs, 0 GPUs], [pid=5110], 23 s, 1 iter, 0.092 loss, 0.971 acc
 - TRAIN_FN_1_lr=0.022625,momentum=0.34345:	TERMINATED, [3 CPUs, 0 GPUs], [pid=5147], 23 s, 1 iter, 0.152 loss, 0.953 acc
 - TRAIN_FN_2_lr=0.065667,momentum=0.17842:	TERMINATED, [3 CPUs, 0 GPUs], [pid=5151], 24 s, 1 iter, 0.103 loss, 0.969 acc
 - TRAIN_FN_3_lr=0.0046286,momentum=0.82255:	TERMINATED, [3 CPUs, 0 GPUs], [pid=5148], 24 s, 1 iter, 0.181 loss, 0.946 acc
 - TRAIN_FN_4_lr=0.05494,momentum=0.88637:	TERMINATED, [3 CPUs, 0 GPUs], [pid=5152], 23 s, 1 iter, 0.246 loss, 0.929 acc
 - TRAIN_FN_5_lr=0.037683,momentum=0.49114:	TERMINATED, [3 CPUs, 0 GPUs], [pid=5112], 23 s, 1 iter, 0.103 loss, 0.968 acc
 - TRAIN_FN_6_lr=0.0049163,momentum=0.13329:	TERMINATED, [3 CPUs, 0 GPUs], [pid=5144], 22 s, 1 iter, 0.509 loss, 0.868 acc
 - TRAIN_FN_7_lr=0.012478,momentum=0.11843:	TERMINATED, [3 CPUs, 0 GPUs], [pid=5114], 22 s, 1 iter, 0.254 loss, 0.926 acc
 - TRAIN_FN_8_lr=0.032357,momentum=0.61504:	TERMINATED, [3 CPUs, 0 GPUs], [pid=4513], 22 s, 1 iter, 0.1 loss, 0.969 acc
 - TRAIN_FN_9_lr=0.079978,momentum=0.83846:	TERMINATED, [3 CPUs, 0 GPUs], [pid=4531], 22 s, 1 iter, 0.184 loss, 0.947 acc


Process finished with exit code 0
```



### mnist_pytorch_trainable
**转换PyTorch MNIST示例以使用Tune和Trainable API。 还使用HyperBandScheduler并在最后检查模型。**
源码：

```
# Original Code here:
# https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from ray.tune import Trainable

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument(
    '--batch-size',
    type=int,
    default=64,
    metavar='N',
    help='input batch size for training (default: 64)')
parser.add_argument(
    '--test-batch-size',
    type=int,
    default=1000,
    metavar='N',
    help='input batch size for testing (default: 1000)')
parser.add_argument(
    '--epochs',
    type=int,
    default=1,
    metavar='N',
    help='number of epochs to train (default: 1)')
parser.add_argument(
    '--lr',
    type=float,
    default=0.01,
    metavar='LR',
    help='learning rate (default: 0.01)')
parser.add_argument(
    '--momentum',
    type=float,
    default=0.5,
    metavar='M',
    help='SGD momentum (default: 0.5)')
parser.add_argument(
    '--no-cuda',
    action='store_true',
    default=False,
    help='disables CUDA training')
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)')
parser.add_argument(
    '--smoke-test', action="store_true", help="Finish quickly for testing")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

＃ Trainable API 类
class TrainMNIST(Trainable):
    def _setup(self, config):
        args = config.pop("args")
        vars(args).update(config)
        args.cuda = not args.no_cuda and torch.cuda.is_available()

        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                '~/data',
                train=True,
                download=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307, ), (0.3081, ))
                ])),
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                '~/data',
                train=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307, ), (0.3081, ))
                ])),
            batch_size=args.test_batch_size,
            shuffle=True,
            **kwargs)

        self.model = Net()
        if args.cuda:
            self.model.cuda()

        self.optimizer = optim.SGD(
            self.model.parameters(), lr=args.lr, momentum=args.momentum)
        self.args = args

    def _train_iteration(self):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()

    def _test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                if self.args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = self.model(data)
                # sum up batch loss
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                # get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(
                    target.data.view_as(pred)).long().cpu().sum()

        test_loss = test_loss / len(self.test_loader.dataset)
        accuracy = correct.item() / len(self.test_loader.dataset)
        return {"mean_loss": test_loss, "mean_accuracy": accuracy}

    def _train(self):
        self._train_iteration()
        return self._test()

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(checkpoint_path)


if __name__ == "__main__":
    datasets.MNIST('~/data', train=True, download=True)
    args = parser.parse_args()

    import numpy as np
    import ray
    from ray import tune
    from ray.tune.schedulers import HyperBandScheduler

    ray.init()
    sched = HyperBandScheduler(
        time_attr="training_iteration", reward_attr="neg_mean_loss")
    tune.run(
        TrainMNIST,
        scheduler=sched,
        **{
            "stop": {
                "mean_accuracy": 0.95,
                "training_iteration": 1 if args.smoke_test else 20,
            },
            "resources_per_trial": {
                "cpu": 3,
                # "gpu": int(not args.no_cuda)
            },
            "num_samples": 1 if args.smoke_test else 20,
            "checkpoint_at_end": True,
            "config": {
                "args": args,
                "lr": tune.sample_from(
                    lambda spec: np.random.uniform(0.001, 0.1)),
                "momentum": tune.sample_from(
                    lambda spec: np.random.uniform(0.1, 0.9)),
            }
        })

```

运行结果：

```
/usr/bin/python3.5 /home/kangkang/PycharmProjects/ray/python/ray/tune/examples/mnist_pytorch_trainable.py
2019-04-24 20:36:49,395	INFO node.py:423 -- Process STDOUT and STDERR is being redirected to /tmp/ray/session_2019-04-24_20-36-49_26310/logs.
2019-04-24 20:36:49,500	INFO services.py:363 -- Waiting for redis server at 127.0.0.1:51648 to respond...
2019-04-24 20:36:49,616	INFO services.py:363 -- Waiting for redis server at 127.0.0.1:15287 to respond...
2019-04-24 20:36:49,617	INFO services.py:760 -- Starting Redis shard with 3.35 GB max memory.
2019-04-24 20:36:49,633	INFO services.py:1384 -- Starting the Plasma object store with 5.03 GB memory using /dev/shm.
2019-04-24 20:36:49,741	INFO tune.py:64 -- Did not find checkpoint file in /home/kangkang/ray_results/TrainMNIST.
2019-04-24 20:36:49,741	INFO tune.py:211 -- Starting a new experiment.
== Status ==
Using HyperBand: num_stopped=0 total_brackets=0
Round #0:
Resources requested: 0/8 CPUs, 0/0 GPUs
Memory usage on this node: 5.2/16.8 GB

== Status ==
Using HyperBand: num_stopped=0 total_brackets=3
Round #0:
  Bracket(Max Size (n)=5, Milestone (r)=81, completed=0.0%): {PENDING: 4, RUNNING: 1} 
  Bracket(Max Size (n)=8, Milestone (r)=27, completed=0.0%): {PENDING: 8} 
  Bracket(Max Size (n)=15, Milestone (r)=9, completed=0.0%): {PENDING: 7} 
Resources requested: 3/8 CPUs, 0/0 GPUs
Memory usage on this node: 5.4/16.8 GB
Result logdir: /home/kangkang/ray_results/TrainMNIST
Number of trials: 20 ({'RUNNING': 1, 'PENDING': 19})
PENDING trials:
 - TrainMNIST_0_lr=0.011026,momentum=0.79764:	PENDING
 - TrainMNIST_1_lr=0.039852,momentum=0.23715:	PENDING
 - TrainMNIST_3_lr=0.017233,momentum=0.20601:	PENDING
 - TrainMNIST_4_lr=0.076475,momentum=0.31586:	PENDING
 - TrainMNIST_5_lr=0.020496,momentum=0.10799:	PENDING
 - TrainMNIST_6_lr=0.081365,momentum=0.76826:	PENDING
 - TrainMNIST_7_lr=0.091581,momentum=0.51798:	PENDING
 - TrainMNIST_8_lr=0.013477,momentum=0.27381:	PENDING
 - TrainMNIST_9_lr=0.046978,momentum=0.45581:	PENDING
 - TrainMNIST_10_lr=0.024256,momentum=0.65206:	PENDING
 - TrainMNIST_11_lr=0.032366,momentum=0.67866:	PENDING
 - TrainMNIST_12_lr=0.056358,momentum=0.71433:	PENDING
 - TrainMNIST_13_lr=0.072232,momentum=0.30116:	PENDING
 - TrainMNIST_14_lr=0.01279,momentum=0.22866:	PENDING
 - TrainMNIST_15_lr=0.071809,momentum=0.66429:	PENDING
 - TrainMNIST_16_lr=0.086842,momentum=0.53263:	PENDING
2019-04-24 20:36:51,069	WARNING util.py:62 -- The `start_trial` operation took 1.1444847583770752 seconds to complete, which may be a performance bottleneck.
 - TrainMNIST_17_lr=0.054844,momentum=0.41178:	PENDING
 - TrainMNIST_18_lr=0.064185,momentum=0.72111:	PENDING
 - TrainMNIST_19_lr=0.08816,momentum=0.69577:	PENDING
RUNNING trials:
 - TrainMNIST_2_lr=0.019918,momentum=0.85528:	RUNNING

Result for TrainMNIST_4_lr=0.076475,momentum=0.31586:
  date: 2019-04-24_20-37-14
  done: true
  experiment_id: fc8592228c364db3b08483f9edd63782
  hostname: kangkang-1994
  iterations_since_restore: 1
  mean_accuracy: 0.9701
  mean_loss: 0.09734744338989258
  neg_mean_loss: -0.09734744338989258
  node_ip: 192.168.4.102
  pid: 26357
  time_since_restore: 23.453298568725586
  time_this_iter_s: 23.453298568725586
  time_total_s: 23.453298568725586
  timestamp: 1556109434
  timesteps_since_restore: 0
  training_iteration: 1
  ......
  ......
  ......
  == Status ==
Using HyperBand: num_stopped=0 total_brackets=3
Round #0:
  Bracket(Max Size (n)=5, Milestone (r)=81, completed=100.0%): {TERMINATED: 5} 
  Bracket(Max Size (n)=8, Milestone (r)=27, completed=0.5%): {RUNNING: 1, TERMINATED: 7} 
  Bracket(Max Size (n)=5, Milestone (r)=36, completed=0.2%): {TERMINATED: 7} 
Resources requested: 3/8 CPUs, 0/0 GPUs
Memory usage on this node: 5.4/16.8 GB
Result logdir: /home/kangkang/ray_results/TrainMNIST
Number of trials: 20 ({'RUNNING': 1, 'TERMINATED': 19})
RUNNING trials:
 - TrainMNIST_7_lr=0.091581,momentum=0.51798:	RUNNING
TERMINATED trials:
 - TrainMNIST_0_lr=0.011026,momentum=0.79764:	TERMINATED, [3 CPUs, 0 GPUs], [pid=26360], 22 s, 1 iter, 0.124 loss, 0.964 acc
 - TrainMNIST_1_lr=0.039852,momentum=0.23715:	TERMINATED, [3 CPUs, 0 GPUs], [pid=26358], 22 s, 1 iter, 0.122 loss, 0.962 acc
 - TrainMNIST_2_lr=0.019918,momentum=0.85528:	TERMINATED, [3 CPUs, 0 GPUs], [pid=26353], 23 s, 1 iter, 0.0872 loss, 0.973 acc
 - TrainMNIST_3_lr=0.017233,momentum=0.20601:	TERMINATED, [3 CPUs, 0 GPUs], [pid=26356], 44 s, 2 iter, 0.126 loss, 0.962 acc
 - TrainMNIST_4_lr=0.076475,momentum=0.31586:	TERMINATED, [3 CPUs, 0 GPUs], [pid=26357], 23 s, 1 iter, 0.0973 loss, 0.97 acc
 - TrainMNIST_5_lr=0.020496,momentum=0.10799:	TERMINATED, [3 CPUs, 0 GPUs], [pid=26354], 46 s, 2 iter, 0.125 loss, 0.962 acc
 - TrainMNIST_6_lr=0.081365,momentum=0.76826:	TERMINATED, [3 CPUs, 0 GPUs], [pid=762], 24 s, 1 iter, 0.121 loss, 0.965 acc
 - TrainMNIST_8_lr=0.013477,momentum=0.27381:	TERMINATED, [3 CPUs, 0 GPUs], [pid=787], 46 s, 2 iter, 0.137 loss, 0.958 acc
 - TrainMNIST_9_lr=0.046978,momentum=0.45581:	TERMINATED, [3 CPUs, 0 GPUs], [pid=704], 24 s, 1 iter, 0.0964 loss, 0.972 acc
 - TrainMNIST_10_lr=0.024256,momentum=0.65206:	TERMINATED, [3 CPUs, 0 GPUs], [pid=26355], 23 s, 1 iter, 0.107 loss, 0.968 acc
 - TrainMNIST_11_lr=0.032366,momentum=0.67866:	TERMINATED, [3 CPUs, 0 GPUs], [pid=734], 24 s, 1 iter, 0.0966 loss, 0.971 acc
 - TrainMNIST_12_lr=0.056358,momentum=0.71433:	TERMINATED, [3 CPUs, 0 GPUs], [pid=26359], 23 s, 1 iter, 0.102 loss, 0.97 acc
 - TrainMNIST_13_lr=0.072232,momentum=0.30116:	TERMINATED, [3 CPUs, 0 GPUs], [pid=1320], 21 s, 1 iter, 0.0965 loss, 0.969 acc
 - TrainMNIST_14_lr=0.01279,momentum=0.22866:	TERMINATED, [3 CPUs, 0 GPUs], [pid=1254], 50 s, 2 iter, 0.147 loss, 0.956 acc
 - TrainMNIST_15_lr=0.071809,momentum=0.66429:	TERMINATED, [3 CPUs, 0 GPUs], [pid=1312], 21 s, 1 iter, 0.0873 loss, 0.973 acc
 - TrainMNIST_16_lr=0.086842,momentum=0.53263:	TERMINATED, [3 CPUs, 0 GPUs], [pid=1344], 21 s, 1 iter, 0.097 loss, 0.971 acc
 - TrainMNIST_17_lr=0.054844,momentum=0.41178:	TERMINATED, [3 CPUs, 0 GPUs], [pid=25339], 22 s, 1 iter, 0.094 loss, 0.971 acc
 - TrainMNIST_18_lr=0.064185,momentum=0.72111:	TERMINATED, [3 CPUs, 0 GPUs], [pid=1292], 28 s, 1 iter, 0.0931 loss, 0.973 acc
 - TrainMNIST_19_lr=0.08816,momentum=0.69577:	TERMINATED, [3 CPUs, 0 GPUs], [pid=25365], 23 s, 1 iter, 0.0913 loss, 0.973 acc

Result for TrainMNIST_7_lr=0.091581,momentum=0.51798:
  date: 2019-04-24_20-41-35
  done: true
  experiment_id: 3bfd303ffb3a4b81b5917919c2ad388b
  hostname: kangkang-1994
  iterations_since_restore: 1
  mean_accuracy: 0.9655
  mean_loss: 0.11476650772094726
  neg_mean_loss: -0.11476650772094726
  node_ip: 192.168.4.102
  pid: 754
  time_since_restore: 22.205914735794067
  time_this_iter_s: 22.205914735794067
  time_total_s: 22.205914735794067
  timestamp: 1556109695
  timesteps_since_restore: 0
  training_iteration: 1
  
2019-04-24 20:41:35,979	INFO ray_trial_executor.py:178 -- Destroying actor for trial TrainMNIST_7_lr=0.091581,momentum=0.51798. If your trainable is slow to initialize, consider setting reuse_actors=True to reduce actor creation overheads.
== Status ==
Using HyperBand: num_stopped=0 total_brackets=3
Round #0:
  Bracket(Max Size (n)=5, Milestone (r)=81, completed=100.0%): {TERMINATED: 5} 
  Bracket(Max Size (n)=3, Milestone (r)=81, completed=100.0%): {TERMINATED: 8} 
  Bracket(Max Size (n)=5, Milestone (r)=36, completed=0.2%): {TERMINATED: 7} 
Resources requested: 0/8 CPUs, 0/0 GPUs
Memory usage on this node: 5.2/16.8 GB
Result logdir: /home/kangkang/ray_results/TrainMNIST
Number of trials: 20 ({'TERMINATED': 20})
TERMINATED trials:
 - TrainMNIST_0_lr=0.011026,momentum=0.79764:	TERMINATED, [3 CPUs, 0 GPUs], [pid=26360], 22 s, 1 iter, 0.124 loss, 0.964 acc
 - TrainMNIST_1_lr=0.039852,momentum=0.23715:	TERMINATED, [3 CPUs, 0 GPUs], [pid=26358], 22 s, 1 iter, 0.122 loss, 0.962 acc
 - TrainMNIST_2_lr=0.019918,momentum=0.85528:	TERMINATED, [3 CPUs, 0 GPUs], [pid=26353], 23 s, 1 iter, 0.0872 loss, 0.973 acc
 - TrainMNIST_3_lr=0.017233,momentum=0.20601:	TERMINATED, [3 CPUs, 0 GPUs], [pid=26356], 44 s, 2 iter, 0.126 loss, 0.962 acc
 - TrainMNIST_4_lr=0.076475,momentum=0.31586:	TERMINATED, [3 CPUs, 0 GPUs], [pid=26357], 23 s, 1 iter, 0.0973 loss, 0.97 acc
 - TrainMNIST_5_lr=0.020496,momentum=0.10799:	TERMINATED, [3 CPUs, 0 GPUs], [pid=26354], 46 s, 2 iter, 0.125 loss, 0.962 acc
 - TrainMNIST_6_lr=0.081365,momentum=0.76826:	TERMINATED, [3 CPUs, 0 GPUs], [pid=762], 24 s, 1 iter, 0.121 loss, 0.965 acc
 - TrainMNIST_7_lr=0.091581,momentum=0.51798:	TERMINATED, [3 CPUs, 0 GPUs], [pid=754], 22 s, 1 iter, 0.115 loss, 0.966 acc
 - TrainMNIST_8_lr=0.013477,momentum=0.27381:	TERMINATED, [3 CPUs, 0 GPUs], [pid=787], 46 s, 2 iter, 0.137 loss, 0.958 acc
 - TrainMNIST_9_lr=0.046978,momentum=0.45581:	TERMINATED, [3 CPUs, 0 GPUs], [pid=704], 24 s, 1 iter, 0.0964 loss, 0.972 acc
 - TrainMNIST_10_lr=0.024256,momentum=0.65206:	TERMINATED, [3 CPUs, 0 GPUs], [pid=26355], 23 s, 1 iter, 0.107 loss, 0.968 acc
 - TrainMNIST_11_lr=0.032366,momentum=0.67866:	TERMINATED, [3 CPUs, 0 GPUs], [pid=734], 24 s, 1 iter, 0.0966 loss, 0.971 acc
 - TrainMNIST_12_lr=0.056358,momentum=0.71433:	TERMINATED, [3 CPUs, 0 GPUs], [pid=26359], 23 s, 1 iter, 0.102 loss, 0.97 acc
 - TrainMNIST_13_lr=0.072232,momentum=0.30116:	TERMINATED, [3 CPUs, 0 GPUs], [pid=1320], 21 s, 1 iter, 0.0965 loss, 0.969 acc
 - TrainMNIST_14_lr=0.01279,momentum=0.22866:	TERMINATED, [3 CPUs, 0 GPUs], [pid=1254], 50 s, 2 iter, 0.147 loss, 0.956 acc
 - TrainMNIST_15_lr=0.071809,momentum=0.66429:	TERMINATED, [3 CPUs, 0 GPUs], [pid=1312], 21 s, 1 iter, 0.0873 loss, 0.973 acc
 - TrainMNIST_16_lr=0.086842,momentum=0.53263:	TERMINATED, [3 CPUs, 0 GPUs], [pid=1344], 21 s, 1 iter, 0.097 loss, 0.971 acc
 - TrainMNIST_17_lr=0.054844,momentum=0.41178:	TERMINATED, [3 CPUs, 0 GPUs], [pid=25339], 22 s, 1 iter, 0.094 loss, 0.971 acc
 - TrainMNIST_18_lr=0.064185,momentum=0.72111:	TERMINATED, [3 CPUs, 0 GPUs], [pid=1292], 28 s, 1 iter, 0.0931 loss, 0.973 acc
 - TrainMNIST_19_lr=0.08816,momentum=0.69577:	TERMINATED, [3 CPUs, 0 GPUs], [pid=25365], 23 s, 1 iter, 0.0913 loss, 0.973 acc


Process finished with exit code 0

```






















