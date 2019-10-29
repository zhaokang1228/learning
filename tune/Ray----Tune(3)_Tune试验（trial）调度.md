本篇介绍Tune中trial调度算法。主要介绍了Population Based Training (PBT)、Asynchronous HyperBand、 HyperBand及实现、Median Stopping Rule四种试验调度。


默认情况下，Tune使用FIFOSchscheduler类按顺序安排试验。不过，您还可以指定一个自定义调度算法，该算法可以提前停止试验或干扰参数。

```
tune.run( ... , scheduler=AsyncHyperBandScheduler())
```
Tune包括早期停止算法的分布式实现，例如中值停止规则，HyperBand和HyperBand的异步版本。这些算法具有很高的资源效率，在很多情况下都优于贝叶斯优化方法。目前，所有调度程序都采用一个度量标准`metric`，该度量标准是您的Trainable的结果字典中返回的值，并且根据模式`mode`而最大化或最小化。(老版本中是所有调度程序都接受一个`reward_attr`，即目标参数标准，该值假定为**最大化**。)。由于此篇写的比较早，故后边的例子主要是按照老版本进行说明的。

目前可用的试验调度程序:

 - Population Based Training (PBT)
 - Asynchronous HyperBand
 - HyperBand--------HyperBand Implementation Details
 - Median Stopping Rule
下面介绍这四种调度算法。

## １．Population Based Training (PBT)
Population Based Training (PBT)的分布式实现。这可以通过设置`tune.run`的`scheduler`参数来启动，例如：
```
pbt_scheduler = PopulationBasedTraining(
        time_attr='time_total_s',
        reward_attr='mean_accuracy',
        perturbation_interval=600.0,
        hyperparam_mutations={
            "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            "alpha": lambda: random.uniform(0.0, 1.0),
            ...
        })
tune.run( ... , scheduler=pbt_scheduler)
```
启用PBT调度程序后，每个试验变体都将被视为总体的成员。 定期检查性能最佳试验（这需要Trainable支持检查点，故用此调度的训练使用基于函数API时需要在函数中加上检查点相关代码）。 **低性能的试验克隆了表现最佳者的试验的参数并添加配置，以期发现更好的参数变体。**
可以通过运行[一个小 PBT 例子](https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/pbt_example.py)了解PBT的工作。当在PBT模型下进行训练时，单个试验可能会在其生命周期中看到许多不同的超参数，并将其记录在`result.json`文件。单个实验过程中优化LR调度的PBT生成了下图：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190413201420547.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzI1NTk2Mg==,size_16,color_FFFFFF,t_70)

```
class ray.tune.schedulers.PopulationBasedTraining(time_attr='time_total_s', 
		reward_attr='episode_reward_mean', perturbation_interval=60.0, 
		hyperparam_mutations={}, resample_probability=0.25, custom_explore_fn=None)
```
[PopulationBasedTraining源码](https://ray.readthedocs.io/en/latest/_modules/ray/tune/schedulers/pbt.html#PopulationBasedTraining)，[PBT论文](https://deepmind.com/blog/population-based-training-neural-networks)
PBT并行地训练一组模型(或代理)。性能较差的模型会周期性地克隆性能最好的模型的状态（参数），并对它们的超参数进行随机突变，希望能够得到超过当前性能最好的模型。

与其他超参数搜索算法不同，PBT在训练过程中会改变超参数。这支持非常快的超参数发现，并自动发现良好的退火调度。
此Tune PBT实现将所有添加的试验视为PBT总体的一部分。如果试验的数量超过了集群的能力，将对试验进行时间复用，以平衡整个实验的训练进展。

**参数：**
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
custom_explore_fn (func) –您还可以指定一个定制的探测函数。在应用了hyperparam_mutations的内
	置扰动之后，这个函数被调用为f(config)，并且应该返回根据需要更新的配置。您必须指定至少一个
	hyperparam_mutations或custom_explore_fn。
log_config (bool) – 是否在每个训练运行过程时将每个模型的ray配置记录到local_dir。
	 允许重建调度的配置信息。
```
Example

```
>>> pbt = PopulationBasedTraining(
>>>     time_attr="training_iteration",
>>>     reward_attr="episode_reward_mean",
>>>     perturbation_interval=10,  # every 10 `time_attr` units
>>>                                # (training_iterations in this case)   每一个time_attr 进行10次
>>>     hyperparam_mutations={
>>>         # Perturb factor1 by scaling it by 0.8 or 1.2. Resampling
>>>         # resets it to a value sampled from the lambda function.
>>>         "factor_1": lambda: random.uniform(0.0, 20.0),
>>>         # Perturb factor2 by changing it to an adjacent value, e.g.
>>>         # 10 -> 1 or 10 -> 100. Resampling will choose at random.
>>>         "factor_2": [1, 10, 100, 1000, 10000],
>>>     })
>>> run_experiments({...}, scheduler=pbt)
```
## ２．Asynchronous（异步） HyperBand
通过设置`tune.run`的`scheduler`参数，可以使用HyperBand 调度器的异步版本。

```
async_hb_scheduler = AsyncHyperBandScheduler(
    time_attr='training_iteration',
    reward_attr='episode_reward_mean',
    max_t=100,
    grace_period=10,
    reduction_factor=3,
    brackets=3)
tune.run( ... , scheduler=async_hb_scheduler)
```
与原始版本的HyperBand相比，该实现提供了更好的并行性，并避免了消除过程中的掉队问题。在[async_hyperband_example.py](https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/async_hyperband_example.py)中可以找到一个这样的例子。我们建议在标准HyperBand 调度程序上使用此功能。

```
class ray.tune.schedulers.AsyncHyperBandScheduler(time_attr='training_iteration', 
					reward_attr='episode_reward_mean', max_t=100, grace_period=10, 
					reduction_factor=3, brackets=3)
```
[AsyncHyperBandScheduler类源码](https://ray.readthedocs.io/en/latest/_modules/ray/tune/schedulers/async_hyperband.html#AsyncHyperBandScheduler)，[Asynchronous HyperBand论文](https://openreview.net/forum?id=S1Y7OOlRZ)
这应该提供与 HyperBand类似的理论性能，但是避免了 HyperBand所面临的问题（消除过程中的掉队问题）。一个实现细节是，当使用多个方括号时，试验分配到方括号是随机进行的，具有超过软最大值的概率。
**参数：**

```
time_attr (str) – 用于比较时间的训练结果。注意，您可以传递一些非时间的东西，比如
	training_iteration作为进度的度量，惟一的要求是属性应该单调地增加。
reward_attr (str) – 训练结果目标值属性。与time_attr一样，这可以引用任何客观值。停止过程将使用
	此属性。
max_t (float) – 每次训练的最大时间单位。max_t时间单位(由time_attr决定)通过后，将停止试验。
	此参数相当于tune.run()中的参数stop的作用。
grace_period (float) – 至少在这个时候停止试验。这些单元与time_attr所命名的属性相同。
reduction_factor (float) –用于设定减半率和用量。这只是一个无单位标量。
brackets (int) –括号数量。 每个括号具有不同的减半率，由减少系数指定。
```

## ３．HyperBand
请注意，HyperBand调度程序要求您的trainable支持检查点，这在Tune用户指南中进行了描述。检查点使调度程序能够将许多并发测试多路复用到一个有限大小的集群上。

Tune还实现了HyperBand的标准版本。你可以这样使用:

```
tune.run( ... , scheduler=HyperBandScheduler())
```
一个例子可以在[hyperband_example.py](https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/hyperband_example.py)中找到。一个这样的hyperband运行过程如下所示。

```
== Status ==
Using HyperBand: num_stopped=0 total_brackets=5
Round #0:
  Bracket(n=5, r=100, completed=80%): {'PAUSED': 4, 'PENDING': 1}
  Bracket(n=8, r=33, completed=23%): {'PAUSED': 4, 'PENDING': 4}
  Bracket(n=15, r=11, completed=4%): {'RUNNING': 2, 'PAUSED': 2, 'PENDING': 11}
  Bracket(n=34, r=3, completed=0%): {'RUNNING': 2, 'PENDING': 32}
  Bracket(n=81, r=1, completed=0%): {'PENDING': 38}
Resources used: 4/4 CPUs, 0/0 GPUs
Result logdir: ~/ray_results/hyperband_test
PAUSED trials:
 - my_class_0_height=99,width=43:   PAUSED [pid=11664], 0 s, 100 ts, 97.1 rew
 - my_class_11_height=85,width=81:  PAUSED [pid=11771], 0 s, 33 ts, 32.8 rew
 - my_class_12_height=0,width=52:   PAUSED [pid=11785], 0 s, 33 ts, 0 rew
 - my_class_19_height=44,width=88:  PAUSED [pid=11811], 0 s, 11 ts, 5.47 rew
 - my_class_27_height=96,width=84:  PAUSED [pid=11840], 0 s, 11 ts, 12.5 rew
  ... 5 more not shown
PENDING trials:
 - my_class_10_height=12,width=25:  PENDING
 - my_class_13_height=90,width=45:  PENDING
 - my_class_14_height=69,width=45:  PENDING
 - my_class_15_height=41,width=11:  PENDING
 - my_class_16_height=57,width=69:  PENDING
  ... 81 more not shown
RUNNING trials:
 - my_class_23_height=75,width=51:  RUNNING [pid=11843], 0 s, 1 ts, 1.47 rew
 - my_class_26_height=16,width=48:  RUNNING
 - my_class_31_height=40,width=10:  RUNNING
 - my_class_53_height=28,width=96:  RUNNING
```
**控制台打印信息：**

    Round　 大试验块，里边最多有五个试验块（Bracket），最多１４３个试验样本。　
            运行的时候是按大实验块的编号顺序执行，每次最多运行５个试验块
    Bracket 试验块　里边从第一个实验块开始，依次存5,8,15,34,81 个试验（样本），
            其中样本数量按照前开始依次存入，只有最后一个试验快可以试验样本不填充满。
            具体　试验块（Bracket）个数(ｎ+1)是由`(5/(n+1))*3**n >= num_samples`,取ｎ最小满足此公式的整数，
            这个括号内可以容纳的训练个数为`(5/(n+1))*3**n`个（可以不容纳满即最后一个括号可以不容纳满）。
            当试验块（Bracket）个数大于５后，ｎ=n-1，重新开始计.
            
            Max Size (n)　表示当前试验块（取样）中没有运行结束的试验个数（当此试验块没有运行时，此值代表此试验块最大
           		能容纳多少试验）， 注：当此试验块里的试验运行结束后，此值代表此实验块最后运行结束的试验个数。
            Milestone (r)　表示试验将在下一次减半之前运行的迭代次数。最后依次增加达到max_t的大小。
            completed   表示近似进度指标。  
   		 Bracket(试验块)后边的大括号内代表当前试验块中试验的状态
   		 	（RUNNING　正在运行　PAUSED　等待　TERMINATED　运行终止）和对应的个数

```
class ray.tune.schedulers.HyperBandScheduler(time_attr='training_iteration', 
		reward_attr='episode_reward_mean', max_t=81)
```
[HyperBandScheduler类的源码](https://ray.readthedocs.io/en/latest/_modules/ray/tune/schedulers/hyperband.html#HyperBandScheduler)，[HyperBand论文](https://people.eecs.berkeley.edu/~kjamieson/hyperband.html)

HyperBandScheduler早期停止使用超带优化算法的试验。它将试验分为不同大小的括号，并定期在每个括号内提前停止性能较差的试验。
要使用这个Tune的HyperBand实现，您所需要做的就是指定一个试验可以运行max_t的最大时间长度、时间单位time_attr和报告的目标值reward_attr的名称。我们根据给定的值自动确定其他超带参数的合理值。

例如，要根据episode_mean_reward attr将试验限制为10分钟并提前停止，请构造：`HyperBand('time_total_s', 'episode_reward_mean', max_t=600)`
请注意，Tune的停止标准将与HyperBand的早期停止机制一起应用。
**参数：**

```
time_attr (str) – 训练结果用于比较时间。注意，您可以传递一些非时间的东西，比如
	training_iteration作为进度的度量，惟一的要求是属性应该单调地增加。
reward_attr (str) – 训练结果目标值属性。与time_attr一样，这可以引用任何客观值。停止过程将使
	用此属性。
max_t (int) –每次试验的最大时间单位。max_t时间单位(由time_attr决定)通过后，将停止试验。调度
	程序将在此时间通过后终止试验。注意，这与原始HyperBand论文中提到的max_t的语义不同。
```
#### HyperBand实现细节
实现细节可能会稍微偏离理论，但重点是提高可用性。注:R、s_max、eta为论文给出的HyperBand参数。详情查看[本论文](https://people.eecs.berkeley.edu/~kjamieson/hyperband.html)了解上下文。

 1. `s_max`(表示`number of brackets - 1`)和`eta`(表示下采样率)都是固定的。在许多实际设置中，`R`可以设置得相当大，比如`R >= 200`, `R`表示某个资源单元和通常训练迭代的次数。为了简单起见，假设`eta = 3`。在`R = 200`和`R = 1000`之间改变`R`，就会产生一个巨大的范围，需要大量的试验来填充所有的括号。
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20190413221432199.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzI1NTk2Mg==,size_16,color_FFFFFF,t_70)
  另一方面，在R = 300处保持R不变，并且改变eta也会导致HyperBand构型，这不是很直观:
  ![在这里插入图片描述](https://img-blog.csdnimg.cn/20190413221639901.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzI1NTk2Mg==,size_16,color_FFFFFF,t_70)
该实现采用与论文给出的示例相同的配置，并公开`max_t`, `max_t`不是论文中的参数。
 2. 后文中计算`n_0`的例子实际上与论文给出的算法略有不同。在这个实现中，我们根据论文实现`n_0`(如下例中为n):
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190413222234499.png)
 3.还有一些具体的实现细节，比如如何将试验放在括号中，这些在本文中没有涉及。该实现将试验按照较小的括号放在括号内——这意味着试验数量越少，早期停止的次数就越少。
## ４．Median Stopping Rule（中值停止规则）
中值停止规则实现了一个简单的策略，即如果一个试验的性能在相同时间点低于其他试验的中值，则停止该试验。你可以设置`scheduler`参数如下:

```
tune.run( ... , scheduler=MedianStoppingRule())
```

```
class ray.tune.schedulers.MedianStoppingRule(time_attr='time_total_s', reward_attr='episode_reward_mean', grace_period=60.0, min_samples_required=3, hard_stop=True, verbose=True)
```
[MedianStoppingRule类源代码](https://ray.readthedocs.io/en/latest/_modules/ray/tune/schedulers/median_stopping_rule.html#MedianStoppingRule),[Median Stopping Rule论文](https://research.google.com/pubs/pub46180.html)
**参数：**
```
time_attr (str) –训练结果用于比较时间。注意，您可以传递一些非时间的东西，比如
	training_iteration作为进度的度量，惟一的要求是属性应该单调地增加。
reward_attr (str) –训练结果目标值属性。与time_attr一样，这可以引用任何应该随时间增长的客
	观值。
grace_period (float) –至少在这个时候停止试验。这些单元与time_attr所命名的属性
	相同。
min_samples_required (int) –计算中值的最小样本。
hard_stop (bool) – 如果错误，暂停试验而不是停止试验。当所有其他试验完成时，暂停的试验将重新开始并允许运行FIFO。
verbose (bool) – 如果为真，将在每次试验报告时输出中值和最佳结果。默认值为True。
```


此篇主要参考Ray官网，如有错误，请阅读者提出指正，谢谢！
原英文链接：https://ray.readthedocs.io/en/latest/tune-schedulers.html


